"""RA-DC-PPO integration for the pinned official NavRL training code."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.tensordict import TensorDict
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from radcppo_navrl import (
    ConservativeRiskSelector,
    ExponentialCorrectionSmoother,
    InterventionGatedBetaActor,
    MonotonicQuantileCritic,
    PlainBetaActor,
    ReachabilityTeacher,
    ReachabilityTeacherConfig,
    counterfactual_calibration_loss,
    normalized_safety_td_target,
    observation_safety_cost,
    quantile_huber_loss,
    upper_cvar,
    upper_tail_dominance_margin,
)
from utils import GAE, IndependentBeta, ValueNorm, make_batch, make_mlp, vec_to_world


def _make_feature_extractor(
    device,
    *,
    cnn_key: str = "_cnn_feature",
    dynamic_key: str = "_dynamic_obstacle_feature",
    feature_key: str = "_feature",
):
    lidar_network = nn.Sequential(
        nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
        nn.ELU(),
        nn.LazyConv2d(
            out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]
        ),
        nn.ELU(),
        nn.LazyConv2d(
            out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]
        ),
        nn.ELU(),
        Rearrange("n c w h -> n (c w h)"),
        nn.LazyLinear(128),
        nn.LayerNorm(128),
    ).to(device)
    dynamic_network = nn.Sequential(
        Rearrange("n c w h -> n (c w h)"), make_mlp([128, 64])
    ).to(device)
    return TensorDictSequential(
        TensorDictModule(
            lidar_network,
            [("agents", "observation", "lidar")],
            [cnn_key],
        ),
        TensorDictModule(
            dynamic_network,
            [("agents", "observation", "dynamic_obstacle")],
            [dynamic_key],
        ),
        CatTensors(
            [
                cnn_key,
                ("agents", "observation", "state"),
                dynamic_key,
            ],
            feature_key,
            del_keys=False,
        ),
        TensorDictModule(make_mlp([256, 256]), [feature_key], [feature_key]),
    ).to(device)


class RADCPPO(TensorDictModuleBase):
    """NavRL PPO with projection distillation and a distributional safety critic."""

    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.rollout_mode = "actor"
        method_cfg = cfg.method
        self.feature_extractor = _make_feature_extractor(device)
        self.use_decoupled_safety_encoder = bool(
            getattr(method_cfg, "use_decoupled_safety_encoder", False)
        )
        self.freeze_pretrained_task_policy = bool(
            getattr(method_cfg, "freeze_pretrained_task_policy", False)
        )
        self.safety_feature_extractor = (
            _make_feature_extractor(
                device,
                cnn_key="_safety_cnn_feature",
                dynamic_key="_safety_dynamic_obstacle_feature",
                feature_key="_safety_feature",
            )
            if self.use_decoupled_safety_encoder
            else None
        )

        self.n_agents, self.action_dim = action_spec.shape
        actor_adapter_cfg = cfg.method.actor_adapter
        if bool(actor_adapter_cfg.enabled):
            self.actor_head = InterventionGatedBetaActor(
                self.action_dim,
                maximum_logit_bias=float(actor_adapter_cfg.maximum_logit_bias),
                gate_mode=str(actor_adapter_cfg.gate_mode),
                gate_temperature=float(
                    getattr(actor_adapter_cfg, "gate_temperature", 1.0)
                ),
                adapter_parameterization=str(
                    getattr(actor_adapter_cfg, "parameterization", "logit")
                ),
                maximum_mean_shift=float(
                    getattr(actor_adapter_cfg, "maximum_mean_shift", 0.05)
                ),
            )
        else:
            self.actor_head = PlainBetaActor(self.action_dim)
        actor_feature_keys = (
            ["_feature", "_safety_feature"]
            if self.use_decoupled_safety_encoder
            else ["_feature"]
        )
        self.actor = ProbabilisticActor(
            TensorDictModule(
                self.actor_head,
                actor_feature_keys,
                ["alpha", "beta", "method_safety_gate", "method_safety_residual"],
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True,
        ).to(device)
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]
        ).to(device)
        self.value_norm = ValueNorm(1).to(device)

        teacher_cfg = ReachabilityTeacherConfig(
            horizon_s=float(method_cfg.teacher.horizon_s),
            horizon_steps=int(method_cfg.teacher.horizon_steps),
            max_acceleration_mps2=float(method_cfg.teacher.max_acceleration_mps2),
            max_speed_mps=float(cfg.actor.action_limit),
            static_point_budget=int(method_cfg.teacher.static_point_budget),
            counterfactual_count=int(method_cfg.counterfactual_count),
            rollout_model=str(method_cfg.teacher.rollout_model),
        )
        self.teacher = ReachabilityTeacher(teacher_cfg).to(device)
        selector_cfg = getattr(method_cfg, "risk_selector", None)
        self.risk_selector = ConservativeRiskSelector(
            action_limit=float(cfg.actor.action_limit),
            action_step_mps=float(
                getattr(selector_cfg, "action_step_mps", 0.08)
            ),
            activation_threshold=float(
                getattr(selector_cfg, "activation_threshold", 0.20)
            ),
            minimum_risk_reduction=float(
                getattr(selector_cfg, "minimum_risk_reduction", 0.015)
            ),
            deviation_penalty=float(
                getattr(selector_cfg, "deviation_penalty", 0.002)
            ),
        ).to(device)
        self.risk_projection_blend = float(
            getattr(selector_cfg, "projection_blend", 1.0)
        )
        if not 0.0 < self.risk_projection_blend <= 1.0:
            raise ValueError("risk projection blend must be in (0, 1]")
        self.risk_correction_smoother = ExponentialCorrectionSmoother(
            coefficient=float(
                getattr(selector_cfg, "projection_smoothing", 0.8)
            )
        )
        self.require_tail_dominance = bool(
            getattr(selector_cfg, "require_tail_dominance", False)
        )
        self.safety_critic = MonotonicQuantileCritic(
            num_quantiles=int(method_cfg.num_quantiles),
            hidden_dim=int(method_cfg.safety_critic_hidden_dim),
        ).to(device)
        self.register_buffer(
            "safety_lambda",
            torch.tensor(float(method_cfg.initial_safety_lambda), device=device),
        )
        self.register_buffer(
            "training_update_count",
            torch.zeros((), dtype=torch.long, device=device),
        )
        self.safety_warmup_updates = int(
            getattr(method_cfg, "safety_warmup_updates", 0)
        )
        if self.safety_warmup_updates < 0:
            raise ValueError("safety_warmup_updates must be non-negative")
        self._auxiliary_scale = 1.0

        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)
        safety_features = self.safety_features(dummy_input)
        self.safety_critic(
            safety_features, dummy_input["method_executed_action_local"]
        )

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)
        if self.safety_feature_extractor is not None:
            self.safety_feature_extractor.apply(init_)
        self.safety_critic.apply(init_)
        self.safety_critic.reset_output_prior(
            prior_probability=float(method_cfg.initial_safety_probability)
        )
        self.target_safety_critic = copy.deepcopy(self.safety_critic).requires_grad_(False)
        self.target_safety_feature_extractor = (
            copy.deepcopy(self.safety_feature_extractor).requires_grad_(False)
            if self.safety_feature_extractor is not None
            else None
        )
        self.actor_head.reset_safety_adapter(
            gate_bias=float(method_cfg.actor_adapter.initial_gate_bias)
        )

        if self.freeze_pretrained_task_policy:
            self.feature_extractor.requires_grad_(False)
            self.actor_head.alpha_layer.requires_grad_(False)
            self.actor_head.beta_layer.requires_grad_(False)

        feature_parameters = [
            parameter
            for parameter in self.feature_extractor.parameters()
            if parameter.requires_grad
        ]
        self.feature_extractor_optim = (
            torch.optim.Adam(
                feature_parameters, lr=cfg.feature_extractor.learning_rate
            )
            if feature_parameters
            else None
        )
        task_actor_parameters = [
            parameter
            for module in (self.actor_head.alpha_layer, self.actor_head.beta_layer)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        adapter_parameters = (
            [
                parameter
                for module in (
                    self.actor_head.gate_layer,
                    self.actor_head.residual_layer,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            if bool(actor_adapter_cfg.enabled)
            else []
        )
        classified_parameter_ids = {
            id(parameter)
            for parameter in task_actor_parameters + adapter_parameters
        }
        unclassified_actor_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if parameter.requires_grad and id(parameter) not in classified_parameter_ids
        ]
        if unclassified_actor_parameters:
            raise RuntimeError(
                "Actor optimizer partition omitted trainable parameters"
            )
        if not task_actor_parameters and not adapter_parameters:
            raise ValueError("The actor configuration leaves no trainable parameters")
        actor_parameter_groups = []
        if task_actor_parameters:
            actor_parameter_groups.append(
                {
                    "params": task_actor_parameters,
                    "lr": float(cfg.actor.learning_rate),
                    "name": "task_actor",
                }
            )
        if adapter_parameters:
            adapter_learning_rate = float(
                getattr(
                    actor_adapter_cfg,
                    "learning_rate",
                    cfg.actor.learning_rate,
                )
            )
            actor_parameter_groups.append(
                {
                    "params": adapter_parameters,
                    "lr": adapter_learning_rate,
                    "configured_lr": adapter_learning_rate,
                    "name": "safety_adapter",
                }
            )
        self.actor_optim = torch.optim.Adam(actor_parameter_groups)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic.learning_rate
        )
        safety_parameters = list(self.safety_critic.parameters())
        if self.safety_feature_extractor is not None:
            safety_parameters.extend(self.safety_feature_extractor.parameters())
        self.safety_critic_optim = torch.optim.Adam(
            safety_parameters, lr=method_cfg.safety_critic_learning_rate
        )
        self.sync_training_schedule()

    @torch.no_grad()
    def initialize_safety_feature_extractor_from_task(self) -> None:
        """Warm-start the independent safety encoder from the loaded NavRL encoder."""

        if self.safety_feature_extractor is None:
            return
        self.safety_feature_extractor.load_state_dict(
            self.feature_extractor.state_dict(), strict=True
        )
        self.target_safety_feature_extractor.load_state_dict(
            self.safety_feature_extractor.state_dict(), strict=True
        )

    def safety_features(self, tensordict):
        if self.safety_feature_extractor is None:
            return tensordict["_feature"]
        self.safety_feature_extractor(tensordict)
        return tensordict["_safety_feature"]

    def target_safety_features(self, tensordict):
        if self.target_safety_feature_extractor is None:
            return tensordict["_feature"]
        self.target_safety_feature_extractor(tensordict)
        return tensordict["_safety_feature"]

    def set_rollout_mode(self, mode: str) -> None:
        if mode not in {
            "actor",
            "risk_guided",
            "risk_projected",
            "risk_projected_blend",
            "risk_projected_smooth",
            "projected",
        }:
            raise ValueError(f"Unknown rollout mode: {mode}")
        self.rollout_mode = mode
        self.risk_correction_smoother.reset()

    def _set_safety_adapter_active(self, active: bool) -> None:
        if hasattr(self.actor_head, "set_runtime_adapter_active"):
            self.actor_head.set_runtime_adapter_active(active)
        for group in self.actor_optim.param_groups:
            if group.get("name") == "safety_adapter":
                group["lr"] = float(group["configured_lr"]) if active else 0.0

    def sync_training_schedule(self) -> bool:
        active = int(self.training_update_count.item()) >= self.safety_warmup_updates
        self._set_safety_adapter_active(active)
        self._auxiliary_scale = float(active)
        return active

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        safety_features = self.safety_features(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        action_limit = float(self.cfg.actor.action_limit)
        normalized = tensordict["agents", "action_normalized"]
        nominal_local = 2.0 * normalized * action_limit - action_limit
        adapted_mean_local = (
            2.0
            * (tensordict["alpha"] / (tensordict["alpha"] + tensordict["beta"]))
            * action_limit
            - action_limit
        )
        base_mean_local = self.actor_head.base_mean_action(
            tensordict["_feature"], action_limit
        )
        observation = tensordict["agents", "observation"]
        teacher = self.teacher(
            nominal_local,
            observation["state"],
            observation["lidar"],
            observation["dynamic_obstacle"],
        )
        tensordict["method_nominal_action_local"] = nominal_local
        tensordict["method_actor_mean_action_local"] = adapted_mean_local
        tensordict["method_base_action_local"] = base_mean_local
        tensordict["method_adapter_shift"] = (
            adapted_mean_local - base_mean_local
        ).norm(dim=-1, keepdim=True)
        tensordict["method_teacher_action_local"] = teacher.action
        tensordict["method_projection_gap"] = teacher.projection_gap
        tensordict["method_teacher_intervention"] = teacher.intervention.float()
        tensordict["method_teacher_feasible"] = teacher.feasible.float()
        tensordict["method_counterfactual_actions"] = teacher.candidate_actions
        tensordict["method_counterfactual_risks"] = teacher.candidate_risks

        risk_used = torch.zeros_like(teacher.feasible)
        risk_shift = torch.zeros_like(teacher.projection_gap)
        nominal_predicted_risk = torch.zeros_like(teacher.projection_gap)
        selected_predicted_risk = torch.zeros_like(teacher.projection_gap)
        predicted_risk_reduction = torch.zeros_like(teacher.projection_gap)
        tail_dominance_margin = torch.zeros_like(teacher.projection_gap)
        if self.rollout_mode == "risk_guided":
            risk_candidates = self.risk_selector.candidate_actions(base_mean_local)
            risk_features = safety_features.unsqueeze(-2).expand(
                *risk_candidates.shape[:-1], safety_features.shape[-1]
            )
            risk_quantiles = self.safety_critic(risk_features, risk_candidates)
            risk_measure = str(getattr(self.cfg.method, "risk_measure", "cvar"))
            if risk_measure == "mean":
                candidate_risks = risk_quantiles.mean(dim=-1, keepdim=True)
            elif risk_measure == "cvar":
                candidate_risks = upper_cvar(
                    risk_quantiles,
                    alpha=float(getattr(self.cfg.method, "cvar_alpha", 0.9)),
                )
            else:
                raise ValueError(f"Unknown risk_measure: {risk_measure}")
            selection = self.risk_selector.select(risk_candidates, candidate_risks)
            use_teacher = torch.zeros_like(teacher.feasible)
            executed_local = selection.action
            risk_used = selection.used.float()
            risk_shift = selection.shift
            nominal_predicted_risk = selection.nominal_risk
            selected_predicted_risk = selection.selected_risk
            predicted_risk_reduction = selection.risk_reduction
        elif self.rollout_mode in {
            "risk_projected",
            "risk_projected_blend",
            "risk_projected_smooth",
        }:
            risk_candidates = torch.stack((base_mean_local, teacher.action), dim=-2)
            risk_features = safety_features.unsqueeze(-2).expand(
                *risk_candidates.shape[:-1], safety_features.shape[-1]
            )
            risk_quantiles = self.safety_critic(risk_features, risk_candidates)
            risk_measure = str(getattr(self.cfg.method, "risk_measure", "cvar"))
            if risk_measure == "mean":
                candidate_risks = risk_quantiles.mean(dim=-1, keepdim=True)
            elif risk_measure == "cvar":
                candidate_risks = upper_cvar(
                    risk_quantiles,
                    alpha=float(getattr(self.cfg.method, "cvar_alpha", 0.9)),
                )
            else:
                raise ValueError(f"Unknown risk_measure: {risk_measure}")
            nominal_predicted_risk = candidate_risks[..., 0, :]
            teacher_predicted_risk = candidate_risks[..., 1, :]
            predicted_risk_reduction = (
                nominal_predicted_risk - teacher_predicted_risk
            )
            tail_dominance_margin = upper_tail_dominance_margin(
                risk_quantiles[..., 0, :],
                risk_quantiles[..., 1, :],
                alpha=float(getattr(self.cfg.method, "cvar_alpha", 0.9)),
            )
            gate_risk_reduction = (
                tail_dominance_margin
                if self.require_tail_dominance
                else predicted_risk_reduction
            )
            learned_gate = (
                (nominal_predicted_risk >= self.risk_selector.activation_threshold)
                & (
                    gate_risk_reduction
                    >= self.risk_selector.minimum_risk_reduction
                )
            )
            use_teacher = (~teacher.feasible) | (teacher.intervention & learned_gate)
            if self.rollout_mode == "risk_projected_blend":
                feasible_blend = torch.full_like(
                    teacher.projection_gap, self.risk_projection_blend
                )
                blend = torch.where(
                    ~teacher.feasible,
                    torch.ones_like(feasible_blend),
                    feasible_blend,
                )
                blended_teacher = base_mean_local + blend * (
                    teacher.action - base_mean_local
                )
            elif self.rollout_mode == "risk_projected_smooth":
                blended_teacher, _ = self.risk_correction_smoother(
                    base_mean_local, teacher.action, use_teacher
                )
            else:
                blended_teacher = teacher.action
            if self.rollout_mode == "risk_projected_smooth":
                executed_local = blended_teacher
            else:
                executed_local = torch.where(
                    use_teacher, blended_teacher, base_mean_local
                )
            risk_used = use_teacher.float()
            if self.rollout_mode == "risk_projected_smooth":
                risk_shift = (executed_local - base_mean_local).norm(
                    dim=-1, keepdim=True
                )
            else:
                risk_shift = torch.where(
                    use_teacher,
                    (executed_local - base_mean_local).norm(dim=-1, keepdim=True),
                    torch.zeros_like(teacher.projection_gap),
                )
            if self.rollout_mode in {
                "risk_projected_blend",
                "risk_projected_smooth",
            }:
                executed_quantiles = self.safety_critic(
                    safety_features, executed_local
                )
                if risk_measure == "mean":
                    executed_risk = executed_quantiles.mean(dim=-1, keepdim=True)
                else:
                    executed_risk = upper_cvar(
                        executed_quantiles,
                        alpha=float(getattr(self.cfg.method, "cvar_alpha", 0.9)),
                    )
                selected_predicted_risk = executed_risk
            else:
                selected_predicted_risk = torch.where(
                    use_teacher, teacher_predicted_risk, nominal_predicted_risk
                )
            predicted_risk_reduction = (
                nominal_predicted_risk - selected_predicted_risk
            )
        elif self.rollout_mode == "projected":
            # Preserve nominal actions that the teacher already considers safe.
            # When no candidate is feasible, execute the best recovery action.
            use_teacher = teacher.intervention | ~teacher.feasible
            executed_local = torch.where(use_teacher, teacher.action, nominal_local)
        else:
            use_teacher = torch.zeros_like(teacher.feasible)
            executed_local = nominal_local
        tensordict["method_teacher_used"] = use_teacher.float()
        tensordict["method_risk_selector_used"] = risk_used
        tensordict["method_risk_selector_shift"] = risk_shift
        tensordict["method_nominal_predicted_risk"] = nominal_predicted_risk
        tensordict["method_selected_predicted_risk"] = selected_predicted_risk
        tensordict["method_predicted_risk_reduction"] = predicted_risk_reduction
        tensordict["method_tail_dominance_margin"] = tail_dominance_margin
        tensordict["method_executed_action_local"] = executed_local
        tensordict["agents", "action"] = vec_to_world(
            executed_local, observation["direction"]
        )
        return tensordict

    def train(self, tensordict):
        safety_training_active = self.sync_training_schedule()
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
            if self.safety_feature_extractor is not None:
                next_tensordict = torch.vmap(
                    self.safety_feature_extractor
                )(next_tensordict)
            next_dist = self.actor.get_dist(next_tensordict)
            next_action_local = self._distribution_mean_action(next_dist)
            if self.target_safety_feature_extractor is None:
                next_safety_features = next_tensordict["_feature"]
            else:
                next_tensordict = torch.vmap(
                    self.target_safety_feature_extractor
                )(next_tensordict)
                next_safety_features = next_tensordict["_safety_feature"]
            next_safety_quantiles = self.target_safety_critic(
                next_safety_features, next_action_local
            )

        rewards = tensordict["next", "agents", "reward"]
        dones = tensordict["next", "terminated"]
        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(next_values)
        advantage, returns = self.gae(rewards, dones, values, next_values)
        advantage = (advantage - advantage.mean()) / advantage.std().clip(1e-7)
        self.value_norm.update(returns)
        tensordict["adv"] = advantage
        tensordict["ret"] = self.value_norm.normalize(returns)

        next_observation = tensordict["next", "agents", "observation"]
        collision = tensordict["next", "stats", "collision"]
        safety_cost = observation_safety_cost(
            next_observation["lidar"],
            next_observation["dynamic_obstacle"],
            collision,
        )
        cost_discount = float(self.cfg.method.cost_discount)
        safety_target = normalized_safety_td_target(
            safety_cost,
            next_safety_quantiles,
            dones,
            discount=cost_discount,
        )
        tensordict["method_safety_cost"] = safety_cost
        tensordict["method_safety_target_quantiles"] = safety_target
        with torch.no_grad():
            training_features = torch.vmap(self.feature_extractor)(tensordict)
            base_mean_action = self.actor_head.base_mean_action(
                training_features["_feature"], float(self.cfg.actor.action_limit)
            )
            observation = training_features["agents", "observation"]
            distillation_teacher = self.teacher(
                base_mean_action,
                observation["state"],
                observation["lidar"],
                observation["dynamic_obstacle"],
            )
        tensordict["method_distillation_teacher_action"] = (
            distillation_teacher.action
        )
        tensordict["method_distillation_intervention"] = (
            distillation_teacher.intervention.float()
        )
        tensordict["method_distillation_feasible"] = (
            distillation_teacher.feasible.float()
        )
        tensordict["method_distillation_gap"] = distillation_teacher.projection_gap
        tensordict["method_distillation_counterfactual_actions"] = (
            distillation_teacher.candidate_actions
        )
        tensordict["method_distillation_counterfactual_risks"] = (
            distillation_teacher.candidate_risks
        )

        infos = []
        for _ in range(self.cfg.training_epoch_num):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()
        mean_constraint_gap = infos["constraint_gap"].mean()
        if safety_training_active and bool(self.cfg.method.adaptive_safety_lambda):
            with torch.no_grad():
                self.safety_lambda.add_(
                    float(self.cfg.method.dual_learning_rate) * mean_constraint_gap
                ).clamp_(0.0, float(self.cfg.method.max_safety_lambda))
        infos = infos.apply(torch.mean, batch_size=[])
        infos["safety_lambda"] = self.safety_lambda.detach().clone()
        infos["safety_training_active"] = torch.tensor(
            float(safety_training_active), device=self.device
        )
        self.training_update_count.add_(1)
        infos["training_update_count"] = self.training_update_count.detach().clone()
        self.sync_training_schedule()
        return {key: value.item() for key, value in infos.items()}

    def _update(self, tensordict):
        self.feature_extractor(tensordict)
        safety_features = self.safety_features(tensordict)
        action_dist = self.actor.get_dist(tensordict)
        log_probs = action_dist.log_prob(tensordict["agents", "action_normalized"])
        entropy_loss = -self.cfg.entropy_loss_coefficient * action_dist.entropy().mean()

        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        advantage = tensordict["adv"]
        surrogate_1 = advantage * ratio
        surrogate_2 = advantage * ratio.clamp(
            1.0 - self.cfg.actor.clip_ratio, 1.0 + self.cfg.actor.clip_ratio
        )
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean() * self.action_dim

        old_value = tensordict["state_value"]
        returns = tensordict["ret"]
        value = self.critic(tensordict)["state_value"]
        value_clipped = old_value + (value - old_value).clamp(
            -self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio
        )
        reward_critic_loss = torch.max(
            self.critic_loss_fn(returns, value_clipped),
            self.critic_loss_fn(returns, value),
        )

        mean_action = self._distribution_mean_action(action_dist)
        isolate_auxiliary_gradients = bool(
            getattr(self.cfg.method, "isolate_auxiliary_gradients", False)
        )
        if isolate_auxiliary_gradients and bool(
            self.cfg.method.actor_adapter.enabled
        ):
            distillation_action, distillation_gate = (
                self.actor_head.isolated_adapter_action(
                    tensordict["_feature"],
                    float(self.cfg.actor.action_limit),
                    safety_features=(
                        safety_features
                        if self.use_decoupled_safety_encoder
                        else None
                    ),
                )
            )
        else:
            distillation_action = mean_action
            distillation_gate = tensordict["method_safety_gate"]
        intervention_weight = (
            tensordict["method_distillation_intervention"]
            * tensordict["method_distillation_feasible"]
        )
        gap_weight = 1.0 + (
            tensordict["method_distillation_gap"]
            / float(self.cfg.method.distillation_gap_scale)
        ).clamp(max=4.0)
        distillation_error = F.smooth_l1_loss(
            distillation_action,
            tensordict["method_distillation_teacher_action"],
            reduction="none",
        ).sum(-1, keepdim=True)
        distillation_loss = (
            self._auxiliary_scale
            * float(self.cfg.method.distillation_coefficient)
            * (distillation_error * intervention_weight * gap_weight).sum()
            / intervention_weight.sum().clamp_min(1.0)
        )
        base_action = self.actor_head.base_mean_action(
            tensordict["_feature"].detach(), float(self.cfg.actor.action_limit)
        ).detach()
        safe_weight = (
            (1.0 - tensordict["method_distillation_intervention"])
            * tensordict["method_distillation_feasible"]
        )
        safe_anchor_error = F.smooth_l1_loss(
            distillation_action,
            base_action,
            reduction="none",
        ).sum(-1, keepdim=True)
        safe_anchor_loss = (
            self._auxiliary_scale
            * float(getattr(self.cfg.method, "safe_anchor_coefficient", 0.0))
            * float(bool(self.cfg.method.actor_adapter.enabled))
            * (safe_anchor_error * safe_weight).sum()
            / safe_weight.sum().clamp_min(1.0)
        )
        adapter_shift_error = F.smooth_l1_loss(
            distillation_action,
            base_action,
            reduction="none",
        ).sum(-1, keepdim=True)
        adapter_shift_loss = (
            self._auxiliary_scale
            * float(getattr(self.cfg.method, "adapter_shift_coefficient", 0.0))
            * float(bool(self.cfg.method.actor_adapter.enabled))
            * adapter_shift_error.mean()
        )
        gate_target = tensordict["method_distillation_intervention"]
        gate_error = F.binary_cross_entropy(
            distillation_gate.clamp(1e-6, 1.0 - 1e-6),
            gate_target,
            reduction="none",
        )
        gate_loss = (
            self._auxiliary_scale
            * float(self.cfg.method.actor_adapter.gate_coefficient)
            * (gate_error * tensordict["method_distillation_feasible"]).sum()
            / tensordict["method_distillation_feasible"].sum().clamp_min(1.0)
        )
        safe_gate_loss = (
            self._auxiliary_scale
            * float(getattr(self.cfg.method, "safe_gate_coefficient", 0.0))
            * float(bool(self.cfg.method.actor_adapter.enabled))
            * (
                distillation_gate
                * (1.0 - gate_target)
                * tensordict["method_distillation_feasible"]
            ).sum()
            / tensordict["method_distillation_feasible"].sum().clamp_min(1.0)
        )

        if isolate_auxiliary_gradients and not self.use_decoupled_safety_encoder:
            safety_features = safety_features.detach()
        predicted_safety = self.safety_critic(
            safety_features, tensordict["method_executed_action_local"]
        )
        if int(self.cfg.method.num_quantiles) == 1:
            safety_critic_loss = F.smooth_l1_loss(
                predicted_safety,
                tensordict["method_safety_target_quantiles"].detach(),
            )
        else:
            safety_critic_loss = quantile_huber_loss(
                predicted_safety,
                tensordict["method_safety_target_quantiles"],
                kappa=float(self.cfg.method.quantile_huber_kappa),
            )
        counterfactual_actions = tensordict[
            "method_distillation_counterfactual_actions"
        ]
        counterfactual_features = safety_features.unsqueeze(-2).expand(
            *counterfactual_actions.shape[:-1], safety_features.shape[-1]
        )
        counterfactual_prediction = self.safety_critic(
            counterfactual_features, counterfactual_actions
        )
        counterfactual_loss = counterfactual_calibration_loss(
            counterfactual_prediction,
            tensordict["method_distillation_counterfactual_risks"],
        )

        self.safety_critic.requires_grad_(False)
        risk_action = (
            distillation_action
            if isolate_auxiliary_gradients
            and bool(self.cfg.method.actor_adapter.enabled)
            else mean_action
        )
        policy_risk_quantiles = self.safety_critic(
            safety_features.detach(), risk_action
        )
        self.safety_critic.requires_grad_(True)
        if str(self.cfg.method.risk_measure) == "mean":
            policy_risk = policy_risk_quantiles.mean(dim=-1, keepdim=True)
        elif str(self.cfg.method.risk_measure) == "cvar":
            policy_risk = upper_cvar(
                policy_risk_quantiles, alpha=float(self.cfg.method.cvar_alpha)
            )
        else:
            raise ValueError(f"Unknown risk_measure: {self.cfg.method.risk_measure}")
        constraint_gap = policy_risk.mean() - float(self.cfg.method.safety_budget)
        safety_actor_loss = (
            self._auxiliary_scale * self.safety_lambda.detach() * constraint_gap
        )

        loss = (
            entropy_loss
            + actor_loss
            + reward_critic_loss
            + distillation_loss
            + safe_anchor_loss
            + adapter_shift_loss
            + gate_loss
            + safe_gate_loss
            + safety_actor_loss
            + float(self.cfg.method.safety_critic_coefficient) * safety_critic_loss
            + float(self.cfg.method.counterfactual_coefficient) * counterfactual_loss
        )

        if self.feature_extractor_optim is not None:
            self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        self.safety_critic_optim.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5.0)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5.0)
        safety_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.safety_critic.parameters(), 5.0
        )
        if self.feature_extractor_optim is not None:
            self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        self.safety_critic_optim.step()
        self._soft_update_safety_target()

        explained_variance = 1.0 - F.mse_loss(value, returns) / returns.var().clamp_min(1e-7)
        return TensorDict(
            {
                "actor_loss": actor_loss.detach(),
                "reward_critic_loss": reward_critic_loss.detach(),
                "distillation_loss": distillation_loss.detach(),
                "safe_anchor_loss": safe_anchor_loss.detach(),
                "adapter_shift_loss": adapter_shift_loss.detach(),
                "gate_loss": gate_loss.detach(),
                "safe_gate_loss": safe_gate_loss.detach(),
                "safety_actor_loss": safety_actor_loss.detach(),
                "safety_critic_loss": safety_critic_loss.detach(),
                "counterfactual_loss": counterfactual_loss.detach(),
                "policy_risk": policy_risk.mean().detach(),
                "constraint_gap": constraint_gap.detach(),
                "safety_cost": tensordict["method_safety_cost"].mean().detach(),
                "safety_lambda": self.safety_lambda.detach().clone(),
                "intervention_rate": tensordict[
                    "method_distillation_intervention"
                ].mean().detach(),
                "teacher_use_rate": tensordict["method_teacher_used"].mean().detach(),
                "projection_gap": tensordict[
                    "method_distillation_gap"
                ].mean().detach(),
                "predicted_intervention": tensordict[
                    "method_safety_gate"
                ].mean().detach(),
                "actor_grad_norm": actor_grad_norm.detach(),
                "critic_grad_norm": critic_grad_norm.detach(),
                "safety_grad_norm": safety_grad_norm.detach(),
                "explained_var": explained_variance.detach(),
            },
            [],
        )

    def _distribution_mean_action(self, distribution):
        action_limit = float(self.cfg.actor.action_limit)
        return 2.0 * distribution.mean * action_limit - action_limit

    @torch.no_grad()
    def _soft_update_safety_target(self) -> None:
        tau = float(self.cfg.method.safety_target_tau)
        for target, source in zip(
            self.target_safety_critic.parameters(), self.safety_critic.parameters()
        ):
            target.lerp_(source, tau)
        if self.target_safety_feature_extractor is not None:
            for target, source in zip(
                self.target_safety_feature_extractor.parameters(),
                self.safety_feature_extractor.parameters(),
            ):
                target.lerp_(source, tau)
