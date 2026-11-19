"""Evaluate multiple checkpoints in one counterbalanced Isaac Sim process."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from evaluate_v19_canonical import (
    PROJECT_ROOT,
    bind_navrl_canonical_dynamic_scene,
    collect_episode_records,
    file_sha256,
    seed_process,
    source_sha256_manifest,
)
from isaac_app import (
    close_simulation_app,
    create_simulation_app,
    validate_navrl_dynamic_obstacles,
)
from radcppo_navrl.checkpointing import (filter_shape_compatible_state_dict,
    materialize_shared_actor_head_aliases, load_fresh_nominal_checkpoint)


ALLOWED_INFERENCE_IRRELEVANT_CHECKPOINT_KEYS = {
    "reference_safety_budget",
    "reference_budget_update_count",
    "safety_gae.gamma",
    "safety_gae.lmbda",
}


class OfficialNavRLPolicyAdapter:
    """Expose the pinned native NavRL PPO actor through the v19 rollout keys."""

    valid_modes = {"actor", "official_vo", "actor_vo"}

    def __init__(self, policy) -> None:
        self.base_policy = policy
        self.cfg = policy.cfg

    def __getattr__(self, name):
        return getattr(self.base_policy, name)

    def set_rollout_mode(self, mode: str) -> None:
        if mode not in self.valid_modes:
            raise ValueError(f"Unknown official NavRL rollout mode: {mode}")

    def __call__(self, tensordict):
        tensordict = self.base_policy(tensordict)
        normalized = tensordict["agents", "action_normalized"]
        local_action = (
            2.0 * normalized * float(self.cfg.actor.action_limit)
            - float(self.cfg.actor.action_limit)
        )
        scalar = local_action[..., :1]
        zeros = torch.zeros_like(scalar)
        zero_vector = torch.zeros_like(local_action)
        ones = torch.ones_like(scalar)
        tensordict["method_executed_action_local"] = local_action
        tensordict["method_safety_gate"] = zeros
        tensordict["method_safety_residual"] = zero_vector
        tensordict["method_adapter_shift"] = zeros
        tensordict["method_risk_selector_used"] = zeros
        tensordict["method_risk_selector_shift"] = zeros
        tensordict["method_predicted_risk_reduction"] = zeros
        tensordict["method_tail_dominance_margin"] = zeros
        tensordict["method_teacher_intervention"] = zeros
        tensordict["method_teacher_feasible"] = ones
        tensordict["method_teacher_used"] = zeros
        tensordict["method_projection_gap"] = zeros
        tensordict["method_previous_action_local"] = local_action
        tensordict["method_action_delta_local"] = zero_vector
        return tensordict


def load_suite_manifest(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("checkpoint suite must contain at least two entries")
    required = {"name", "checkpoint", "mode"}
    names = set()
    entries = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"invalid checkpoint suite entry {index}")
        entry: dict[str, object] = {key: str(item[key]) for key in required}
        if not entry["name"] or entry["name"] in names:
            raise ValueError("checkpoint suite names must be non-empty and unique")
        checkpoint = Path(str(entry["checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint does not exist: {checkpoint}")
        entry["checkpoint"] = str(checkpoint)
        algo_overrides = item.get("algo_overrides", {})
        if not isinstance(algo_overrides, dict):
            raise ValueError(
                f"checkpoint suite entry {index} algo_overrides must be an object"
            )
        entry["algo_overrides"] = algo_overrides
        policy_kind = str(item.get("policy_kind", "radcppo"))
        if policy_kind not in {"radcppo", "official_navrl"}:
            raise ValueError(
                f"checkpoint suite entry {index} has unknown policy kind "
                f"{policy_kind!r}"
            )
        entry["policy_kind"] = policy_kind
        if "training_seed" in item:
            if type(item["training_seed"]) is not int or item["training_seed"] < 0:
                raise ValueError("training_seed must be a nonnegative integer")
            entry["training_seed"] = item["training_seed"]
        if "condition" in item:
            if not isinstance(item["condition"], str) or not item["condition"].strip():
                raise ValueError("condition must be a nonempty string")
            entry["condition"] = item["condition"]
        shield = str(item.get("shield", "none"))
        if shield not in {"none", "official_vo"}:
            raise ValueError(
                f"checkpoint suite entry {index} has unknown shield {shield!r}"
            )
        if shield == "official_vo" and entry["mode"] != "official_vo":
            raise ValueError("official_vo shield entries must use mode official_vo")
        if shield == "none" and entry["mode"] == "official_vo":
            raise ValueError("official_vo mode requires shield official_vo")
        entry["shield"] = shield
        names.add(entry["name"])
        entries.append(entry)
    return entries


def counterbalanced_order(
    count: int,
    seed_index: int,
    scheme: str = "alternating_reverse",
) -> list[int]:
    order = list(range(count))
    if scheme == "alternating_reverse":
        return order if seed_index % 2 == 0 else list(reversed(order))
    if scheme == "permutation_cycle":
        rotation = seed_index % count
        order = order[rotation:] + order[:rotation]
        return order if (seed_index // count) % 2 == 0 else list(reversed(order))
    raise ValueError(f"unknown checkpoint suite order scheme {scheme!r}")


@hydra.main(config_path="cfg", config_name="train", version_base=None)
def main(cfg):
    if not cfg.checkpoint_suite_manifest:
        raise ValueError("Set checkpoint_suite_manifest=/absolute/path.json")
    if cfg.process_replicate is None:
        raise ValueError("Set process_replicate to the independent Isaac run index")
    manifest_path = Path(cfg.checkpoint_suite_manifest).resolve()
    entries = load_suite_manifest(manifest_path)
    validate_navrl_dynamic_obstacles(int(cfg.env_dyn.num_obstacles))
    simulation_app = create_simulation_app(headless=True)
    success = False
    try:
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from ppo import PPO as OfficialPPO
        from radcppo_adapter import RADCPPO
        from torchrl.envs.transforms import Compose, TransformedEnv

        from env import NavigationEnv

        seed_process(int(cfg.seed))
        env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed_env = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))
        ).train()
        canonical_scene_metadata = bind_navrl_canonical_dynamic_scene(
            transformed_env,
            str(cfg.canonical_dynamic_scene_snapshot or ""),
            str(cfg.canonical_dynamic_scene_snapshot_sha256 or ""),
        )

        official_vo_classes = None
        if any(entry["shield"] == "official_vo" for entry in entries):
            from radcppo_navrl.official_vo import (
                IsaacOfficialVONumericalShield,
                OfficialVOShieldedPolicy,
            )

            official_vo_classes = (
                IsaacOfficialVONumericalShield,
                OfficialVOShieldedPolicy,
            )

        policies = []
        load_metadata = []
        deployment_diagnostics = []
        for entry in entries:
            seed_process(int(cfg.seed))
            policy_algo = OmegaConf.create(
                OmegaConf.to_container(cfg.algo, resolve=False)
            )
            OmegaConf.set_struct(policy_algo, True)
            policy_algo = OmegaConf.merge(
                policy_algo,
                OmegaConf.create(entry["algo_overrides"]),
            )
            checkpoint_path = Path(entry["checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            is_official = entry["policy_kind"] == "official_navrl"
            restored_actor_head_aliases = []
            if is_official:
                native_policy = OfficialPPO(
                    policy_algo,
                    transformed_env.observation_spec,
                    transformed_env.action_spec,
                    cfg.device,
                )
                native_report = native_policy.load_state_dict(
                    checkpoint, strict=True
                )
                if native_report.missing_keys or native_report.unexpected_keys:
                    raise RuntimeError(
                        f"Official NavRL checkpoint load mismatch for {entry['name']}: "
                        f"{native_report}"
                    )
                policy = OfficialNavRLPolicyAdapter(native_policy)
                skipped = []
                ignored_training_state_keys = []
                missing_task_keys = []
                load_report = native_report
                safety_critic_ready = False
            else:
                policy = RADCPPO(
                    policy_algo,
                    transformed_env.observation_spec,
                    transformed_env.action_spec,
                    cfg.device,
                )
                checkpoint, restored_actor_head_aliases = materialize_shared_actor_head_aliases(policy, checkpoint)
                compatible, skipped = filter_shape_compatible_state_dict(
                    policy, checkpoint,
                    allowed_mismatch_prefixes=("safety_critic.", "target_safety_critic."))
                # Only documented training-only state can be omitted at inference.
                ignored_training_state_keys = sorted(key for key in compatible
                    if key not in policy.state_dict()
                    and key in ALLOWED_INFERENCE_IRRELEVANT_CHECKPOINT_KEYS)
                compatible = {key: value for key, value in compatible.items()
                              if key not in ignored_training_state_keys}
                load_report, extra_aliases, strict_skipped = load_fresh_nominal_checkpoint(policy, compatible)
                restored_actor_head_aliases.extend(extra_aliases)
                skipped.extend(strict_skipped)
                missing_task_keys = []
                policy.sync_training_schedule()
                safety_critic_ready = (
                    any(key.startswith("safety_critic.") for key in compatible)
                    and (
                        float(policy_algo.method.safety_critic_coefficient) > 0.0
                        or float(policy_algo.method.counterfactual_coefficient) > 0.0
                    )
                )
            if entry["mode"].startswith("risk_") and not safety_critic_ready:
                raise RuntimeError(
                    f"{entry['name']} requests learned-risk mode without a critic"
                )
            shield = None
            rollout_policy = policy
            if entry["shield"] == "official_vo":
                if official_vo_classes is None:
                    raise RuntimeError("official VO classes were not loaded")
                IsaacOfficialVONumericalShield, OfficialVOShieldedPolicy = (
                    official_vo_classes
                )
                vertical_angles = tuple(
                    float(value)
                    for value in np.linspace(
                        float(cfg.sensor.lidar_vfov[0]),
                        float(cfg.sensor.lidar_vfov[1]),
                        int(cfg.sensor.lidar_vbeams),
                    )
                )
                static_point_budget = int(
                    OmegaConf.select(
                        cfg, "official_vo.static_point_budget", default=48
                    )
                )
                shield = IsaacOfficialVONumericalShield(
                    sensor_range_m=float(cfg.sensor.lidar_range),
                    horizontal_beams=int(360.0 / float(cfg.sensor.lidar_hres)),
                    vertical_angles_deg=vertical_angles,
                    static_point_budget=static_point_budget,
                    action_limit_mps=float(policy_algo.actor.action_limit),
                )
                rollout_policy = OfficialVOShieldedPolicy(policy, shield)
            policies.append((rollout_policy, safety_critic_ready, shield))
            load_metadata.append(
                {
                    **entry,
                    "resolved_algo": OmegaConf.to_container(
                        policy_algo, resolve=True
                    ),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "missing_checkpoint_keys": list(load_report.missing_keys),
                    "missing_task_policy_keys": missing_task_keys,
                    "unexpected_checkpoint_keys": list(
                        load_report.unexpected_keys
                    ),
                    "skipped_shape_mismatch_keys": skipped,
                    "ignored_inference_irrelevant_checkpoint_keys": (
                        ignored_training_state_keys
                    ),
                    "native_official_policy": is_official,
                    "restored_actor_head_aliases": restored_actor_head_aliases,
                }
            )

        records = []
        mechanism_enabled = bool(
            OmegaConf.select(cfg, "mechanism_analysis.enabled", default=False)
        )
        mechanism_records = []
        evaluation_order = []
        evaluation_state_resets = []
        order_scheme = str(
            OmegaConf.select(
                cfg,
                "checkpoint_suite_order_scheme",
                default="alternating_reverse",
            )
        )
        for seed_index, seed in enumerate(cfg.evaluation_seeds):
            order = counterbalanced_order(
                len(entries), seed_index, scheme=order_scheme
            )
            evaluation_order.append(
                {
                    "seed": int(seed),
                    "checkpoint_order": [entries[index]["name"] for index in order],
                }
            )
            for index in order:
                entry = entries[index]
                policy, safety_critic_ready, shield = policies[index]
                print(
                    f"Starting evaluation checkpoint={entry['name']} "
                    f"mode={entry['mode']} seed={seed}",
                    flush=True,
                )
                seed_records = collect_episode_records(
                    transformed_env,
                    policy,
                    cfg,
                    entry["mode"],
                    int(seed),
                    safety_critic_ready=safety_critic_ready,
                    mechanism_records=(mechanism_records if mechanism_enabled else None),
                    trace_name=str(entry["name"]),
                    trace_training_seed=entry.get("training_seed"),
                    trace_condition=str(entry.get("condition", "canonical_D80")),
                    trace_checkpoint_sha256=file_sha256(Path(entry["checkpoint"])),
                )
                reset_diagnostics = getattr(
                    transformed_env.base_env,
                    "_navrl_last_evaluation_reset",
                    None,
                )
                if not isinstance(reset_diagnostics, dict):
                    raise RuntimeError(
                        "NavRL canonical evaluation reset diagnostics are missing"
                    )
                evaluation_state_resets.append(
                    {
                        "checkpoint": entry["name"],
                        "seed": int(seed),
                        **reset_diagnostics,
                    }
                )
                for record in seed_records:
                    record["mode"] = entry["name"]
                if mechanism_enabled:
                    mechanism_record = mechanism_records[-1]
                    mechanism_record["checkpoint"] = entry["name"]
                    mechanism_record["mode"] = entry["name"]
                    mechanism_record["training_seed_index"] = int(
                        str(entry["name"]).rsplit("_s", 1)[-1]
                    )
                    resolved_method = load_metadata[index]["resolved_algo"]["method"]
                    mechanism_record["risk_measure"] = str(
                        resolved_method["risk_measure"]
                    )
                    mechanism_record["gate_mode"] = str(
                        resolved_method["actor_adapter"]["gate_mode"]
                    )
                    mechanism_record["counterfactual_coefficient"] = float(
                        resolved_method["counterfactual_coefficient"]
                    )
                records.extend(seed_records)
                if shield is not None:
                    deployment_diagnostics.append(
                        {
                            "checkpoint": entry["name"],
                            "seed": int(seed),
                            **shield.diagnostics(),
                        }
                    )
                print(
                    f"Finished evaluation checkpoint={entry['name']} "
                    f"seed={seed} episodes={len(seed_records)}",
                    flush=True,
                )

        if not records:
            raise RuntimeError("Checkpoint suite produced no episode records")
        output_dir = Path(cfg.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "episodes.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        mechanism_csv_path = None
        if mechanism_enabled:
            if not mechanism_records:
                raise RuntimeError("mechanism analysis produced no records")
            mechanism_csv_path = output_dir / "mechanism_seed_metrics.csv"
            with mechanism_csv_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(mechanism_records[0])
                )
                writer.writeheader()
                writer.writerows(mechanism_records)
        metadata = {
            "schema_version": 1,
            "official_navrl_commit": (
                "3725bcc2e7c1be4ecf1455d922299ae85042603a"
            ),
            "isaac_sim_version": "2023.1.0-hotfix.1",
            "actual_timing": {"configured_physics_dt_s":float(cfg.sim.dt),
                "physics_dt_s":float(env.dt),"control_dt_s":float(env.dt)*int(cfg.sim.substeps)},
            "execution_environment": {
                "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(cfg.device),
                **{key: os.environ.get(key) for key in (
                    "NAVRL_HEADLESS_NO_VIEWPORT", "NAVRL_ISAAC_EXPERIENCE", "NAVRL_OFFICIAL_ROOT")},
            },
            "single_isaac_process": True,
            "process_replicate": int(cfg.process_replicate),
            "counterbalanced_order": True,
            "checkpoint_suite_order_scheme": order_scheme,
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "checkpoints": load_metadata,
            "evaluation_order": evaluation_order,
            "evaluation_state_resets": evaluation_state_resets,
            "canonical_dynamic_scene": canonical_scene_metadata,
            "deployment_diagnostics": deployment_diagnostics,
            "episode_count": len(records),
            "mechanism_analysis": {
                "enabled": mechanism_enabled,
                "record_count": len(mechanism_records),
                "seed_metric_file": (
                    mechanism_csv_path.name if mechanism_csv_path else None
                ),
                "counterfactual_ranking_margin": float(
                    OmegaConf.select(
                        cfg,
                        "mechanism_analysis.counterfactual_ranking_margin",
                        default=0.05,
                    )
                ),
                "tail_fraction": float(
                    OmegaConf.select(
                        cfg, "mechanism_analysis.tail_fraction", default=0.10
                    )
                ),
                "inference_chunk_size": int(
                    OmegaConf.select(
                        cfg,
                        "mechanism_analysis.inference_chunk_size",
                        default=4096,
                    )
                ),
            },
            "outcome_definitions": {
                "goal_reach": (
                    "NavRL stats.reach_goal sampled at the first done step"
                ),
                "success": (
                    "goal is reached strictly before unsafe termination and "
                    "at or before the first done step"
                ),
                "first_goal_step": (
                    "zero-based first goal step in the first episode, or -1 "
                    "if absent"
                ),
                "first_unsafe_step": (
                    "zero-based first unsafe termination step in the first "
                    "episode, or -1 if absent"
                ),
                "first_done_step": (
                    "zero-based first done step in the first episode"
                ),
                "event_step_absence_sentinel": -1,
                "primary_gate_metric": "success",
                "official_compatibility_metric": "goal_reach",
            },
            "source_sha256": source_sha256_manifest(),
            "suite_source_sha256": file_sha256(Path(__file__).resolve()),
            "v19_evaluation_harness_sha256": {
                relative_path: file_sha256(PROJECT_ROOT / relative_path)
                for relative_path in (
                    "methods/radcppo_navrl/evaluate_v19_canonical.py",
                    "methods/radcppo_navrl/evaluate_v19_checkpoint_suite.py",
                    "src/radcppo_navrl/canonical_scene.py",
                    "src/radcppo_navrl/evaluation_outcomes.py",
                )
            },
            "deployment_source_sha256": (
                {
                    "src/radcppo_navrl/official_vo.py": file_sha256(
                        PROJECT_ROOT / "src/radcppo_navrl/official_vo.py"
                    ),
                    "src/navrl_q2/projectors3d.py": file_sha256(
                        PROJECT_ROOT / "src/navrl_q2/projectors3d.py"
                    ),
                }
                if any(entry["shield"] == "official_vo" for entry in entries)
                else {}
            ),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(records)} episodes to {csv_path}", flush=True)
        success = True
    finally:
        close_simulation_app(simulation_app, success=success)


if __name__ == "__main__":
    main()
