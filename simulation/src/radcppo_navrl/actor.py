"""Intervention-gated actor head for reachability-aware NavRL training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class InterventionGatedBetaActor(nn.Module):
    """Bias a Beta policy distribution with a learned safety residual.

    ``alpha_layer`` and ``beta_layer`` intentionally retain the field names of
    the official NavRL ``BetaActor`` so its actor checkpoint weights can be
    loaded directly with ``strict=False``. The new gate and residual layers are
    initialized to preserve the baseline distribution at the start of tuning.
    """

    def __init__(
        self,
        action_dim: int,
        maximum_logit_bias: float = 1.5,
        gate_mode: str = "learned",
        gate_temperature: float = 1.0,
        adapter_parameterization: str = "logit",
        maximum_mean_shift: float = 0.05,
    ) -> None:
        super().__init__()
        if gate_mode not in {"learned", "always_on"}:
            raise ValueError("gate_mode must be 'learned' or 'always_on'")
        if adapter_parameterization not in {"logit", "mean"}:
            raise ValueError(
                "adapter_parameterization must be 'logit' or 'mean'"
            )
        self.maximum_logit_bias = maximum_logit_bias
        self.gate_mode = gate_mode
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if maximum_mean_shift < 0.0:
            raise ValueError("maximum_mean_shift must be non-negative")
        self.gate_temperature = gate_temperature
        self.adapter_parameterization = adapter_parameterization
        self.maximum_mean_shift = maximum_mean_shift
        self.runtime_adapter_active = True
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.gate_layer = nn.LazyLinear(1)
        self.residual_layer = nn.LazyLinear(action_dim)
        self.softplus = nn.Softplus()

    def forward(
        self,
        features: Tensor,
        safety_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha_raw = self.alpha_layer(features)
        beta_raw = self.beta_layer(features)
        base_alpha = 1.0 + self.softplus(alpha_raw) + 1e-6
        base_beta = 1.0 + self.softplus(beta_raw) + 1e-6
        base_mean = base_alpha / (base_alpha + base_beta)
        adapter_context = features if safety_features is None else safety_features
        adapter_features = torch.cat(
            (adapter_context.detach(), base_mean.detach()), dim=-1
        )
        learned_gate = torch.sigmoid(
            self.gate_layer(adapter_features) / self.gate_temperature
        )
        safety_gate = (
            learned_gate if self.gate_mode == "learned" else torch.ones_like(learned_gate)
        )
        residual_scale = (
            self.maximum_mean_shift
            if self.adapter_parameterization == "mean"
            else self.maximum_logit_bias
        )
        safety_residual = torch.tanh(
            self.residual_layer(adapter_features)
        ) * residual_scale
        if not self.runtime_adapter_active:
            return base_alpha, base_beta, safety_gate, safety_residual
        gated_residual = safety_gate * safety_residual
        if self.adapter_parameterization == "mean":
            concentration = base_alpha + base_beta
            lower_mean = (1.0 + 1e-6) / concentration
            upper_mean = 1.0 - lower_mean
            adapted_mean = torch.minimum(
                torch.maximum(base_mean + gated_residual, lower_mean),
                upper_mean,
            )
            mean_shift = adapted_mean - base_mean
            alpha = base_alpha + mean_shift * concentration
            beta = base_beta - mean_shift * concentration
        else:
            alpha = 1.0 + self.softplus(alpha_raw + gated_residual) + 1e-6
            beta = 1.0 + self.softplus(beta_raw - gated_residual) + 1e-6
        return alpha, beta, safety_gate, safety_residual

    def set_runtime_adapter_active(self, active: bool) -> None:
        self.runtime_adapter_active = bool(active)

    def base_mean_action(self, features: Tensor, action_limit: float) -> Tensor:
        """Return the deterministic action of the pretrained task head."""

        alpha = 1.0 + self.softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.softplus(self.beta_layer(features)) + 1e-6
        normalized_mean = alpha / (alpha + beta)
        return 2.0 * normalized_mean * action_limit - action_limit

    def isolated_adapter_action(
        self,
        features: Tensor,
        action_limit: float,
        safety_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute the adapted mean action with gradients limited to the adapter."""

        task_features = features.detach()
        alpha_raw = self.alpha_layer(task_features).detach()
        beta_raw = self.beta_layer(task_features).detach()
        base_alpha = 1.0 + self.softplus(alpha_raw) + 1e-6
        base_beta = 1.0 + self.softplus(beta_raw) + 1e-6
        base_mean = base_alpha / (base_alpha + base_beta)
        adapter_context = (
            task_features
            if safety_features is None
            else safety_features.detach()
        )
        adapter_input = torch.cat((adapter_context, base_mean), dim=-1)
        learned_gate = torch.sigmoid(
            self.gate_layer(adapter_input) / self.gate_temperature
        )
        safety_gate = (
            learned_gate
            if self.gate_mode == "learned"
            else torch.ones_like(learned_gate)
        )
        residual_scale = (
            self.maximum_mean_shift
            if self.adapter_parameterization == "mean"
            else self.maximum_logit_bias
        )
        safety_residual = torch.tanh(
            self.residual_layer(adapter_input)
        ) * residual_scale
        gated_residual = safety_gate * safety_residual
        if self.adapter_parameterization == "mean":
            concentration = base_alpha + base_beta
            lower_mean = (1.0 + 1e-6) / concentration
            upper_mean = 1.0 - lower_mean
            adapted_mean = torch.minimum(
                torch.maximum(base_mean + gated_residual, lower_mean),
                upper_mean,
            )
            mean_shift = adapted_mean - base_mean
            alpha = base_alpha + mean_shift * concentration
            beta = base_beta - mean_shift * concentration
        else:
            alpha = 1.0 + self.softplus(alpha_raw + gated_residual) + 1e-6
            beta = 1.0 + self.softplus(beta_raw - gated_residual) + 1e-6
        normalized_mean = alpha / (alpha + beta)
        local_action = 2.0 * normalized_mean * action_limit - action_limit
        return local_action, safety_gate

    @torch.no_grad()
    def reset_safety_adapter(self, gate_bias: float = -2.0) -> None:
        """Start from the loaded NavRL task policy with a near-inactive adapter."""

        self.gate_layer.weight.zero_()
        self.gate_layer.bias.fill_(gate_bias)
        self.residual_layer.weight.zero_()
        self.residual_layer.bias.zero_()


class PlainBetaActor(nn.Module):
    """Official NavRL Beta head with adapter-compatible diagnostic outputs."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.softplus = nn.Softplus()

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = 1.0 + self.softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.softplus(self.beta_layer(features)) + 1e-6
        gate = features.new_zeros(*features.shape[:-1], 1)
        residual = features.new_zeros(*features.shape[:-1], self.action_dim)
        return alpha, beta, gate, residual

    def base_mean_action(self, features: Tensor, action_limit: float) -> Tensor:
        alpha = 1.0 + self.softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.softplus(self.beta_layer(features)) + 1e-6
        normalized_mean = alpha / (alpha + beta)
        return 2.0 * normalized_mean * action_limit - action_limit

    @torch.no_grad()
    def reset_safety_adapter(self, gate_bias: float = -2.0) -> None:
        del gate_bias
