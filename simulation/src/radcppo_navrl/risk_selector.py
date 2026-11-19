"""Conservative policy improvement driven by a learned safety critic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RiskSelection:
    action: Tensor
    used: Tensor
    shift: Tensor
    nominal_risk: Tensor
    selected_risk: Tensor
    risk_reduction: Tensor
    selected_index: Tensor


class ConservativeRiskSelector(nn.Module):
    """Select a bounded local action perturbation using predicted risk.

    The selector has no obstacle-geometry or PC-ARP dependency. It probes a
    fixed local lattice around the frozen task action and accepts a proposal
    only when the learned critic predicts a material risk reduction.
    """

    def __init__(
        self,
        *,
        action_limit: float,
        action_step_mps: float = 0.08,
        activation_threshold: float = 0.20,
        minimum_risk_reduction: float = 0.015,
        deviation_penalty: float = 0.002,
    ) -> None:
        super().__init__()
        if action_limit <= 0.0:
            raise ValueError("action_limit must be positive")
        if action_step_mps <= 0.0:
            raise ValueError("action_step_mps must be positive")
        if not 0.0 <= activation_threshold <= 1.0:
            raise ValueError("activation_threshold must be in [0, 1]")
        if minimum_risk_reduction < 0.0:
            raise ValueError("minimum_risk_reduction must be non-negative")
        if deviation_penalty < 0.0:
            raise ValueError("deviation_penalty must be non-negative")

        self.action_limit = float(action_limit)
        self.action_step_mps = float(action_step_mps)
        self.activation_threshold = float(activation_threshold)
        self.minimum_risk_reduction = float(minimum_risk_reduction)
        self.deviation_penalty = float(deviation_penalty)

        diagonal = 1.0 / math.sqrt(2.0)
        fixed_directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [diagonal, diagonal, 0.0],
                [diagonal, -diagonal, 0.0],
                [-diagonal, diagonal, 0.0],
                [-diagonal, -diagonal, 0.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("fixed_directions", fixed_directions)

    def candidate_actions(self, nominal_action: Tensor) -> Tensor:
        """Return the nominal action followed by bounded 3-D perturbations."""

        if nominal_action.shape[-1] != 3:
            raise ValueError("nominal_action must have three components")
        fixed = self.fixed_directions.to(dtype=nominal_action.dtype)
        fixed = fixed.view(*([1] * (nominal_action.ndim - 1)), -1, 3)
        fixed = fixed.expand(*nominal_action.shape[:-1], -1, -1)

        speed_direction = nominal_action / nominal_action.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        speed_directions = torch.stack((-speed_direction, speed_direction), dim=-2)
        directions = torch.cat((fixed, speed_directions), dim=-2)
        proposals = nominal_action.unsqueeze(-2) + self.action_step_mps * directions
        proposals = proposals.clamp(-self.action_limit, self.action_limit)
        return torch.cat((nominal_action.unsqueeze(-2), proposals), dim=-2)

    def select(self, candidate_actions: Tensor, candidate_risks: Tensor) -> RiskSelection:
        """Choose a proposal and reject it unless all conservative gates pass."""

        if candidate_actions.shape[-1] != 3:
            raise ValueError("candidate_actions must have three action components")
        if candidate_risks.shape[:-1] != candidate_actions.shape[:-1]:
            raise ValueError("candidate risk and action batch shapes must match")
        if candidate_risks.shape[-1] != 1:
            raise ValueError("candidate_risks must end in a singleton dimension")
        if candidate_actions.shape[-2] < 2:
            raise ValueError("at least one proposal is required")

        nominal = candidate_actions[..., 0, :]
        nominal_risk = candidate_risks[..., 0, :]
        shift = (candidate_actions - nominal.unsqueeze(-2)).norm(
            dim=-1, keepdim=True
        )
        normalized_shift = shift / self.action_step_mps
        score = candidate_risks + self.deviation_penalty * normalized_shift.square()
        selected_index = score.argmin(dim=-2)
        action_index = selected_index.unsqueeze(-1).expand(*selected_index.shape, 3)
        selected_action = candidate_actions.gather(-2, action_index).squeeze(-2)
        selected_risk = candidate_risks.gather(
            -2, selected_index.unsqueeze(-1)
        ).squeeze(-2)
        selected_shift = (selected_action - nominal).norm(dim=-1, keepdim=True)
        risk_reduction = nominal_risk - selected_risk
        use = (
            (selected_index != 0)
            & (nominal_risk >= self.activation_threshold)
            & (risk_reduction >= self.minimum_risk_reduction)
        )
        action = torch.where(use, selected_action, nominal)
        actual_shift = torch.where(use, selected_shift, torch.zeros_like(selected_shift))
        actual_selected_risk = torch.where(use, selected_risk, nominal_risk)
        actual_reduction = nominal_risk - actual_selected_risk
        return RiskSelection(
            action=action,
            used=use,
            shift=actual_shift,
            nominal_risk=nominal_risk,
            selected_risk=actual_selected_risk,
            risk_reduction=actual_reduction,
            selected_index=selected_index,
        )


class ExponentialCorrectionSmoother(nn.Module):
    """Smooth an auxiliary action correction across control steps."""

    def __init__(self, coefficient: float = 0.8) -> None:
        super().__init__()
        if not 0.0 <= coefficient < 1.0:
            raise ValueError("smoothing coefficient must be in [0, 1)")
        self.coefficient = float(coefficient)
        self._correction: Tensor | None = None

    def reset(self) -> None:
        self._correction = None

    def forward(
        self,
        nominal_action: Tensor,
        auxiliary_action: Tensor,
        use_auxiliary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target = torch.where(
            use_auxiliary,
            auxiliary_action - nominal_action,
            torch.zeros_like(nominal_action),
        )
        previous = self._correction
        if (
            previous is None
            or previous.shape != target.shape
            or previous.device != target.device
            or previous.dtype != target.dtype
        ):
            previous = torch.zeros_like(target)
        correction = self.coefficient * previous + (1.0 - self.coefficient) * target
        self._correction = correction.detach()
        return nominal_action + correction, correction
