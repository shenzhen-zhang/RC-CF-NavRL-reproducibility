"""Distributional safety critic and losses for RA-DC-PPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MonotonicQuantileCritic(nn.Module):
    """Action-conditioned critic for normalized discounted safety cost.

    The output is bounded to ``[0, 1]`` and ordered by construction.  The
    latent increments keep a stable identity for every quantile, unlike
    sorting unconstrained outputs after each forward pass.
    """

    def __init__(self, num_quantiles: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        if num_quantiles < 1:
            raise ValueError("num_quantiles must be at least 1")
        self.num_quantiles = num_quantiles
        self.network = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_quantiles),
        )

    def forward(self, features: Tensor, actions: Tensor) -> Tensor:
        raw = self.network(torch.cat((features, actions), dim=-1))
        if self.num_quantiles == 1:
            return torch.sigmoid(raw)
        base_logit = raw[..., :1]
        positive_increments = F.softplus(raw[..., 1:]) / self.num_quantiles
        ordered_logits = torch.cat(
            (
                base_logit,
                base_logit + positive_increments.cumsum(dim=-1),
            ),
            dim=-1,
        )
        return torch.sigmoid(ordered_logits)

    @torch.no_grad()
    def reset_output_prior(
        self,
        prior_probability: float = 0.02,
        increment_logit: float = -4.0,
    ) -> None:
        """Initialize the safety distribution near a low-risk prior."""

        if not 0.0 < prior_probability < 1.0:
            raise ValueError("prior_probability must be in (0, 1)")
        output_layer = self.network[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("The final safety-critic layer must be linear")
        prior_logit = torch.logit(
            torch.tensor(
                prior_probability,
                device=output_layer.bias.device,
                dtype=output_layer.bias.dtype,
            )
        )
        output_layer.bias[0].copy_(prior_logit)
        if self.num_quantiles > 1:
            output_layer.bias[1:].fill_(increment_logit)


def quantile_huber_loss(
    prediction: Tensor,
    target_samples: Tensor,
    *,
    kappa: float = 1.0,
) -> Tensor:
    """Pairwise quantile-regression Huber loss used by distributional TD."""

    if prediction.shape[-1] < 2:
        raise ValueError("prediction must contain multiple quantiles")
    num_quantiles = prediction.shape[-1]
    taus = (
        torch.arange(num_quantiles, device=prediction.device, dtype=prediction.dtype) + 0.5
    ) / num_quantiles
    delta = target_samples.detach().unsqueeze(-2) - prediction.unsqueeze(-1)
    abs_delta = delta.abs()
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.square(),
        kappa * (abs_delta - 0.5 * kappa),
    )
    weight = (taus.view(*([1] * (delta.ndim - 2)), num_quantiles, 1) - (delta < 0).float()).abs()
    return (weight * huber / kappa).mean()


def upper_cvar(quantiles: Tensor, alpha: float = 0.9) -> Tensor:
    """Mean of the upper cost tail represented by ordered quantiles."""

    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    start = min(int(alpha * quantiles.shape[-1]), quantiles.shape[-1] - 1)
    return quantiles[..., start:].mean(dim=-1, keepdim=True)


def upper_tail_dominance_margin(
    nominal_quantiles: Tensor,
    candidate_quantiles: Tensor,
    *,
    alpha: float = 0.9,
) -> Tensor:
    """Worst paired risk reduction across the ordered upper-cost tail."""

    if nominal_quantiles.shape != candidate_quantiles.shape:
        raise ValueError("nominal and candidate quantiles must have identical shapes")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    start = min(
        int(alpha * nominal_quantiles.shape[-1]),
        nominal_quantiles.shape[-1] - 1,
    )
    paired_reduction = (
        nominal_quantiles[..., start:] - candidate_quantiles[..., start:]
    )
    return paired_reduction.amin(dim=-1, keepdim=True)


def normalized_safety_td_target(
    cost: Tensor,
    next_quantiles: Tensor,
    terminated: Tensor,
    *,
    discount: float,
) -> Tensor:
    """Build a bounded safety target with unsafe termination as absorbing cost."""

    if not 0.0 <= discount < 1.0:
        raise ValueError("discount must be in [0, 1)")
    target = (1.0 - discount) * cost + discount * (
        1.0 - terminated.float()
    ) * next_quantiles
    return torch.where(terminated.bool(), torch.ones_like(target), target).clamp(0.0, 1.0)


def counterfactual_calibration_loss(
    predicted_quantiles: Tensor,
    target_risk: Tensor,
    *,
    ranking_margin: float = 0.05,
) -> Tensor:
    """Calibrate candidate risk means and preserve within-state risk ordering."""

    predicted_mean = predicted_quantiles.mean(dim=-1, keepdim=True)
    calibration = F.smooth_l1_loss(predicted_mean, target_risk)
    if predicted_mean.shape[-2] < 2:
        return calibration
    predicted_delta = predicted_mean[..., 1:, :] - predicted_mean[..., :-1, :]
    target_delta = target_risk[..., 1:, :] - target_risk[..., :-1, :]
    comparable = target_delta.abs() > ranking_margin
    ranking = F.softplus(-predicted_delta * target_delta.sign())
    ranking = torch.where(comparable, ranking, torch.zeros_like(ranking))
    denominator = comparable.float().sum().clamp_min(1.0)
    return calibration + ranking.sum() / denominator
