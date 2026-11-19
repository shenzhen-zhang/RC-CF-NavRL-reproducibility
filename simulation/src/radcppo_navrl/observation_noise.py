"""Deterministic sensor-noise injection for NavRL policy evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ObservationNoiseConfig:
    """Noise applied to policy inputs while preserving true outcome observations."""

    enabled: bool = False
    lidar_std_m: float = 0.0
    dynamic_position_std_m: float = 0.0
    dynamic_velocity_std_mps: float = 0.0
    sensor_range_m: float = 4.0
    preserve_lidar_no_return: bool = True
    lidar_hit_threshold_m: float = 1e-6
    seed_offset: int = 19_871_011

    def validate(self) -> ObservationNoiseConfig:
        for name in (
            "lidar_std_m",
            "dynamic_position_std_m",
            "dynamic_velocity_std_mps",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.sensor_range_m <= 0.0:
            raise ValueError("sensor_range_m must be positive")
        if self.lidar_hit_threshold_m < 0.0:
            raise ValueError("lidar_hit_threshold_m must be non-negative")
        if self.lidar_hit_threshold_m >= self.sensor_range_m:
            raise ValueError("lidar_hit_threshold_m must be smaller than sensor_range_m")
        if self.enabled and not any(
            value > 0.0
            for value in (
                self.lidar_std_m,
                self.dynamic_position_std_m,
                self.dynamic_velocity_std_mps,
            )
        ):
            raise ValueError("enabled observation noise requires a positive standard deviation")
        return self

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def _normal_like(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def apply_navrl_observation_noise(
    observation,
    config: ObservationNoiseConfig,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Mutate a cloned NavRL observation and return auditable noise diagnostics.

    NavRL stores LiDAR as ``sensor_range - measured_distance``. Noise is
    therefore applied in the physical distance domain and encoded back. Rays
    with no return are kept at zero by default so zero-mean noise cannot create
    a systematic field of phantom obstacles at the encoding boundary.
    """

    config.validate()
    if not config.enabled:
        return {}

    diagnostics: dict[str, torch.Tensor] = {}

    lidar = observation["lidar"]
    if config.lidar_std_m > 0.0:
        original_proximity = lidar.clone()
        hit_mask = original_proximity > config.lidar_hit_threshold_m
        measured_distance = config.sensor_range_m - original_proximity
        range_error = config.lidar_std_m * _normal_like(lidar, generator)
        noisy_distance = (measured_distance + range_error).clamp(0.0, config.sensor_range_m)
        noisy_proximity = config.sensor_range_m - noisy_distance
        if config.preserve_lidar_no_return:
            noisy_proximity = torch.where(hit_mask, noisy_proximity, original_proximity)
        lidar.copy_(noisy_proximity)
        applied_range_error = original_proximity - noisy_proximity
        diagnostics.update(
            {
                "lidar_hit_fraction": hit_mask.to(lidar.dtype).mean(),
                "lidar_range_error_mean_m": _masked_mean(applied_range_error, hit_mask),
                "lidar_range_error_rms_m": _masked_mean(
                    applied_range_error.square(), hit_mask
                ).sqrt(),
                "lidar_proximity_shift_mean_m": _masked_mean(
                    noisy_proximity - original_proximity, hit_mask
                ),
                "lidar_no_return_change_fraction": ((~hit_mask) & (lidar != original_proximity))
                .to(lidar.dtype)
                .mean(),
            }
        )

    dynamic = observation["dynamic_obstacle"]
    valid = dynamic.abs().sum(dim=-1, keepdim=True) > 1e-6
    if config.dynamic_position_std_m > 0.0:
        direction = dynamic[..., :3]
        distance_2d = dynamic[..., 3:4]
        distance_z = dynamic[..., 4:5]
        distance_3d = torch.sqrt(distance_2d.square() + distance_z.square())
        position = direction * distance_3d
        original_position = position.clone()
        position.add_(config.dynamic_position_std_m * _normal_like(position, generator))
        position = torch.where(valid, position, torch.zeros_like(position))
        updated_distance_3d = position.norm(dim=-1, keepdim=True)
        updated_direction = position / updated_distance_3d.clamp_min(1e-6)
        dynamic[..., :3] = torch.where(
            valid, updated_direction, torch.zeros_like(updated_direction)
        )
        dynamic[..., 3:4] = torch.where(
            valid,
            position[..., :2].norm(dim=-1, keepdim=True),
            torch.zeros_like(distance_2d),
        )
        dynamic[..., 4:5] = torch.where(valid, position[..., 2:3], torch.zeros_like(distance_z))
        position_error = (position - original_position).norm(dim=-1, keepdim=True)
        diagnostics["dynamic_position_error_rms_m"] = _masked_mean(
            position_error.square(), valid
        ).sqrt()

    if config.dynamic_velocity_std_mps > 0.0:
        velocity = dynamic[..., 5:8]
        original_velocity = velocity.clone()
        velocity.add_(config.dynamic_velocity_std_mps * _normal_like(velocity, generator))
        velocity.copy_(torch.where(valid, velocity, torch.zeros_like(velocity)))
        velocity_error = (velocity - original_velocity).norm(dim=-1, keepdim=True)
        diagnostics["dynamic_velocity_error_rms_mps"] = _masked_mean(
            velocity_error.square(), valid
        ).sqrt()

    return diagnostics


class ObservationNoisePolicy:
    """Wrap a policy so only its input observation is corrupted."""

    def __init__(self, base_policy, config: ObservationNoiseConfig, device: str) -> None:
        self.base_policy = base_policy
        self.config = config.validate()
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._last_noise_seed: int | None = None
        self._diagnostic_sums: dict[str, torch.Tensor] = {}
        self._diagnostic_call_count = 0

    def __getattr__(self, name):
        return getattr(self.base_policy, name)

    def set_rollout_mode(self, mode: str) -> None:
        self.base_policy.set_rollout_mode(mode)

    def reset_observation_noise(self, evaluation_seed: int) -> int:
        noise_seed = (int(evaluation_seed) + int(self.config.seed_offset)) % (2**63 - 1)
        self._generator.manual_seed(noise_seed)
        self._last_noise_seed = noise_seed
        self._diagnostic_sums = {}
        self._diagnostic_call_count = 0
        return noise_seed

    def observation_noise_diagnostics(self) -> dict[str, float | int]:
        if self._diagnostic_call_count == 0:
            return {"observation_noise_step_count": 0}
        return {
            "observation_noise_step_count": self._diagnostic_call_count,
            **{
                name: float((total / self._diagnostic_call_count).item())
                for name, total in sorted(self._diagnostic_sums.items())
            },
        }

    def __call__(self, tensordict):
        if self._last_noise_seed is None:
            raise RuntimeError("observation-noise generator was not reset for this rollout")
        true_observation = tensordict["agents", "observation"]
        working = tensordict.clone(recurse=True)
        diagnostics = apply_navrl_observation_noise(
            working["agents", "observation"],
            self.config,
            self._generator,
        )
        for name, value in diagnostics.items():
            if name not in self._diagnostic_sums:
                self._diagnostic_sums[name] = torch.zeros_like(value)
            self._diagnostic_sums[name].add_(value)
        self._diagnostic_call_count += 1
        result = self.base_policy(working)
        result["agents", "observation"] = true_observation
        return result
