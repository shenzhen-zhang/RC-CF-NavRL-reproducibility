"""Vectorized PC-ARP teacher used during NavRL training.

The teacher consumes the same single-step structured observations as NavRL. It
does not add a temporal encoder or another sensor modality. Candidate commands
are evaluated with an acceleration-limited short-horizon rollout so that the
training labels describe executable motion instead of instantaneous velocity
jumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ReachabilityTeacherConfig:
    sensor_range_m: float = 4.0
    horizontal_beams: int = 36
    vertical_angles_deg: tuple[float, ...] = (-10.0, 0.0, 10.0, 20.0)
    horizon_s: float = 0.8
    horizon_steps: int = 4
    max_acceleration_mps2: float = 8.0
    max_speed_mps: float = 2.0
    robot_radius_m: float = 0.30
    static_margin_m: float = 0.10
    dynamic_margin_m: float = 0.14
    static_point_budget: int = 48
    yaw_offsets_deg: tuple[float, ...] = (-60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0)
    pitch_offsets_deg: tuple[float, ...] = (-20.0, 0.0, 20.0)
    speed_scales: tuple[float, ...] = (0.5, 1.0)
    minimum_lattice_speed_mps: float = 0.8
    policy_weight: float = 1.0
    smoothness_weight: float = 0.10
    progress_weight: float = 0.32
    clearance_weight: float = 0.08
    preferred_clearance_m: float = 0.35
    intervention_threshold_mps: float = 0.05
    counterfactual_count: int = 12
    risk_temperature_m: float = 0.15
    rollout_model: str = "acceleration_limited"


@dataclass
class TeacherOutput:
    action: Tensor
    clearance: Tensor
    projection_gap: Tensor
    intervention: Tensor
    feasible: Tensor
    candidate_actions: Tensor
    candidate_risks: Tensor
    selected_index: Tensor


class ReachabilityTeacher(nn.Module):
    """Generate PC-ARP actions and counterfactual candidate supervision."""

    def __init__(self, config: ReachabilityTeacherConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReachabilityTeacherConfig()
        if self.config.rollout_model not in {
            "acceleration_limited",
            "instant_velocity",
        }:
            raise ValueError(
                "rollout_model must be 'acceleration_limited' or 'instant_velocity'"
            )
        self.register_buffer("ray_directions", _make_ray_directions(self.config))
        self.register_buffer(
            "times",
            torch.linspace(
                self.config.horizon_s / self.config.horizon_steps,
                self.config.horizon_s,
                self.config.horizon_steps,
            ),
        )
        yaw, pitch, scale = torch.meshgrid(
            torch.tensor(self.config.yaw_offsets_deg).deg2rad(),
            torch.tensor(self.config.pitch_offsets_deg).deg2rad(),
            torch.tensor(self.config.speed_scales),
            indexing="ij",
        )
        self.register_buffer("candidate_yaw_offsets", yaw.reshape(-1))
        self.register_buffer("candidate_pitch_offsets", pitch.reshape(-1))
        self.register_buffer("candidate_speed_scales", scale.reshape(-1))

    @torch.no_grad()
    def forward(
        self,
        nominal_action: Tensor,
        state: Tensor,
        lidar: Tensor,
        dynamic_obstacle: Tensor,
    ) -> TeacherOutput:
        leading_shape = nominal_action.shape[:-1]
        nominal = nominal_action.reshape(-1, 3)
        state_flat = state.reshape(-1, state.shape[-1])
        lidar_flat = lidar.reshape(-1, self.config.horizontal_beams, len(self.config.vertical_angles_deg))
        dynamic_flat = dynamic_obstacle.reshape(-1, dynamic_obstacle.shape[-2], dynamic_obstacle.shape[-1])

        candidates = self._candidate_commands(nominal, state_flat)
        trajectories = self._reachable_trajectories(candidates, state_flat[..., 5:8])
        clearance = self._minimum_clearance(trajectories, lidar_flat, dynamic_flat)

        goal_unit = _normalize(state_flat[..., :3])
        current_velocity = state_flat[..., 5:8]
        nominal_gap = (candidates - nominal[:, None, :]).square().sum(-1)
        smoothness = (candidates - current_velocity[:, None, :]).square().sum(-1)
        progress = (candidates * goal_unit[:, None, :]).sum(-1)
        clearance_cost = (self.config.preferred_clearance_m - clearance).clamp_min(0.0).square()
        cost = (
            self.config.policy_weight * nominal_gap
            + self.config.smoothness_weight * smoothness
            - self.config.progress_weight * progress
            + self.config.clearance_weight * clearance_cost
        )

        feasible_mask = clearance >= 0.0
        any_feasible = feasible_mask.any(-1)
        feasible_choice = torch.where(feasible_mask, cost, torch.full_like(cost, torch.inf)).argmin(-1)
        fallback_choice = (clearance + 0.08 * progress).argmax(-1)
        selected_index = torch.where(any_feasible, feasible_choice, fallback_choice)
        batch_index = torch.arange(len(nominal), device=nominal.device)
        selected = candidates[batch_index, selected_index]
        selected_clearance = clearance[batch_index, selected_index]
        projection_gap = (selected - nominal).norm(dim=-1, keepdim=True)
        intervention = projection_gap > self.config.intervention_threshold_mps

        counterfactual_actions, counterfactual_risks = self._counterfactual_subset(
            candidates, clearance, selected, selected_clearance
        )
        return TeacherOutput(
            action=selected.reshape(*leading_shape, 3),
            clearance=selected_clearance.reshape(*leading_shape, 1),
            projection_gap=projection_gap.reshape(*leading_shape, 1),
            intervention=intervention.reshape(*leading_shape, 1),
            feasible=any_feasible.reshape(*leading_shape, 1),
            candidate_actions=counterfactual_actions.reshape(
                *leading_shape, self.config.counterfactual_count, 3
            ),
            candidate_risks=counterfactual_risks.reshape(
                *leading_shape, self.config.counterfactual_count, 1
            ),
            selected_index=selected_index.reshape(*leading_shape, 1),
        )

    def _candidate_commands(self, nominal: Tensor, state: Tensor) -> Tensor:
        speed = nominal.norm(dim=-1, keepdim=True)
        goal_unit = _normalize(state[..., :3])
        direction = torch.where(speed > 0.05, nominal / speed.clamp_min(1e-8), goal_unit)
        horizontal = direction.clone()
        horizontal[..., 2] = 0.0
        goal_horizontal = goal_unit.clone()
        goal_horizontal[..., 2] = 0.0
        horizontal = torch.where(
            horizontal.norm(dim=-1, keepdim=True) > 1e-6,
            horizontal,
            goal_horizontal,
        )
        horizontal = _normalize(horizontal)
        base_yaw = torch.atan2(horizontal[..., 1], horizontal[..., 0])
        base_pitch = torch.asin(direction[..., 2].clamp(-1.0, 1.0))

        yaw = base_yaw[:, None] + self.candidate_yaw_offsets[None, :]
        pitch = (
            base_pitch[:, None] + self.candidate_pitch_offsets[None, :]
        ).clamp(-math.radians(55.0), math.radians(55.0))
        directions = torch.stack(
            (
                torch.cos(pitch) * torch.cos(yaw),
                torch.cos(pitch) * torch.sin(yaw),
                torch.sin(pitch),
            ),
            dim=-1,
        )
        base_speed = speed.clamp_min(self.config.minimum_lattice_speed_mps)
        lattice = directions * base_speed[:, None, :] * self.candidate_speed_scales[None, :, None]
        extras = torch.stack(
            (nominal, goal_unit * base_speed, torch.zeros_like(nominal)), dim=1
        )
        candidates = torch.cat((extras, lattice), dim=1)
        norm = candidates.norm(dim=-1, keepdim=True)
        return candidates * (self.config.max_speed_mps / norm.clamp_min(1e-8)).clamp_max(1.0)

    def _reachable_trajectories(self, candidates: Tensor, current_velocity: Tensor) -> Tensor:
        if self.config.rollout_model == "instant_velocity":
            return candidates[:, :, None, :] * self.times.view(1, 1, -1, 1)
        delta = candidates - current_velocity[:, None, :]
        delta_norm = delta.norm(dim=-1, keepdim=True)
        acceleration_direction = delta / delta_norm.clamp_min(1e-8)
        time_to_target = delta_norm / self.config.max_acceleration_mps2
        times = self.times.view(1, 1, -1, 1)
        acceleration_time = torch.minimum(times, time_to_target[:, :, None, :])
        return (
            current_velocity[:, None, None, :] * acceleration_time
            + 0.5
            * self.config.max_acceleration_mps2
            * acceleration_direction[:, :, None, :]
            * acceleration_time.square()
            + candidates[:, :, None, :] * (times - acceleration_time)
        )

    def _minimum_clearance(
        self, trajectories: Tensor, lidar: Tensor, dynamic_obstacle: Tensor
    ) -> Tensor:
        static_clearance = self._static_clearance(trajectories, lidar)
        dynamic_clearance = self._dynamic_clearance(trajectories, dynamic_obstacle)
        return torch.minimum(static_clearance, dynamic_clearance)

    def _static_clearance(self, trajectories: Tensor, lidar: Tensor) -> Tensor:
        ranges = self.config.sensor_range_m - lidar.reshape(len(lidar), -1)
        point_budget = min(self.config.static_point_budget, ranges.shape[-1])
        nearest_range, nearest_index = torch.topk(
            ranges, k=point_budget, dim=-1, largest=False, sorted=False
        )
        directions = self.ray_directions[nearest_index]
        points = directions * nearest_range[..., None]
        valid = nearest_range < self.config.sensor_range_m - 1e-3
        distance = (
            trajectories[:, :, :, None, :] - points[:, None, None, :, :]
        ).norm(dim=-1)
        clearance = distance - (self.config.robot_radius_m + self.config.static_margin_m)
        clearance = torch.where(valid[:, None, None, :], clearance, torch.inf)
        return clearance.amin(dim=(-1, -2))

    def _dynamic_clearance(self, trajectories: Tensor, dynamic_obstacle: Tensor) -> Tensor:
        direction = dynamic_obstacle[..., :3]
        distance_2d = dynamic_obstacle[..., 3]
        distance_z = dynamic_obstacle[..., 4]
        total_distance = torch.sqrt(distance_2d.square() + distance_z.square())
        valid = dynamic_obstacle.abs().sum(dim=-1) > 1e-6
        relative_position = direction * total_distance[..., None]
        obstacle_velocity = dynamic_obstacle[..., 5:8]
        obstacle_future = (
            relative_position[:, None, :, :]
            + self.times[None, :, None, None] * obstacle_velocity[:, None, :, :]
        )
        relative = trajectories[:, :, :, None, :] - obstacle_future[:, None, :, :, :]

        width_category = dynamic_obstacle[..., 8].round().clamp(0.0, 3.0)
        obstacle_radius = 0.125 * (width_category + 1.0)
        inflation = self.config.robot_radius_m + self.config.dynamic_margin_m
        cylinder_clearance = relative[..., :2].norm(dim=-1) - (
            obstacle_radius[:, None, None, :] + inflation
        )

        half_extent = torch.stack(
            (obstacle_radius, obstacle_radius, torch.full_like(obstacle_radius, 0.5)), dim=-1
        )
        q = relative.abs() - (half_extent[:, None, None, :, :] + inflation)
        box_clearance = q.clamp_min(0.0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0.0)
        is_three_dimensional = dynamic_obstacle[..., 9] > 0.5
        clearance = torch.where(
            is_three_dimensional[:, None, None, :], box_clearance, cylinder_clearance
        )
        clearance = torch.where(valid[:, None, None, :], clearance, torch.inf)
        return clearance.amin(dim=(-1, -2))

    def _counterfactual_subset(
        self,
        candidates: Tensor,
        clearance: Tensor,
        selected: Tensor,
        selected_clearance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        count = self.config.counterfactual_count
        if count < 2:
            raise ValueError("counterfactual_count must be at least 2")
        base_indices = torch.linspace(
            0, candidates.shape[1] - 1, count - 1, device=candidates.device
        ).round().long()
        sampled_actions = candidates[:, base_indices]
        sampled_clearance = clearance[:, base_indices]
        actions = torch.cat((sampled_actions, selected[:, None, :]), dim=1)
        clearances = torch.cat((sampled_clearance, selected_clearance[:, None]), dim=1)
        risks = torch.sigmoid(
            (self.config.preferred_clearance_m - clearances)
            / self.config.risk_temperature_m
        )
        return actions, risks[..., None]


def observation_safety_cost(
    lidar: Tensor,
    dynamic_obstacle: Tensor,
    collision: Tensor | None = None,
    *,
    sensor_range_m: float = 4.0,
    robot_radius_m: float = 0.30,
    preferred_clearance_m: float = 0.35,
    temperature_m: float = 0.15,
) -> Tensor:
    """Dense transition cost used by the safety critic.

    Collision remains the highest cost, while near-clearance states provide a
    non-sparse learning signal before an episode terminates.
    """

    clearance = observation_min_clearance(
        lidar,
        dynamic_obstacle,
        sensor_range_m=sensor_range_m,
        robot_radius_m=robot_radius_m,
    )
    cost = torch.sigmoid((preferred_clearance_m - clearance) / temperature_m).unsqueeze(-1)
    if collision is not None:
        cost = torch.maximum(cost, collision.float().reshape_as(cost))
    return cost.clamp(0.0, 1.0)


def observation_min_clearance(
    lidar: Tensor,
    dynamic_obstacle: Tensor,
    *,
    sensor_range_m: float = 4.0,
    robot_radius_m: float = 0.30,
) -> Tensor:
    """Approximate current structured-observation clearance in metres."""

    static_distance = sensor_range_m - lidar.flatten(start_dim=-3).amax(dim=-1)
    static_clearance = static_distance - robot_radius_m
    dynamic = dynamic_obstacle.squeeze(-3)
    dynamic_distance = torch.sqrt(dynamic[..., 3].square() + dynamic[..., 4].square())
    valid = dynamic.abs().sum(dim=-1) > 1e-6
    relative_position = dynamic[..., :3] * dynamic_distance[..., None]
    radius = 0.125 * (dynamic[..., 8].round().clamp(0.0, 3.0) + 1.0)
    cylinder_clearance = relative_position[..., :2].norm(dim=-1) - (
        radius + robot_radius_m
    )
    half_extent = torch.stack(
        (radius, radius, torch.full_like(radius, 0.5)), dim=-1
    )
    q = relative_position.abs() - (half_extent + robot_radius_m)
    box_clearance = q.clamp_min(0.0).norm(dim=-1) + q.amax(dim=-1).clamp_max(0.0)
    is_three_dimensional = dynamic[..., 9] > 0.5
    dynamic_clearance = torch.where(
        is_three_dimensional, box_clearance, cylinder_clearance
    )
    dynamic_clearance = torch.where(valid, dynamic_clearance, torch.inf).amin(dim=-1)
    return torch.minimum(static_clearance, dynamic_clearance)


def _make_ray_directions(config: ReachabilityTeacherConfig) -> Tensor:
    yaw = torch.arange(config.horizontal_beams, dtype=torch.float32) * (
        2.0 * math.pi / config.horizontal_beams
    )
    pitch = torch.tensor(config.vertical_angles_deg, dtype=torch.float32).deg2rad()
    yaw_grid, pitch_grid = torch.meshgrid(yaw, pitch, indexing="ij")
    return torch.stack(
        (
            torch.cos(pitch_grid) * torch.cos(yaw_grid),
            torch.cos(pitch_grid) * torch.sin(yaw_grid),
            torch.sin(pitch_grid),
        ),
        dim=-1,
    ).reshape(-1, 3)


def _normalize(vector: Tensor) -> Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
