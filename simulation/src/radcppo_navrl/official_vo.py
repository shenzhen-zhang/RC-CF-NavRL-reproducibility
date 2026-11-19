"""Observation-compatible Isaac Sim port of NavRL's public VO shield."""

from __future__ import annotations

import atexit
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from navrl_q2.projectors3d import OfficialVO3DProjector, SafetyContext3D

_OFFICIAL_VO_EXECUTOR: ProcessPoolExecutor | None = None
_OFFICIAL_VO_WORKER_COUNT = 0


def _pool_identity(value: int) -> int:
    return value


def prepare_official_vo_worker_pool(worker_count: int) -> None:
    """Start CPU projection workers before Isaac initializes CUDA."""

    global _OFFICIAL_VO_EXECUTOR, _OFFICIAL_VO_WORKER_COUNT
    if worker_count < 2:
        return
    if _OFFICIAL_VO_EXECUTOR is not None:
        if worker_count != _OFFICIAL_VO_WORKER_COUNT:
            raise RuntimeError(
                "Official VO worker pool is already initialized with "
                f"{_OFFICIAL_VO_WORKER_COUNT} workers"
            )
        return
    context = multiprocessing.get_context("forkserver")
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    )
    warmed = list(executor.map(_pool_identity, range(worker_count), chunksize=1))
    if warmed != list(range(worker_count)):
        executor.shutdown(wait=True, cancel_futures=True)
        raise RuntimeError("Official VO worker-pool warmup failed")
    _OFFICIAL_VO_EXECUTOR = executor
    _OFFICIAL_VO_WORKER_COUNT = worker_count


def shutdown_official_vo_worker_pool() -> None:
    global _OFFICIAL_VO_EXECUTOR, _OFFICIAL_VO_WORKER_COUNT
    executor = _OFFICIAL_VO_EXECUTOR
    _OFFICIAL_VO_EXECUTOR = None
    _OFFICIAL_VO_WORKER_COUNT = 0
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown_official_vo_worker_pool)


@dataclass(frozen=True)
class OfficialVOBatchResult:
    action: Tensor
    intervention: Tensor
    feasible: Tensor
    shift: Tensor


@dataclass(frozen=True)
class _OfficialVOWorkItem:
    nominal: np.ndarray
    state: np.ndarray
    ray_directions: np.ndarray
    ray_ranges: np.ndarray
    dynamic_positions: np.ndarray
    dynamic_velocities: np.ndarray
    dynamic_radii: np.ndarray
    dynamic_valid: np.ndarray
    dynamic_heights: np.ndarray
    robot_radius_m: float
    horizon_s: float
    time_step_s: float
    safety_distance_m: float
    max_speed_mps: float


def _unused_raycast(directions: np.ndarray) -> np.ndarray:
    raise RuntimeError(f"Official VO unexpectedly requested {len(directions)} rays")


def _project_official_vo_work_item(
    item: _OfficialVOWorkItem,
) -> tuple[np.ndarray, int, int]:
    projector = OfficialVO3DProjector(
        horizon_s=item.horizon_s,
        time_step_s=item.time_step_s,
        safety_distance_m=item.safety_distance_m,
        max_speed_mps=item.max_speed_mps,
    )
    context = SafetyContext3D(
        position=np.zeros(3, dtype=np.float64),
        velocity=item.state[5:8],
        goal=item.state[:3],
        ray_directions=item.ray_directions,
        ray_ranges=item.ray_ranges,
        dynamic_positions=item.dynamic_positions,
        dynamic_velocities=item.dynamic_velocities,
        dynamic_radii=item.dynamic_radii,
        dynamic_valid=item.dynamic_valid,
        dt=item.time_step_s,
        robot_radius=item.robot_radius_m,
        raycast=_unused_raycast,
        dynamic_heights=item.dynamic_heights,
    )
    selected = projector.project(item.nominal, context)
    return selected, projector.interventions, projector.no_feasible


class IsaacOfficialVONumericalShield:
    """Apply the public 3-D VO geometry to a batch of NavRL observations.

    The ROS service and DBSCAN transport are not available inside the pinned
    Isaac Sim image. Static input is therefore represented by the nearest
    observed ray endpoints, while the released ORCA planes, vertical dynamic
    obstacle decomposition, LP solver and action bounds are retained.
    """

    def __init__(
        self,
        *,
        sensor_range_m: float = 4.0,
        horizontal_beams: int = 36,
        vertical_angles_deg: tuple[float, ...] = (-10.0, 0.0, 10.0, 20.0),
        static_point_budget: int = 48,
        robot_radius_m: float = 0.30,
        dynamic_width_resolution_m: float = 0.25,
        dynamic_3d_height_m: float = 1.0,
        dynamic_2d_height_m: float = 5.0,
        horizon_s: float = 2.0,
        time_step_s: float = 0.05,
        safety_distance_m: float = 0.30,
        action_limit_mps: float = 2.0,
    ) -> None:
        if sensor_range_m <= 0.0:
            raise ValueError("sensor_range_m must be positive")
        if horizontal_beams < 1 or len(vertical_angles_deg) < 1:
            raise ValueError("ray layout must contain at least one beam")
        if static_point_budget < 0:
            raise ValueError("static_point_budget must be non-negative")
        if robot_radius_m <= 0.0:
            raise ValueError("robot_radius_m must be positive")
        if dynamic_width_resolution_m <= 0.0:
            raise ValueError("dynamic_width_resolution_m must be positive")
        if action_limit_mps <= 0.0:
            raise ValueError("action_limit_mps must be positive")

        self.sensor_range_m = float(sensor_range_m)
        self.static_point_budget = int(static_point_budget)
        self.robot_radius_m = float(robot_radius_m)
        self.dynamic_width_resolution_m = float(dynamic_width_resolution_m)
        self.dynamic_3d_height_m = float(dynamic_3d_height_m)
        self.dynamic_2d_height_m = float(dynamic_2d_height_m)
        self.ray_directions = _make_ray_directions(
            horizontal_beams, vertical_angles_deg
        )
        self.projector = OfficialVO3DProjector(
            horizon_s=horizon_s,
            time_step_s=time_step_s,
            safety_distance_m=safety_distance_m,
            max_speed_mps=action_limit_mps * math.sqrt(3.0),
        )

    def reset(self) -> None:
        self.projector.reset()

    def diagnostics(self) -> dict[str, float | int | str]:
        values = self.projector.diagnostics()
        values.update(
            {
                "sensor_range_m": self.sensor_range_m,
                "static_point_budget": self.static_point_budget,
                "ray_count": len(self.ray_directions),
                "horizon_s": self.projector.horizon_s,
                "time_step_s": self.projector.time_step_s,
                "safety_distance_m": self.projector.safety_distance_m,
                "max_speed_mps": self.projector.max_speed_mps,
                "static_input_adapter": "nearest_ray_endpoints",
                "execution_backend": (
                    "forkserver_process_pool"
                    if _OFFICIAL_VO_EXECUTOR is not None
                    else "serial"
                ),
                "parallel_worker_count": _OFFICIAL_VO_WORKER_COUNT,
            }
        )
        return values

    @torch.no_grad()
    def project(
        self,
        nominal_action: Tensor,
        state: Tensor,
        lidar: Tensor,
        dynamic_obstacle: Tensor,
    ) -> OfficialVOBatchResult:
        if nominal_action.shape[-1] != 3:
            raise ValueError("nominal_action must have three components")
        leading_shape = nominal_action.shape[:-1]
        batch_size = math.prod(leading_shape) if leading_shape else 1
        nominal_np = (
            nominal_action.detach().reshape(batch_size, 3).cpu().double().numpy()
        )
        state_np = state.detach().reshape(batch_size, -1).cpu().double().numpy()
        lidar_np = lidar.detach().reshape(batch_size, -1).cpu().double().numpy()
        dynamic_np = (
            dynamic_obstacle.detach()
            .reshape(batch_size, dynamic_obstacle.shape[-2], dynamic_obstacle.shape[-1])
            .cpu()
            .double()
            .numpy()
        )
        if lidar_np.shape[-1] != len(self.ray_directions):
            raise ValueError(
                "lidar ray count does not match the registered VO ray layout"
            )
        if state_np.shape[-1] < 8 or dynamic_np.shape[-1] < 10:
            raise ValueError("NavRL structured observation has an invalid shape")

        work_items = []
        for batch_index in range(batch_size):
            ray_ranges = np.clip(
                self.sensor_range_m - lidar_np[batch_index],
                0.0,
                self.sensor_range_m,
            )
            ray_ranges = self._select_static_points(ray_ranges)
            dynamic = dynamic_np[batch_index]
            dynamic_distance = np.sqrt(
                np.square(dynamic[:, 3]) + np.square(dynamic[:, 4])
            )
            dynamic_positions = dynamic[:, :3] * dynamic_distance[:, None]
            dynamic_velocities = dynamic[:, 5:8]
            width = self.dynamic_width_resolution_m * (
                np.clip(np.rint(dynamic[:, 8]), 0.0, 3.0) + 1.0
            )
            dynamic_radii = width / math.sqrt(2.0)
            dynamic_valid = np.abs(dynamic).sum(axis=-1) > 1e-6
            dynamic_heights = np.where(
                dynamic[:, 9] > 0.5,
                self.dynamic_3d_height_m,
                self.dynamic_2d_height_m,
            )
            work_items.append(
                _OfficialVOWorkItem(
                    nominal=nominal_np[batch_index],
                    state=state_np[batch_index],
                    ray_directions=self.ray_directions,
                    ray_ranges=ray_ranges,
                    dynamic_positions=dynamic_positions,
                    dynamic_velocities=dynamic_velocities,
                    dynamic_radii=dynamic_radii,
                    dynamic_valid=dynamic_valid,
                    dynamic_heights=dynamic_heights,
                    robot_radius_m=self.robot_radius_m,
                    horizon_s=self.projector.horizon_s,
                    time_step_s=self.projector.time_step_s,
                    safety_distance_m=self.projector.safety_distance_m,
                    max_speed_mps=self.projector.max_speed_mps,
                )
            )

        executor = _OFFICIAL_VO_EXECUTOR
        if executor is None or batch_size == 1:
            projected = map(_project_official_vo_work_item, work_items)
        else:
            projected = executor.map(
                _project_official_vo_work_item,
                work_items,
                chunksize=1,
            )
        selected = np.empty_like(nominal_np)
        feasible = np.ones((batch_size, 1), dtype=np.bool_)
        interventions = 0
        no_feasible = 0
        for batch_index, (action, intervention, failed) in enumerate(projected):
            selected[batch_index] = action
            feasible[batch_index, 0] = not bool(failed)
            interventions += int(intervention)
            no_feasible += int(failed)
        self.projector.decisions += batch_size
        self.projector.interventions += interventions
        self.projector.no_feasible += no_feasible

        selected_tensor = torch.as_tensor(
            selected,
            device=nominal_action.device,
            dtype=nominal_action.dtype,
        ).reshape(*leading_shape, 3)
        feasible_tensor = torch.as_tensor(
            feasible,
            device=nominal_action.device,
            dtype=torch.bool,
        ).reshape(*leading_shape, 1)
        shift = (selected_tensor - nominal_action).norm(dim=-1, keepdim=True)
        return OfficialVOBatchResult(
            action=selected_tensor,
            intervention=shift > 1e-3,
            feasible=feasible_tensor,
            shift=shift,
        )

    def _select_static_points(self, ray_ranges: np.ndarray) -> np.ndarray:
        if self.static_point_budget == 0:
            return np.full_like(ray_ranges, self.sensor_range_m)
        hit_indices = np.flatnonzero(ray_ranges < self.sensor_range_m - 1e-3)
        if len(hit_indices) <= self.static_point_budget:
            return ray_ranges
        nearest_local = np.argpartition(
            ray_ranges[hit_indices], self.static_point_budget - 1
        )[: self.static_point_budget]
        selected_indices = hit_indices[nearest_local]
        selected = np.full_like(ray_ranges, self.sensor_range_m)
        selected[selected_indices] = ray_ranges[selected_indices]
        return selected


class OfficialVOShieldedPolicy:
    """Wrap a NavRL-compatible policy without changing its checkpoint."""

    valid_modes: ClassVar[set[str]] = {"official_vo", "actor_vo"}

    def __init__(self, base_policy, shield: IsaacOfficialVONumericalShield) -> None:
        self.base_policy = base_policy
        self.shield = shield
        self._previous_action: Tensor | None = None

    def __getattr__(self, name):
        return getattr(self.base_policy, name)

    def set_rollout_mode(self, mode: str) -> None:
        if mode not in self.valid_modes:
            raise ValueError(f"Unknown VO rollout mode: {mode}")
        self.base_policy.set_rollout_mode("actor")
        self.shield.reset()
        self._previous_action = None

    def __call__(self, tensordict):
        tensordict = self.base_policy(tensordict)
        observation = tensordict["agents", "observation"]
        nominal = tensordict["method_executed_action_local"]
        result = self.shield.project(
            nominal,
            observation["state"],
            observation["lidar"],
            observation["dynamic_obstacle"],
        )
        previous = self._previous_action
        if (
            previous is None
            or previous.shape != result.action.shape
            or previous.device != result.action.device
        ):
            previous = nominal.detach()

        tensordict["method_vo_nominal_action_local"] = nominal
        tensordict["method_vo_intervention"] = result.intervention.float()
        tensordict["method_vo_feasible"] = result.feasible.float()
        tensordict["method_vo_shift"] = result.shift
        tensordict["method_teacher_action_local"] = result.action
        tensordict["method_teacher_intervention"] = result.intervention.float()
        tensordict["method_teacher_feasible"] = result.feasible.float()
        tensordict["method_teacher_used"] = result.intervention.float()
        tensordict["method_projection_gap"] = result.shift
        tensordict["method_executed_action_local"] = result.action
        tensordict["method_previous_action_local"] = previous
        tensordict["method_action_delta_local"] = result.action - previous
        tensordict["agents", "action"] = _vec_to_world(
            result.action, observation["direction"]
        )
        self._previous_action = result.action.detach()
        if hasattr(self.base_policy, "_previous_action_local"):
            self.base_policy._previous_action_local = result.action.detach()
        return tensordict


def _make_ray_directions(
    horizontal_beams: int, vertical_angles_deg: tuple[float, ...]
) -> np.ndarray:
    yaw = np.arange(horizontal_beams, dtype=np.float64) * (
        2.0 * math.pi / horizontal_beams
    )
    pitch = np.deg2rad(np.asarray(vertical_angles_deg, dtype=np.float64))
    yaw_grid, pitch_grid = np.meshgrid(yaw, pitch, indexing="ij")
    return np.stack(
        (
            np.cos(pitch_grid) * np.cos(yaw_grid),
            np.cos(pitch_grid) * np.sin(yaw_grid),
            np.sin(pitch_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)


def _vec_to_world(vector: Tensor, goal_direction: Tensor) -> Tensor:
    while goal_direction.ndim > vector.ndim and goal_direction.shape[-2] == 1:
        goal_direction = goal_direction.squeeze(-2)
    while goal_direction.ndim < vector.ndim:
        goal_direction = goal_direction.unsqueeze(-2)
    goal_direction = goal_direction.expand_as(vector)
    goal_x = goal_direction / goal_direction.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    world_z = torch.zeros_like(goal_x)
    world_z[..., 2] = 1.0
    goal_y = torch.cross(world_z, goal_x, dim=-1)
    goal_y = goal_y / goal_y.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    goal_z = torch.cross(goal_x, goal_y, dim=-1)
    goal_z = goal_z / goal_z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (
        vector[..., 0:1] * goal_x
        + vector[..., 1:2] * goal_y
        + vector[..., 2:3] * goal_z
    )
