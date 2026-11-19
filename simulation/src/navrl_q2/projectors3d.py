"""Deployment-time 3-D safety projectors for the frozen NavRL actor.

The projectors consume only the structured geometry available to NavRL.  They
do not retrain the actor or use observation histories.  The final UARP variant
adds budgeted ray refinement and split-conformal prediction tubes to a common
sampling-based reachability projector.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SafetyContext3D:
    position: Array
    velocity: Array
    goal: Array
    ray_directions: Array
    ray_ranges: Array
    dynamic_positions: Array
    dynamic_velocities: Array
    dynamic_radii: Array
    dynamic_valid: Array
    dt: float
    robot_radius: float
    raycast: Callable[[Array], Array]
    dynamic_heights: Array | None = None


@dataclass(frozen=True)
class ConformalTube:
    """Finite-sample calibrated residual radii.

    The scalar dynamic and tracking scores are multiplied by ``tau / H``.
    Static-ray error is time independent.
    """

    alpha: float = 0.10
    dynamic_q_m: float = 0.0
    tracking_q_m: float = 0.0
    static_q_m: float = 0.0
    calibration_size: int = 0


@dataclass(frozen=True)
class Reachability3DConfig:
    horizon_s: float = 1.6
    horizon_steps: int = 8
    max_acceleration_mps2: float = 8.0
    max_speed_mps: float = 2.0
    static_margin_m: float = 0.10
    dynamic_margin_m: float = 0.14
    feasibility_margin_m: float = 0.0
    sensor_range_m: float = 4.0
    dynamic_slots: int = 5
    adaptive_rays: bool = False
    adaptive_ray_budget: int = 24
    conformal_tube: ConformalTube | None = None
    policy_weight: float = 1.0
    smoothness_weight: float = 0.10
    progress_weight: float = 0.32
    clearance_weight: float = 0.08


class BaseProjector3D:
    name = "base"

    def __init__(self) -> None:
        self.decisions = 0
        self.interventions = 0
        self.no_feasible = 0

    def reset(self) -> None:
        self.decisions = 0
        self.interventions = 0
        self.no_feasible = 0

    def _record(self, nominal: Array, selected: Array, feasible: bool = True) -> Array:
        self.decisions += 1
        self.interventions += int(np.linalg.norm(selected - nominal) > 1e-3)
        self.no_feasible += int(not feasible)
        return selected.astype(np.float32, copy=False)

    def diagnostics(self) -> dict[str, float | int | str]:
        decisions = max(self.decisions, 1)
        return {
            "projector": self.name,
            "decisions": self.decisions,
            "intervention_rate": self.interventions / decisions,
            "no_feasible_rate": self.no_feasible / decisions,
        }


class OfficialVO3DProjector(BaseProjector3D):
    """Numerical port of NavRL's public 3-D ORCA-plane safety shield.

    Plane construction follows ``safeAction.cpp`` at the pinned public commit.
    The public ``linearProgram1``--``linearProgram4`` solver is ported directly
    to NumPy, without retaining the ROS transport dependency. Static ray
    endpoints remain an observation-compatible point-sphere approximation of
    the released node's DBSCAN preprocessing.
    """

    name = "navrl_official_vo_3d_port"

    def __init__(
        self,
        *,
        horizon_s: float = 2.0,
        time_step_s: float = 0.05,
        safety_distance_m: float = 0.30,
        max_speed_mps: float = 2.0 * math.sqrt(3.0),
    ) -> None:
        super().__init__()
        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        if time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        if safety_distance_m < 0.0:
            raise ValueError("safety_distance_m must be non-negative")
        if max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive")
        self.horizon_s = horizon_s
        self.time_step_s = time_step_s
        self.safety_distance_m = safety_distance_m
        self.max_speed_mps = max_speed_mps

    def project(self, nominal_action: Array, context: SafetyContext3D) -> Array:
        nominal = _clip_norm(nominal_action, self.max_speed_mps)
        positions: list[Array] = []
        velocities: list[Array] = []
        radii: list[float] = []
        agent_radii: list[float] = []

        hit = context.ray_ranges < 3.999
        for point in (
            context.position[None, :]
            + context.ray_directions[hit] * context.ray_ranges[hit, None]
        ):
            positions.append(point)
            velocities.append(np.zeros(3, dtype=np.float64))
            radii.append(0.10)
            agent_radii.append(0.0)

        for index in np.flatnonzero(context.dynamic_valid):
            if np.linalg.norm(context.dynamic_positions[index] - context.position) > 4.0:
                continue
            obstacle_position = context.dynamic_positions[index]
            obstacle_velocity = context.dynamic_velocities[index]
            obstacle_radius = float(context.dynamic_radii[index])
            obstacle_height = (
                float(context.dynamic_heights[index])
                if context.dynamic_heights is not None
                else 0.0
            )
            obstacle_width = obstacle_radius * math.sqrt(2.0)
            if obstacle_height > obstacle_width > 1e-8:
                cube_count = max(1, math.ceil(obstacle_height / obstacle_width))
                remaining_height = obstacle_height - (cube_count - 1) * obstacle_width
                z_start = (
                    obstacle_position[2]
                    - obstacle_height / 2.0
                    + remaining_height / 2.0
                )
                obstacle_centers = []
                for cube_index in range(cube_count):
                    center = obstacle_position.copy()
                    center[2] = z_start + cube_index * obstacle_width
                    obstacle_centers.append(center)
            else:
                obstacle_centers = [obstacle_position]
            for center in obstacle_centers:
                positions.append(center)
                velocities.append(obstacle_velocity)
                radii.append(obstacle_radius)
                agent_radii.append(
                    context.robot_radius + self.safety_distance_m
                )

        planes: list[tuple[Array, Array]] = []
        in_any_vo = False
        for position, velocity, radius, agent_radius in zip(
            positions, velocities, radii, agent_radii, strict=True
        ):
            plane, in_vo = _orca_plane(
                context.position,
                nominal,
                agent_radius,
                position,
                velocity,
                radius,
                self.horizon_s,
                self.time_step_s,
            )
            if plane is not None:
                planes.append(plane)
                in_any_vo |= in_vo

        if not in_any_vo:
            return self._record(nominal, nominal)
        selected, feasible = _solve_official_orca(
            nominal, planes, self.max_speed_mps
        )
        return self._record(nominal, selected, feasible)


class CBFQP3DProjector(BaseProjector3D):
    """Discrete-time control-barrier velocity projection baseline."""

    name = "cbf_qp_3d"

    def __init__(self, *, alpha: float = 1.5, margin_m: float = 0.20) -> None:
        super().__init__()
        self.alpha = alpha
        self.margin_m = margin_m

    def project(self, nominal_action: Array, context: SafetyContext3D) -> Array:
        nominal = _clip_norm(nominal_action, 2.0)
        planes: list[tuple[Array, Array]] = []
        hit = context.ray_ranges < 3.999
        hit_points = (
            context.position[None, :]
            + context.ray_directions[hit] * context.ray_ranges[hit, None]
        )
        for point in hit_points:
            relative = point - context.position
            distance = float(np.linalg.norm(relative))
            if distance < 1e-6:
                continue
            toward = relative / distance
            clearance = distance - context.robot_radius - self.margin_m
            # toward . v <= alpha * clearance
            normal = -toward
            plane_point = toward * (self.alpha * max(clearance, 0.0))
            planes.append((plane_point, normal))

        for index in np.flatnonzero(context.dynamic_valid):
            relative = context.dynamic_positions[index] - context.position
            distance = float(np.linalg.norm(relative))
            if distance > 4.0 or distance < 1e-6:
                continue
            toward = relative / distance
            clearance = (
                distance
                - context.robot_radius
                - context.dynamic_radii[index]
                - self.margin_m
            )
            upper = float(toward @ context.dynamic_velocities[index]) + self.alpha * max(
                clearance, 0.0
            )
            planes.append((toward * upper, -toward))

        selected, feasible = _project_halfspaces(nominal, planes, 2.0)
        if not feasible:
            selected = np.zeros(3, dtype=np.float64)
        return self._record(nominal, selected, feasible)


class ReachabilityProjector3D(BaseProjector3D):
    """Policy-consistent sampling MPC with optional adaptive rays/tubes."""

    def __init__(self, config: Reachability3DConfig | None = None) -> None:
        super().__init__()
        self.config = config or Reachability3DConfig()
        if self.config.adaptive_rays and self.config.conformal_tube is not None:
            self.name = "uarp_navrl_3d"
        elif self.config.adaptive_rays:
            self.name = "adaptive_ray_reachability_3d"
        else:
            self.name = "sampling_mpc_3d"

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        tube = self.config.conformal_tube
        values.update(
            {
                "adaptive_ray_budget": (
                    self.config.adaptive_ray_budget if self.config.adaptive_rays else 0
                ),
                "horizon_s": self.config.horizon_s,
                "horizon_steps": self.config.horizon_steps,
                "candidate_count": 120,
                "conformal_alpha": tube.alpha if tube else None,
                "conformal_dynamic_q_m": tube.dynamic_q_m if tube else 0.0,
                "conformal_tracking_q_m": tube.tracking_q_m if tube else 0.0,
                "conformal_static_q_m": tube.static_q_m if tube else 0.0,
            }
        )
        return values

    def project(self, nominal_action: Array, context: SafetyContext3D) -> Array:
        config = self.config
        nominal = _clip_norm(nominal_action, config.max_speed_mps)
        goal_delta = context.goal - context.position
        goal_unit = goal_delta / max(float(np.linalg.norm(goal_delta)), 1e-8)
        candidates = _candidate_lattice_3d(
            nominal,
            goal_unit,
            context.velocity,
            config.max_acceleration_mps2 * context.dt,
            config.max_speed_mps,
        )
        ray_directions, ray_ranges = _refined_rays(context, config)
        risk_clearance = _minimum_clearance(
            candidates, context, ray_directions, ray_ranges, config
        )
        if config.conformal_tube is not None:
            nominal_config = replace(config, conformal_tube=None)
            feasibility_clearance = _minimum_clearance(
                candidates, context, ray_directions, ray_ranges, nominal_config
            )
        else:
            feasibility_clearance = risk_clearance
        feasible_mask = feasibility_clearance >= config.feasibility_margin_m
        any_feasible = bool(feasible_mask.any())

        nominal_deviation = np.sum((candidates - nominal[None, :]) ** 2, axis=-1)
        smoothness = np.sum((candidates - context.velocity[None, :]) ** 2, axis=-1)
        progress = candidates @ goal_unit
        clearance_cost = np.maximum(0.35 - risk_clearance, 0.0) ** 2
        cost = (
            config.policy_weight * nominal_deviation
            + config.smoothness_weight * smoothness
            - config.progress_weight * progress
            + config.clearance_weight * clearance_cost
        )
        if any_feasible:
            choice = int(np.argmin(np.where(feasible_mask, cost, np.inf)))
        else:
            choice = int(np.argmax(feasibility_clearance + 0.08 * progress))
        return self._record(nominal, candidates[choice], any_feasible)


def _candidate_lattice_3d(
    nominal: Array,
    goal_unit: Array,
    current_velocity: Array,
    max_delta: float,
    max_speed: float,
) -> Array:
    speed = float(np.linalg.norm(nominal))
    direction = nominal / max(speed, 1e-8) if speed > 0.05 else goal_unit.copy()
    horizontal = direction.copy()
    horizontal[2] = 0.0
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = goal_unit.copy()
        horizontal[2] = 0.0
    horizontal /= max(float(np.linalg.norm(horizontal)), 1e-8)
    base_yaw = math.atan2(float(horizontal[1]), float(horizontal[0]))
    base_pitch = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
    yaw_offsets = np.deg2rad(
        np.asarray((-90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90))
    )
    pitch_offsets = np.deg2rad(np.asarray((-25.0, 0.0, 25.0)))
    scales = np.asarray((0.35, 0.65, 1.0))
    directions = []
    for yaw_offset in yaw_offsets:
        for pitch_offset in pitch_offsets:
            pitch = float(np.clip(base_pitch + pitch_offset, -math.radians(55), math.radians(55)))
            yaw = base_yaw + yaw_offset
            directions.append(
                (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
            )
    directions_array = np.asarray(directions, dtype=np.float64)
    base_speed = max(speed, 0.8)
    lattice = (directions_array[:, None, :] * (base_speed * scales)[None, :, None]).reshape(
        -1, 3
    )
    extras = np.stack((nominal, goal_unit * base_speed, np.zeros(3)), axis=0)
    candidates = np.concatenate((extras, lattice), axis=0)
    delta = candidates - current_velocity[None, :]
    norms = np.linalg.norm(delta, axis=-1, keepdims=True)
    candidates = current_velocity[None, :] + delta * np.minimum(1.0, max_delta / np.maximum(norms, 1e-8))
    norms = np.linalg.norm(candidates, axis=-1, keepdims=True)
    return candidates * np.minimum(1.0, max_speed / np.maximum(norms, 1e-8))


def _refined_rays(
    context: SafetyContext3D, config: Reachability3DConfig
) -> tuple[Array, Array]:
    directions = context.ray_directions
    ranges = context.ray_ranges
    if not config.adaptive_rays or config.adaptive_ray_budget <= 0:
        return directions, ranges

    goal = context.goal - context.position
    goal /= max(float(np.linalg.norm(goal)), 1e-8)
    goal_yaw = math.atan2(float(goal[1]), float(goal[0]))
    goal_pitch = math.asin(float(np.clip(goal[2], -1.0, 1.0)))
    budget = config.adaptive_ray_budget
    forward_count = min(12, budget)
    forward_yaw = np.deg2rad(np.linspace(-27.5, 27.5, forward_count)) + goal_yaw
    forward = np.stack(
        (
            np.cos(goal_pitch) * np.cos(forward_yaw),
            np.cos(goal_pitch) * np.sin(forward_yaw),
            np.full(forward_count, math.sin(goal_pitch)),
        ),
        axis=-1,
    )
    extras = [forward]
    remaining = budget - forward_count
    if remaining > 0:
        discontinuity = np.abs(np.roll(ranges, 4) - np.roll(ranges, -4)) + 0.5 * (
            config.sensor_range_m - ranges
        )
        indices = np.argsort(discontinuity)[-remaining:]
        base = directions[indices]
        yaw = np.arctan2(base[:, 1], base[:, 0]) + np.deg2rad(
            np.where(indices % 2 == 0, 5.0, -5.0)
        )
        pitch = np.arcsin(np.clip(base[:, 2], -1.0, 1.0))
        refined = np.stack(
            (np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)),
            axis=-1,
        )
        extras.append(refined)
    extra_directions = np.concatenate(extras, axis=0)
    extra_ranges = context.raycast(extra_directions)
    return (
        np.concatenate((directions, extra_directions), axis=0),
        np.concatenate((ranges, extra_ranges), axis=0),
    )


def _minimum_clearance(
    candidates: Array,
    context: SafetyContext3D,
    ray_directions: Array,
    ray_ranges: Array,
    config: Reachability3DConfig,
) -> Array:
    times = np.linspace(context.dt, config.horizon_s, config.horizon_steps)
    robot_future = candidates[:, None, :] * times[None, :, None]
    tube = config.conformal_tube
    static_tube = tube.static_q_m if tube else 0.0

    hit = ray_ranges < config.sensor_range_m - 1e-3
    if hit.any():
        hit_points = ray_directions[hit] * ray_ranges[hit, None]
        static_distance = np.linalg.norm(
            robot_future[:, :, None, :] - hit_points[None, None, :, :], axis=-1
        ) - (context.robot_radius + config.static_margin_m + static_tube)
        static_clearance = static_distance.min(axis=(1, 2))
    else:
        static_clearance = np.full(len(candidates), np.inf)

    valid_indices = np.flatnonzero(context.dynamic_valid)
    if not len(valid_indices):
        return static_clearance
    relative_all = context.dynamic_positions[valid_indices] - context.position[None, :]
    distances = np.linalg.norm(relative_all, axis=-1)
    order = np.argsort(distances)
    order = order[distances[order] < config.sensor_range_m][: config.dynamic_slots]
    if not len(order):
        return static_clearance
    selected = valid_indices[order]
    relative = context.dynamic_positions[selected] - context.position[None, :]
    obstacle_future = relative[None, :, :] + times[:, None, None] * context.dynamic_velocities[
        selected
    ][None, :, :]
    # A nonzero intercept covers observation latency at the first node.
    scale = 0.25 + 0.75 * times / config.horizon_s
    uncertainty = (
        (tube.dynamic_q_m + tube.tracking_q_m) * scale if tube else np.zeros_like(times)
    )
    dynamic_distance = np.linalg.norm(
        robot_future[:, :, None, :] - obstacle_future[None, :, :, :], axis=-1
    ) - (
        context.robot_radius
        + context.dynamic_radii[selected][None, None, :]
        + config.dynamic_margin_m
        + uncertainty[None, :, None]
    )
    dynamic_clearance = dynamic_distance.min(axis=(1, 2))
    return np.minimum(static_clearance, dynamic_clearance)


def _orca_plane(
    agent_position: Array,
    agent_velocity: Array,
    agent_radius: float,
    obstacle_position: Array,
    obstacle_velocity: Array,
    obstacle_radius: float,
    horizon: float,
    dt: float,
) -> tuple[tuple[Array, Array] | None, bool]:
    """Port ``safeAction::Agent::getORCAPlane`` with finite guards."""

    relative_position = obstacle_position - agent_position
    relative_velocity = agent_velocity - obstacle_velocity
    distance_sq = float(relative_position @ relative_position)
    combined_radius = agent_radius + obstacle_radius
    if distance_sq < 1e-12:
        normal = np.asarray((1.0, 0.0, 0.0))
        return (agent_velocity + normal * combined_radius / dt, normal), True

    in_vo = False
    if distance_sq > combined_radius**2:
        inv_horizon = 1.0 / horizon
        w = relative_velocity - inv_horizon * relative_position
        w_sq = float(w @ w)
        dot_product = float(w @ relative_position)
        relative_position_norm = math.sqrt(distance_sq)
        relative_velocity_norm = float(np.linalg.norm(relative_velocity))
        angle_vo = math.asin(
            np.clip(combined_radius / relative_position_norm, -1.0, 1.0)
        )
        if relative_velocity_norm > 1e-12:
            angle_velocity = math.acos(
                np.clip(
                    float(relative_position @ relative_velocity)
                    / (relative_position_norm * relative_velocity_norm),
                    -1.0,
                    1.0,
                )
            )
        else:
            angle_velocity = math.inf

        if dot_product < 0.0 and dot_product**2 > combined_radius**2 * w_sq:
            distance_at_horizon = float(
                np.linalg.norm(relative_velocity * horizon - relative_position)
            )
            in_vo = distance_at_horizon < combined_radius
            w_length = math.sqrt(max(w_sq, 1e-12))
            normal = w / w_length
            u = (combined_radius * inv_horizon - w_length) * normal
        else:
            in_vo = angle_velocity < angle_vo
            a = distance_sq
            b = float(relative_position @ relative_velocity)
            cross_sq = float(np.linalg.norm(np.cross(relative_position, relative_velocity)) ** 2)
            c = float(relative_velocity @ relative_velocity) - cross_sq / max(
                distance_sq - combined_radius**2, 1e-12
            )
            discriminant = max(b * b - a * c, 0.0)
            t = (b + math.sqrt(discriminant)) / a
            w = relative_velocity - t * relative_position
            w_length = float(np.linalg.norm(w))
            if w_length < 1e-10:
                # The released C++ normalizes this zero vector for exactly
                # collinear encounters. Use the inward cone-limit direction
                # so the port remains finite and slows a head-on command.
                normal = -relative_position / relative_position_norm
            else:
                normal = w / w_length
            u = (combined_radius * t - w_length) * normal
    else:
        w = relative_velocity - relative_position / dt
        w_length = float(np.linalg.norm(w))
        if w_length < 1e-10:
            normal = -relative_position / math.sqrt(distance_sq)
        else:
            normal = w / w_length
        u = (combined_radius / dt - w_length) * normal
        in_vo = True
    return (agent_velocity + u, normal), in_vo


def _linear_program_1(
    planes: list[tuple[Array, Array]],
    plane_no: int,
    line_point: Array,
    line_direction: Array,
    radius: float,
    optimal_velocity: Array,
    direction_opt: bool,
    result: Array,
) -> tuple[bool, Array]:
    dot_product = float(line_point @ line_direction)
    discriminant = dot_product**2 + radius**2 - float(line_point @ line_point)
    if discriminant < 0.0:
        return False, result

    sqrt_discriminant = math.sqrt(discriminant)
    left = -dot_product - sqrt_discriminant
    right = -dot_product + sqrt_discriminant
    for point, normal in planes[:plane_no]:
        numerator = float((point - line_point) @ normal)
        denominator = float(line_direction @ normal)
        if denominator**2 <= 1e-12:
            if numerator > 0.0:
                return False, result
            continue
        value = numerator / denominator
        if denominator >= 0.0:
            left = max(left, value)
        else:
            right = min(right, value)
        if left > right:
            return False, result

    if direction_opt:
        value = right if float(optimal_velocity @ line_direction) > 0.0 else left
    else:
        value = float(line_direction @ (optimal_velocity - line_point))
        value = min(max(value, left), right)
    return True, line_point + value * line_direction


def _linear_program_2(
    planes: list[tuple[Array, Array]],
    plane_no: int,
    radius: float,
    optimal_velocity: Array,
    direction_opt: bool,
    result: Array,
) -> tuple[bool, Array]:
    point, normal = planes[plane_no]
    plane_distance = float(point @ normal)
    plane_distance_sq = plane_distance**2
    radius_sq = radius**2
    if plane_distance_sq > radius_sq:
        return False, result

    plane_radius_sq = radius_sq - plane_distance_sq
    plane_center = plane_distance * normal
    if direction_opt:
        plane_optimal_velocity = optimal_velocity - float(
            optimal_velocity @ normal
        ) * normal
        plane_optimal_norm_sq = float(
            plane_optimal_velocity @ plane_optimal_velocity
        )
        if plane_optimal_norm_sq <= 1e-12:
            candidate = plane_center.copy()
        else:
            candidate = plane_center + math.sqrt(
                plane_radius_sq / plane_optimal_norm_sq
            ) * plane_optimal_velocity
    else:
        candidate = optimal_velocity + float(
            (point - optimal_velocity) @ normal
        ) * normal
        if float(candidate @ candidate) > radius_sq:
            plane_result = candidate - plane_center
            plane_result_norm_sq = float(plane_result @ plane_result)
            if plane_result_norm_sq <= 1e-12:
                return False, result
            candidate = plane_center + math.sqrt(
                plane_radius_sq / plane_result_norm_sq
            ) * plane_result

    for index in range(plane_no):
        prior_point, prior_normal = planes[index]
        if float(prior_normal @ (prior_point - candidate)) <= 0.0:
            continue
        cross_product = np.cross(prior_normal, normal)
        cross_norm_sq = float(cross_product @ cross_product)
        if cross_norm_sq <= 1e-12:
            return False, result
        line_direction = cross_product / math.sqrt(cross_norm_sq)
        line_normal = np.cross(line_direction, normal)
        denominator = float(line_normal @ prior_normal)
        if abs(denominator) <= 1e-12:
            return False, result
        line_point = point + (
            float((prior_point - point) @ prior_normal) / denominator
        ) * line_normal
        success, candidate = _linear_program_1(
            planes,
            index,
            line_point,
            line_direction,
            radius,
            optimal_velocity,
            direction_opt,
            candidate,
        )
        if not success:
            return False, result
    return True, candidate


def _linear_program_3(
    planes: list[tuple[Array, Array]],
    radius: float,
    optimal_velocity: Array,
    direction_opt: bool,
) -> tuple[int, Array]:
    if direction_opt:
        result = optimal_velocity * radius
    elif float(optimal_velocity @ optimal_velocity) > radius**2:
        result = _clip_norm(optimal_velocity, radius)
    else:
        result = optimal_velocity.astype(np.float64, copy=True)

    for index, (point, normal) in enumerate(planes):
        if float(normal @ (point - result)) <= 0.0:
            continue
        previous = result.copy()
        success, result = _linear_program_2(
            planes,
            index,
            radius,
            optimal_velocity,
            direction_opt,
            result,
        )
        if not success:
            return index, previous
    return len(planes), result


def _linear_program_4(
    planes: list[tuple[Array, Array]],
    begin_plane: int,
    radius: float,
    result: Array,
) -> Array:
    distance = 0.0
    for index in range(begin_plane, len(planes)):
        point, normal = planes[index]
        violation = float(normal @ (point - result))
        if violation <= distance:
            continue

        projected_planes: list[tuple[Array, Array]] = []
        for prior_index in range(index):
            prior_point, prior_normal = planes[prior_index]
            cross_product = np.cross(prior_normal, normal)
            cross_norm_sq = float(cross_product @ cross_product)
            if cross_norm_sq <= 1e-12:
                if float(normal @ prior_normal) > 0.0:
                    continue
                projected_point = 0.5 * (point + prior_point)
            else:
                line_normal = np.cross(cross_product, normal)
                denominator = float(line_normal @ prior_normal)
                if abs(denominator) <= 1e-12:
                    continue
                projected_point = point + (
                    float((prior_point - point) @ prior_normal) / denominator
                ) * line_normal

            projected_normal = prior_normal - normal
            projected_normal_norm = float(np.linalg.norm(projected_normal))
            if projected_normal_norm <= 1e-12:
                continue
            projected_planes.append(
                (projected_point, projected_normal / projected_normal_norm)
            )

        previous = result.copy()
        plane_fail, candidate = _linear_program_3(
            projected_planes,
            radius,
            normal,
            True,
        )
        if plane_fail < len(projected_planes):
            result = previous
        else:
            result = candidate
        distance = float(normal @ (point - result))
    return result


def _solve_official_orca(
    preferred: Array,
    planes: list[tuple[Array, Array]],
    max_speed: float,
) -> tuple[Array, bool]:
    """Port the released 3-D LP solver and report residual feasibility."""

    preferred = np.asarray(preferred, dtype=np.float64)
    plane_fail, result = _linear_program_3(
        planes, max_speed, preferred, False
    )
    if plane_fail < len(planes):
        result = _linear_program_4(planes, plane_fail, max_speed, result)
    feasible = all(
        float(normal @ (point - result)) <= 2e-6
        for point, normal in planes
    )
    return result, feasible


def _project_halfspaces(
    preferred: Array,
    planes: list[tuple[Array, Array]],
    max_speed: float,
    *,
    iterations: int = 40,
) -> tuple[Array, bool]:
    velocity = _clip_norm(preferred, max_speed).astype(np.float64)
    for _ in range(iterations):
        max_violation = 0.0
        for point, normal in planes:
            violation = float((velocity - point) @ normal)
            if violation < 0.0:
                velocity -= violation * normal
                velocity = _clip_norm(velocity, max_speed)
                max_violation = max(max_violation, -violation)
        if max_violation < 1e-6:
            break
    feasible = all(float((velocity - point) @ normal) >= -2e-3 for point, normal in planes)
    return velocity, feasible


def _clip_norm(vector: Array, limit: float) -> Array:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm < 1e-12:
        return vector.copy()
    return vector * (limit / norm)
