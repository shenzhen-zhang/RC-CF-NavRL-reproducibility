"""Benchmark frozen NavRL/v19 actor and optional safety-layer paths."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from isaac_app import close_simulation_app, create_simulation_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = Path(os.environ["NAVRL_OFFICIAL_ROOT"]).resolve()
OFFICIAL_SCRIPTS = OFFICIAL_ROOT / "training" / "scripts"
if not OFFICIAL_SCRIPTS.is_dir():
    raise RuntimeError(f"Pinned NavRL scripts were not found at {OFFICIAL_SCRIPTS}")
sys.path.insert(0, str(OFFICIAL_SCRIPTS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def unique_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def parameter_summary(modules: Iterable[torch.nn.Module]) -> dict[str, float | int]:
    parameters = unique_parameters(modules)
    count = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    storage_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    return {
        "parameter_count": int(count),
        "trainable_parameter_count": int(trainable),
        "parameter_storage_mib": float(storage_bytes / (1024**2)),
    }


@torch.inference_mode()
def benchmark_path(
    forward: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    if warmup < 1 or iterations < 2:
        raise ValueError("benchmark requires warmup >= 1 and iterations >= 2")
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize(device)
    cuda_times_ms: list[float] = []
    wall_times_ms: list[float] = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter_ns()
        start_event.record()
        output = forward()
        end_event.record()
        end_event.synchronize()
        wall_end = time.perf_counter_ns()
        if not torch.isfinite(output).all():
            raise RuntimeError("runtime benchmark produced a non-finite action")
        cuda_times_ms.append(float(start_event.elapsed_time(end_event)))
        wall_times_ms.append((wall_end - wall_start) / 1_000_000.0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    for _ in range(10):
        forward()
    torch.cuda.synchronize(device)
    peak_delta = max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
    return {
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "cuda_mean_ms": float(statistics.fmean(cuda_times_ms)),
        "cuda_median_ms": float(statistics.median(cuda_times_ms)),
        "cuda_p95_ms": float(percentile(cuda_times_ms, 0.95)),
        "wall_mean_ms": float(statistics.fmean(wall_times_ms)),
        "wall_median_ms": float(statistics.median(wall_times_ms)),
        "wall_p95_ms": float(percentile(wall_times_ms, 0.95)),
        "wall_p99_ms": float(percentile(wall_times_ms, 0.99)),
        "peak_working_memory_mib": float(peak_delta / (1024**2)),
    }


@hydra.main(config_path="cfg", config_name="train", version_base=None)
def main(cfg) -> None:
    if not cfg.checkpoint:
        raise ValueError("Set checkpoint=/absolute/path/to/v19_checkpoint.pt")
    if not torch.cuda.is_available():
        raise RuntimeError("The runtime benchmark requires CUDA")
    simulation_app = create_simulation_app(headless=True)
    success = False
    try:
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from radcppo_adapter import RADCPPO
        from radcppo_navrl.official_vo import IsaacOfficialVONumericalShield
        from torchrl.envs.transforms import Compose, TransformedEnv

        from env import NavigationEnv

        device = torch.device(str(cfg.device))
        env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, env.drone.params).to(device)
        transformed_env = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))
        ).eval()
        policy = RADCPPO(
            cfg.algo,
            transformed_env.observation_spec,
            transformed_env.action_spec,
            device,
        )
        torch.nn.Module.train(policy, False)
        checkpoint_path = Path(str(cfg.checkpoint)).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_report = policy.load_state_dict(checkpoint, strict=False)
        if load_report.unexpected_keys:
            raise RuntimeError(f"unexpected checkpoint keys: {load_report.unexpected_keys}")
        missing_task = [
            key
            for key in load_report.missing_keys
            if key.startswith(
                (
                    "feature_extractor.",
                    "actor_head.alpha_layer.",
                    "actor_head.beta_layer.",
                )
            )
        ]
        if missing_task:
            raise RuntimeError(f"missing task-policy keys: {missing_task}")
        policy.sync_training_schedule()
        if policy.safety_feature_extractor is None:
            raise RuntimeError("v19 benchmark requires the decoupled safety encoder")

        vertical_angles = tuple(
            float(value)
            for value in np.linspace(
                float(cfg.sensor.lidar_vfov[0]),
                float(cfg.sensor.lidar_vfov[1]),
                int(cfg.sensor.lidar_vbeams),
            )
        )

        def make_vo():
            return IsaacOfficialVONumericalShield(
                sensor_range_m=float(cfg.sensor.lidar_range),
                horizontal_beams=int(360.0 / float(cfg.sensor.lidar_hres)),
                vertical_angles_deg=vertical_angles,
                static_point_budget=int(
                    OmegaConf.select(cfg, "official_vo.static_point_budget", default=48)
                ),
                action_limit_mps=float(cfg.algo.actor.action_limit),
            )

        navrl_vo = make_vo()
        v19_vo = make_vo()
        reset_td = transformed_env.reset().to(device)
        available_batch = int(reset_td.batch_size[0])
        batch_sizes = [
            int(value)
            for value in OmegaConf.select(
                cfg, "benchmark_batch_sizes", default=[1, available_batch]
            )
        ]
        warmup = int(OmegaConf.select(cfg, "benchmark_warmup", default=30))
        iterations = int(OmegaConf.select(cfg, "benchmark_iterations", default=100))
        action_limit = float(cfg.algo.actor.action_limit)

        baseline_modules = [
            policy.feature_extractor,
            policy.actor_head.alpha_layer,
            policy.actor_head.beta_layer,
        ]
        method_modules = [
            policy.feature_extractor,
            policy.safety_feature_extractor,
            policy.actor_head,
        ]
        paths = (
            "navrl_actor",
            "navrl_official_vo",
            "v19_actor",
            "v19_official_vo",
            "v19_pcarp",
        )
        report: dict[str, object] = {
            "schema_version": 1,
            "benchmark_definition": (
                "Minimum deployment graph on one fixed Isaac observation batch. "
                "NavRL actor uses the frozen common-start task encoder and Beta head "
                "embedded in the v19 checkpoint. v19 actor additionally uses its "
                "decoupled safety encoder and learned gate/residual head. VO and "
                "PC-ARP rows add only their safety calculation. Physics and low-level "
                "controller stepping are excluded."
            ),
            "official_navrl_commit": (
                "3725bcc2e7c1be4ecf1455d922299ae85042603a"
            ),
            "isaac_sim_version": "2023.1.0-hotfix.1",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "missing_checkpoint_keys": list(load_report.missing_keys),
            "path_names": list(paths),
            "batch_sizes": batch_sizes,
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "navrl_actor_parameters": parameter_summary(baseline_modules),
            "v19_actor_parameters": parameter_summary(method_modules),
            "official_vo_parameters": navrl_vo.diagnostics(),
            "source_sha256": {
                "methods/radcppo_navrl/benchmark_v19_deployment_runtime.py": sha256(
                    Path(__file__).resolve()
                ),
                "methods/radcppo_navrl/radcppo_adapter.py": sha256(
                    PROJECT_ROOT / "methods/radcppo_navrl/radcppo_adapter.py"
                ),
                "src/radcppo_navrl/actor.py": sha256(
                    PROJECT_ROOT / "src/radcppo_navrl/actor.py"
                ),
                "src/radcppo_navrl/official_vo.py": sha256(
                    PROJECT_ROOT / "src/radcppo_navrl/official_vo.py"
                ),
                "src/radcppo_navrl/reachability_teacher.py": sha256(
                    PROJECT_ROOT / "src/radcppo_navrl/reachability_teacher.py"
                ),
            },
            "batches": {},
        }

        for batch_size in batch_sizes:
            if batch_size < 1 or batch_size > available_batch:
                raise ValueError(
                    f"benchmark batch {batch_size} is outside [1, {available_batch}]"
                )
            navrl_vo.reset()
            v19_vo.reset()
            baseline_td = reset_td[:batch_size].clone()
            method_td = reset_td[:batch_size].clone()

            def navrl_actor() -> torch.Tensor:
                policy.feature_extractor(baseline_td)
                return policy.actor_head.base_mean_action(
                    baseline_td["_feature"], action_limit
                )

            def v19_actor() -> torch.Tensor:
                policy.feature_extractor(method_td)
                policy.safety_feature_extractor(method_td)
                alpha, beta, _, _ = policy.actor_head(
                    method_td["_feature"], method_td["_safety_feature"]
                )
                return 2.0 * (alpha / (alpha + beta)) * action_limit - action_limit

            def observation(tensordict):
                return tensordict["agents", "observation"]

            def navrl_with_vo() -> torch.Tensor:
                nominal = navrl_actor()
                obs = observation(baseline_td)
                return navrl_vo.project(
                    nominal, obs["state"], obs["lidar"], obs["dynamic_obstacle"]
                ).action

            def v19_with_vo() -> torch.Tensor:
                nominal = v19_actor()
                obs = observation(method_td)
                return v19_vo.project(
                    nominal, obs["state"], obs["lidar"], obs["dynamic_obstacle"]
                ).action

            def v19_with_pcarp() -> torch.Tensor:
                nominal = v19_actor()
                obs = observation(method_td)
                teacher = policy.teacher(
                    nominal, obs["state"], obs["lidar"], obs["dynamic_obstacle"]
                )
                use_teacher = teacher.intervention | ~teacher.feasible
                return torch.where(use_teacher, teacher.action, nominal)

            forwards = {
                "navrl_actor": navrl_actor,
                "navrl_official_vo": navrl_with_vo,
                "v19_actor": v19_actor,
                "v19_official_vo": v19_with_vo,
                "v19_pcarp": v19_with_pcarp,
            }
            report["batches"][str(batch_size)] = {
                path: benchmark_path(
                    forwards[path], warmup=warmup, iterations=iterations, device=device
                )
                for path in paths
            }

        navrl_count = int(report["navrl_actor_parameters"]["parameter_count"])
        v19_count = int(report["v19_actor_parameters"]["parameter_count"])
        report["actor_parameter_overhead"] = {
            "absolute": v19_count - navrl_count,
            "relative_percent": 100.0 * (v19_count - navrl_count) / navrl_count,
        }
        output_path = Path(
            str(
                OmegaConf.select(
                    cfg,
                    "benchmark_output",
                    default="results/v19_runtime/runtime_report.json",
                )
            )
        ).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "passed", "output": str(output_path)}))
        success = True
    finally:
        close_simulation_app(simulation_app, success=success)


if __name__ == "__main__":
    main()
