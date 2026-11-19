"""Train RA-DC-PPO in the pinned official NavRL Isaac Sim environment."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from radcppo_navrl.checkpointing import filter_shape_compatible_state_dict

from isaac_app import (
    close_simulation_app,
    create_simulation_app,
    validate_navrl_dynamic_obstacles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = Path(
    os.environ.get(
        "NAVRL_OFFICIAL_ROOT",
        PROJECT_ROOT / "external" / "NavRL_official_3725bcc" / "isaac-training",
    )
).resolve()
OFFICIAL_SCRIPTS = OFFICIAL_ROOT / "training" / "scripts"
if not OFFICIAL_SCRIPTS.is_dir():
    raise RuntimeError(f"Pinned NavRL scripts were not found at {OFFICIAL_SCRIPTS}")
sys.path.insert(0, str(OFFICIAL_SCRIPTS))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256_manifest() -> dict[str, str]:
    relative_paths = (
        "methods/radcppo_navrl/radcppo_adapter.py",
        "methods/radcppo_navrl/train.py",
        "methods/radcppo_navrl/cfg/train.yaml",
        "methods/radcppo_navrl/cfg/ppo.yaml",
        "src/radcppo_navrl/actor.py",
        "src/radcppo_navrl/distributional.py",
        "src/radcppo_navrl/reachability_teacher.py",
        "src/radcppo_navrl/risk_selector.py",
    )
    return {
        relative_path: file_sha256(PROJECT_ROOT / relative_path)
        for relative_path in relative_paths
    }


def seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="cfg", config_name="train", version_base=None)
def main(cfg):
    if not cfg.checkpoint and not bool(cfg.allow_from_scratch):
        raise ValueError(
            "RA-DC-PPO defaults to released NavRL checkpoint fine-tuning. "
            "Set checkpoint=/absolute/path/navrl_checkpoint.pt or explicitly "
            "set allow_from_scratch=true."
        )
    validate_navrl_dynamic_obstacles(int(cfg.env_dyn.num_obstacles))
    simulation_app = create_simulation_app(headless=bool(cfg.headless))
    collector = None
    run = None
    metrics_handle = None
    success = False
    try:
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl import EpisodeStats, SyncDataCollector
        from omni_drones.utils.torchrl.transforms import VelController
        from radcppo_adapter import RADCPPO
        from torchrl.envs.transforms import Compose, TransformedEnv
        from torchrl.envs.utils import ExplorationType
        from utils import evaluate

        from env import NavigationEnv

        run_id = cfg.wandb.run_id or wandb.util.generate_id()
        output_dir = Path(cfg.train_output_dir).resolve() / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
        (output_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )
        metrics_handle = (output_dir / "metrics.jsonl").open(
            "a" if cfg.wandb.run_id else "w", encoding="utf-8"
        )

        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=resolved_config,
            mode=cfg.wandb.mode,
            id=run_id,
            resume="must" if cfg.wandb.run_id else None,
        )

        # Terrain and obstacle assets are sampled while NavigationEnv is built.
        seed_process(int(cfg.seed))
        env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        transformed_env = TransformedEnv(
            env, Compose(VelController(controller, yaw_control=False))
        ).train()
        policy = RADCPPO(
            cfg.algo,
            transformed_env.observation_spec,
            transformed_env.action_spec,
            cfg.device,
        )
        policy.set_rollout_mode("actor")
        initial_checkpoint = None
        initial_checkpoint_sha256 = None
        initial_checkpoint_shape_skipped_keys: list[str] = []
        if cfg.checkpoint:
            checkpoint_path = Path(cfg.checkpoint).resolve()
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            compatible_checkpoint, initial_checkpoint_shape_skipped_keys = (
                filter_shape_compatible_state_dict(
                    policy,
                    checkpoint,
                    allowed_mismatch_prefixes=(
                        "safety_critic.",
                        "target_safety_critic.",
                    ),
                )
            )
            load_report = policy.load_state_dict(compatible_checkpoint, strict=False)
            if load_report.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected checkpoint keys: {load_report.unexpected_keys}"
                )
            if bool(cfg.algo.method.use_decoupled_safety_encoder) and any(
                key.startswith("safety_feature_extractor.")
                for key in load_report.missing_keys
            ):
                policy.initialize_safety_feature_extractor_from_task()
            policy.sync_training_schedule()
            run.config.update(
                {
                    "initial_checkpoint": str(checkpoint_path),
                    "initial_checkpoint_sha256": file_sha256(checkpoint_path),
                    "initial_checkpoint_missing_keys": list(load_report.missing_keys),
                    "initial_checkpoint_shape_skipped_keys": (
                        initial_checkpoint_shape_skipped_keys
                    ),
                },
                allow_val_change=True,
            )
            initial_checkpoint = str(checkpoint_path)
            initial_checkpoint_sha256 = file_sha256(checkpoint_path)

        episode_stats_keys = [
            key
            for key in transformed_env.observation_spec.keys(True, True)
            if isinstance(key, tuple) and key[0] == "stats"
        ]
        episode_stats = EpisodeStats(episode_stats_keys)
        # Seed immediately before the collector's first reset. The full and
        # ablated policies instantiate different modules and therefore consume
        # different amounts of RNG state during construction.
        seed_process(int(cfg.seed))
        transformed_env.set_seed(int(cfg.seed), static_seed=True)
        collector = SyncDataCollector(
            transformed_env,
            policy=policy,
            frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,
            total_frames=cfg.max_frame_num,
            device=cfg.device,
            return_same_td=True,
            exploration_type=ExplorationType.RANDOM,
        )

        completed_iterations = 0
        for iteration, data in enumerate(collector):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            info.update(policy.train(data))
            non_finite = {
                key: value
                for key, value in info.items()
                if isinstance(value, (int, float)) and not math.isfinite(float(value))
            }
            if non_finite:
                raise RuntimeError(f"Non-finite training metrics: {non_finite}")
            episode_stats.add(data)
            if len(episode_stats) >= transformed_env.num_envs:
                info.update(
                    {
                        "train/"
                        + (".".join(key) if isinstance(key, tuple) else key): value.float()
                        .mean()
                        .item()
                        for key, value in episode_stats.pop().items(True, True)
                    }
                )

            eval_interval = int(cfg.eval_interval)
            if eval_interval > 0 and (iteration + 1) % eval_interval == 0:
                policy.set_rollout_mode("actor")
                env.enable_render(True)
                env.eval()
                eval_info = evaluate(
                    env=transformed_env,
                    policy=policy,
                    seed=cfg.seed,
                    cfg=cfg,
                    exploration_type=ExplorationType.MEAN,
                )
                info.update(
                    {f"actor_only/{key}": value for key, value in eval_info.items()}
                )
                policy.set_rollout_mode("actor")
                env.enable_render(not cfg.headless)
                env.train()
                env.reset()

            run.log(info)
            scalar_info = {
                key: value
                for key, value in info.items()
                if isinstance(value, (str, bool, int, float))
            }
            metrics_handle.write(
                json.dumps(
                    {"iteration": iteration + 1, **scalar_info},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            metrics_handle.flush()
            completed_iterations = iteration + 1
            save_interval = int(cfg.save_interval)
            if save_interval > 0 and (iteration + 1) % save_interval == 0:
                torch.save(
                    policy.state_dict(),
                    output_dir / f"checkpoint_{iteration + 1}.pt",
                )

        final_checkpoint = output_dir / "checkpoint_final.pt"
        torch.save(policy.state_dict(), final_checkpoint)
        metadata = {
            "status": "completed",
            "run_id": run_id,
            "official_navrl_commit": "3725bcc2e7c1be4ecf1455d922299ae85042603a",
            "initial_checkpoint": initial_checkpoint,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "initial_checkpoint_shape_skipped_keys": (
                initial_checkpoint_shape_skipped_keys
            ),
            "seed": int(cfg.seed),
            "requested_max_frames": int(float(cfg.max_frame_num)),
            "completed_frames": int(collector._frames),
            "completed_iterations": completed_iterations,
            "source_sha256": source_sha256_manifest(),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        success = True
    finally:
        if metrics_handle is not None:
            metrics_handle.close()
        if collector is not None:
            collector.shutdown()
        if run is not None:
            wandb.finish()
        close_simulation_app(simulation_app, success=success)


if __name__ == "__main__":
    main()
