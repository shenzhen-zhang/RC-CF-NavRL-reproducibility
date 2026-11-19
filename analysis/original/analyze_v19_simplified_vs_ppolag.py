#!/usr/bin/env python3
"""Compare the mean-risk simplified v19 model with matched PPO-Lagrangian."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


HISTORICAL_SEEDS = (1183109, 1294127, 1405151, 1516183, 1627211, 1738247)
EXTENSION_SEEDS = (
    1849289,
    1960313,
    2071349,
    2182387,
    2293429,
    2404471,
    2515519,
    2626573,
    2737631,
    2848697,
    2959769,
    3070849,
)
EVALUATION_SEEDS = HISTORICAL_SEEDS + EXTENSION_SEEDS
TRAINING_SEEDS = (0, 1, 2, 3, 4)
METRICS = (
    "success",
    "goal_reach",
    "collision",
    "near_miss",
    "path_efficiency",
    "min_clearance_m",
    "empirical_cost_cvar",
    "mean_command_acceleration_mps2",
    "mean_command_jerk_mps3",
    "action_oscillation_rate",
    "stop_rate",
)
BINARY_METRICS = {"success", "goal_reach", "collision", "near_miss"}


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = {"mode", "seed", "env_index", "straight_distance_m", *METRICS}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"required columns missing: {path}")
            rows.extend(reader)
    if not rows:
        raise ValueError("no evaluation rows supplied")
    return rows


def index_mode(
    rows: list[dict[str, str]], mode: str
) -> dict[tuple[int, int], dict[str, str]]:
    indexed: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if row["mode"] != mode:
            continue
        key = (int(row["seed"]), int(row["env_index"]))
        if key in indexed:
            raise ValueError(f"duplicate key {mode}/{key}")
        indexed[key] = row
    expected = {(seed, env) for seed in EVALUATION_SEEDS for env in range(64)}
    if set(indexed) != expected:
        missing = len(expected - set(indexed))
        extra = len(set(indexed) - expected)
        raise ValueError(f"{mode}: key mismatch; missing={missing}, extra={extra}")
    return indexed


def exact_mcnemar(method: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    method_only = int(np.sum((method == 1.0) & (baseline == 0.0)))
    baseline_only = int(np.sum((method == 0.0) & (baseline == 1.0)))
    discordant = method_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail_count = min(method_only, baseline_only)
        tail = sum(math.comb(discordant, k) for k in range(tail_count + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "method_only": method_only,
        "baseline_only": baseline_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def hierarchical_ci(
    blocks: dict[int, dict[int, np.ndarray]],
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    training = np.asarray(TRAINING_SEEDS, dtype=np.int64)
    scenes = np.asarray(EVALUATION_SEEDS, dtype=np.int64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_training = rng.choice(training, size=len(training), replace=True)
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        estimates[index] = np.concatenate(
            [
                blocks[int(training_seed)][int(scene_seed)]
                for training_seed in sampled_training
                for scene_seed in sampled_scenes
            ]
        ).mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def analyze(rows: list[dict[str, str]], samples: int, seed: int) -> dict[str, object]:
    pairs = {}
    for training_seed in TRAINING_SEEDS:
        method = index_mode(rows, f"mean_risk_s{training_seed}")
        baseline = index_mode(rows, f"ppolag_matched_s{training_seed}")
        if any(
            abs(
                float(method[key]["straight_distance_m"])
                - float(baseline[key]["straight_distance_m"])
            )
            > 1e-6
            for key in method
        ):
            raise ValueError(f"training seed {training_seed}: pairing drift")
        pairs[training_seed] = (method, baseline)

    metrics: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        blocks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        all_method = []
        all_baseline = []
        per_training_seed = {}
        for training_seed, (method, baseline) in pairs.items():
            seed_method = []
            seed_baseline = []
            for evaluation_seed in EVALUATION_SEEDS:
                keys = [(evaluation_seed, env_index) for env_index in range(64)]
                method_values = np.asarray(
                    [float(method[key][metric]) for key in keys], dtype=np.float64
                )
                baseline_values = np.asarray(
                    [float(baseline[key][metric]) for key in keys], dtype=np.float64
                )
                if not np.isfinite(method_values).all() or not np.isfinite(
                    baseline_values
                ).all():
                    raise ValueError(f"non-finite values for {metric}")
                blocks[training_seed][evaluation_seed] = (
                    method_values - baseline_values
                )
                seed_method.append(method_values)
                seed_baseline.append(baseline_values)
            method_values = np.concatenate(seed_method)
            baseline_values = np.concatenate(seed_baseline)
            per_training_seed[training_seed] = float(
                (method_values - baseline_values).mean()
            )
            all_method.append(method_values)
            all_baseline.append(baseline_values)

        method_values = np.concatenate(all_method)
        baseline_values = np.concatenate(all_baseline)
        differences = method_values - baseline_values
        low, high = hierarchical_ci(
            blocks, samples, np.random.default_rng(seed + metric_index)
        )
        mean_difference = float(differences.mean())
        metric_report: dict[str, object] = {
            "method_mean": float(method_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "paired_difference": mean_difference,
            "hierarchical_bootstrap_ci95": [low, high],
            "training_seed_mean_sd": float(
                np.std(list(per_training_seed.values()), ddof=1)
            ),
            "directionally_consistent_training_seed_count": int(
                sum(
                    np.sign(value) == np.sign(mean_difference)
                    for value in per_training_seed.values()
                )
            ),
            "per_training_seed_difference": per_training_seed,
        }
        if metric in BINARY_METRICS:
            metric_report["mcnemar"] = exact_mcnemar(
                method_values, baseline_values
            )
        metrics[metric] = metric_report

    return {
        "status": "pass",
        "method": "RC-CF-NavRL simplified (mean risk)",
        "baseline": "matched PPO-Lagrangian",
        "training_seeds": list(TRAINING_SEEDS),
        "historical_evaluation_seeds": list(HISTORICAL_SEEDS),
        "extension_evaluation_seeds": list(EXTENSION_SEEDS),
        "evaluation_seed_count": len(EVALUATION_SEEDS),
        "paired_episode_count": len(TRAINING_SEEDS) * len(EVALUATION_SEEDS) * 64,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "inference_unit": (
            "training-seed and evaluation-seed hierarchical paired bootstrap"
        ),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()
    if args.bootstrap != 20_000:
        raise ValueError("the fixed bootstrap sample count is 20000")
    rows = read_rows(args.episodes)
    report = analyze(rows, args.bootstrap, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
