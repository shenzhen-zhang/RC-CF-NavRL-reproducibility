#!/usr/bin/env python3
"""Analyze simplified-model performance on three unseen layouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LAYOUT_SEEDS = (17, 29, 43)
EVALUATION_SEEDS = (1183109, 1294127, 1405151, 1516183, 1627211, 1738247)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mode", "seed", "env_index", "straight_distance_m", *METRICS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"required columns missing: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty evaluation file: {path}")
    return rows


def parse_tagged_paths(values: list[str]) -> dict[int, list[Path]]:
    paths: dict[int, list[Path]] = defaultdict(list)
    for value in values:
        tag, separator, raw_path = value.partition("=")
        if not separator or not tag.isdigit():
            raise ValueError(f"path must be layout=path: {value}")
        layout = int(tag)
        if layout not in LAYOUT_SEEDS:
            raise ValueError(f"invalid layout: {layout}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[layout].append(path)
    if set(paths) != set(LAYOUT_SEEDS):
        raise ValueError("exactly layouts 17, 29, and 43 are required")
    return paths


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
        raise ValueError(f"{mode}: key mismatch")
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


def three_level_ci(
    blocks: dict[int, dict[int, dict[int, np.ndarray]]],
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    training = np.asarray(TRAINING_SEEDS, dtype=np.int64)
    layouts = np.asarray(LAYOUT_SEEDS, dtype=np.int64)
    scenes = np.asarray(EVALUATION_SEEDS, dtype=np.int64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_training = rng.choice(training, size=len(training), replace=True)
        sampled_layouts = rng.choice(layouts, size=len(layouts), replace=True)
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        estimates[index] = np.concatenate(
            [
                blocks[int(train)][int(layout)][int(scene)]
                for train in sampled_training
                for layout in sampled_layouts
                for scene in sampled_scenes
            ]
        ).mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def combine_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_rows(path))
    return rows


def analyze(
    method_paths: dict[int, list[Path]],
    baseline_paths: dict[int, list[Path]],
    samples: int,
    seed: int,
) -> dict[str, object]:
    method_rows = {layout: combine_rows(paths) for layout, paths in method_paths.items()}
    baseline_rows = {layout: combine_rows(paths) for layout, paths in baseline_paths.items()}
    pairs = {}
    for layout in LAYOUT_SEEDS:
        method = {
            train: index_mode(method_rows[layout], f"mean_risk_s{train}")
            for train in TRAINING_SEEDS
        }
        baseline = {
            train: index_mode(baseline_rows[layout], f"ppolag_matched_s{train}")
            for train in TRAINING_SEEDS
        }
        for train in TRAINING_SEEDS:
            if any(
                abs(
                    float(method[train][key]["straight_distance_m"])
                    - float(baseline[train][key]["straight_distance_m"])
                )
                > 1e-6
                for key in method[train]
            ):
                raise ValueError(f"layout {layout}, training seed {train}: pairing drift")
        pairs[layout] = (method, baseline)

    metrics: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        blocks: dict[int, dict[int, dict[int, np.ndarray]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        all_method = []
        all_baseline = []
        per_training = {train: [] for train in TRAINING_SEEDS}
        per_layout = {layout: [] for layout in LAYOUT_SEEDS}
        for layout in LAYOUT_SEEDS:
            method, baseline = pairs[layout]
            for train in TRAINING_SEEDS:
                for scene in EVALUATION_SEEDS:
                    keys = [(scene, env) for env in range(64)]
                    method_values = np.asarray(
                        [float(method[train][key][metric]) for key in keys],
                        dtype=np.float64,
                    )
                    baseline_values = np.asarray(
                        [float(baseline[train][key][metric]) for key in keys],
                        dtype=np.float64,
                    )
                    if not np.isfinite(method_values).all() or not np.isfinite(
                        baseline_values
                    ).all():
                        raise ValueError(f"non-finite values for {metric}")
                    difference = method_values - baseline_values
                    blocks[train][layout][scene] = difference
                    all_method.append(method_values)
                    all_baseline.append(baseline_values)
                    per_training[train].append(float(difference.mean()))
                    per_layout[layout].append(float(difference.mean()))
        method_values = np.concatenate(all_method)
        baseline_values = np.concatenate(all_baseline)
        differences = method_values - baseline_values
        low, high = three_level_ci(
            blocks, samples, np.random.default_rng(seed + metric_index)
        )
        mean_difference = float(differences.mean())
        training_means = {
            train: float(np.mean(values)) for train, values in per_training.items()
        }
        layout_means = {
            layout: float(np.mean(values)) for layout, values in per_layout.items()
        }
        report: dict[str, object] = {
            "method_mean": float(method_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "paired_difference": mean_difference,
            "three_level_bootstrap_ci95": [low, high],
            "directionally_consistent_training_seed_count": int(
                sum(np.sign(value) == np.sign(mean_difference) for value in training_means.values())
            ),
            "per_training_seed_difference": training_means,
            "per_layout_seed_difference": layout_means,
        }
        if metric in BINARY_METRICS:
            report["mcnemar"] = exact_mcnemar(method_values, baseline_values)
        metrics[metric] = report

    return {
        "status": "pass",
        "method": "RC-CF-NavRL simplified (mean risk)",
        "baseline": "matched PPO-Lagrangian",
        "training_seeds": list(TRAINING_SEEDS),
        "layout_creation_seeds": list(LAYOUT_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "paired_episode_count": len(TRAINING_SEEDS) * len(LAYOUT_SEEDS) * len(EVALUATION_SEEDS) * 64,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "inference_unit": "training seed, unseen layout seed, and evaluation seed",
        "input_method_files": {
            str(key): [str(path.resolve()) for path in value]
            for key, value in method_paths.items()
        },
        "input_baseline_files": {
            str(key): [str(path.resolve()) for path in value]
            for key, value in baseline_paths.items()
        },
        "input_sha256": {
            "method": {
                str(key): [sha256_file(path) for path in value]
                for key, value in method_paths.items()
            },
            "baseline": {
                str(key): [sha256_file(path) for path in value]
                for key, value in baseline_paths.items()
            },
        },
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()
    if args.bootstrap != 20_000:
        raise ValueError("the fixed bootstrap sample count is 20000")
    method_paths = parse_tagged_paths(args.method)
    baseline_paths = parse_tagged_paths(args.baseline)
    report = analyze(method_paths, baseline_paths, args.bootstrap, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
