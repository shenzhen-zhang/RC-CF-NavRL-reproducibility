"""Run preregistered statistics for final Mean-Risk core ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
TRAINING_SEEDS = tuple(range(5))
EVALUATION_SEEDS = (
    1183109,
    1294127,
    1405151,
    1516183,
    1627211,
    1738247,
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

COMPARISONS = {
    "no_gate": "mean_risk_no_gate_s{seed}",
    "no_counterfactual": "mean_risk_no_counterfactual_s{seed}",
}
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"mode", "seed", "env_index", "straight_distance_m", *METRICS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("evaluation rows are incomplete")
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
        raise ValueError(f"{mode}: paired episode key mismatch")
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
        "full_only": method_only,
        "ablation_only": baseline_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def hierarchical_ci(
    block_means: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_training = rng.integers(0, len(TRAINING_SEEDS), len(TRAINING_SEEDS))
        sampled_scenes = rng.integers(0, len(EVALUATION_SEEDS), len(EVALUATION_SEEDS))
        estimates[index] = block_means[np.ix_(sampled_training, sampled_scenes)].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def analyze_comparison(
    rows: list[dict[str, str]], comparison: str, bootstrap: int, bootstrap_seed: int
) -> dict[str, object]:
    full_by_seed = {
        seed: index_mode(rows, f"mean_risk_s{seed}") for seed in TRAINING_SEEDS
    }
    ablation_by_seed = {
        seed: index_mode(rows, COMPARISONS[comparison].format(seed=seed))
        for seed in TRAINING_SEEDS
    }
    metrics: dict[str, object] = {}
    for metric_index, metric in enumerate(METRICS):
        full_blocks = np.empty(
            (len(TRAINING_SEEDS), len(EVALUATION_SEEDS), 64), dtype=np.float64
        )
        ablation_blocks = np.empty_like(full_blocks)
        for train_index, train_seed in enumerate(TRAINING_SEEDS):
            full = full_by_seed[train_seed]
            ablation = ablation_by_seed[train_seed]
            for scene_index, scene_seed in enumerate(EVALUATION_SEEDS):
                keys = [(scene_seed, env_index) for env_index in range(64)]
                if any(
                    abs(
                        float(full[key]["straight_distance_m"])
                        - float(ablation[key]["straight_distance_m"])
                    )
                    > 1e-6
                    for key in keys
                ):
                    raise ValueError("paired initial distance drift")
                full_blocks[train_index, scene_index] = [
                    float(full[key][metric]) for key in keys
                ]
                ablation_blocks[train_index, scene_index] = [
                    float(ablation[key][metric]) for key in keys
                ]
        if not np.all(np.isfinite(full_blocks)) or not np.all(
            np.isfinite(ablation_blocks)
        ):
            raise ValueError(f"non-finite values in {comparison}/{metric}")
        differences = full_blocks - ablation_blocks
        block_means = differences.mean(axis=2)
        ci_low, ci_high = hierarchical_ci(
            block_means, bootstrap, bootstrap_seed + metric_index
        )
        full_flat = full_blocks.ravel()
        ablation_flat = ablation_blocks.ravel()
        per_training_seed = {
            str(seed): float(block_means[index].mean())
            for index, seed in enumerate(TRAINING_SEEDS)
        }
        mean_difference = float(differences.mean())
        result: dict[str, object] = {
            "full_mean": float(full_flat.mean()),
            "ablation_mean": float(ablation_flat.mean()),
            "paired_difference_full_minus_ablation": mean_difference,
            "hierarchical_bootstrap_ci95": [ci_low, ci_high],
            "training_seed_difference_sd": float(
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
            result["mcnemar"] = exact_mcnemar(full_flat, ablation_flat)
        metrics[metric] = result
    return {
        "comparison": f"mean_risk_full_vs_{comparison}",
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "paired_episode_count": len(TRAINING_SEEDS) * len(EVALUATION_SEEDS) * 64,
        "bootstrap_samples": bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "inference_unit": "training-seed and evaluation-seed hierarchical paired bootstrap",
        "metrics": metrics,
    }


def write_summary_csv(report: dict[str, object], output: Path) -> None:
    fieldnames = (
        "comparison",
        "metric",
        "full_mean",
        "ablation_mean",
        "paired_difference_full_minus_ablation",
        "ci95_low",
        "ci95_high",
        "mcnemar_exact_two_sided_p",
    )
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for comparison, comparison_report in report["comparisons"].items():
            for metric, result in comparison_report["metrics"].items():
                ci_low, ci_high = result["hierarchical_bootstrap_ci95"]
                writer.writerow(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "full_mean": result["full_mean"],
                        "ablation_mean": result["ablation_mean"],
                        "paired_difference_full_minus_ablation": result[
                            "paired_difference_full_minus_ablation"
                        ],
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "mcnemar_exact_two_sided_p": result.get("mcnemar", {}).get(
                            "exact_two_sided_p", ""
                        ),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    args = parser.parse_args()
    audit = json.loads(args.evaluation_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("episode_count") != 17_280:
        raise ValueError("evaluation integrity audit has not passed")
    if args.bootstrap != 20_000:
        raise ValueError("formal analysis requires exactly 20000 bootstrap samples")
    rows = read_rows(args.episodes)
    report: dict[str, object] = {
        "status": "passed",
        "protocol": "v19_mean_risk_matched_core_ablations_20260811_v1",
        "analysis_after_integrity_audit": True,
        "checkpoint_selection_after_results": False,
        "negative_results_retained": True,
        "comparisons": {
            comparison: analyze_comparison(
                rows, comparison, args.bootstrap, args.bootstrap_seed
            )
            for comparison in COMPARISONS
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mean_risk_core_ablation_statistics.json"
    csv_path = args.output_dir / "mean_risk_core_ablation_summary.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_summary_csv(report, csv_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "comparison_count": len(COMPARISONS),
                "paired_episode_count_per_comparison": 5760,
                "bootstrap_samples": args.bootstrap,
            }
        )
    )


if __name__ == "__main__":
    main()
