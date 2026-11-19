#!/usr/bin/env python3
"""Analyze preregistered v19 mechanism metrics at the training-seed level."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANT_PREFIXES = {
    "full": "v19_exact_s",
    "no_gate": "no_gate_s",
    "no_counterfactual": "no_counterfactual_s",
    "mean_risk": "mean_risk_s",
}


def variant_of(checkpoint: str) -> str:
    for variant, prefix in VARIANT_PREFIXES.items():
        if checkpoint.startswith(prefix):
            return variant
    raise ValueError(f"unknown checkpoint name: {checkpoint}")


def bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("bootstrap input must contain five finite training-seed values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "training_seed_values": [float(value) for value in values],
    }


def load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                converted: dict[str, object] = dict(row)
                for key, value in list(converted.items()):
                    if key in {"checkpoint", "mode", "risk_measure", "gate_mode"}:
                        continue
                    converted[key] = float(value)
                rows.append(converted)
    return rows


def aggregate_training_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        checkpoint = str(row["checkpoint"])
        variant = variant_of(checkpoint)
        training_seed = int(float(row["training_seed_index"]))
        grouped[(variant, training_seed)].append(row)
    expected = {(variant, seed) for variant in VARIANT_PREFIXES for seed in range(5)}
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise ValueError(f"training-seed coverage mismatch; missing={missing}, extra={extra}")

    numeric_metrics = [
        key
        for key, value in rows[0].items()
        if isinstance(value, float)
        and key not in {"evaluation_seed", "training_seed_index"}
    ]
    output = []
    for (variant, training_seed), group in sorted(grouped.items()):
        evaluation_seeds = {int(float(row["evaluation_seed"])) for row in group}
        if len(evaluation_seeds) != 6:
            raise ValueError(
                f"{variant} seed {training_seed} must contain six evaluation seeds"
            )
        record: dict[str, object] = {
            "variant": variant,
            "training_seed_index": training_seed,
            "evaluation_seed_count": len(evaluation_seeds),
        }
        for metric in numeric_metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {metric} for {variant} seed {training_seed}")
            record[metric] = float(values.mean())
        output.append(record)
    return output


def values_by_variant(
    aggregates: list[dict[str, object]], variant: str, metric: str
) -> np.ndarray:
    rows = sorted(
        (row for row in aggregates if row["variant"] == variant),
        key=lambda row: int(row["training_seed_index"]),
    )
    return np.asarray([float(row[metric]) for row in rows], dtype=np.float64)


def analyze(
    aggregates: list[dict[str, object]], bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, object]:
    full_gate_auc = values_by_variant(aggregates, "full", "gate_auroc")
    no_gate_auc = values_by_variant(aggregates, "no_gate", "gate_auroc")
    full_gate_selectivity = values_by_variant(
        aggregates, "full", "gate_danger_safe_difference"
    )
    gate_auc_delta = bootstrap_mean(
        full_gate_auc - no_gate_auc, bootstrap_samples, bootstrap_seed + 1
    )
    gate_selectivity = bootstrap_mean(
        full_gate_selectivity, bootstrap_samples, bootstrap_seed + 2
    )
    retain_gate = (
        gate_auc_delta["ci95_low"] > 0.0
        and gate_selectivity["ci95_low"] > 0.0
    )

    full_rank = values_by_variant(
        aggregates, "full", "counterfactual_pairwise_accuracy"
    )
    no_cf_rank = values_by_variant(
        aggregates, "no_counterfactual", "counterfactual_pairwise_accuracy"
    )
    full_mae = values_by_variant(
        aggregates, "full", "counterfactual_calibration_mae"
    )
    no_cf_mae = values_by_variant(
        aggregates, "no_counterfactual", "counterfactual_calibration_mae"
    )
    rank_delta = bootstrap_mean(
        full_rank - no_cf_rank, bootstrap_samples, bootstrap_seed + 3
    )
    calibration_delta = bootstrap_mean(
        full_mae - no_cf_mae, bootstrap_samples, bootstrap_seed + 4
    )
    full_rank_level = bootstrap_mean(
        full_rank, bootstrap_samples, bootstrap_seed + 5
    )
    retain_counterfactual = (
        full_rank_level["ci95_low"] > 0.5
        and (
            rank_delta["ci95_low"] > 0.0
            or calibration_delta["ci95_high"] < 0.0
        )
    )

    cvar_auc_advantage = values_by_variant(
        aggregates, "full", "tail_auroc_cvar_minus_mean"
    )
    cvar_primary = bootstrap_mean(
        cvar_auc_advantage, bootstrap_samples, bootstrap_seed + 6
    )
    full_q90 = values_by_variant(aggregates, "full", "realized_return_q90")
    mean_risk_q90 = values_by_variant(
        aggregates, "mean_risk", "realized_return_q90"
    )
    q90_delta = bootstrap_mean(
        full_q90 - mean_risk_q90, bootstrap_samples, bootstrap_seed + 7
    )
    retain_cvar = cvar_primary["ci95_low"] > 0.0

    decisions = {
        "risk_conditioned_gate": "retain" if retain_gate else "simplify",
        "counterfactual_supervision": (
            "retain" if retain_counterfactual else "simplify"
        ),
        "cvar_risk_measure": "retain" if retain_cvar else "simplify",
    }
    return {
        "schema_version": 1,
        "analysis_unit": "training seed after averaging six evaluation seeds",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "registered_thresholds": {
            "gate": (
                "lower 95% CI > 0 for full-minus-fixed-open-gate AUROC and "
                "for full danger-minus-safe gate probability"
            ),
            "counterfactual": (
                "full ranking lower 95% CI > 0.5 and either ranking improvement "
                "lower CI > 0 or calibration-MAE change upper CI < 0"
            ),
            "cvar": "lower 95% CI > 0 for CVaR-minus-mean tail AUROC",
        },
        "gate": {
            "full_minus_no_gate_auroc": gate_auc_delta,
            "full_danger_minus_safe_gate": gate_selectivity,
        },
        "counterfactual": {
            "full_pairwise_accuracy": full_rank_level,
            "full_minus_no_counterfactual_pairwise_accuracy": rank_delta,
            "full_minus_no_counterfactual_calibration_mae": calibration_delta,
        },
        "cvar": {
            "full_cvar_minus_mean_tail_auroc": cvar_primary,
            "full_minus_mean_risk_realized_return_q90": q90_delta,
            "q90_comparison_role": "secondary; not used for the keep/remove gate",
        },
        "module_decisions": decisions,
        "recommended_simplification": [
            module for module, decision in decisions.items() if decision == "simplify"
        ],
    }


def write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.bootstrap_samples != 20000:
        raise ValueError("the preregistered bootstrap sample count is 20000")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    aggregates = aggregate_training_seeds(rows)
    report = analyze(aggregates, args.bootstrap_samples, args.bootstrap_seed)
    write_aggregate_csv(args.output / "训练种子机制指标汇总.csv", aggregates)
    (args.output / "三个模块机制分析报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["RC-CF-CVaR-NavRL v19 模块去留建议", ""]
    chinese = {
        "risk_conditioned_gate": "风险条件门控",
        "counterfactual_supervision": "反事实监督",
        "cvar_risk_measure": "CVaR尾部风险",
    }
    for module, decision in report["module_decisions"].items():
        lines.append(f"{chinese[module]}：{'保留' if decision == 'retain' else '建议简化'}")
    lines.extend(
        [
            "",
            "说明：以上判断严格按分析前写定的95%置信区间门槛生成，",
            "不使用单次最好结果，也不删除负结果。",
        ]
    )
    (args.output / "模块去留建议.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
