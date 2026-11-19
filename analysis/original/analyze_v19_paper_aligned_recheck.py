"""Analyze fully audited paper-aligned density and corrected-noise results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

DIFFICULTY_LEVELS = ("d60", "d80", "d100", "d120")
NOISE_LEVELS = ("mild", "moderate")
NOISE_REALIZATIONS = ("n0", "n1", "n2")
METRICS = ("success", "goal_reach", "collision", "path_efficiency", "min_clearance_m")
PRIMARY_METRICS = ("success", "collision")
MODE_RE = re.compile(
    r"^(?:(?P<official>official_navrl)_(?P<official_variant>actor|vo)_r0|"
    r"(?P<family>mean_risk|ppolag)_s(?P<training_seed>[0-4])_actor)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mode(mode: str) -> tuple[str, str, int | None, int]:
    match = MODE_RE.fullmatch(mode)
    if match is None:
        raise ValueError(f"unexpected mode: {mode}")
    if match.group("official"):
        return "official_navrl", str(match.group("official_variant")), None, 0
    training_seed = int(str(match.group("training_seed")))
    return str(match.group("family")), "actor", training_seed, training_seed


def _float(row: dict[str, object], metric: str) -> float:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _load_stage(result_root: Path, stage: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    audit_path = result_root / "integrity_audits" / f"{stage}.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing integrity audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if audit.get("status") != "passed" or audit.get("outcome_values_modified") is not False:
        raise ValueError(f"stage audit did not pass: {audit_path}")

    episodes_path = result_root / "evaluation" / stage / "episodes.csv"
    rows: list[dict[str, object]] = []
    with episodes_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"mode", "seed", "env_index", *PRIMARY_METRICS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{episodes_path} is missing columns: {sorted(missing)}")
        for raw in reader:
            family, variant, training_seed, stratum = parse_mode(str(raw["mode"]))
            parsed: dict[str, object] = dict(raw)
            parsed.update(
                {
                    "family": family,
                    "variant": variant,
                    "training_seed": training_seed,
                    "stratum": stratum,
                    "evaluation_seed": int(str(raw["seed"])),
                    "env_index": int(str(raw["env_index"])),
                    "stage": stage,
                }
            )
            rows.append(parsed)
    if len(rows) != int(audit["episode_count"]):
        raise ValueError(f"episode/audit count mismatch for {stage}")
    return rows, {"path": str(audit_path), "sha256": sha256(audit_path), **audit}


def _select(
    rows: list[dict[str, object]],
    family: str,
    variant: str,
    stratum: int,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and row["variant"] == variant
        and int(row["stratum"]) == stratum
    ]
    if len(selected) != 384:
        raise ValueError(
            f"expected 384 episodes for {family}/{variant}/stratum={stratum}, "
            f"observed {len(selected)}"
        )
    return selected


def _values_by_key(rows: list[dict[str, object]], metric: str) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["evaluation_seed"]), int(row["env_index"]))
        if key in output:
            raise ValueError(f"duplicate evaluation key: {key}")
        output[key] = _float(row, metric)
    return output


def _paired_difference(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    metric: str,
) -> np.ndarray:
    left_values = _values_by_key(left, metric)
    right_values = _values_by_key(right, metric)
    if set(left_values) != set(right_values):
        raise ValueError("paired evaluation keys do not match")
    values = np.asarray(
        [left_values[key] - right_values[key] for key in sorted(left_values)],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _bootstrap_strata(
    arrays: list[np.ndarray], samples: int, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    if not arrays or any(array.size == 0 for array in arrays):
        raise ValueError("bootstrap strata must be non-empty")
    estimate = float(np.mean([float(array.mean()) for array in arrays]))
    draws = np.empty(samples, dtype=float)
    for draw in range(samples):
        sampled_strata = rng.integers(0, len(arrays), size=len(arrays))
        means = []
        for index in sampled_strata:
            array = arrays[int(index)]
            sampled = array[rng.integers(0, array.size, size=array.size)]
            means.append(float(sampled.mean()))
        draws[draw] = float(np.mean(means))
    return (
        estimate,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int(sum(array.size for array in arrays)),
    )


def _bootstrap_scalar_means(
    values: list[float], samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value, value
    indices = rng.integers(0, array.size, size=(samples, array.size))
    draws = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _summary_rows(
    rows: list[dict[str, object]],
    condition: str,
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    methods = (
        ("official_navrl", "actor", [0]),
        ("official_navrl", "vo", [0]),
        ("ppolag", "actor", list(range(5))),
        ("mean_risk", "actor", list(range(5))),
    )
    for family, variant, strata in methods:
        selected_by_stratum = [
            _select(rows, family, variant, stratum) for stratum in strata
        ]
        for metric in METRICS:
            stratum_means = [
                float(np.nanmean([_float(row, metric) for row in selected]))
                for selected in selected_by_stratum
            ]
            mean, low, high = _bootstrap_scalar_means(stratum_means, samples, rng)
            record: dict[str, object] = {
                "condition": condition,
                "family": family,
                "variant": variant,
                "metric": metric,
                "five_seed_mean_or_single_checkpoint_value": mean,
                "ci95_low": low,
                "ci95_high": high,
                "training_seed_count": 0 if family == "official_navrl" else len(strata),
                "episode_count": sum(len(selected) for selected in selected_by_stratum),
            }
            if metric == "success" and family != "official_navrl":
                best_index = int(np.nanargmax(stratum_means))
                record["highest_frozen_checkpoint_success"] = stratum_means[best_index]
                record["highest_success_training_seed"] = strata[best_index]
            elif metric == "success":
                record["highest_frozen_checkpoint_success"] = mean
                record["highest_success_training_seed"] = "publisher_saved_best"
            output.append(record)
    return output


def _paired_effect(
    rows: list[dict[str, object]],
    condition: str,
    comparison: str,
    left: tuple[str, str],
    right: tuple[str, str],
    strata: list[int],
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    output = []
    for metric in PRIMARY_METRICS:
        arrays = []
        for stratum in strata:
            left_rows = _select(rows, left[0], left[1], stratum)
            right_rows = _select(rows, right[0], right[1], stratum)
            arrays.append(_paired_difference(left_rows, right_rows, metric))
        estimate, low, high, pair_count = _bootstrap_strata(arrays, samples, rng)
        output.append(
            {
                "condition": condition,
                "comparison": comparison,
                "metric": metric,
                "difference_definition": "left minus right",
                "difference": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "stratum_count": len(strata),
                "paired_episode_count": pair_count,
                "composite_key": "training_seed + evaluation_seed + env_index",
            }
        )
    return output


def _condition_change(
    current: list[dict[str, object]],
    reference: list[dict[str, object]],
    condition: str,
    family: str,
    variant: str,
    strata: list[int],
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    output = []
    for metric in PRIMARY_METRICS:
        arrays = []
        for stratum in strata:
            current_rows = _select(current, family, variant, stratum)
            reference_rows = _select(reference, family, variant, stratum)
            arrays.append(_paired_difference(current_rows, reference_rows, metric))
        estimate, low, high, pair_count = _bootstrap_strata(arrays, samples, rng)
        output.append(
            {
                "condition": condition,
                "comparison": f"{family}_{variant}_condition_change",
                "metric": metric,
                "difference_definition": "current condition minus reference condition",
                "difference": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "stratum_count": len(strata),
                "paired_episode_count": pair_count,
                "composite_key": "training_seed + evaluation_seed + env_index",
            }
        )
    return output


def _condition_interaction(
    current: list[dict[str, object]],
    reference: list[dict[str, object]],
    condition: str,
    comparison: str,
    left: tuple[str, str],
    right: tuple[str, str],
    strata: list[int],
    samples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    output = []
    for metric in PRIMARY_METRICS:
        arrays = []
        for stratum in strata:
            left_change = _paired_difference(
                _select(current, left[0], left[1], stratum),
                _select(reference, left[0], left[1], stratum),
                metric,
            )
            right_change = _paired_difference(
                _select(current, right[0], right[1], stratum),
                _select(reference, right[0], right[1], stratum),
                metric,
            )
            if left_change.shape != right_change.shape:
                raise ValueError("condition-interaction pairing shapes do not match")
            arrays.append(left_change - right_change)
        estimate, low, high, pair_count = _bootstrap_strata(arrays, samples, rng)
        output.append(
            {
                "condition": condition,
                "comparison": comparison,
                "metric": metric,
                "difference_definition": (
                    "(current left - reference left) minus "
                    "(current right - reference right)"
                ),
                "difference": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "stratum_count": len(strata),
                "paired_episode_count": pair_count,
                "composite_key": "training_seed + evaluation_seed + env_index",
            }
        )
    return output


def _noise_diagnostics(result_root: Path, stages: list[str]) -> list[dict[str, object]]:
    fields = (
        "lidar_hit_fraction",
        "lidar_range_error_mean_m",
        "lidar_range_error_rms_m",
        "lidar_proximity_shift_mean_m",
        "lidar_no_return_change_fraction",
        "dynamic_position_error_rms_m",
        "dynamic_velocity_error_rms_mps",
    )
    output = []
    for stage in stages:
        path = result_root / "evaluation" / stage / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
        resets = metadata.get("evaluation_noise_resets", [])
        if not resets:
            raise ValueError(f"noise diagnostics are missing for {stage}")
        record: dict[str, object] = {"condition": stage, "reset_count": len(resets)}
        for field in fields:
            values = np.asarray([float(item[field]) for item in resets], dtype=float)
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_max_abs"] = float(np.max(np.abs(values)))
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_difficulty(summary: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        ("official_navrl", "actor"): "NavRL",
        ("official_navrl", "vo"): "NavRL + VO",
        ("ppolag", "actor"): "PPO-Lagrangian",
        ("mean_risk", "actor"): "RC-CF-NavRL",
    }
    colors = ("#3B6FB6", "#2E8B57", "#D17A22", "#B23A48")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, PRIMARY_METRICS, strict=True):
        for (key, label), color in zip(labels.items(), colors, strict=True):
            values = [
                next(
                    float(item["five_seed_mean_or_single_checkpoint_value"])
                    for item in summary
                    if item["condition"] == f"difficulty_{level}"
                    and item["family"] == key[0]
                    and item["variant"] == key[1]
                    and item["metric"] == metric
                )
                for level in DIFFICULTY_LEVELS
            ]
            axis.plot(DIFFICULTY_LEVELS, values, marker="o", label=label, color=color)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")

    stages = [f"difficulty_{level}" for level in DIFFICULTY_LEVELS]
    noise_stages = [
        f"noise_{level}_{realization}"
        for level in NOISE_LEVELS
        for realization in NOISE_REALIZATIONS
    ]
    loaded = {stage: _load_stage(args.result_root, stage) for stage in [*stages, *noise_stages]}
    freeze_path = args.result_root / "manifests" / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    if freeze.get("protocol") != "v19_paper_aligned_exact_recheck_20260810_v2":
        raise ValueError("unexpected paper-aligned freeze protocol")

    rng = np.random.default_rng(args.bootstrap_seed)
    summaries: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    for stage in stages:
        rows = loaded[stage][0]
        summaries.extend(_summary_rows(rows, stage, args.bootstrap, rng))
        effects.extend(
            _paired_effect(
                rows,
                stage,
                "mean_risk_vs_ppolag",
                ("mean_risk", "actor"),
                ("ppolag", "actor"),
                list(range(5)),
                args.bootstrap,
                rng,
            )
        )
        effects.extend(
            _paired_effect(
                rows,
                stage,
                "official_vo_vs_actor",
                ("official_navrl", "vo"),
                ("official_navrl", "actor"),
                [0],
                args.bootstrap,
                rng,
            )
        )

    d60_rows = loaded["difficulty_d60"][0]
    d120_rows = loaded["difficulty_d120"][0]
    for stage in stages[1:]:
        for family, variant, strata in (
            ("official_navrl", "actor", [0]),
            ("official_navrl", "vo", [0]),
            ("ppolag", "actor", list(range(5))),
            ("mean_risk", "actor", list(range(5))),
        ):
            effects.extend(
                _condition_change(
                    loaded[stage][0],
                    d60_rows,
                    f"{stage}_vs_difficulty_d60",
                    family,
                    variant,
                    strata,
                    args.bootstrap,
                    rng,
                )
            )
        effects.extend(
            _condition_interaction(
                loaded[stage][0],
                d60_rows,
                f"{stage}_vs_difficulty_d60",
                "mean_risk_vs_ppolag_degradation_interaction",
                ("mean_risk", "actor"),
                ("ppolag", "actor"),
                list(range(5)),
                args.bootstrap,
                rng,
            )
        )

    for stage in noise_stages:
        rows = loaded[stage][0]
        summaries.extend(_summary_rows(rows, stage, args.bootstrap, rng))
        effects.extend(
            _paired_effect(
                rows,
                stage,
                "mean_risk_vs_ppolag",
                ("mean_risk", "actor"),
                ("ppolag", "actor"),
                list(range(5)),
                args.bootstrap,
                rng,
            )
        )
        effects.extend(
            _paired_effect(
                rows,
                stage,
                "official_vo_vs_actor",
                ("official_navrl", "vo"),
                ("official_navrl", "actor"),
                [0],
                args.bootstrap,
                rng,
            )
        )
        for family, variant, strata in (
            ("official_navrl", "actor", [0]),
            ("official_navrl", "vo", [0]),
            ("ppolag", "actor", list(range(5))),
            ("mean_risk", "actor", list(range(5))),
        ):
            effects.extend(
                _condition_change(
                    rows,
                    d120_rows,
                    f"{stage}_vs_difficulty_d120",
                    family,
                    variant,
                    strata,
                    args.bootstrap,
                    rng,
                )
            )
        effects.extend(
            _condition_interaction(
                rows,
                d120_rows,
                f"{stage}_vs_difficulty_d120",
                "mean_risk_vs_ppolag_noise_interaction",
                ("mean_risk", "actor"),
                ("ppolag", "actor"),
                list(range(5)),
                args.bootstrap,
                rng,
            )
        )

    diagnostics = _noise_diagnostics(args.result_root, noise_stages)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paper_aligned_summary.csv", summaries)
    _write_csv(args.output_dir / "paper_aligned_paired_effects.csv", effects)
    _write_csv(args.output_dir / "corrected_noise_diagnostics.csv", diagnostics)
    _plot_difficulty(summaries, args.output_dir / "paper_aligned_difficulty.png")

    audits = [loaded[stage][1] for stage in [*stages, *noise_stages]]
    report = {
        "status": "complete",
        "scope": "paper_aligned_60_80_100_120_and_corrected_noise",
        "freeze_sha256": sha256(freeze_path),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "summary": summaries,
        "paired_effects": effects,
        "corrected_noise_diagnostics": diagnostics,
        "integrity_audits": audits,
        "outcome_values_modified": False,
        "negative_results_retained": True,
    }
    report_path = args.output_dir / "paper_aligned_strict_analysis.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    analysis_audit = {
        "status": "passed",
        "required_stage_count": len(stages) + len(noise_stages),
        "integrity_audit_count": len(audits),
        "all_upstream_audits_passed": True,
        "composite_pairing_key": (
            "training_seed_or_official_replicate + evaluation_seed + env_index; "
            "noise realization retained as a separate condition"
        ),
        "highest_success_rule": (
            "post-evaluation descriptive maximum across all five preregistered frozen "
            "training-seed checkpoints for every trainable method; reported beside, "
            "and never substituted for, the five-seed mean and confidence interval"
        ),
        "condition_interaction_definition": (
            "(current RC-CF-NavRL - reference RC-CF-NavRL) minus "
            "(current PPO-Lagrangian - reference PPO-Lagrangian)"
        ),
        "test_set_checkpoint_selection_for_primary_analysis": False,
        "supplementary_posthoc_descriptive_maximum": True,
        "primary_statistic": "five-training-seed mean with 95% confidence interval",
        # Compatibility alias consumed by already-running finalize/export watchers.
        "test_set_checkpoint_selection": False,
        "test_set_checkpoint_selection_scope": (
            "no checkpoint was selected or excluded after test outcomes; the separately "
            "labelled maximum is a post-evaluation descriptive statistic only"
        ),
        "outcome_values_modified": False,
        "analysis_sha256": sha256(report_path),
    }
    (args.output_dir / "paper_aligned_analysis_audit.json").write_text(
        json.dumps(analysis_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
