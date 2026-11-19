"""Strict offline analysis for the preregistered D60/D120 VO extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STAGES = ("d60", "d120")
TRAINING_SEEDS = tuple(range(5))
EPISODES_PER_MODE = 384
METRICS = ("success", "collision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_audit(path: Path, expected_episode_count: int) -> dict[str, object]:
    audit = json.loads(path.read_text(encoding="utf-8-sig"))
    if audit.get("status") != "passed":
        raise ValueError(f"audit did not pass: {path}")
    if audit.get("outcome_values_modified") is not False:
        raise ValueError(f"outcome integrity failed: {path}")
    if int(audit.get("episode_count", -1)) != expected_episode_count:
        raise ValueError(f"unexpected audit episode count: {path}")
    return {"path": str(path), "sha256": sha256(path), **audit}


def _load_modes(
    path: Path, expected_modes: set[str]
) -> dict[str, dict[tuple[int, int], dict[str, float]]]:
    rows = {mode: {} for mode in expected_modes}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"mode", "seed", "env_index", *METRICS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        for row in reader:
            mode = str(row["mode"])
            if mode not in rows:
                continue
            key = (int(row["seed"]), int(row["env_index"]))
            if key in rows[mode]:
                raise ValueError(f"duplicate evaluation key: {mode}/{key}")
            rows[mode][key] = {metric: float(row[metric]) for metric in METRICS}

    expected_keys = {(seed, env_index) for seed in _evaluation_seeds() for env_index in range(64)}
    for mode, values in rows.items():
        if set(values) != expected_keys:
            raise ValueError(f"evaluation keys mismatch for {mode}")
        if any(
            not np.isfinite(value)
            for record in values.values()
            for value in record.values()
        ):
            raise ValueError(f"non-finite outcomes in {mode}")
    return rows


def _evaluation_seeds() -> tuple[int, ...]:
    return (1183109, 1294127, 1405151, 1516183, 1627211, 1738247)


def _seed_arrays(
    rows: dict[str, dict[tuple[int, int], dict[str, float]]],
    template: str,
    metric: str,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for seed in TRAINING_SEEDS:
        values = rows[template.format(seed=seed)]
        array = np.asarray(
            [values[key][metric] for key in sorted(values)], dtype=np.float64
        )
        if array.shape != (EPISODES_PER_MODE,):
            raise ValueError(f"unexpected episode count for {template.format(seed=seed)}")
        arrays.append(array)
    return arrays


def _paired_arrays(
    left: dict[str, dict[tuple[int, int], dict[str, float]]],
    left_template: str,
    right: dict[str, dict[tuple[int, int], dict[str, float]]],
    right_template: str,
    metric: str,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for seed in TRAINING_SEEDS:
        left_values = left[left_template.format(seed=seed)]
        right_values = right[right_template.format(seed=seed)]
        if set(left_values) != set(right_values):
            raise ValueError("paired evaluation keys do not match")
        array = np.asarray(
            [left_values[key][metric] - right_values[key][metric] for key in sorted(left_values)],
            dtype=np.float64,
        )
        arrays.append(array)
    return arrays


def _bootstrap_scalar(
    seed_arrays: list[np.ndarray], samples: int, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    if len(seed_arrays) != len(TRAINING_SEEDS):
        raise ValueError("expected all five frozen training seeds")
    stacked = np.vstack(seed_arrays)
    if stacked.shape != (len(TRAINING_SEEDS), EPISODES_PER_MODE):
        raise ValueError("unexpected stratum shape")
    if not np.all(np.isfinite(stacked)):
        raise ValueError("non-finite bootstrap inputs")
    estimate = float(stacked.mean(axis=1).mean())
    sampled_seed_indices = rng.integers(
        0, stacked.shape[0], size=(samples, stacked.shape[0])
    )
    sampled_episode_indices = rng.integers(
        0, stacked.shape[1], size=(samples, stacked.shape[0], stacked.shape[1])
    )
    draws = stacked[sampled_seed_indices[:, :, None], sampled_episode_indices].mean(
        axis=(1, 2)
    )
    return (
        estimate,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        int(stacked.size),
    )


def analyze(
    source_result_root: Path,
    extension_result_root: Path,
    bootstrap: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")

    actor_modes = {
        f"{family}_s{seed}_actor"
        for family in ("mean_risk", "ppolag")
        for seed in TRAINING_SEEDS
    }
    shield_modes = {
        f"{family}_s{seed}_official_vo"
        for family in ("mean_risk", "ppolag")
        for seed in TRAINING_SEEDS
    }
    source_rows: dict[str, dict[str, dict[tuple[int, int], dict[str, float]]]] = {}
    shield_rows: dict[str, dict[str, dict[tuple[int, int], dict[str, float]]]] = {}
    source_audits = []
    shield_audits = []
    for stage in STAGES:
        source_stage = f"difficulty_{stage}"
        source_audit_path = source_result_root / "integrity_audits" / f"{source_stage}.json"
        shield_audit_path = extension_result_root / "integrity_audits" / f"{stage}_official_vo.json"
        source_audits.append(_read_audit(source_audit_path, expected_episode_count=4608))
        shield_audits.append(_read_audit(shield_audit_path, expected_episode_count=3840))
        source_rows[stage] = _load_modes(
            source_result_root / "evaluation" / source_stage / "episodes.csv", actor_modes
        )
        shield_rows[stage] = _load_modes(
            extension_result_root / "04_逐回合结果" / f"{stage}_official_vo" / "episodes.csv",
            shield_modes,
        )

    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    for stage in STAGES:
        for family, source, template, deployment in (
            ("mean_risk", source_rows[stage], "mean_risk_s{seed}_actor", "actor"),
            ("mean_risk", shield_rows[stage], "mean_risk_s{seed}_official_vo", "official_vo"),
            ("ppolag", source_rows[stage], "ppolag_s{seed}_actor", "actor"),
            ("ppolag", shield_rows[stage], "ppolag_s{seed}_official_vo", "official_vo"),
        ):
            for metric in METRICS:
                mean, low, high, episode_count = _bootstrap_scalar(
                    _seed_arrays(source, template, metric), bootstrap, rng
                )
                summaries.append(
                    {
                        "condition": stage,
                        "family": family,
                        "deployment": deployment,
                        "metric": metric,
                        "five_seed_mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "training_seed_count": len(TRAINING_SEEDS),
                        "episode_count": episode_count,
                    }
                )

        comparisons = (
            (
                "mean_risk_official_vo_vs_actor",
                shield_rows[stage],
                "mean_risk_s{seed}_official_vo",
                source_rows[stage],
                "mean_risk_s{seed}_actor",
            ),
            (
                "ppolag_official_vo_vs_actor",
                shield_rows[stage],
                "ppolag_s{seed}_official_vo",
                source_rows[stage],
                "ppolag_s{seed}_actor",
            ),
            (
                "mean_risk_official_vo_vs_ppolag_official_vo",
                shield_rows[stage],
                "mean_risk_s{seed}_official_vo",
                shield_rows[stage],
                "ppolag_s{seed}_official_vo",
            ),
        )
        for comparison, left, left_template, right, right_template in comparisons:
            for metric in METRICS:
                mean, low, high, episode_count = _bootstrap_scalar(
                    _paired_arrays(left, left_template, right, right_template, metric),
                    bootstrap,
                    rng,
                )
                effects.append(
                    {
                        "condition": stage,
                        "comparison": comparison,
                        "metric": metric,
                        "difference_definition": "left minus right",
                        "difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "training_seed_count": len(TRAINING_SEEDS),
                        "paired_episode_count": episode_count,
                        "composite_key": "training_seed + evaluation_seed + env_index",
                    }
                )

    audit = {
        "status": "passed",
        "scope": "preregistered_exact_d60_d120_official_vo_deployment_extension",
        "source_protocol": "v19_paper_aligned_exact_recheck_20260810_v2",
        "extension_protocol": "v19_paper_aligned_shield_extension_20260811_v1",
        "source_audits": source_audits,
        "extension_audits": shield_audits,
        "source_stage_count": len(source_audits),
        "extension_stage_count": len(shield_audits),
        "training_seed_count": len(TRAINING_SEEDS),
        "evaluation_seed_count": len(_evaluation_seeds()),
        "bootstrap_samples": bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "test_set_checkpoint_selection": False,
        "outcome_values_modified": False,
        "negative_results_retained": True,
        "claim_boundary": (
            "deployment-shield effects are reported separately and must not be "
            "attributed to actor-only training"
        ),
    }
    return summaries, effects, audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result-root", type=Path, required=True)
    parser.add_argument("--extension-result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    args = parser.parse_args()
    summaries, effects, audit = analyze(
        args.source_result_root,
        args.extension_result_root,
        args.bootstrap,
        args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "deployment_shield_summary.csv", summaries)
    _write_csv(args.output_dir / "deployment_shield_paired_effects.csv", effects)
    report = {
        "status": "complete",
        "summary": summaries,
        "paired_effects": effects,
        "outcome_values_modified": False,
        "negative_results_retained": True,
    }
    report_path = args.output_dir / "deployment_shield_strict_analysis.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit["analysis_sha256"] = sha256(report_path)
    (args.output_dir / "deployment_shield_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
