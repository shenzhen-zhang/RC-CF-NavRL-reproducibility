"""Validation helpers for a frozen cross-process NavRL dynamic scene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_DYNAMIC_SCENE_PROTOCOL = "canonical_navrl_dynamic_obstacles_v2"
PINNED_NAVRL_COMMIT = "3725bcc2e7c1be4ecf1455d922299ae85042603a"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_array_sha256(value: Any) -> str:
    array = np.asarray(value, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_canonical_dynamic_scene(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("canonical dynamic scene snapshot must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("canonical dynamic scene schema_version must be 1")
    if payload.get("protocol") != CANONICAL_DYNAMIC_SCENE_PROTOCOL:
        raise ValueError(
            "canonical dynamic scene protocol must be "
            f"{CANONICAL_DYNAMIC_SCENE_PROTOCOL}"
        )
    if payload.get("official_navrl_commit") != PINNED_NAVRL_COMMIT:
        raise ValueError("canonical dynamic scene uses an unexpected NavRL commit")

    origin = np.asarray(payload.get("dynamic_obstacle_origin"), dtype=np.float32)
    size = np.asarray(payload.get("dynamic_obstacle_size"), dtype=np.float32)
    if origin.ndim != 2 or origin.shape[1:] != (3,):
        raise ValueError("dynamic_obstacle_origin must have shape [N, 3]")
    if size.shape != origin.shape:
        raise ValueError("dynamic_obstacle_size must match origin shape")
    if int(payload.get("dynamic_obstacle_count", -1)) != int(origin.shape[0]):
        raise ValueError("dynamic_obstacle_count does not match origin")
    if origin.shape[0] < 8:
        raise ValueError("dynamic obstacle count must be at least 8")
    if not np.isfinite(origin).all() or not np.isfinite(size).all():
        raise ValueError("canonical dynamic scene contains non-finite values")
    if not (size > 0.0).all():
        raise ValueError("dynamic obstacle sizes must be positive")

    observed_origin_sha256 = float32_array_sha256(origin)
    if (
        payload.get("dynamic_obstacle_origin_sha256")
        != observed_origin_sha256
    ):
        raise ValueError("dynamic obstacle origin SHA-256 does not match payload")
    observed_size_sha256 = float32_array_sha256(size)
    if payload.get("dynamic_obstacle_size_sha256") != observed_size_sha256:
        raise ValueError("dynamic obstacle size SHA-256 does not match payload")

    map_range = np.asarray(payload.get("map_range"), dtype=np.float32)
    if map_range.shape != (3,) or not np.isfinite(map_range).all():
        raise ValueError("map_range must contain three finite values")
    if not (map_range > 0.0).all():
        raise ValueError("map_range must be positive")
    if not isinstance(payload.get("creation_seed"), int):
        raise ValueError("creation_seed must be an integer")

    normalized = dict(payload)
    normalized["dynamic_obstacle_origin"] = origin.tolist()
    normalized["dynamic_obstacle_size"] = size.tolist()
    normalized["map_range"] = map_range.tolist()
    return normalized


def build_canonical_dynamic_scene(
    *,
    dynamic_obstacle_origin: Any,
    dynamic_obstacle_size: Any,
    map_range: Any,
    creation_seed: int,
) -> dict[str, Any]:
    origin = np.asarray(dynamic_obstacle_origin, dtype=np.float32)
    size = np.asarray(dynamic_obstacle_size, dtype=np.float32)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": CANONICAL_DYNAMIC_SCENE_PROTOCOL,
        "official_navrl_commit": PINNED_NAVRL_COMMIT,
        "source": (
            "one frozen scene sampled by the pinned public NavRL "
            "NavigationEnv and reused verbatim across Isaac processes"
        ),
        "creation_seed": int(creation_seed),
        "dynamic_obstacle_count": int(origin.shape[0]),
        "dynamic_obstacle_origin": origin.tolist(),
        "dynamic_obstacle_origin_sha256": float32_array_sha256(origin),
        "dynamic_obstacle_size": size.tolist(),
        "dynamic_obstacle_size_sha256": float32_array_sha256(size),
        "map_range": np.asarray(map_range, dtype=np.float32).tolist(),
    }
    return validate_canonical_dynamic_scene(payload)


def load_canonical_dynamic_scene(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("expected canonical scene SHA-256 must be provided")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"canonical dynamic scene not found: {resolved}")
    observed_sha256 = file_sha256(resolved)
    if observed_sha256 != expected_sha256.lower():
        raise ValueError(
            "canonical dynamic scene file SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, observed {observed_sha256}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return validate_canonical_dynamic_scene(payload), observed_sha256
