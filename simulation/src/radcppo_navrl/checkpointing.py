"""Checkpoint compatibility helpers for architecture ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_SHARED_ACTOR_HEAD_ALIASES = {
    "actor_head.alpha_layer.weight": "actor.module.0.module.alpha_layer.weight",
    "actor_head.alpha_layer.bias": "actor.module.0.module.alpha_layer.bias",
    "actor_head.beta_layer.weight": "actor.module.0.module.beta_layer.weight",
    "actor_head.beta_layer.bias": "actor.module.0.module.beta_layer.bias",
}


def materialize_shared_actor_head_aliases(
    module: Any,
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Restore canonical names for actor-head tensors saved through TorchRL.

    The pinned NavRL checkpoint registers the same Beta head through the
    TorchRL actor wrapper only, while the adapted policy exposes both the
    direct ``actor_head`` path and the wrapper path. Copying the source tensor
    under the direct alias makes strict missing-key checks reflect the shared
    module instead of the serialization path.
    """

    current_state = module.state_dict()
    materialized = dict(checkpoint)
    restored: list[dict[str, str]] = []
    for canonical, serialized in _SHARED_ACTOR_HEAD_ALIASES.items():
        if canonical in materialized or serialized not in materialized:
            continue
        if canonical not in current_state or serialized not in current_state:
            raise RuntimeError(
                "Actor-head alias is not shared by the target model: "
                f"{serialized} -> {canonical}"
            )
        source_shape = tuple(materialized[serialized].shape)
        canonical_shape = tuple(current_state[canonical].shape)
        serialized_shape = tuple(current_state[serialized].shape)
        if source_shape != canonical_shape or source_shape != serialized_shape:
            raise RuntimeError(
                "Actor-head alias shape mismatch for "
                f"{serialized} -> {canonical}: checkpoint={source_shape}, "
                f"canonical={canonical_shape}, serialized={serialized_shape}"
            )
        materialized[canonical] = materialized[serialized]
        restored.append({"source": serialized, "target": canonical})
    return materialized, restored


def filter_shape_compatible_state_dict(
    module: Any,
    checkpoint: Mapping[str, Any],
    *,
    allowed_mismatch_prefixes: Sequence[str] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Skip explicitly allowed auxiliary incompatibilities.

    Keys outside ``allowed_mismatch_prefixes`` remain in the returned state
    dictionary when they are unknown, so ``load_state_dict`` still reports
    unexpected task-policy parameters. This keeps task-weight loading strict
    while allowing an obsolete auxiliary architecture to be replaced.
    """

    current_state = module.state_dict()
    compatible: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in checkpoint.items():
        current_value = current_state.get(key)
        if current_value is None:
            if key.startswith(tuple(allowed_mismatch_prefixes)):
                skipped.append(key)
                continue
            compatible[key] = value
            continue
        try:
            current_shape = tuple(current_value.shape)
            checkpoint_shape = tuple(value.shape)
        except RuntimeError:
            compatible[key] = value
            continue
        if current_shape == checkpoint_shape:
            compatible[key] = value
            continue
        if key.startswith(tuple(allowed_mismatch_prefixes)):
            skipped.append(key)
            continue
        raise RuntimeError(
            f"Checkpoint shape mismatch for {key}: "
            f"checkpoint={checkpoint_shape}, model={current_shape}"
        )
    return compatible, sorted(skipped)
