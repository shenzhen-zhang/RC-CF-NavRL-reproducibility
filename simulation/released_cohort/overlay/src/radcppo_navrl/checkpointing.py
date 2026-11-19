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


_AUXILIARY_MISSING_PREFIXES = (
    "safety_critic.", "target_safety_critic.", "safety_feature_extractor.",
    "target_safety_feature_extractor.", "actor_head.gate_layer.",
    "actor_head.residual_layer.", "actor.module.0.module.gate_layer.",
    "actor.module.0.module.residual_layer.",
)
_AUXILIARY_MISSING_KEYS = {"safety_lambda", "training_update_count"}
_FRESH_DETERMINISTIC_BUFFER_KEYS = frozenset({
    "teacher.ray_directions", "teacher.times", "teacher.candidate_yaw_offsets",
    "teacher.candidate_pitch_offsets", "teacher.candidate_speed_scales",
    "risk_selector.fixed_directions",
})


def load_fresh_nominal_checkpoint(module, checkpoint):
    """Strictly load every nominal tensor; only named NEW safety state may miss.

    Public NavRL serializes the Beta head via TorchRL's wrapper. Both alias
    paths must agree when both exist. No nominal encoder/head/critic/value
    normalization tensor may remain randomly initialized.
    """
    import torch
    for canonical, serialized in _SHARED_ACTOR_HEAD_ALIASES.items():
        if canonical in checkpoint and serialized in checkpoint:
            if checkpoint[canonical].shape != checkpoint[serialized].shape or not torch.equal(
                    checkpoint[canonical].detach().cpu(), checkpoint[serialized].detach().cpu()):
                raise RuntimeError(f"Conflicting shared actor-head aliases: {canonical}")
    materialized, aliases = materialize_shared_actor_head_aliases(module, checkpoint)
    compatible, skipped = filter_shape_compatible_state_dict(
        module, materialized,
        allowed_mismatch_prefixes=("safety_critic.", "target_safety_critic."))
    report = module.load_state_dict(compatible, strict=False)
    if report.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {report.unexpected_keys}")
    # The released policy predates these fixed geometry/time/lattice tensors.
    # Keep their deterministic constructor values, but do not exempt an entire
    # teacher namespace or a trainable parameter that reuses a listed name.
    registered_buffers = dict(module.named_buffers())
    for key in _FRESH_DETERMINISTIC_BUFFER_KEYS.intersection(report.missing_keys):
        value = registered_buffers.get(key)
        if value is None or value.requires_grad:
            raise RuntimeError(f"Fresh auxiliary state must be a registered non-learnable buffer: {key}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Fresh deterministic buffer is non-finite: {key}")
    forbidden = [key for key in report.missing_keys
                 if key not in _AUXILIARY_MISSING_KEYS
                 and key not in _FRESH_DETERMINISTIC_BUFFER_KEYS
                 and not key.startswith(_AUXILIARY_MISSING_PREFIXES)]
    if forbidden:
        raise RuntimeError(f"Checkpoint did not load nominal/non-auxiliary tensors: {forbidden}")
    current = module.state_dict()
    nominal_prefixes = ("feature_extractor.", "actor_head.alpha_layer.",
                        "actor_head.beta_layer.")
    nominal_keys = [key for key in current if key.startswith(nominal_prefixes)]
    if not nominal_keys:
        raise RuntimeError("Target has no recognized nominal encoder/head tensors")
    for key in nominal_keys:
        if key not in compatible or not torch.equal(current[key].detach().cpu(), compatible[key].detach().cpu()):
            raise RuntimeError(f"Loaded nominal tensor does not exactly equal checkpoint: {key}")
    return report, aliases, skipped


def nominal_tensor_hashes(module):
    import hashlib
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in module.state_dict().items()
            if key.startswith(("feature_extractor.", "actor_head.alpha_layer.", "actor_head.beta_layer."))}


def verify_nominal_integrity(initial, final, *, require_frozen):
    changed = sorted(key for key in initial.keys() | final.keys() if initial.get(key) != final.get(key))
    if require_frozen and (not initial or changed):
        raise RuntimeError("Frozen nominal integrity failed before saving final checkpoint: " + str(changed))
    return changed, True if require_frozen else None
