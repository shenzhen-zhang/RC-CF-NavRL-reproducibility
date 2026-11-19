"""Exact, balanced allocation across NavRL's eight obstacle categories."""

from __future__ import annotations

from collections.abc import Sequence


NAVRL_DYNAMIC_OBSTACLE_CATEGORY_COUNT = 8

# For a four-obstacle remainder, this order adds one obstacle to every width
# bin while keeping the cuboid and cylinder families exactly balanced.
NAVRL_DYNAMIC_OBSTACLE_REMAINDER_ORDER = (0, 5, 2, 7, 4, 1, 6, 3)


def balanced_dynamic_obstacle_category_counts(total_count: int) -> tuple[int, ...]:
    """Return category counts whose sum is exactly ``total_count``.

    Every category receives either floor(N / 8) or ceil(N / 8) obstacles.
    The fixed remainder order preserves coverage of both shape families and
    all four width bins without introducing a post-evaluation choice.
    """
    total_count = int(total_count)
    if total_count < 0:
        raise ValueError("dynamic obstacle count must be non-negative")
    base_count, remainder = divmod(
        total_count,
        NAVRL_DYNAMIC_OBSTACLE_CATEGORY_COUNT,
    )
    counts = [base_count] * NAVRL_DYNAMIC_OBSTACLE_CATEGORY_COUNT
    for category_index in NAVRL_DYNAMIC_OBSTACLE_REMAINDER_ORDER[:remainder]:
        counts[category_index] += 1
    if sum(counts) != total_count or max(counts, default=0) - min(counts, default=0) > 1:
        raise RuntimeError("dynamic obstacle category allocation is inconsistent")
    return tuple(counts)


def dynamic_obstacle_category_slices(
    category_counts: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Convert category counts into contiguous half-open state slices."""
    counts = tuple(int(value) for value in category_counts)
    if len(counts) != NAVRL_DYNAMIC_OBSTACLE_CATEGORY_COUNT:
        raise ValueError("NavRL dynamic obstacle allocation must have 8 categories")
    if any(value < 0 for value in counts):
        raise ValueError("dynamic obstacle category counts must be non-negative")
    slices: list[tuple[int, int]] = []
    offset = 0
    for count in counts:
        slices.append((offset, offset + count))
        offset += count
    return tuple(slices)
