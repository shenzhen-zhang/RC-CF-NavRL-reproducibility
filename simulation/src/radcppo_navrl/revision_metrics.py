"""Observed episode costs for the reviewer study; no critic predictions.

The finite stopped return matches manuscript Eq. (3), with zero cost after
the chosen stopping event. The absorbing return replaces an unsafe terminal
target by 1. It matches the TD recursion only over that supplied sequence.
The original environment does not terminate on first goal: a first-goal-
stopped value therefore differs from its full training episode target.
These estimands must never share a column name.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def observed_cost_returns(
    costs: Sequence[float], *, unsafe_terminal: bool, discount: float = 0.99
) -> dict[str, float | int | bool]:
    if not 0 <= discount < 1:
        raise ValueError('discount must be in [0, 1)')
    values = [float(x) for x in costs]
    if not values or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('costs must be a nonempty finite sequence in [0, 1]')
    finite = math.fsum((1 - discount) * discount ** t * c for t, c in enumerate(values))
    # TD at an unsafe last transition is exactly 1, regardless of c_T.
    absorbing = (
        math.fsum((1 - discount) * discount ** t * c for t, c in enumerate(values[:-1]))
        + discount ** (len(values) - 1)
        if unsafe_terminal else finite
    )
    return {
        'cost_num_transitions': len(values),
        'cost_discount': discount,
        'observed_cost_finite_stopped': finite,
        'observed_cost_unsafe_absorbing': absorbing,
        'cost_unsafe_terminal': bool(unsafe_terminal),
        'observed_cost_sum': math.fsum(values),
    }


def possible_cost_bounds(mean_cost: float, length: int, discount: float = 0.99):
    """Sharp bounds using only sum(cost), length, and 0 <= cost <= 1.

    Latest-first allocation minimizes a decreasing weighted sum; earliest-
    first maximizes it. This does not recover an observed discounted return.
    """
    if length < 1 or int(length) != length or not 0 <= mean_cost <= 1 or not 0 <= discount < 1:
        raise ValueError('invalid cost-bound input')
    mass = mean_cost * length
    integer = min(int(mass), length)
    fraction = mass - integer
    if discount == 0:
        return max(0.0, mass - (length - 1)), min(1.0, mass)
    high = 1 - discount ** integer
    low = discount ** (length - integer) * (1 - discount ** integer)
    if fraction > 0 and integer < length:
        high += (1 - discount) * fraction * discount ** integer
        low += (1 - discount) * fraction * discount ** (length - integer - 1)
    return low, high
