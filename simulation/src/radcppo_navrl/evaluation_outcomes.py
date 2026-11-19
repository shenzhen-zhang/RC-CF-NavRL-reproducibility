"""Pure outcome definitions shared by NavRL evaluation and protocol tests."""

from __future__ import annotations


def first_episode_step(step: int, *, done_step: int, num_steps: int) -> int:
    """Return a zero-based event step in the first episode or -1 if absent."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if step < 0 or done_step < 0:
        raise ValueError("outcome step indices must be non-negative")
    if done_step >= num_steps or step >= num_steps or step > done_step:
        return -1
    return step


def classify_navrl_outcomes(
    *,
    goal_step: int,
    unsafe_step: int,
    done_step: int,
    num_steps: int,
    reached_goal_at_done: bool,
) -> tuple[bool, bool]:
    """Return official goal-at-done and collision-free first-goal success."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if min(goal_step, unsafe_step, done_step) < 0:
        raise ValueError("outcome step indices must be non-negative")

    rollout_done = done_step < num_steps
    goal_reach = rollout_done and bool(reached_goal_at_done)
    success = (
        rollout_done
        and goal_step < num_steps
        and goal_step < unsafe_step
        and goal_step <= done_step
    )
    return goal_reach, success
