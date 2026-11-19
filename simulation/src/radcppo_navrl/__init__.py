"""Training-side reachability-aware constrained PPO components for NavRL."""

from .actor import InterventionGatedBetaActor, PlainBetaActor

from .distributional import (
    MonotonicQuantileCritic,
    counterfactual_calibration_loss,
    normalized_safety_td_target,
    quantile_huber_loss,
    upper_cvar,
    upper_tail_dominance_margin,
)
from .reachability_teacher import (
    ReachabilityTeacher,
    ReachabilityTeacherConfig,
    TeacherOutput,
    observation_min_clearance,
    observation_safety_cost,
)
from .risk_selector import (
    ConservativeRiskSelector,
    ExponentialCorrectionSmoother,
    RiskSelection,
)

__all__ = [
    "InterventionGatedBetaActor",
    "MonotonicQuantileCritic",
    "PlainBetaActor",
    "ReachabilityTeacher",
    "ReachabilityTeacherConfig",
    "ConservativeRiskSelector",
    "ExponentialCorrectionSmoother",
    "RiskSelection",
    "TeacherOutput",
    "counterfactual_calibration_loss",
    "normalized_safety_td_target",
    "observation_min_clearance",
    "observation_safety_cost",
    "quantile_huber_loss",
    "upper_cvar",
    "upper_tail_dominance_margin",
]
