# Data dictionary

CSV files use UTF-8 and JSON files use named fields. The [raw data index](../raw/README.md) groups files by experiment. The [released input manifest](released-inputs.json) records source checksums. A `recorded://` path identifies a source in the original experiment environment.

Treat `nan`, unavailable interval endpoints and explicit unavailable flags as missing. Values retain the units of the original exports.

## Episode identifiers

| Field | Meaning |
| --- | --- |
| `mode` | Policy family and training seed. `mean_risk_s0` through `mean_risk_s4` identify original RC-CF-NavRL runs. `ppolag_matched_s0` through `ppolag_matched_s4` identify the matched PPO-Lagrangian runs. |
| `seed`, `env_index` | Shared evaluation randomization seed and fixed environment case (0–63). |
| `family`, `variant`, `deployment`, `condition`, `comparison`, `contrast`, `composite_key` | Study and comparison labels. Read `difference_definition` for the subtraction order. |
| `training_seed_index`, `training_seeds`, `evaluation_seeds`, `stratum_count`, `episode_count`, `paired_episode_count`, `referenced_paired_differences` | Seed identifiers or counts. Fields ending in `_count` are also counts. |

## Navigation outcomes

| Field | Meaning and units |
| --- | --- |
| `success`, `goal_reach`, `collision`, `out_of_bounds`, `truncated`, `near_miss` | Episode indicators (0/1). Averages are proportions. The manuscript defines first-event and termination conventions. |
| `first_goal_step`, `first_unsafe_step`, `first_done_step`, `episode_len`, `anchor_step` | Simulator steps. A negative first-event step indicates no recorded event. |
| `time_to_goal_s` | Time in seconds. Failed flights can have missing values. |
| `path_length_m`, `straight_distance_m`, `min_clearance_m` | Metres. The simulator's surface clearance measurement can be negative. |
| `path_efficiency` | Straight distance divided by path length for a success. Failed flights receive zero. |
| `return`, `mean_safety_cost`, `empirical_cost_cvar`, `predicted_cost_cvar` | Dimensionless task return, recorded costs or model predictions. |
| Mean-command acceleration, jerk, and command shifts | Acceleration uses m/s² and jerk uses m/s³. Fields ending in `_mps` use m/s. |
| `action_oscillation_rate`, `stop_rate`, intervention, selector, teacher and feasibility rates | Dimensionless proportions. |

## Estimates, intervals and timing

- `estimate`, `difference`, `paired_difference`, `full_mean`, `ablation_mean`, `method_mean`, `baseline_mean`, `five_seed_mean` and `five_seed_mean_or_single_checkpoint_value` use the named metric's native unit.
- `ci95_low`, `ci95_high`, `hierarchical_bootstrap_ci95` and `bootstrap_ci_two_sided` give 95% interval endpoints. The fixed original NavRL checkpoint has no interval across training seeds. Conditional sensitivity estimates have their own labels.
- Multiply success or collision proportions by 100 for percentages and their differences by 100 for percentage points. The teacher figure converts metre differences to centimetres.
- Probabilities and p-values are dimensionless. GPU and batch sizes describe the benchmark protocol.
- `wall_mean_ms` and `cuda_mean_ms` use milliseconds. The timing table takes the mean and sample SD of three repeat means. Memory and storage fields ending in MiB use 2²⁰ bytes.

## Local teacher replay

| Field | Meaning |
| --- | --- |
| `source`, `family` | Checkpoints that generated the replay anchors. |
| `duration` | 50- or 49-step replay. |
| Collision, clearance and progress contrasts | Teacher-selected minus original nominal command. Collision uses proportions and distance uses metres. |
| AUROC, average precision and candidate-order accuracy | Dimensionless measures of prediction or within-anchor ranking. |
| `qualified_fraction`, `qualified_count` | Fraction of the planned anchor/environment panel and count of accepted anchors. Bootstrap flags identify rare-event or undefined intervals. |

## Mechanism diagnostics

`gate_mean_danger` and `gate_mean_safe` are dimensionless mean gates grouped by the teacher's intervention label. The labels describe the teacher’s decision for each observation. Brier score, accuracy, calibration and quantile coverage errors are dimensionless. Adapter and gated residual norms use m/s. Figure 9 converts adapter shifts to mm/s.

Each archived diagnostic aggregates six evaluation seeds per training seed. See the [diagnostic provenance](../evidence/mechanism_diagnostics_provenance.json) for source fields and aggregation.

## Geometry and trajectories

Scene and flight metadata identify axes, obstacle positions and sizes, and scene hashes. Positions, goals and dimensions use metres. Velocities use m/s, times use seconds and steps use simulator indices.

The flight plot uses records through termination and omits the automatic reset. The plotted illustration omits static obstacles and expands the vertical display scale. The archives preserve the original coordinates. Figures 1 and 2 are conceptual diagrams.

## Control-step safety costs

Each schema-v3 NPZ in the [cost archive](ARCHIVES.md) covers one evaluation seed and checkpoint. It records the checkpoint name and hash, training and evaluation seeds, source hashes and cost settings.

| Field | Meaning |
| --- | --- |
| `safety_cost`, `clearance_m` | Arrays indexed by environment and time. Costs are dimensionless in [0, 1]. Clearance uses metres. |
| `collision`, `reached_goal`, `terminated`, `truncated`, `done` | Boolean event masks indexed by environment and time. |
| `final_steps` | Zero-based index of the first resolved event for each environment. |

The clearance sigmoid uses a preferred clearance of 0.35 m and a temperature of 0.15 m. The normalized return uses a discount of 0.99. The [cost analysis](../analysis/reproduce_costs.py) computes finite first-event return, first-event unsafe absorption and termination continuation bounds. Bounds for the wider component comparisons use the [five planned missing blocks](../analysis/supplementary/missing-cases.json).
