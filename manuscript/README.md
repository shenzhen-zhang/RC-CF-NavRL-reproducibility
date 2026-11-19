# Figure and table index

Numbering follows the 21-page revision sent to *Drones* on November 19, 2026. The [artifact manifest](artifacts.json) identifies the submitted files and PDF checksums. Regenerated plots can differ in rendering across environments.

## Figures

| Submitted figure | Data and regeneration |
| --- | --- |
| [1. Framework](figures/figure-1.pdf) | Conceptual diagram. Submitted PDF. |
| [2. Local encounter](figures/figure-2.pdf) | Conceptual diagram. Submitted PDF. |
| [3. Simulation environment](figures/figure-3.pdf) | [Scene export](../visuals/data/d80_snapshot.json), [screenshot](../figures/canonical_d80_scene_cropped.png), [script](../visuals/environment_setup/build_environment_setup.py). This D80 snapshot belongs to the density experiment. |
| [4. Navigation outcomes](figures/figure-4.pdf) | [Original episodes](../raw/original/primary_paired_episodes.csv), [released-checkpoint episodes](../raw/released_checkpoint/navigation/), [original estimates](../evidence/canonical_paired_effects.json), [released estimates](../evidence/primary_complete18.csv), [plot script](../build/reorganize_evidence_figures.py). |
| [5. Example flights](figures/figure-5.pdf) | [Recorded trajectories](../raw/flight_case/), [plot inputs](../visuals/data/flight_case.json), [script](../visuals/flight_case/build_flight_case.py). |
| [6. Moving-obstacle density](figures/figure-6.pdf) | [Episode data](../raw/original/density_noise/), [scene snapshots](../raw/scenes/), [estimates](../evidence/paper_aligned_summary.csv), [plot script](../build/reorganize_evidence_figures.py). |
| [7. Unseen layouts and observation noise](figures/figure-7.pdf) | [Unseen-layout episodes](../raw/original/unseen_layouts/), [noise episodes](../raw/original/density_noise/), [layout effects](../visuals/data/ood_paired_effects.json), [noise effects](../evidence/paper_aligned_paired_effects.csv), [plot script](../build/reorganize_evidence_figures.py). |
| [8. Local teacher replay](figures/figure-8.pdf) | [Replay data](../raw/released_checkpoint/replay/), [analysis](../analysis/replay/analyze_teacher_replay.py), [estimates](../evidence/teacher_all_reported_metrics.csv), [plot script](../visuals/teacher_validation/build_effect_matrix.py). |
| [9. Learned mechanisms](figures/figure-9.pdf) | [Seed-level diagnostic export](../evidence/mechanism_seed_diagnostics.csv), [provenance](../evidence/mechanism_diagnostics_provenance.json), [plot script](../build/reorganize_evidence_figures.py). |

## Tables

All five use the [table generator](../build/build_current_tables.py).

| Submitted table | Data |
| --- | --- |
| [1. Policy latency](tables/table-1.tex) | [Three timing repeats](../raw/runtime/), [timing aggregates](../evidence/runtime_complexity_summary.json). |
| [A1. Adaptation settings](tables/table-A1.tex) | [Configuration files](../simulation/methods/radcppo_navrl/cfg/), [released-cohort study plan](../analysis/supplementary/study_plan.json). |
| [A2. Component effects](tables/table-A2.tex) | [Original episodes](../raw/original/canonical/), [original paired effects](../visuals/data/mean_risk_core_ablation_statistics.json), [released common-17 effects](../evidence/scope17_paired_effects_disclosed.json). |
| [A3. Teacher replay](tables/table-A3.tex) | [Candidate outcomes and anchors](../raw/released_checkpoint/replay/), [replay estimates](../evidence/teacher_all_reported_metrics.csv), [qualification coverage](../evidence/teacher_qualification_coverage.csv). |
| [A4. Velocity-obstacle protection](tables/table-A4.tex) | [Recorded VO episodes](../raw/original/deployment/), [frozen paired effects and intervals](../evidence/deployment_shield_paired_effects.csv). |

## Analysis coverage

- The two main outcomes use all 18 planned evaluation seeds. The wider released-checkpoint component matrix has [five missing checkpoint–evaluation blocks](../analysis/supplementary/missing-cases.json), totaling 320 episodes. Its paired contrasts use the common 17 seeds. The [18-seed sensitivity results](../processed/supplementary/完整18主要比较/bounded_missing18_sensitivity.json) bound the effect of those missing outcomes.
- Figure 9 uses the recorded seed-level diagnostics. Table A2 uses frozen original-cohort intervals and recomputed common-17 intervals. Table A4 and the finite-episode cost summaries use frozen estimates. Step-level cost traces are outside the package.
- The [input manifest](../metadata/released-inputs.json) tracks the released cohort and teacher replay inputs. Path fields marked `recorded://` point to source records from the original experiment environment.
