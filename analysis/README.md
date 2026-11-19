# Analysis

Run `python analysis/reproduce.py --quick` from the repository root for the two navigation cohorts, flight and timing checks, and figure and table rendering. For density, noise, unseen layouts, component ablations, VO deployment, and teacher replay, run the same command without `--quick` and choose a fresh `--output` directory.

| Directory | Analysis |
| --- | --- |
| [`original/`](original/) | Original-cohort paired navigation, density, noise, unseen layouts, component ablations, and VO deployment |
| [`supplementary/`](supplementary/) | Released-checkpoint cohort, common-17 component analysis, missing-outcome bounds, and finite-episode cost |
| [`replay/`](replay/) | Teacher replay validation and analysis of recorded candidates |

See the [figure and table index](../manuscript/README.md) for the scope of each check.

For the separate cost check, pass the original trace ZIP to [reproduce_costs.py](reproduce_costs.py). See the [archive guide](../metadata/ARCHIVES.md) for the command and input details.
