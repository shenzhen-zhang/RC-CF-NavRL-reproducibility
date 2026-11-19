# Recorded experiment data

| Directory | Records |
| --- | --- |
| [`original/`](original/) | 11,520 episode rows for the 5,760 original primary pairs, plus density, noise, unseen-layout, VO, and 17,280 matched component-ablation episode rows |
| [`released_checkpoint/navigation/`](released_checkpoint/navigation/) | 104,512 episode rows across 341 shards for the second cohort |
| [`released_checkpoint/replay/`](released_checkpoint/replay/) | Candidate outcomes and anchor arrays from 20 local replay jobs |
| [`flight_case/`](flight_case/) | Four policy-specific trajectories for the displayed flight example |
| [`runtime/`](runtime/) | Three recorded latency benchmark repeats |
| [`scenes/`](scenes/) | Moving-obstacle scene snapshots for the density tests |

The released-checkpoint component matrix has [five missing checkpoint–evaluation blocks](../analysis/supplementary/missing-cases.json) totaling 320 episodes. Its main comparisons cover all 18 planned evaluation seeds. Component comparisons use the 17 seeds shared by all variants.

See the [data dictionary](../metadata/DATA_DICTIONARY.md) for field definitions and units.

See the [data archive guide](../metadata/ARCHIVES.md) for the separate full control-step traces and policy checkpoints.
