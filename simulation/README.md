# Isaac Sim implementation

The experiments used NVIDIA Isaac Sim `2023.1.0-hotfix.1` and [NavRL at commit `3725bcc`](https://github.com/Zhefan-Xu/NavRL/tree/3725bcc2e7c1be4ecf1455d922299ae85042603a). Follow the pinned NavRL setup for OmniDrones, PyTorch, and simulator dependencies. Set `NAVRL_OFFICIAL_ROOT` to that checkout's `isaac-training` directory and add this repository's `simulation/src` to `PYTHONPATH`.

| File | Role |
| --- | --- |
| [`methods/radcppo_navrl/train.py`](methods/radcppo_navrl/train.py) | Training entry point for the correction policy and matched PPO-Lagrangian baseline |
| [`methods/radcppo_navrl/evaluate_v19_canonical.py`](methods/radcppo_navrl/evaluate_v19_canonical.py) | Matched-seed evaluation and per-episode CSV export |
| [`methods/radcppo_navrl/benchmark_v19_deployment_runtime.py`](methods/radcppo_navrl/benchmark_v19_deployment_runtime.py) | Actor and online correction timing |
| [`src/radcppo_navrl/`](src/radcppo_navrl/) | Actor correction, cost critic, teacher, and episode outcome logic |
| [`configs/`](configs/) | Original canonical D80 snapshot and three unseen-layout snapshots |

The original cohort started from a matched stage-0 checkpoint. The second cohort started from [NavRL's public checkpoint](https://github.com/Zhefan-Xu/NavRL/blob/3725bcc2e7c1be4ecf1455d922299ae85042603a/quick-demos/ckpts/navrl_checkpoint.pt). The paper's [training settings](../manuscript/tables/table-A1.tex) and [released-cohort study plan](../analysis/supplementary/study_plan.json) record the experiment overrides. Use those settings when reading the Hydra configuration defaults.

## Reproducing the GPU experiments

The [CPU workflow](../README.md#reproduce-on-cpu) verifies results from recorded data. Rerunning the GPU experiments requires Isaac Sim, the matching scenes, and the checkpoint for each policy. For the second cohort, use the [released-checkpoint setup](released_cohort/README.md), which includes the original training scene geometry, runtime patches, and ordered evaluation manifests. The [archive guide](../metadata/ARCHIVES.md) describes the separate weight archive. The original five-seed cohort’s stage-0 and adapted checkpoints were unavailable in the checked backups.
