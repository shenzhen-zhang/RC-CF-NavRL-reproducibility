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

This release was verified by recomputing from recorded experiment data on CPU. Rerunning training or simulation evaluation requires the simulator, checkpoints, and matching scene setup. The original stage-0 checkpoint and adapted policy weights are outside this package.
