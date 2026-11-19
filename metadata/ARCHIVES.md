# Data archives

The [CPU workflow](../README.md#reproduce-on-cpu) uses the recorded data in GitHub. The three archives below are being uploaded to Zenodo. This page will link to the public dataset once the upload is verified.

| Archive | Size | Contents |
| --- | ---: | --- |
| `episode-and-replay-data.zip` | 238 MB | The recorded episode, teacher replay, trajectory, timing and scene data, archived estimates, and field definitions |
| `released-cohort-checkpoints.zip` | 260 MB | 90 final policy checkpoints, with variant, training seed and SHA-256 for each. The NavRL initializer remains at its [pinned upstream location](navrl-checkpoint.json). |
| `released-cohort-step-cost-traces.zip` | 1.51 GB | 1,633 original NPZ files with per-step clearance, cost and event masks. Each NPZ covers one checkpoint and evaluation seed with 64 environment cases. |

## Recompute the safety costs

The cost workflow uses the 535 available full/no-gate/no-counterfactual blocks, checks recorded checkpoint hashes against the [weight manifest](released-checkpoints.csv), and verifies the four reported F11 cost estimands. Five planned ablation blocks are absent. The [figure and table index](../manuscript/README.md) describes the affected analysis.

```bash
mkdir -p recomputed
python analysis/reproduce_costs.py /path/to/released-cohort-step-cost-traces.zip \
  --output recomputed/costs.json
```

Pass the ZIP as downloaded or assembled. The [data dictionary](DATA_DICTIONARY.md) gives the trace fields and units.
