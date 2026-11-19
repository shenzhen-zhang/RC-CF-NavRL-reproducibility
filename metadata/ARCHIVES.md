# Data archives

The CPU workflow starts from this GitHub repository. The larger archives listed below are prepared for a separate Zenodo deposit. A download link will be added when the record is public.

| Archive | Contents |
| --- | --- |
| `episode-and-replay-data.zip` | The recorded episode, teacher replay, trajectory, timing and scene data, archived estimates, and field definitions |
| `released-cohort-checkpoints.zip` | 90 final policy checkpoints, with variant, training seed and SHA-256 for each. The NavRL initializer remains at its [pinned upstream location](navrl-checkpoint.json). |
| `released-cohort-step-cost-traces.zip` | 1,633 original NPZ files with per-step clearance, cost and event masks. Each NPZ covers one checkpoint and evaluation seed with 64 environment cases. |
| `SHA256SUMS.txt` | Archive-level SHA-256 checksums. Each ZIP also contains a per-file manifest. |

The cost workflow uses the 535 available full/no-gate/no-counterfactual blocks, checks recorded checkpoint hashes against the [weight manifest](released-checkpoints.csv), and verifies the four reported F11 cost estimands. Five planned ablation blocks are absent. The [figure and table index](../manuscript/README.md) describes the affected analysis.

```bash
python analysis/reproduce_costs.py /path/to/released-cohort-step-cost-traces.zip \
  --output recomputed/costs.json
```

Create the output's parent directory first, or pass a filename in an existing directory. The ZIP can stay compressed. The [data dictionary](DATA_DICTIONARY.md) gives the trace fields and units.
