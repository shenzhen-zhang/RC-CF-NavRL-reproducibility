# Data archives

Download the three archives from [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.24362473). The [CPU workflow](../README.md#reproduce-on-cpu) uses the episode and replay data already included in GitHub.

| Archive | Size | Contents |
| --- | ---: | --- |
| `episode-and-replay-data.zip` | 238 MB | The recorded episode, teacher replay, trajectory, timing and scene data, archived estimates, and field definitions |
| `released-cohort-checkpoints.zip` | 260 MB | 90 final policy checkpoints, with variant, training seed and SHA-256 for each. The NavRL initializer remains at its [pinned upstream location](navrl-checkpoint.json). |
| `released-cohort-step-cost-traces.zip` | 1.51 GB | 1,633 original NPZ files with per-step clearance, cost and event masks. Each NPZ covers one checkpoint and evaluation seed with 64 environment cases. |

## Download and join the parts

Zenodo stores each ZIP in numbered parts. Run only the download and join commands for the archive you need, in an empty directory.

```bash
record=https://zenodo.org/records/24362473/files

# Episode and replay records
curl -fL --fail-early --retry 3 --remote-name-all "$record/episode-and-replay-data.zip.part-[01-11]" &&
  cat episode-and-replay-data.zip.part-* > episode-and-replay-data.zip

# Final checkpoints
curl -fL --fail-early --retry 3 --remote-name-all "$record/released-cohort-checkpoints.zip.part-[01-12]" &&
  cat released-cohort-checkpoints.zip.part-* > released-cohort-checkpoints.zip

# Per-step cost traces
curl -fL --fail-early --retry 3 --remote-name-all "$record/released-cohort-step-cost-traces.zip.part-[01-66]" &&
  cat released-cohort-step-cost-traces.zip.part-* > released-cohort-step-cost-traces.zip
```

Compare the assembled ZIP's SHA-256 with [SHA256SUMS.txt](https://zenodo.org/records/24362473/files/SHA256SUMS.txt) before extracting it.

## Recompute the safety costs

The cost workflow uses the 535 available full/no-gate/no-counterfactual blocks, checks recorded checkpoint hashes against the [weight manifest](released-checkpoints.csv), and verifies the four reported F11 cost estimands. Five planned ablation blocks are absent. The [figure and table index](../manuscript/README.md) describes the affected analysis.

```bash
mkdir -p recomputed
python analysis/reproduce_costs.py /path/to/released-cohort-step-cost-traces.zip \
  --output recomputed/costs.json
```

Pass the ZIP as downloaded or assembled. The [data dictionary](DATA_DICTIONARY.md) gives the trace fields and units.
