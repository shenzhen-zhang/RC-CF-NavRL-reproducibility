# RC-CF-NavRL

Research code and reproducibility data for **Learning Bounded Velocity Corrections for UAV Navigation among Moving Obstacles** by Shenzhen Zhang and Shan Wang. RC-CF-NavRL learns a bounded velocity correction for a pretrained UAV navigation policy using local counterfactual supervision.

![RC-CF-NavRL framework](manuscript/overview.png)

## Reproduce on CPU

Use Python 3.10–3.12. The commands were tested on Linux with Python 3.12.

```bash
git clone https://github.com/shenzhen-zhang/RC-CF-NavRL-reproducibility.git
cd RC-CF-NavRL-reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/reproduce.py --quick --output recomputed/quick
```

The quick check recomputes the two navigation cohorts and checks the flight trajectory and timing aggregates. It also renders seven figures and rebuilds all five tables. Run the full CPU check to also recompute the density, noise, unseen-layout, and teacher replay analyses.

```bash
python analysis/reproduce.py --output recomputed/full
```

Statistics are written to the specified output directory. Regenerated plots are under `visuals/*/output/`, and tables are under `visuals/tables/`. Choose a new output directory for each run. The teacher replay bootstrap can take several minutes.

## Main result

Across 5,760 paired episodes in the original cohort, RC-CF-NavRL increased success from **37.62% to 45.97%** and reduced collisions from **59.53% to 52.15%** relative to matched PPO-Lagrangian. The separate released-checkpoint cohort and uncertainty intervals are shown in [Figure 4](manuscript/figures/figure-4.pdf).

## Find the files

| Where | Contents |
| --- | --- |
| [Figure and table index](manuscript/README.md) | Submitted figures and tables, with a link to the data and script for each |
| [Raw data](raw/) · [Data dictionary](metadata/DATA_DICTIONARY.md) | Per-episode CSVs, flight and teacher replay NPZs, scenes, and timing repeats |
| [Analysis](analysis/) · [Evidence](evidence/) · [Supplementary estimates](processed/supplementary/) | Statistical code, reference estimates, and protocol exports |
| [Simulation](simulation/README.md) | Policy implementation, training and evaluation entry points, Isaac Sim setup |

The package reproduces reported analyses from recorded outputs. The adapted policy weights, original-cohort initialization, and step-level cost traces are outside this release. Some diagnostics and appendix intervals use frozen estimates as documented in the [figure and table index](manuscript/README.md).

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. Project code is [MIT](LICENSE), and the accompanying data and author-created figures and tables are [CC BY 4.0](metadata/LICENSE-CC-BY-4.0.txt). See the [NavRL license](simulation/LICENSE.NavRL) for upstream portions.
