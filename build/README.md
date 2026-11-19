# Figure and table generators

- [`reorganize_evidence_figures.py`](reorganize_evidence_figures.py) renders Figures 4, 6, 7, and 9 from the archived analysis outputs.
- [`build_current_tables.py`](build_current_tables.py) renders Tables 1 and A1–A4 into `visuals/tables/`.

The [CPU entry point](../analysis/reproduce.py) runs both scripts. It checks the newly generated LaTeX tables against the submitted copies.
