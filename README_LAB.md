# tickfire lab

This is the experimental platform for synthetic price generation.

Key ideas:

- Keep raw data (`data/rawprice`, `out/bars`, `out/bars_indicators`).
- Treat each training+generation as a *run* with its own folder.
- Standardize how we log config, timings, metrics, and charts.
- Add sweeps on top so we can explore many combinations of:
  - timeframes
  - window sizes
  - feature palettes
  - model families & sizes
  - training budgets

Over time we will wire:

- `lab/engine/run_experiment.py`  -> orchestrates a full run
- `lab/features/*`                -> define feature palettes
- `lab/models/*`                  -> build models given config
- `lab/metrics/*`                 -> compute metrics from real vs synthetic
- `lab/compare/*`                 -> build leaderboards and prune runs

## Quick start

- **Phase-0 pipeline (ticks → GRU → synthetic 15s)**
  ```bash
  python scripts/run_phase0_pipeline.py
  ```
- **Same pipeline via the lab engine (artifacts copied into runs/):**
  ```bash
  python scripts/run_phase0_experiment.py
  ```
- **Stubbed bars/indicator workflow (creates run folders and metrics placeholders):**
  ```bash
  python scripts/run_single.py          # one stub
  python scripts/run_sweep.py           # small sweep of stub configs
  ```

For a full tour of the directory layout, run commands, and the next improvements to tackle, see [`docs/LAB_RUNBOOK.md`](docs/LAB_RUNBOOK.md).
