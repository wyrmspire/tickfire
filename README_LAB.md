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
