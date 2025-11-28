# Tickfire Lab Runbook

This runbook captures how the lab is structured today, how to run what already exists, and the next upgrades to prioritize as we scale automated model search for MES price movement.

## Repository map (quick reference)

- `data/` — local-only raw inputs (not versioned)
  - `rawprice/` — MDP3 tick JSON; source of truth for candles.
- `out/` — generated artifacts (also local-only)
  - `bars/` and `bars_indicators/` — prebuilt multi-timeframe bars + indicators.
- `ml/` — phase-0 reference pipeline (ticks → 15s candles → features → GRU training → generation).
- `lab/` — evolving experiment platform
  - `engine/` — orchestrates a run via `run_experiment`.
  - `features/`, `models/`, `metrics/`, `compare/` — registries and utilities (largely stubs now).
  - `utils/` — path helpers.
- `scripts/` — entrypoints for single runs, sweeps, and legacy phase-0 helpers.
- `runs/` — per-run outputs written by the lab engine (created on demand).
- `ui/` — placeholder for future visualization (not wired yet).

## How to run what exists today

### 1) Phase-0 (legacy raw tick pipeline)

Use this when you want the known-good GRU reference path that starts from raw MDP3 trades.

```bash
python scripts/run_phase0_pipeline.py
```

This simply chains the `ml/` scripts in-place. Make sure the expected raw files are under `data/rawprice/`.

To execute the same logic through the lab engine (so outputs land in `runs/sweeps/...`):

```bash
python scripts/run_phase0_experiment.py
```

That wraps `lab.engine.run_experiment` with `experiment_kind="phase0_raw_ml"` and copies artifacts into the run folder.

### 2) Lab engine stubs (bars/indicator workflow)

The future direction is the `bars_multi_tf` workflow, which will consume prebuilt bars/indicators and handle sweeps. The current code is a stub but already produces consistent run folders for downstream tooling.

- Single stub run:

  ```bash
  python scripts/run_single.py
  ```

- Stub sweep (three configs with different timeframes/indicator sets/windows):

  ```bash
  python scripts/run_sweep.py
  ```

Both commands will write under `runs/sweeps/<sweep_id>/run_*` and emit `config.json`, `status.json`, and `eval/metrics.json` placeholders so we can start building dashboards and comparisons.

### 3) Inspecting structure

For a quick sanity check of directory contents:

```bash
python scripts/print_structure.py
```

## Data expectations

- Raw MDP3 trade files live outside of version control in `data/rawprice/` (e.g., `glbx-mdp3-20250318.trades.json`).
- Derived artifacts (bars, indicators, datasets, generated candles) should also stay untracked under `out/` or `runs/`.
- The repo ships with no data; ensure local paths match the scripts before running.

## What to improve next (priority list)

1. **Bars+indicator dataset builder** — Implement the `bars_multi_tf` pipeline to read `out/bars/` and `out/bars_indicators/`, assemble feature frames for configurable windows/horizons, and cache datasets under `data/compiled/`.
2. **Feature palettes** — Fill `lab/features/palettes.py` with concrete palettes (F0–F5) and a helper to materialize them from bar data, including multi-timeframe joins.
3. **Model registry** — Flesh out `lab/models/registry.py` to instantiate GRU/Transformer variants and surface parameter counts for sweep bookkeeping.
4. **Metrics + comparisons** — Add return/volatility/shape metrics in `lab/metrics/` and leaderboards in `lab/compare/` so sweeps produce sortable summaries.
5. **Sweep orchestration** — Extend `scripts/run_sweep.py` to accept CLI/JSON definitions, handle resume/skip, and optionally distribute runs.
6. **Logging & artifacts** — Standardize per-phase timers, error logging, and artifact manifests (datasets, checkpoints, generated bars) in `run_experiment` outputs.
7. **UI hook-up** — Point a lightweight viewer (CLI or Streamlit) at `runs/sweeps/**/` to visualize metrics and generated series side-by-side.

As these pieces land, the lab will move from stubbed scaffolding to a repeatable, automated search surface for the best MES price-generation models.
