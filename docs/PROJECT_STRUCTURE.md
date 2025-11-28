# tickfire Project Structure (current)

This document describes how the repo is laid out **right now**, after the initial lab bootstrap and before we fully integrate everything.

## Top-level

- `data/`
  - `rawprice/`  
    - Raw MDP3 trade files for MES (e.g. `glbx-mdp3-20250318.trades.json`).
    - These are the source of truth for price data.

- `out/`
  - `bars/YYYYMMDD/*.csv`  
    - Multi-timeframe OHLCV bars already built (15s, 1m, 5m, 15m, 1h, 4h).
  - `bars_indicators/YYYYMMDD/*_ind.csv`  
    - Same structure but with indicator-enriched bars.

- `ml/`
  - Prototype / phase-0 ML pipeline.
  - Contains working scripts for:
    - building 15s candles from raw trades
    - enriching candles with features
    - building sliding window datasets
    - defining the GRU model
    - training the GRU
    - generating synthetic 15s candles from a trained GRU
  - This is the code that produced the **early good small GRU** you mentioned.

- `lab/`
  - New **platform** skeleton for multi-run experiments:
    - `lab/engine/`   — experiment runner orchestration.
    - `lab/features/` — feature palettes (to be wired).
    - `lab/models/`   — model registry (to be wired).
    - `lab/metrics/`  — return & OHLC metrics (to be wired).
    - `lab/compare/`  — sweep-level leaderboards (to be wired).
    - `lab/utils/`    — path helpers, etc.
  - Right now mostly stubs and scaffolding.

- `runs/`
  - `runs/sweeps/` — each sweep will get its own subfolder.
  - `runs/best/`   — will store symlinks or copies of top runs later.

- `scripts/`
  - Entrypoints for running:
    - a single experiment (`run_single.py`)
    - a sweep of experiments (`run_sweep.py`)
    - comparison / pruning scripts (stubs for now).

- `ui/`
  - Placeholder for a future viewer (CLI / Streamlit / Next.js).
  - Will read `runs/sweeps/**/metrics.json` and charts for side-by-side inspection.

- `README_LAB.md`
  - High-level description of the "lab" concept.

## Intent going forward

- **Do not** put raw/generated data under version control.
- Use `lab/` as the main home for incremental improvements.
- Treat `ml/` as our **phase-0 reference implementation** that we will gradually refactor into `lab/engine` and `lab/features`.
