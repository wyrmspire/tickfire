"""
Path helpers for the lab.

Centralized place to define:
- where raw data lives (e.g. data/rawprice, out/bars, out/bars_indicators)
- where compiled datasets should be cached (e.g. data/compiled)
- where runs / sweeps live
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAWPRICE_DIR = DATA_DIR / "rawprice"
OUT_DIR = PROJECT_ROOT / "out"
BARS_DIR = OUT_DIR / "bars"
BARS_IND_DIR = OUT_DIR / "bars_indicators"
RUNS_DIR = PROJECT_ROOT / "runs"
SWEEPS_DIR = RUNS_DIR / "sweeps"
BEST_DIR = RUNS_DIR / "best"
