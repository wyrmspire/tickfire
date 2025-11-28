"""
Phase-0 pipeline runner.

This wraps the existing ml/ scripts into a single command:

    python scripts/run_phase0_pipeline.py

It will:

1. Change into the ml/ directory (so relative paths like ../data/rawprice work).
2. Import and call:
   - build_dataset_v2.main()
   - train_gru.main()
   - generate_gru.main()

All configs (symbol, date, window, etc) are taken from the ml scripts themselves
for now. We'll gradually refactor those into the new lab/ engine/config system.
"""

from pathlib import Path
import os
import sys


def main() -> None:
    # Project root = this file's parent, then parent (scripts/ -> root)
    root = Path(__file__).resolve().parents[1]
    ml_dir = root / "ml"

    if not ml_dir.exists():
        raise SystemExit(f"[phase0] ml/ directory not found at: {ml_dir}")

    print(f"[phase0] Project root: {root}")
    print(f"[phase0] Using ml dir: {ml_dir}")

    # Make sure imports like `from model_gru import PriceGenGRU` work
    sys.path.insert(0, str(ml_dir))

    # Change working directory so relative paths inside ml scripts resolve:
    # - RAW_DIR = Path(\"../data/rawprice\")
    # - OUT_DIR = Path(\"out\")
    os.chdir(ml_dir)
    print(f"[phase0] Changed cwd to: {Path.cwd()}")

    # Lazy imports after cwd change, so their module-level Path configs behave
    try:
        import build_dataset_v2
    except ImportError as e:
        raise SystemExit(f"[phase0] Failed to import build_dataset_v2: {e}")

    try:
        import train_gru
    except ImportError as e:
        raise SystemExit(f"[phase0] Failed to import train_gru: {e}")

    try:
        import generate_gru
    except ImportError as e:
        raise SystemExit(f"[phase0] Failed to import generate_gru: {e}")

    # ------------------------------------------------------------------
    # STEP 1: build dataset (ticks -> 15s -> features -> windows -> npz)
    # ------------------------------------------------------------------
    if hasattr(build_dataset_v2, "main"):
        print("[phase0] STEP 1: build_dataset_v2.main()")
        build_dataset_v2.main()
    else:
        print("[phase0] WARNING: build_dataset_v2.main not found, skipping.")

    # ------------------------------------------------------------------
    # STEP 2: train GRU on that dataset
    # ------------------------------------------------------------------
    if hasattr(train_gru, "main"):
        print("[phase0] STEP 2: train_gru.main()")
        train_gru.main()
    else:
        print("[phase0] WARNING: train_gru.main not found, skipping.")

    # ------------------------------------------------------------------
    # STEP 3: generate synthetic 15s candles using trained GRU
    # ------------------------------------------------------------------
    if hasattr(generate_gru, "main"):
        print("[phase0] STEP 3: generate_gru.main()")
        generate_gru.main()
    else:
        print("[phase0] WARNING: generate_gru.main not found, skipping.")

    print("[phase0] Pipeline complete (phase-0).")
    print("[phase0] Check ml/out or top-level out/ for generated files.")


if __name__ == "__main__":
    main()
