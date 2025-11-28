"""
Inspect a lab run's dataset.

Usage (from repo root):

    python scripts/inspect_run_dataset.py runs/sweeps/manual_single/<run_dir>
    python scripts/inspect_run_dataset.py runs/sweeps/dev_sweep_stub/run_000_bars_apr07_apr25_15s_indfull_v1_w256_h1
"""

from pathlib import Path
import sys
import json
import numpy as np


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_run_dataset.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[inspect] Run dir does not exist: {run_dir}")
        sys.exit(1)

    status_path = run_dir / "status.json"
    if not status_path.exists():
        print(f"[inspect] No status.json in {run_dir}")
        sys.exit(1)

    with status_path.open("r") as f:
        status = json.load(f)

    ds_info = status.get("dataset", {})
    ds_path = ds_info.get("dataset_path")
    if not ds_path:
        print("[inspect] No dataset_path in status.json")
        sys.exit(1)

    ds_path = Path(ds_path)
    if not ds_path.exists():
        print(f"[inspect] Dataset file missing: {ds_path}")
        sys.exit(1)

    print("=== RUN INFO ===")
    print(f"name:              {status.get('name')}")
    print(f"sweep_id:          {status.get('sweep_id')}")
    print(f"experiment_kind:   {status.get('experiment_kind')}")
    print(f"data_family:       {status.get('data_family')}")
    print(f"base_timeframe:    {status.get('base_timeframe')}")
    print(f"indicator_set:     {status.get('indicator_set')}")
    print(f"window:            {status.get('window')}")
    print(f"horizon:           {status.get('horizon')}")
    print()

    print("=== DATASET ===")
    print(f"path:              {ds_path}")
    print(f"num_windows:       {ds_info.get('num_windows')}")
    print(f"feature_dim:       {ds_info.get('feature_dim')}")
    print(f"features_used:     {ds_info.get('features_used')}")
    print(f"missing_from_pal:  {ds_info.get('features_missing_from_palette')}")
    print()

    npz = np.load(ds_path)
    X = npz["X"]
    Y = npz["Y"]

    print("=== SHAPES ===")
    print(f"X.shape:           {X.shape}   # [N, window, feat_dim]")
    print(f"Y.shape:           {Y.shape}   # [N]")
    print()

    if Y.size > 0:
        print("=== TARGET (Y) STATS ===")
        print(f"Y min:             {float(Y.min()): .6f}")
        print(f"Y max:             {float(Y.max()): .6f}")
        print(f"Y mean:            {float(Y.mean()): .6f}")
        print(f"Y std:             {float(Y.std()): .6f}")
    else:
        print("=== TARGET (Y) STATS ===")
        print("Y is empty.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
