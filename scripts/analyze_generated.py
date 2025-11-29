from pathlib import Path
import sys
import json

import pandas as pd


def analyze_generated(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    if "is_generated" not in df.columns:
        raise ValueError("CSV missing 'is_generated' column")

    gen = df[df["is_generated"] == 1].copy()
    ctx = df[df["is_generated"] == 0].copy()

    if gen.empty:
        return {"status": "no_generated_rows"}

    gen["close_ret"] = gen["close"].pct_change()

    # Basic stats
    start_close = float(gen["close"].iloc[0])
    end_close = float(gen["close"].iloc[-1])
    total_drift = (end_close / start_close) - 1.0 if start_close != 0 else float("nan")

    rets = gen["close_ret"].dropna()

    metrics = {
        "status": "ok",
        "num_context_rows": int(len(ctx)),
        "num_generated_rows": int(len(gen)),
        "start_ts": str(gen["ts"].iloc[0]),
        "end_ts": str(gen["ts"].iloc[-1]),
        "start_close": start_close,
        "end_close": end_close,
        "total_drift_pct": float(total_drift * 100.0),
        "mean_ret_per_bar": float(rets.mean()) if not rets.empty else 0.0,
        "std_ret_per_bar": float(rets.std()) if not rets.empty else 0.0,
        "max_abs_ret_per_bar": float(rets.abs().max()) if not rets.empty else 0.0,
    }

    # Simple flags
    metrics["flag_explosive"] = metrics["max_abs_ret_per_bar"] > 0.03  # >3% per 15s
    metrics["flag_huge_drift"] = abs(metrics["total_drift_pct"]) > 5.0  # >5% over whole sim

    return metrics


def find_latest_manual_test(root: Path) -> Path:
    sweeps_root = root / "runs" / "sweeps" / "manual_test"
    if not sweeps_root.exists():
        raise FileNotFoundError(f"No manual_test sweeps at {sweeps_root}")

    runs = sorted(sweeps_root.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No runs found under {sweeps_root}")

    return runs[-1]


def main():
    root = Path(__file__).resolve().parents[1]

    # Optional arg: run directory
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
    else:
        run_dir = find_latest_manual_test(root)

    csv_path = run_dir / "generated_sequence.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"generated_sequence.csv not found in {run_dir}")

    metrics = analyze_generated(csv_path)

    # Print to stdout
    print(json.dumps(metrics, indent=2))

    # Save to results folder
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    out_path = results_root / f"{run_dir.name}_analysis.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[analyze_generated] Saved analysis to {out_path}")


if __name__ == "__main__":
    main()
