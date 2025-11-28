"""
Run a single bars+indicators experiment via the lab engine.

Usage (from repo root):

    python scripts/run_single.py
"""

from pathlib import Path
from datetime import datetime
import sys

# Ensure project root (which contains the `lab` package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main() -> None:
    sweep_id = "manual_single"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / "sweeps" / sweep_id / f"run_000_manual_single_{ts}"

    cfg = ExperimentConfig(
        name=f"manual_single_{ts}",
        sweep_id=sweep_id,
        experiment_kind="bars_multi_tf",
        data_family="bars_apr07_apr25",
        base_timeframe="15s",
        context_timeframes=["5m", "15m"],  # just to populate the field
        indicator_set="full_v1",
        window=256,
        horizon=1,
        out_dir=run_dir,
    )

    result = run_experiment(cfg)
    print("[run_single] Experiment completed.")
    print(result)


if __name__ == "__main__":
    main()
