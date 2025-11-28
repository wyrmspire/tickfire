"""
Run a single phase-0 experiment through the lab engine.

Usage (from repo root):

    python scripts/run_phase0_experiment.py

This will:
- ensure the project root is on sys.path (so `import lab` works)
- create a run directory under runs/sweeps/phase0_dev/run_<timestamp>/
- call lab.engine.run_experiment(...) with:

    experiment_kind = "phase0_raw_ml"
    data_family     = "raw_mdp3"

so it's explicit that this is the legacy raw-tick pipeline, not the
bars+indicators setup.
"""

from datetime import datetime
from pathlib import Path
import sys

# Make sure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main() -> None:
    sweep_id = "phase0_dev"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sweep_id}_{ts}"

    run_dir = PROJECT_ROOT / "runs" / "sweeps" / sweep_id / f"run_{ts}"

    print(f"[run_phase0_experiment] Starting experiment: {run_name}")
    print(f"[run_phase0_experiment] Run directory: {run_dir}")

    cfg = ExperimentConfig(
        name=run_name,
        sweep_id=sweep_id,
        experiment_kind="phase0_raw_ml",
        data_family="raw_mdp3",
        base_timeframe="15s",
        # context_timeframes / indicator_set keep their defaults for this kind
        out_dir=run_dir,
    )

    info = run_experiment(cfg)

    print("[run_phase0_experiment] Experiment finished.")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
