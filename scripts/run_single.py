"""
Run a single experiment (stub).

Eventually this script will:
- parse a config (CLI args or a small JSON)
- call lab.engine.run_experiment(...)
- print where outputs were written
"""

from pathlib import Path
from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main():
    sweep_id = "manual_single"
    run_dir = Path("runs") / "sweeps" / sweep_id / "run_000_manual_single"

    cfg = ExperimentConfig(
        name="manual_single",
        sweep_id=sweep_id,
        experiment_kind="bars_multi_tf",
        data_family="bars_apr07_apr25",
        out_dir=run_dir,
    )

    result = run_experiment(cfg)
    print("[run_single] Experiment completed (stub).")
    print(result)


if __name__ == "__main__":
    main()
