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
    cfg = ExperimentConfig(
        name="manual_single",
        out_dir=Path("runs") / "sweeps" / "manual_single" / "run_000_manual_single",
    )
    result = run_experiment(cfg)
    print("[run_single] Experiment completed (stub).")
    print(result)


if __name__ == "__main__":
    main()
