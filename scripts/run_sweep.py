"""
Run a sweep of experiments (stub).

Later:
- define a small search space over configs
- loop over them
- call run_experiment for each
- respect time budgets, allow resume/skip completed runs
"""

from pathlib import Path
from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main():
    sweep_id = "dev_sweep_stub"
    base_dir = Path("runs") / "sweeps" / sweep_id

    configs = [
        ExperimentConfig(
            name=f"{sweep_id}_run_{i:03d}",
            hidden_size=hs,
            num_layers=nl,
            out_dir=base_dir / f"run_{i:03d}_hs{hs}_L{nl}",
        )
        for i, (hs, nl) in enumerate(
            [
                (64, 1),
                (64, 2),
                (128, 1),
                (128, 2),
            ]
        )
    ]

    for cfg in configs:
        print(f"[run_sweep] Running: {cfg.name}")
        result = run_experiment(cfg)
        print("  ->", result)


if __name__ == "__main__":
    main()
