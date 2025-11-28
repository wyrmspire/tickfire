"""
Run a small sweep of bars+indicator experiments using the lab engine.

Usage (from repo root):

    python scripts/run_sweep.py
"""

from pathlib import Path
import sys

# Ensure project root (which contains the `lab` package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main() -> None:
    sweep_id = "dev_sweep_stub"
    base_dir = PROJECT_ROOT / "runs" / "sweeps" / sweep_id

    configs = [
        ExperimentConfig(
            name=f"{sweep_id}_run_{i:03d}",
            sweep_id=sweep_id,
            experiment_kind="bars_multi_tf",
            data_family=data_family,
            base_timeframe=base_tf,
            context_timeframes=context,
            indicator_set=indicator,
            window=window,
            horizon=horizon,
            out_dir=base_dir
            / f"run_{i:03d}_{data_family}_{base_tf}_ind{indicator}_w{window}_h{horizon}",
        )
        for i, (data_family, base_tf, context, indicator, window, horizon) in enumerate(
            [
                ("bars_apr07_apr25", "15s", ["5m", "15m"], "full_v1", 256, 1),
                ("bars_apr07_apr25", "15s", ["5m", "15m"], "basic", 128, 1),
                ("bars_apr07_apr25", "1m", ["5m", "15m", "1h"], "full_v1", 256, 4),
            ]
        )
    ]

    for cfg in configs:
        print(f"[run_sweep] Running: {cfg.name}")
        result = run_experiment(cfg)
        print("  ->", result)


if __name__ == "__main__":
    main()
