import sys
from pathlib import Path
from datetime import datetime

# Ensure repo root is on sys.path (so 'lab' package is importable)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "sweeps" / "manual_test" / f"run_{ts}"

    cfg = ExperimentConfig(
        name=f"manual_test_{ts}",
        sweep_id="manual_test",
        out_dir=run_dir,
        data_family="bars_middle_2days",   # focused 2-day slice
        base_timeframe="15s",
        context_timeframes=["15m"],
        indicator_set="basic",
        window=256,
        horizon=1,
        hidden_size=64,        # keep it modest for fast probes
        num_layers=1,
        batch_size=64,
        learning_rate=1e-4,    # simple, not too aggressive
        epochs=5,              # SMALL wiggle passes
    )

    info = run_experiment(cfg)
    print("[run_manual_experiment] Finished:")
    print(info)


if __name__ == "__main__":
    main()
