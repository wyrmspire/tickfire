
import sys
from pathlib import Path
from datetime import datetime

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "sweeps" / "long_run" / f"run_{ts}"

    # 1 month of 15s candles:
    # 30 days * 24 hours * 60 mins * 4 candles/min = 172,800 steps
    # Let's generate 175,000 steps to be safe.
    
    cfg = ExperimentConfig(
        name=f"long_run_{ts}",
        sweep_id="long_run",
        out_dir=run_dir,
        # Use the larger dataset (Apr 07 - Apr 25 is ~3 weeks)
        data_family="bars_apr07_apr25",
        base_timeframe="15s",
        context_timeframes=["15m"],
        indicator_set="basic",
        window=256,
        horizon=1,
        hidden_size=128,       # Larger model
        num_layers=2,          # Deeper model
        batch_size=256,        # Larger batch for GPU
        learning_rate=1e-4,
        epochs=20,             # More training
        generation_steps=175000, # ~1 month of 15s candles
    )

    print(f"Starting long run: {cfg.name}")
    print(f"Output dir: {run_dir}")
    
    # We need to hack run_experiment.py to support custom generation length?
    # Currently it's hardcoded to 4 hours (960 steps) in run_generation.
    # We should probably make that configurable or patch it here.
    # But for now, let's just run it and see if we can patch it dynamically or if the user wants us to edit the code.
    # The user asked to "generate a month of data".
    # I will modify run_experiment.py to accept a generation length config or argument.
    
    info = run_experiment(cfg)
    print("[run_1month_gen] Finished:")
    print(info)

if __name__ == "__main__":
    main()
