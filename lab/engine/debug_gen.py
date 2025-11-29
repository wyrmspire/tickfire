import sys
sys.path.append("c:\\tickfire")

from lab.engine.run_experiment import run_generation, ExperimentConfig
from pathlib import Path
import json

def debug_generation():
    # Point to one of the failed runs
    run_dir = Path("runs/sweeps/sweep_20251128_185654/model_3_h64_l1_w256")
    
    # Load config to reconstruct cfg object (or just mock it)
    # We need cfg.horizon, cfg.window, etc.
    # Let's just hardcode what we know for model_3
    cfg = ExperimentConfig(
        name="debug",
        out_dir=run_dir,
        data_family="bars_middle_2days",
        context_timeframes=["15m"],
        hidden_size=64,
        num_layers=1,
        window=256,
        horizon=1,
        indicator_set="basic"
    )
    
    # Construct paths
    # Note: dataset name depends on config. 
    # For model_3: window=256, horizon=1, base_tf=15s
    dataset_path = run_dir / "artifacts" / "ds_bars_15s_w256_h1.npz"
    model_path = run_dir / "model.pt"
    
    print(f"Debugging generation for {run_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    
    try:
        result = run_generation(cfg, dataset_path, model_path, run_dir)
        print("Generation Result:", result)
    except Exception as e:
        print("Generation Failed with error:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_generation()
