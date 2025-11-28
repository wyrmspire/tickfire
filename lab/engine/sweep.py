"""
Sweep runner for model maker.

Iterates over a set of configurations and runs experiments.
"""

from pathlib import Path
from datetime import datetime
import itertools
import json

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def run_sweep():
    # Define sweep parameters
    sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path("runs") / "sweeps" / sweep_id
    
    # Hyperparameters to sweep
    hidden_sizes = [32, 64]
    num_layers_list = [1, 2]
    windows = [128, 256]
    
    # Generate all combinations
    combinations = list(itertools.product(hidden_sizes, num_layers_list, windows))
    
    print(f"Starting sweep {sweep_id} with {len(combinations)} configurations.")
    
    results = []
    
    for i, (hidden_size, num_layers, window) in enumerate(combinations):
        exp_name = f"exp_{i:03d}_h{hidden_size}_l{num_layers}_w{window}"
        print(f"Running {exp_name}...")
        
        cfg = ExperimentConfig(
            name=exp_name,
            sweep_id=sweep_id,
            out_dir=base_dir / exp_name,
            hidden_size=hidden_size,
            num_layers=num_layers,
            window=window,
            epochs=5, # Keep it short for demo
        )
        
        try:
            status = run_experiment(cfg)
            
            # Extract key metrics
            best_loss = status.get("training", {}).get("best_val_loss", float("inf"))
            
            results.append({
                "name": exp_name,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "window": window,
                "best_val_loss": best_loss,
                "status": "success"
            })
            
        except Exception as e:
            print(f"Experiment {exp_name} failed: {e}")
            results.append({
                "name": exp_name,
                "status": "failed",
                "error": str(e)
            })
            
    # Save sweep summary
    summary_path = base_dir / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Sweep complete. Summary saved to {summary_path}")
    
    # Print simple table
    print("\nSweep Results:")
    print(f"{'Name':<25} | {'Hidden':<6} | {'Layers':<6} | {'Window':<6} | {'Loss':<10}")
    print("-" * 65)
    for r in results:
        if r["status"] == "success":
            print(f"{r['name']:<25} | {r['hidden_size']:<6} | {r['num_layers']:<6} | {r['window']:<6} | {r['best_val_loss']:.4f}")
        else:
            print(f"{r['name']:<25} | FAILED")


if __name__ == "__main__":
    run_sweep()
