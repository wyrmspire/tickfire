"""
Sweep runner for model maker.

Iterates over a set of configurations and runs experiments.
"""

from pathlib import Path
from datetime import datetime
import json

from lab.engine.run_experiment import ExperimentConfig, run_experiment
from lab.engine.plot_results import plot_model_result


def run_sweep():
    # Define sweep parameters
    sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path("runs") / "sweeps" / sweep_id
    
    # 5 Specific Configurations
    configs = [
        {"hidden_size": 32, "num_layers": 1, "window": 128},
        {"hidden_size": 32, "num_layers": 2, "window": 128},
        {"hidden_size": 64, "num_layers": 1, "window": 256},
        {"hidden_size": 64, "num_layers": 2, "window": 256},
        {"hidden_size": 128, "num_layers": 2, "window": 256},
    ]
    
    print(f"Starting sweep {sweep_id} with {len(configs)} configurations.")
    
    results = []
    
    for i, params in enumerate(configs):
        exp_name = f"model_{i+1}_h{params['hidden_size']}_l{params['num_layers']}_w{params['window']}"
        print(f"Running {exp_name}...")
        
        cfg = ExperimentConfig(
            name=exp_name,
            sweep_id=sweep_id,
            out_dir=base_dir / exp_name,
            data_family="bars_middle_2days",
            context_timeframes=["15m"],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            window=params['window'],
            epochs=5, # Short run for demo
        )
        
        try:
            status = run_experiment(cfg)
            
            # Plot results
            print(f"Generating chart for {exp_name}...")
            plot_model_result(cfg.out_dir)
            
            # Extract key metrics
            best_loss = status.get("training", {}).get("best_val_loss", float("inf"))
            
            results.append({
                "name": exp_name,
                "params": params,
                "best_val_loss": best_loss,
                "status": "success",
                "chart_path": str(cfg.out_dir / "chart.png")
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
    print(f"{'Name':<25} | {'Loss':<10} | {'Chart'}")
    print("-" * 65)
    for r in results:
        if r["status"] == "success":
            print(f"{r['name']:<25} | {r['best_val_loss']:.4f} | {r.get('chart_path', 'N/A')}")
        else:
            print(f"{r['name']:<25} | FAILED")


if __name__ == "__main__":
    run_sweep()
