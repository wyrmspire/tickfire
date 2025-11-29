
from lab.engine.run_experiment import ExperimentConfig, run_experiment
from pathlib import Path
from datetime import datetime

def run_single():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = ExperimentConfig(
        name=f"verify_fix_{ts}",
        sweep_id="verify_fix",
        out_dir=Path("runs") / "verify_fix" / f"run_{ts}",
        data_family="bars_middle_2days",
        context_timeframes=["15m"],
        hidden_size=32,
        num_layers=1,
        window=128,
        epochs=3
    )
    status = run_experiment(cfg)
    print("Run Status:", status["training"]["status"])
    print("Best Loss:", status["training"]["best_val_loss"])

if __name__ == "__main__":
    run_single()
