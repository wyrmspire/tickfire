from pathlib import Path
import sys
import json
from datetime import datetime

# Ensure project root (/c/tickfire) is on sys.path so "lab" is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_id = "manual_test"

    run_dir = ROOT / "runs" / "sweeps" / sweep_id / f"run_{ts}"

    cfg = ExperimentConfig(
        name=f"{sweep_id}_{ts}",
        sweep_id=sweep_id,
        out_dir=run_dir,
        experiment_kind="bars_multi_tf",
        data_family="bars_apr07_apr25",
        base_timeframe="15s",
        context_timeframes=["15m"],
        indicator_set="full_v1",
        window=256,
        horizon=1,
        hidden_size=64,
        num_layers=1,
        batch_size=128,
        learning_rate=1e-4,
        epochs=10,
    )

    print(f"[run_manual_experiment] Running into: {run_dir}")
    status = run_experiment(cfg)

    summary = {
        "run_dir": str(run_dir),
        "dataset_status": status.get("dataset", {}).get("status"),
        "training_status": status.get("training", {}).get("status"),
        "generation_status": status.get("generation", {}).get("status"),
        "generated_csv": status.get("generation", {}).get("csv_path"),
        "best_model_path": status.get("training", {}).get("best_model_path"),
    }

    print("[run_manual_experiment] Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
