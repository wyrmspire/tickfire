#!/usr/bin/env bash
set -e

echo "[tickfire-lab] Fixing run_experiment + run_single + run_sweep..."

# -------------------------------------------------------------------
# 1) Rewrite lab/engine/run_experiment.py with Path-safe JSON config
# -------------------------------------------------------------------
cat > lab/engine/run_experiment.py << 'EOF_RUNEXP'
"""
Unified run executor for experiments.

Currently supports:
- experiment_kind = "bars_multi_tf"
  -> dataset compilation from prebuilt bars + indicators via bars_multi_tf.compile_dataset_bars_multi_tf

This version:
- Converts Path fields in ExperimentConfig to strings before json.dump
- Ensures context_timeframes is always a list (not None)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json
import time

from lab.engine.bars_multi_tf import compile_dataset_bars_multi_tf


@dataclass
class ExperimentConfig:
    name: str
    out_dir: Path

    sweep_id: str = ""

    # which pipeline to use
    experiment_kind: str = "bars_multi_tf"   # future: add others

    # data / TF / indicators
    data_family: str = "bars_apr07_apr25"
    base_timeframe: str = "15s"
    context_timeframes: List[str] | None = None
    indicator_set: str = "basic"
    window: int = 256
    horizon: int = 1

    # model knobs — placeholders for future training
    hidden_size: int = 64
    num_layers: int = 1


def _config_to_jsonable(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Convert dataclass to dict with Paths turned into strings."""
    d = asdict(cfg)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    # ensure context_timeframes is always a list
    if d.get("context_timeframes") is None:
        d["context_timeframes"] = []
    return d


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single experiment.

    For experiment_kind == "bars_multi_tf":

    - compile_dataset_bars_multi_tf(cfg, run_dir)
    - (training / generation / metrics are stubs for now)

    Returns a dict summarizing status, phases, and artifacts.
    """
    run_dir = cfg.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config (JSON-friendly)
    cfg_json = _config_to_jsonable(cfg)
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg_json, f, indent=2)

    phases: Dict[str, float] = {
        "compile_dataset": 0.0,
        "train": 0.0,
        "generate": 0.0,
        "metrics": 0.0,
    }

    status: Dict[str, Any] = {}
    dataset_info: Dict[str, Any] = {}
    train_info: Dict[str, Any] = {"status": "stub"}
    gen_info: Dict[str, Any] = {"status": "stub"}
    metrics_info: Dict[str, Any] = {"status": "stub"}

    # --------------------------------------------------------
    # PHASE 1 — Dataset compilation
    # --------------------------------------------------------
    t0 = time.time()
    if cfg.experiment_kind == "bars_multi_tf":
        dataset_info = compile_dataset_bars_multi_tf(cfg, run_dir)
    else:
        dataset_info = {
            "note": f"experiment_kind={cfg.experiment_kind!r} not implemented",
        }
    phases["compile_dataset"] = time.time() - t0

    # --------------------------------------------------------
    # PHASE 2 — Train (stub)
    # --------------------------------------------------------
    t1 = time.time()
    # Training implementation will go here later.
    phases["train"] = time.time() - t1

    # --------------------------------------------------------
    # PHASE 3 — Generate (stub)
    # --------------------------------------------------------
    t2 = time.time()
    # Generation implementation will go here later.
    phases["generate"] = time.time() - t2

    # --------------------------------------------------------
    # PHASE 4 — Metrics (stub)
    # --------------------------------------------------------
    t3 = time.time()
    # Metrics implementation will go here later.
    phases["metrics"] = time.time() - t3

    # Save metrics
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "metrics.json").open("w") as f:
        json.dump(metrics_info, f, indent=2)

    # Save status
    status = {
        "name": cfg.name,
        "sweep_id": cfg.sweep_id,
        "experiment_kind": cfg.experiment_kind,
        "data_family": cfg.data_family,
        "base_timeframe": cfg.base_timeframe,
        "context_timeframes": cfg_json.get("context_timeframes", []),
        "indicator_set": cfg.indicator_set,
        "window": cfg.window,
        "horizon": cfg.horizon,
        "phases": phases,
        "dataset": dataset_info,
        "training": train_info,
        "generation": gen_info,
        "metrics": metrics_info,
    }

    with (run_dir / "status.json").open("w") as f:
        json.dump(status, f, indent=2)

    return status


if __name__ == "__main__":
    # Simple manual smoke test (not used in normal flow)
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parents[2]
    test_run = root / "runs" / "sweeps" / "manual_test" / f"run_{ts}"

    cfg = ExperimentConfig(
        name=f"manual_test_{ts}",
        sweep_id="manual_test",
        out_dir=test_run,
    )
    info = run_experiment(cfg)
    print("[run_experiment.__main__] Finished:", info)
EOF_RUNEXP

# -------------------------------------------------------------------
# 2) Rewrite scripts/run_single.py cleanly (with sys.path fix)
# -------------------------------------------------------------------
cat > scripts/run_single.py << 'EOF_SINGLE'
"""
Run a single bars+indicators experiment via the lab engine.

Usage (from repo root):

    python scripts/run_single.py
"""

from pathlib import Path
from datetime import datetime
import sys

# Ensure project root (which contains the `lab` package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab.engine.run_experiment import ExperimentConfig, run_experiment


def main() -> None:
    sweep_id = "manual_single"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / "sweeps" / sweep_id / f"run_000_manual_single_{ts}"

    cfg = ExperimentConfig(
        name=f"manual_single_{ts}",
        sweep_id=sweep_id,
        experiment_kind="bars_multi_tf",
        data_family="bars_apr07_apr25",
        base_timeframe="15s",
        context_timeframes=["5m", "15m"],  # just to populate the field
        indicator_set="full_v1",
        window=256,
        horizon=1,
        out_dir=run_dir,
    )

    result = run_experiment(cfg)
    print("[run_single] Experiment completed.")
    print(result)


if __name__ == "__main__":
    main()
EOF_SINGLE

# -------------------------------------------------------------------
# 3) Rewrite scripts/run_sweep.py cleanly (with sys.path fix)
# -------------------------------------------------------------------
cat > scripts/run_sweep.py << 'EOF_SWEEP'
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
EOF_SWEEP

echo "[tickfire-lab] Fix complete. Try:"
echo "  python scripts/run_single.py"
echo "  python scripts/run_sweep.py"
