"""
Experiment runner for the tickfire lab.

This version aligns with the new understanding:

- Experiments are *combinations of knobs*:
  - which data family to use (raw ticks vs prebuilt bars+indicators)
  - which base timeframe and context timeframes
  - which indicator/feature set
  - which training pipeline
  - which generation pipeline

- `experiment_kind` selects a concrete pipeline implementation.

Currently implemented kinds:

- "phase0_raw_ml"
    Uses the existing ml/ pipeline (build_dataset_v2, train_gru, generate_gru)
    starting from raw tick data. This is kept as a legacy / reference path.

Planned / stubbed kinds:

- "bars_multi_tf"
    Will use the prebuilt multi-timeframe bars and indicator files under:
      out/bars/YYYYMMDD/*.csv
      out/bars_indicators/YYYYMMDD/*_ind.csv
    to build datasets and train generators without touching raw ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List

import json
import os
import sys
import time
import traceback

from lab.utils.paths import PROJECT_ROOT


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment run.

    The key idea is that this describes *what kind of pipeline* to run
    and *how* it should be parameterized.

    Knobs:

    - experiment_kind:
        "phase0_raw_ml"   -> use the existing ml/ raw pipeline
        "bars_multi_tf"   -> (future) multi-timeframe bars+indicators pipeline

    - data_family:
        logical grouping of data / date ranges
        e.g. "raw_mdp3", "bars_apr07_apr25"

    - base_timeframe:
        the primary timeframe we model at (e.g. "15s", "1m", "5m").

    - context_timeframes:
        extra timeframes we may pull features from (e.g. 5m, 15m, 1h, 4h).

    - indicator_set:
        which indicator/feature palette to use:
          "none", "basic", "full_v1", etc.

    The existing ml/ pipeline still hardcodes many of its own choices,
    but this config structure reflects where we're going.
    """
    name: str
    sweep_id: str

    # Which pipeline to use
    experiment_kind: str = "bars_multi_tf"  # default future direction

    # Data / TF / indicator knobs
    data_family: str = "bars_apr07_apr25"
    base_timeframe: str = "15s"
    context_timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h"])
    indicator_set: str = "full_v1"

    # Model / training knobs (high-level only for now)
    window: int = 256
    horizon: int = 1
    description: str = (
        "Generic experiment config for tickfire lab. "
        "Specific behavior depends on experiment_kind."
    )

    # Output
    out_dir: Path = PROJECT_ROOT / "runs" / "sweeps" / "UNSPECIFIED" / "run_000"


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


# -------------------------------------------------------------------
# Implementation: experiment kinds
# -------------------------------------------------------------------


def _run_phase0_raw_ml_pipeline(run_dir: Path) -> Dict[str, Any]:
    """
    Run the existing ml/ pipeline as a "phase0_raw_ml" experiment.

    This is the legacy path that starts from raw tick data via:

        ml/build_dataset_v2.py
        ml/train_gru.py
        ml/generate_gru.py

    We keep it so we can compare against future bars+indicators pipelines.
    """
    phases: Dict[str, float] = {}
    errors: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}

    orig_cwd = Path.cwd()
    ml_dir = PROJECT_ROOT / "ml"

    if not ml_dir.exists():
        raise FileNotFoundError(f"ml directory not found at: {ml_dir}")

    try:
        os.chdir(ml_dir)
        if str(ml_dir) not in sys.path:
            sys.path.insert(0, str(ml_dir))

        # STEP 1: build_dataset_v2 (if present)
        t0 = time.time()
        try:
            import build_dataset_v2  # type: ignore

            if hasattr(build_dataset_v2, "main"):
                print("[engine.phase0_raw_ml] STEP 1: build_dataset_v2.main()")
                build_dataset_v2.main()
            else:
                print("[engine.phase0_raw_ml] STEP 1: build_dataset_v2 has no main(), skipping.")
        except Exception as e:  # noqa: BLE001
            errors["build_dataset_v2"] = repr(e)
            errors["build_dataset_v2_traceback"] = traceback.format_exc()
            print("[engine.phase0_raw_ml] WARNING: build_dataset_v2 failed:", e)
        phases["build_dataset_v2"] = time.time() - t0

        # STEP 2: train_gru
        t1 = time.time()
        try:
            import train_gru  # type: ignore

            if hasattr(train_gru, "main"):
                print("[engine.phase0_raw_ml] STEP 2: train_gru.main()")
                train_gru.main()
            else:
                print("[engine.phase0_raw_ml] STEP 2: train_gru has no main(), skipping.")
        except Exception as e:  # noqa: BLE001
            errors["train_gru"] = repr(e)
            errors["train_gru_traceback"] = traceback.format_exc()
            print("[engine.phase0_raw_ml] ERROR: train_gru failed:", e)
        phases["train_gru"] = time.time() - t1

        # STEP 3: generate_gru
        t2 = time.time()
        try:
            import generate_gru  # type: ignore

            if hasattr(generate_gru, "main"):
                print("[engine.phase0_raw_ml] STEP 3: generate_gru.main()")
                generate_gru.main()
            else:
                print("[engine.phase0_raw_ml] STEP 3: generate_gru has no main(), skipping.")
        except Exception as e:  # noqa: BLE001
            errors["generate_gru"] = repr(e)
            errors["generate_gru_traceback"] = traceback.format_exc()
            print("[engine.phase0_raw_ml] ERROR: generate_gru failed:", e)
        phases["generate_gru"] = time.time() - t2

        # Locate artifacts in ml/out
        out_dir = ml_dir / "out"
        if out_dir.exists():
            def newest(pattern: str) -> Path | None:
                files = list(out_dir.glob(pattern))
                if not files:
                    return None
                return max(files, key=lambda p: p.stat().st_mtime)

            ds_file = newest("ds_*.npz")
            model_file = newest("gru*.pt")
            gen_file = newest("gen_15s_*.csv")

            artifacts["dataset"] = str(ds_file) if ds_file else None
            artifacts["model"] = str(model_file) if model_file else None
            artifacts["generated_15s"] = str(gen_file) if gen_file else None

            # Copy into run_dir/artifacts
            art_dir = run_dir / "artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)

            import shutil

            for key, src in [
                ("dataset", ds_file),
                ("model", model_file),
                ("generated_15s", gen_file),
            ]:
                if src is not None and src.exists():
                    dest = art_dir / src.name
                    try:
                        shutil.copy2(src, dest)
                        artifacts[f"{key}_copied"] = str(dest)
                    except Exception as e:  # noqa: BLE001
                        errors[f"copy_{key}"] = repr(e)
                        errors[f"copy_{key}_traceback"] = traceback.format_exc()
        else:
            print(f"[engine.phase0_raw_ml] No ml/out directory found at {out_dir}")

    finally:
        os.chdir(orig_cwd)

    return {
        "phases": phases,
        "errors": errors,
        "artifacts": artifacts,
    }


def _run_bars_multi_tf_stub(run_dir: Path, cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Stub for the future "bars_multi_tf" pipeline.

    This is where we'll:

    - Read bars & indicators from:
        out/bars/YYYYMMDD/*.csv
        out/bars_indicators/YYYYMMDD/*_ind.csv

    - Build datasets using the chosen:
        - base_timeframe (cfg.base_timeframe)
        - context_timeframes (cfg.context_timeframes)
        - indicator_set (cfg.indicator_set)
        - window / horizon

    - Train a generator model on those datasets.
    - Generate 4h of synthetic 15s (or other TF) bars.
    - Compare synthetic vs real and write full metrics.

    For now, this just records that it ran as a stub.
    """
    phases = {
        "compile_dataset": 0.0,
        "train": 0.0,
        "generate": 0.0,
        "compare": 0.0,
    }
    errors: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {
        "note": "bars_multi_tf is not implemented yet; stub only.",
        "data_family": cfg.data_family,
        "base_timeframe": cfg.base_timeframe,
        "context_timeframes": cfg.context_timeframes,
        "indicator_set": cfg.indicator_set,
    }

    # We still create an eval/ dir so metrics.json can live there.
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stub_metrics = {
        "note": "metrics not implemented yet for bars_multi_tf; stub run only."
    }
    _save_json(eval_dir / "metrics.json", stub_metrics)

    return {
        "phases": phases,
        "errors": errors,
        "artifacts": artifacts,
    }


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Top-level entrypoint for running an experiment.

    Behavior is selected via cfg.experiment_kind:

    - "phase0_raw_ml": use existing ml/ raw pipeline (ticks -> 15s -> GRU).
    - "bars_multi_tf": stub for now, will use prebuilt bars+indicators.

    The runner:
    - Ensures the run directory exists.
    - Writes config.json.
    - Dispatches to the appropriate implementation.
    - Writes status.json with timings, artifacts, and any errors.
    - Ensures eval/metrics.json exists (even if stub).
    """
    run_dir = Path(cfg.out_dir)
    eval_dir = run_dir / "eval"
    logs_dir = run_dir / "logs"

    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save config (with Paths turned into strings)
    cfg_dict = asdict(cfg)
    if isinstance(cfg_dict.get("out_dir"), Path):
        cfg_dict["out_dir"] = str(cfg.out_dir)

    config_path = run_dir / "config.json"
    _save_json(config_path, cfg_dict)

    status_path = run_dir / "status.json"
    metrics_path = eval_dir / "metrics.json"

    t0 = time.time()
    state = "ok"
    result: Dict[str, Any] = {}

    try:
        if cfg.experiment_kind == "phase0_raw_ml":
            result = _run_phase0_raw_ml_pipeline(run_dir)
        elif cfg.experiment_kind == "bars_multi_tf":
            result = _run_bars_multi_tf_stub(run_dir, cfg)
        else:
            raise ValueError(f"Unknown experiment_kind: {cfg.experiment_kind!r}")

        if result.get("errors"):
            # If there were any recorded errors, treat as partial_failure
            state = "partial_failure"
    except Exception as e:  # noqa: BLE001
        state = "failed"
        result["exception"] = repr(e)
        result["traceback"] = traceback.format_exc()
        print("[engine] FATAL error in run_experiment:", e)

    total_seconds = time.time() - t0

    status = {
        "name": cfg.name,
        "sweep_id": cfg.sweep_id,
        "experiment_kind": cfg.experiment_kind,
        "data_family": cfg.data_family,
        "base_timeframe": cfg.base_timeframe,
        "context_timeframes": cfg.context_timeframes,
        "indicator_set": cfg.indicator_set,
        "state": state,
        "total_seconds": total_seconds,
        **result,
    }
    _save_json(status_path, status)

    # Ensure metrics.json exists for any kind
    if not metrics_path.exists():
        _save_json(metrics_path, {"note": "metrics not implemented yet"})

    return {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "status_path": str(status_path),
        "metrics_path": str(metrics_path),
    }


if __name__ == "__main__":
    # Simple manual smoke test.
    default_run_dir = PROJECT_ROOT / "runs" / "sweeps" / "manual_test" / "run_manual"
    cfg = ExperimentConfig(
        name="manual_test_raw_phase0",
        sweep_id="manual_test",
        experiment_kind="phase0_raw_ml",
        data_family="raw_mdp3",
        out_dir=default_run_dir,
    )
    info = run_experiment(cfg)
    print("[engine.__main__] Experiment finished:")
    for k, v in info.items():
        print(" ", k, "=", v)
