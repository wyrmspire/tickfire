"""
Unified run executor for experiments.

experiment_kind = "bars_multi_tf":
- compile multi-timeframe dataset via bars_multi_tf.compile_dataset_bars_multi_tf
- train PriceGenGRU on next-step additive OHLCV+V *tick* deltas
- generate synthetic sequences by adding predicted tick deltas to last close/volume,
  then snapping back to 0.25-tick prices and integer volume.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from lab.engine.bars_multi_tf import compile_dataset_bars_multi_tf
from lab.engine.plot_results import plot_model_result
from lab.models.gru import PriceGenGRU


# MES-style tick size
TICK_SIZE = 0.25


def repair_ohlc(o: float, h: float, l: float, c: float):
    """Ensure high/low are consistent with open/close."""
    hi = max(h, o, c, l)
    lo = min(l, o, c, h)
    return o, hi, lo, c


def round_to_tick(x: float, tick: float = TICK_SIZE) -> float:
    """Snap a price to the nearest tick."""
    return float(np.round(x / tick) * tick)


def round_volume(v: float) -> int:
    """Non-negative integer volume."""
    v = max(0.0, v)
    return int(np.round(v))


@dataclass
class ExperimentConfig:
    name: str
    out_dir: Path

    sweep_id: str = ""

    # which pipeline to use
    experiment_kind: str = "bars_multi_tf"

    # data / TF / indicators
    data_family: str = "bars_apr07_apr25"
    base_timeframe: str = "15s"
    context_timeframes: List[str] | None = None
    indicator_set: str = "basic"
    window: int = 256
    horizon: int = 1

    # model knobs
    hidden_size: int = 64
    num_layers: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 10


def _config_to_jsonable(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Convert dataclass to dict with Paths turned into strings."""
    d = asdict(cfg)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    if d.get("context_timeframes") is None:
        d["context_timeframes"] = []
    return d


class PriceSeqDataset(Dataset):
    """
    Wraps the NPZ dataset with X, Y arrays into a PyTorch Dataset.

    X: [N, seq_len, feat_dim]
    Y: [N, horizon, 5] additive deltas in *price space*:

       0: O_next - C_curr
       1: H_next - C_curr
       2: L_next - C_curr
       3: C_next - C_curr
       4: V_next - V_curr

    For training, we convert price deltas -> tick deltas by dividing by TICK_SIZE.
    The model predicts *normalized tick deltas*; at generation time we:
    - denormalize to tick deltas
    - convert to price deltas via tick_size
    - snap to the nearest tick + integer volume
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
        self.feature_cols = data["feature_cols"] if "feature_cols" in data else []
        self.target_cols = data["target_cols"] if "target_cols" in data else []

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        count += X.size(0)

    return total_loss / max(count, 1)


def eval_one_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            total_loss += loss.item() * X.size(0)
            count += X.size(0)

    return total_loss / max(count, 1)


def run_training(cfg: ExperimentConfig, dataset_path: Path, run_dir: Path) -> Dict[str, Any]:
    """
    Train the model using the compiled dataset.

    Important: we convert price deltas -> tick deltas for Y before normalization.
    """
    if not dataset_path.exists():
        return {"status": "failed", "error": f"Dataset not found: {dataset_path}"}

    full_dataset = PriceSeqDataset(dataset_path)

    # Handle NaNs in data (indicators etc.)
    full_dataset.X = np.nan_to_num(full_dataset.X, nan=0.0, posinf=0.0, neginf=0.0)
    full_dataset.Y = np.nan_to_num(full_dataset.Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure Y is 3D [N, horizon, 5]
    if full_dataset.Y.ndim == 2:
        full_dataset.Y = full_dataset.Y[:, None, :]

    N = full_dataset.X.shape[0]

    # Interpret Y as price-space additive deltas
    price_deltas = full_dataset.Y[..., 0:4]  # O/H/L/C deltas in price units
    vol_deltas = full_dataset.Y[..., 4]      # V deltas in volume units

    # Convert price deltas to tick deltas
    tick_deltas = price_deltas / TICK_SIZE

    # Compute scales in tick + volume space
    tick_scale = np.percentile(np.abs(tick_deltas), 95)
    if not np.isfinite(tick_scale) or tick_scale < 0.5:
        tick_scale = 1.0

    vol_scale = np.percentile(np.abs(vol_deltas), 95)
    if not np.isfinite(vol_scale) or vol_scale < 1e-6:
        vol_scale = 1.0

    # Normalize and clamp to a reasonable range (e.g. [-3, 3])
    full_dataset.Y[..., 0:4] = np.clip(tick_deltas / tick_scale, -3.0, 3.0)
    full_dataset.Y[..., 4] = np.clip(vol_deltas / vol_scale, -3.0, 3.0)

    # Normalize inputs (simple z-score)
    x_mean = full_dataset.X.mean(axis=(0, 1))
    x_std = full_dataset.X.std(axis=(0, 1))
    x_std[x_std < 1e-5] = 1.0
    full_dataset.X = (full_dataset.X - x_mean) / x_std

    # Train/val split
    val_frac = 0.1
    val_size = max(1, int(N * val_frac))
    train_size = N - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_training] Using device: {device}")

    model = PriceGenGRU(
        input_dim=full_dataset.X.shape[2],
        hidden_dim=cfg.hidden_size,
        num_layers=cfg.num_layers,
        horizon=cfg.horizon,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_path = run_dir / "model.pt"
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if np.isnan(val_loss):
            print(f"[run_training] Warning: val_loss is NaN at epoch {epoch}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"[run_training] → New best model (val_loss={val_loss:.4f})")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(full_dataset.X.shape[2]),
                    "hidden_dim": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "horizon": cfg.horizon,
                    "feature_cols": getattr(full_dataset, "feature_cols", []),
                    "target_cols": getattr(full_dataset, "target_cols", []),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "tick_scale": float(tick_scale),
                    "tick_size": float(TICK_SIZE),
                    "vol_scale": float(vol_scale),
                },
                best_path,
            )

    return {
        "status": "success",
        "best_val_loss": float(best_val_loss),
        "best_model_path": str(best_path),
        "history": history,
        "device": str(device),
        "input_dim": int(full_dataset.X.shape[2]),
        "tick_scale": float(tick_scale),
        "tick_size": float(TICK_SIZE),
        "vol_scale": float(vol_scale),
    }


def run_generation(cfg: ExperimentConfig, dataset_path: Path, model_path: Path, run_dir: Path) -> Dict[str, Any]:
    """
    Generate synthetic sequences using the trained model.

    The model predicts 5 normalized components per step:

        normalized_targets ≈ [
            tick_delta_O / tick_scale,
            tick_delta_H / tick_scale,
            tick_delta_L / tick_scale,
            tick_delta_C / tick_scale,
            volume_delta   / vol_scale,
        ]

    where tick_delta_* are in *ticks* (price_delta / tick_size).

    We denormalize to tick deltas, convert back to price deltas with tick_size,
    add to last close/volume, then snap prices to the tick grid and volume to ints.
    """
    if not model_path.exists():
        return {"status": "failed", "error": f"Model not found: {model_path}"}
    if not dataset_path.exists():
        return {"status": "failed", "error": f"Dataset not found for generation: {dataset_path}"}

    # Load dataset arrays (raw, unnormalized)
    data = np.load(dataset_path)
    X_all = data["X"].astype(np.float32)
    feature_cols = data["feature_cols"] if "feature_cols" in data else None

    if X_all.shape[0] == 0:
        return {"status": "failed", "error": "Empty dataset X for generation."}

    # Seed window is the last window in the dataset
    seed_window_raw = X_all[-1]
    current_window_raw = seed_window_raw.copy()
    seq_len, feat_dim = current_window_raw.shape

    # Feature indices for OHLCV
    open_idx = 0
    high_idx = 1
    low_idx = 2
    close_idx = 3
    vol_idx = 4

    if feature_cols is not None and len(feature_cols) > 0:
        feat_list = [str(f) for f in feature_cols.tolist()]
        idx_map = {name: i for i, name in enumerate(feat_list)}
        open_idx = idx_map.get("open", open_idx)
        high_idx = idx_map.get("high", high_idx)
        low_idx = idx_map.get("low", low_idx)
        close_idx = idx_map.get("close", close_idx)
        vol_idx = idx_map.get("volume", vol_idx)

    # Model + normalization stats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_generation] Using device: {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = PriceGenGRU(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        horizon=int(ckpt["horizon"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean = ckpt.get("x_mean", None)
    x_std = ckpt.get("x_std", None)
    tick_scale = float(ckpt.get("tick_scale", 1.0))
    tick_size = float(ckpt.get("tick_size", TICK_SIZE))
    vol_scale = float(ckpt.get("vol_scale", 1.0))

    if not np.isfinite(tick_scale) or tick_scale < 0.5:
        tick_scale = 1.0
    if not np.isfinite(tick_size) or tick_size <= 0.0:
        tick_size = TICK_SIZE
    if not np.isfinite(vol_scale) or vol_scale < 1e-6:
        vol_scale = 1.0

    # Time handling: clean grid timestamps
    step_seconds = 15 if cfg.base_timeframe == "15s" else 60
    base_ts = pd.Timestamp.now().floor(f"{step_seconds}S")

    # Context rows for plotting (seed window)
    context_rows = []
    for i in range(seq_len):
        row = current_window_raw[i]
        ts = base_ts - pd.Timedelta(seconds=step_seconds * (seq_len - i))
        context_rows.append(
            {
                "ts": ts,
                "open": float(row[open_idx]),
                "high": float(row[high_idx]),
                "low": float(row[low_idx]),
                "close": float(row[close_idx]),
                "volume": float(row[vol_idx]),
                "is_generated": 0,
            }
        )

    last_ts = base_ts
    generated_rows = []

    # 4 hours worth of steps
    if cfg.base_timeframe == "15s":
        num_steps = 4 * 60 * 4  # 960 steps
    else:
        num_steps = 4 * 60      # 240 steps for 1m

    for step in range(num_steps):
        # Normalize current window with saved stats
        if x_mean is not None and x_std is not None:
            curr_win_norm = (current_window_raw - x_mean) / x_std
        else:
            curr_win_norm = current_window_raw

        x_in = torch.from_numpy(curr_win_norm[None, :, :]).to(device)  # [1, seq_len, feat_dim]
        with torch.no_grad():
            pred = model(x_in)  # [1, horizon, 5]

        pred_vals = pred[0, 0].cpu().numpy()  # [5]

        last_close = float(current_window_raw[-1, close_idx])
        last_vol = float(current_window_raw[-1, vol_idx])

        if not np.isfinite(last_close):
            print("[run_generation] Aborting: invalid last_close.")
            break
        if not np.isfinite(last_vol):
            last_vol = 0.0

        # Denormalize to tick deltas
        delta_ticks_o = float(pred_vals[0]) * tick_scale
        delta_ticks_h = float(pred_vals[1]) * tick_scale
        delta_ticks_l = float(pred_vals[2]) * tick_scale
        delta_ticks_c = float(pred_vals[3]) * tick_scale
        delta_v = float(pred_vals[4]) * vol_scale

        # Convert tick deltas to price deltas
        delta_o = delta_ticks_o * tick_size
        delta_h = delta_ticks_h * tick_size
        delta_l = delta_ticks_l * tick_size
        delta_c = delta_ticks_c * tick_size

        # Compute next OHLCV in price space
        next_o = last_close + delta_o
        next_h = last_close + delta_h
        next_l = last_close + delta_l
        next_c = last_close + delta_c
        next_v = last_vol + delta_v

        vals = [next_o, next_h, next_l, next_c, next_v]
        if not all(np.isfinite(vals)):
            print(f"[run_generation] Aborting: non-finite next values at step {step}.")
            break

        # Reject absurd close magnitudes
        if abs(next_c) > 1e6:
            print(f"[run_generation] Aborting: close magnitude too large at step {step}: {next_c}")
            break

        # Snap to tick grid and integer volume
        next_o = round_to_tick(next_o, tick_size)
        next_h = round_to_tick(next_h, tick_size)
        next_l = round_to_tick(next_l, tick_size)
        next_c = round_to_tick(next_c, tick_size)
        next_v = round_volume(next_v)

        # Repair OHLC consistency AFTER snapping
        next_o, next_h, next_l, next_c = repair_ohlc(next_o, next_h, next_l, next_c)

        # Build next feature row by copying last and overriding OHLCV
        next_full_row = np.array(current_window_raw[-1], copy=True)
        next_full_row[open_idx] = next_o
        next_full_row[high_idx] = next_h
        next_full_row[low_idx] = next_l
        next_full_row[close_idx] = next_c
        next_full_row[vol_idx] = float(next_v)

        # Slide window
        current_window_raw = np.vstack([current_window_raw[1:], next_full_row])

        last_ts = last_ts + pd.Timedelta(seconds=step_seconds)
        generated_rows.append(
            {
                "ts": last_ts,
                "open": float(next_o),
                "high": float(next_h),
                "low": float(next_l),
                "close": float(next_c),
                "volume": float(next_v),
                "is_generated": 1,
            }
        )

    full_df = pd.DataFrame(context_rows + generated_rows)
    out_csv = run_dir / "generated_sequence.csv"
    full_df.to_csv(out_csv, index=False)

    return {
        "status": "success",
        "generated_count": len(generated_rows),
        "csv_path": str(out_csv),
        "aborted_early": len(generated_rows) < num_steps,
    }


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single experiment.
    """
    run_dir = cfg.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_json = _config_to_jsonable(cfg)
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg_json, f, indent=2)

    phases: Dict[str, float] = {
        "compile_dataset": 0.0,
        "train": 0.0,
        "generate": 0.0,
        "metrics": 0.0,
    }

    dataset_info: Dict[str, Any] = {}
    train_info: Dict[str, Any] = {"status": "skipped"}
    gen_info: Dict[str, Any] = {"status": "skipped"}
    metrics_info: Dict[str, Any] = {"status": "stub"}

    # PHASE 1 — Dataset compilation
    t0 = time.time()
    if cfg.experiment_kind == "bars_multi_tf":
        try:
            dataset_info = compile_dataset_bars_multi_tf(cfg, run_dir)
        except Exception as e:
            dataset_info = {"status": "failed", "error": str(e)}
    else:
        dataset_info = {"note": f"experiment_kind={cfg.experiment_kind!r} not implemented"}
    phases["compile_dataset"] = time.time() - t0

    # PHASE 2 — Train
    t1 = time.time()
    dataset_path = dataset_info.get("dataset_path")
    if dataset_path:
        try:
            train_info = run_training(cfg, Path(dataset_path), run_dir)
        except Exception as e:
            train_info = {"status": "failed", "error": str(e)}
    phases["train"] = time.time() - t1

    # PHASE 3 — Generate
    t2 = time.time()
    model_path = train_info.get("best_model_path")
    if model_path and dataset_path:
        try:
            gen_info = run_generation(cfg, Path(dataset_path), Path(model_path), run_dir)
        except Exception as e:
            gen_info = {"status": "failed", "error": str(e)}
    phases["generate"] = time.time() - t2

    # PHASE 3.5 — Plot chart
    if gen_info.get("status") == "success":
        try:
            plot_model_result(run_dir)
        except Exception as e:
            print(f"[run_experiment] plot_model_result failed: {e}")

    # PHASE 4 — Metrics (stub)
    t3 = time.time()
    phases["metrics"] = time.time() - t3

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "metrics.json").open("w") as f:
        json.dump(metrics_info, f, indent=2)

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
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parents[2]
    test_run = root / "runs" / "sweeps" / "manual_test" / f"run_{ts}"

    cfg = ExperimentConfig(
        name=f"manual_test_{ts}",
        sweep_id="manual_test",
        out_dir=test_run,
        epochs=5,
        hidden_size=64,
        num_layers=1,
        batch_size=128,
    )
    info = run_experiment(cfg)
    print("[run_experiment.__main__] Finished:", info)
