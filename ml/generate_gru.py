from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from model_gru import PriceGenGRU


def compute_time_features(ts_utc: pd.Timestamp):
    """
    Given a UTC timestamp, compute:
    - minutes_since_midnight in America/Chicago
    - tod_sin, tod_cos
    - dow (0=Mon)
    - is_rth (approx 08:30–15:15 CT)
    """
    if ts_utc.tz is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_local = ts_utc.tz_convert("America/Chicago")

    minutes_since_midnight = (
        ts_local.hour * 60
        + ts_local.minute
        + ts_local.second / 60.0
    )

    day_minutes = 24 * 60
    angle = 2 * np.pi * minutes_since_midnight / day_minutes
    tod_sin = float(np.sin(angle))
    tod_cos = float(np.cos(angle))

    dow = ts_local.dayofweek

    hour = ts_local.hour
    minute = ts_local.minute
    is_after_open = (hour > 8) or (hour == 8 and minute >= 30)
    is_before_close = (hour < 15) or (hour == 15 and minute <= 15)
    is_rth = 1 if (is_after_open and is_before_close) else 0

    return minutes_since_midnight, tod_sin, tod_cos, dow, is_rth


def build_feature_index_map(feature_cols: np.ndarray) -> Dict[str, int]:
    """Map feature name -> index in feature vector."""
    return {name: i for i, name in enumerate(feature_cols.tolist())}


def repair_ohlc(o, h, l, c):
    """Ensure high/low are consistent with open/close."""
    hi = max(h, o, c, l)
    lo = min(l, o, c, h)
    return o, hi, lo, c


def generate_sequence(
    ds_path: Path,
    model_path: Path,
    num_steps: int = 240,  # 1 hour of 15s candles
    seed_index: int = -1,  # use last window by default
):
    """
    Generate a sequence of synthetic 15s OHLCV candles using the trained GRU.

    - Loads the dataset NPZ to get an initial window and feature definitions.
    - Loads the trained model.
    - Seeds from X[seed_index].
    - Each step:
        - Predict next OHLCV
        - Build a new feature row
        - Slide the window forward
    """
    # Load dataset
    data = np.load(ds_path)
    X_all = data["X"].astype(np.float32)         # [N, seq_len, feat_dim]
    Y_all = data["Y"].astype(np.float32)         # [N, horizon, 5] (not used here)
    indices = data["indices"]                    # [N,]
    feature_cols = data["feature_cols"]          # list of feature names
    target_cols = data["target_cols"]            # ['open','high','low','close','volume']

    N, seq_len, feat_dim = X_all.shape
    _, horizon, tgt_dim = Y_all.shape

    print(f"Loaded dataset: N={N}, seq_len={seq_len}, feat_dim={feat_dim}, horizon={horizon}, tgt_dim={tgt_dim}")

    if seed_index < 0:
        seed_index = N + seed_index  # -1 -> last element
    if not (0 <= seed_index < N):
        raise IndexError(f"seed_index {seed_index} out of range [0, {N})")

    # Load model checkpoint (weights_only=False because we saved extra metadata)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model = PriceGenGRU(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        horizon=int(ckpt["horizon"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build feature index map
    fidx = build_feature_index_map(feature_cols)

    # Identify required feature indexes
    idx_open = fidx["open"]
    idx_high = fidx["high"]
    idx_low = fidx["low"]
    idx_close = fidx["close"]
    idx_volume = fidx["volume"]
    idx_minutes = fidx["minutes_since_midnight"]
    idx_tod_sin = fidx["tod_sin"]
    idx_tod_cos = fidx["tod_cos"]
    idx_dow = fidx["dow"]
    idx_is_rth = fidx["is_rth"]
    idx_range = fidx["range"]
    idx_body = fidx["body"]
    idx_upper_wick = fidx["upper_wick"]
    idx_lower_wick = fidx["lower_wick"]
    idx_vol_mean_w = fidx["vol_mean_w"]
    idx_vol_std_w = fidx["vol_std_w"]
    idx_vol_z_w = fidx["vol_z_w"]

    # Seed window
    X_window = X_all[seed_index].copy()  # [seq_len, feat_dim]

    # Reconstruct the timestamp of the last candle in the seed window
    start_ts_np = indices[seed_index]  # numpy datetime64[ns]
    start_ts = pd.to_datetime(start_ts_np).tz_localize("UTC")
    last_ts = start_ts + pd.Timedelta(seconds=15 * (seq_len - 1))
    print(f"Seed start_ts={start_ts}, last_ts={last_ts}")

    # For rolling volume stats, maintain the last 60 volumes
    vol_hist = X_window[:, idx_volume].tolist()
    if len(vol_hist) >= 60:
        vol_hist = vol_hist[-60:]

    generated_rows = []

    for step in range(num_steps):
        # Model expects batch dimension
        x_batch = torch.from_numpy(X_window[None, :, :])  # [1, seq_len, feat_dim]
        with torch.no_grad():
            pred = model(x_batch)  # [1, horizon, 5]
        next_ohlcv = pred[0, 0].numpy()  # [5]

        o, h, l, c, v = map(float, next_ohlcv)
        if v < 0:
            v = 0.0

        o, h, l, c = repair_ohlc(o, h, l, c)

        next_ts = last_ts + pd.Timedelta(seconds=15)

        # Time features
        minutes_since_midnight, tod_sin, tod_cos, dow, is_rth = compute_time_features(next_ts)

        # Candle geometry
        candle_range = h - l
        body = c - o
        upper_body = max(o, c)
        lower_body = min(o, c)
        upper_wick = h - upper_body
        lower_wick = lower_body - l

        # Update rolling volume stats
        vol_hist.append(v)
        if len(vol_hist) > 60:
            vol_hist.pop(0)
        vol_arr = np.array(vol_hist, dtype=np.float32)
        vol_mean = float(vol_arr.mean())
        vol_std = float(vol_arr.std()) if vol_arr.size > 1 else 0.0
        vol_z = 0.0 if vol_std == 0.0 else (v - vol_mean) / vol_std

        # Build new feature row following feature_cols order
        new_row = np.zeros((feat_dim,), dtype=np.float32)
        new_row[idx_open] = o
        new_row[idx_high] = h
        new_row[idx_low] = l
        new_row[idx_close] = c
        new_row[idx_volume] = v
        new_row[idx_minutes] = minutes_since_midnight
        new_row[idx_tod_sin] = tod_sin
        new_row[idx_tod_cos] = tod_cos
        new_row[idx_dow] = dow
        new_row[idx_is_rth] = is_rth
        new_row[idx_range] = candle_range
        new_row[idx_body] = body
        new_row[idx_upper_wick] = upper_wick
        new_row[idx_lower_wick] = lower_wick
        new_row[idx_vol_mean_w] = vol_mean
        new_row[idx_vol_std_w] = vol_std
        new_row[idx_vol_z_w] = vol_z

        generated_rows.append(
            {
                "ts": next_ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )

        # Slide window: drop first row, append new_row
        X_window = np.vstack([X_window[1:], new_row])
        last_ts = next_ts

    gen_df = pd.DataFrame(generated_rows)
    gen_df = gen_df.set_index("ts").sort_index()
    return gen_df


def main():
    symbol = "MESM5"
    date_str = "20250318"
    window_in = 64
    horizon = 1

    out_dir = Path("out")
    ds_path = out_dir / f"ds_MESM5_20250318_win64_h1.npz"
    model_path = out_dir / f"gru_pricegen_{symbol}_{date_str}.pt"

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Using dataset: {ds_path}")
    print(f"Using model:   {model_path}")

    gen_df_15s = generate_sequence(ds_path, model_path, num_steps=240, seed_index=-1)
    print("Generated 15s candles:")
    print(gen_df_15s.head())

    gen_15s_path = out_dir / f"gen_15s_{symbol}_{date_str}.csv"
    gen_df_15s.to_csv(gen_15s_path, index_label="ts")
    print(f"Wrote generated 15s candles to: {gen_15s_path}")


if __name__ == "__main__":
    main()
