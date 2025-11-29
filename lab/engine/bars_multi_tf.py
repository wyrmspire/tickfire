"""
Dataset builder for bars + indicators.

Consumes:
    out/bars/YYYYMMDD/<tf>.csv
    out/bars_indicators/YYYYMMDD/<tf>_ind.csv

Currently supports:
    - base_timeframe = "15s" or "1m"
    - data_family    = "bars_apr07_apr25"
    - indicator_set  in ["basic", "full_v1"]

Assumptions:
    - indicator CSVs were built from the bars CSVs and are row-aligned.
      We therefore join on index, not on a timestamp key.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Date ranges for each data_family
# ------------------------------------------------------------
DATA_FAMILIES: Dict[str, List[str]] = {
    "bars_apr07_apr25": [
        "20250407", "20250408", "20250409", "20250410", "20250411",
        "20250413", "20250414", "20250415", "20250416", "20250417",
        "20250420", "20250421", "20250422", "20250423", "20250424",
        "20250425",
    ],
    "bars_middle_2days": [
        "20250416", "20250417",
    ],
}

# ------------------------------------------------------------
# Indicator palettes (desired columns)
# ------------------------------------------------------------
PALETTES: Dict[str, List[str]] = {
    # ultra minimal: OHLCV only
    "basic": [
        "open", "high", "low", "close", "volume",
    ],
    # "full" wish list: we will down-select to existing columns
    "full_v1": [
        "open", "high", "low", "close", "volume",
        "dow", "tod_sin", "tod_cos",
        "body", "upper_wick", "lower_wick",
        "atr", "ema20", "ema50", "ema200",
        "vwap", "vol_z",
    ],
}


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def load_bars_for_date(out_dir: Path, date: str, tf: str) -> pd.DataFrame:
    """
    Load base timeframe and indicator CSVs for one date, then join them
    by index (row-aligned), dropping duplicate columns from the indicator side.
    """
    base_path = out_dir / "bars" / date / f"{tf}.csv"
    ind_path = out_dir / "bars_indicators" / date / f"{tf}_ind.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing bars: {base_path}")
    if not ind_path.exists():
        raise FileNotFoundError(f"Missing indicators: {ind_path}")

    bars = pd.read_csv(base_path)
    ind = pd.read_csv(ind_path)

    if len(bars) != len(ind):
        raise ValueError(
            f"Row count mismatch for {date} {tf}: "
            f"bars={len(bars)}, indicators={len(ind)}"
        )

    # Drop any columns in indicators that already exist in bars
    duplicate_cols = [c for c in ind.columns if c in bars.columns]
    ind_extra = ind.drop(columns=duplicate_cols)

    merged = pd.concat(
        [bars.reset_index(drop=True), ind_extra.reset_index(drop=True)],
        axis=1,
    )
    return merged


def join_dates_to_df(out_dir: Path, dates: List[str], tf: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for d in dates:
        frames.append(load_bars_for_date(out_dir, d, tf))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# Window construction
# ------------------------------------------------------------
def make_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) sliding windows.

    X: [N, window, feat_dim] from feature_cols
    Y: [N, 5] next-step targets (returns):

       Let C_t = close at time t, V_t = volume at t,
           O_{t+h}, H_{t+h}, L_{t+h}, C_{t+h}, V_{t+h} be OHLCV at horizon steps ahead.

       Then:

         0: (O_{t+h} - C_t) / C_t
         1: (H_{t+h} - C_t) / C_t
         2: (L_{t+h} - C_t) / C_t
         3: (C_{t+h} - C_t) / C_t
         4: (V_{t+h} - V_t) / (V_t + 1)

    The model will learn to predict these returns.
    """
    values = df[feature_cols].values

    # We need OHLCV for targets.
    req_cols = ["open", "high", "low", "close", "volume"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' for target generation")

    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    vols = df["volume"].values.astype(float)

    # Shifted arrays (future values)
    next_opens = np.roll(opens, -horizon)
    next_highs = np.roll(highs, -horizon)
    next_lows = np.roll(lows, -horizon)
    next_closes = np.roll(closes, -horizon)
    next_vols = np.roll(vols, -horizon)

    # Compute returns relative to CURRENT close / volume
    tgt_open = (next_opens - closes) / closes
    tgt_high = (next_highs - closes) / closes
    tgt_low = (next_lows - closes) / closes
    tgt_close = (next_closes - closes) / closes
    tgt_vol = (next_vols - vols) / (vols + 1.0)

    targets_all = np.stack([tgt_open, tgt_high, tgt_low, tgt_close, tgt_vol], axis=1)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    max_i = len(df) - window - horizon
    for i in range(max_i):
        X_list.append(values[i : i + window])
        # Target is the step AFTER the window (at i + window)
        Y_list.append(targets_all[i + window])

    if not X_list:
        X = np.zeros((0, window, len(feature_cols)), dtype=float)
        Y = np.zeros((0, 5), dtype=float)
    else:
        X = np.stack(X_list)
        Y = np.stack(Y_list)

    return X, Y


# ------------------------------------------------------------
# Feature selection logic
# ------------------------------------------------------------
def resolve_features(df: pd.DataFrame, indicator_set: str) -> Dict[str, Any]:
    """
    Take a desired palette name + DataFrame, and return:
        - actual feature list (intersection with df.columns)
        - list of missing columns from the desired palette
    Guarantees that we at least fall back to OHLCV if available.
    """
    if indicator_set not in PALETTES:
        raise ValueError(f"Unknown indicator_set: {indicator_set!r}")

    desired = PALETTES[indicator_set]
    cols = set(df.columns)

    available = [c for c in desired if c in cols]
    missing = [c for c in desired if c not in cols]

    # always ensure we have at least OHLCV if present
    fallback = [c for c in ["open", "high", "low", "close", "volume"] if c in cols]
    if not available:
        available = fallback

    if not available:
        raise ValueError(
            f"No usable feature columns found in df. "
            f"Desired={desired}, df.columns={list(df.columns)[:20]}..."
        )

    return {
        "features": available,
        "missing": missing,
    }


# ------------------------------------------------------------
# MTF Merging Logic
# ------------------------------------------------------------
def ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'ts' column exists (datetime64) by renaming a likely timestamp
    column if needed.
    """
    if "ts" not in df.columns:
        candidates = ["timestamp", "date", "time", "datetime"]
        for c in candidates:
            if c in df.columns:
                df = df.rename(columns={c: "ts"})
                break

    if "ts" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"])

    return df


def merge_context_data(base_df: pd.DataFrame, context_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Merge context_df onto base_df using merge_asof.
    Assumes both have a 'ts' column or a convertible timestamp column.
    """
    base_df = ensure_ts_column(base_df)
    context_df = ensure_ts_column(context_df)

    if "ts" not in base_df.columns:
        raise ValueError("Base dataframe missing 'ts' column for MTF merge")
    if "ts" not in context_df.columns:
        raise ValueError(f"Context dataframe ({suffix}) missing 'ts' column for MTF merge")

    # Sort for asof merge
    base_df = base_df.sort_values("ts")
    context_df = context_df.sort_values("ts")

    # Rename context columns (except ts) to avoid collisions
    ctx_cols = [c for c in context_df.columns if c != "ts"]
    rename_map = {c: f"{c}_{suffix}" for c in ctx_cols}
    context_df_renamed = context_df.rename(columns=rename_map)

    merged = pd.merge_asof(
        base_df,
        context_df_renamed,
        on="ts",
        direction="backward",
    )

    # Fill NaNs (e.g., base starts before context)
    merged = merged.fillna(0)

    return merged


# ------------------------------------------------------------
# Public entry point for the experiment engine
# ------------------------------------------------------------
def compile_dataset_bars_multi_tf(cfg, run_dir: Path) -> Dict[str, Any]:
    """
    Build a dataset for the given ExperimentConfig using prebuilt bars
    and indicators.

    Returns a dict with metadata and dataset path.
    """
    out_dir = Path("out")

    # Pick date range
    if cfg.data_family not in DATA_FAMILIES:
        raise ValueError(f"Unknown data_family: {cfg.data_family}")

    dates = DATA_FAMILIES[cfg.data_family]

    tf = cfg.base_timeframe
    if tf not in ["15s", "1m"]:
        raise ValueError(f"Unsupported base timeframe for now: {tf!r}")

    # load + concat days for BASE timeframe
    df = join_dates_to_df(out_dir, dates, tf)

    if "close" not in df.columns:
        raise ValueError(f"'close' column is required but missing; df.columns={list(df.columns)}")

    # Ensure we have a usable timestamp axis if possible
    df = ensure_ts_column(df)
    has_ts = "ts" in df.columns
    ts_array = None
    seed_ts = None
    if has_ts:
        ts_series = pd.to_datetime(df["ts"])
        ts_array = ts_series.values.astype("datetime64[ns]")
        if len(ts_array) > 0:
            seed_ts = ts_array[-1]

    # --------------------------------------------------------
    # MTF Context Merging
    # --------------------------------------------------------
    context_tfs = cfg.context_timeframes or []
    for ctx_tf in context_tfs:
        print(f"Loading context data for {ctx_tf}...")
        ctx_df = join_dates_to_df(out_dir, dates, ctx_tf)
        df = merge_context_data(df, ctx_df, suffix=ctx_tf)
        print(f"Merged {ctx_tf} context. New shape: {df.shape}")

    # resolve features against actual columns
    feat_info = resolve_features(df, cfg.indicator_set)
    feature_cols = feat_info["features"]

    # Add context columns (e.g., open_15m, high_15m, ...)
    for ctx_tf in context_tfs:
        suffix = f"_{ctx_tf}"
        ctx_cols = [c for c in df.columns if c.endswith(suffix)]
        feature_cols.extend(ctx_cols)

    # Remove duplicates
    feature_cols = list(dict.fromkeys(feature_cols))
    missing_cols = feat_info["missing"]

    # build sliding windows
    X, Y = make_sliding_windows(
        df=df,
        feature_cols=feature_cols,
        horizon=cfg.horizon,
        window=cfg.window,
    )

    # Save dataset
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    out_path = art_dir / f"ds_bars_{tf}_w{cfg.window}_h{cfg.horizon}.npz"

    save_dict: Dict[str, Any] = {
        "X": X,
        "Y": Y,
        "feature_cols": np.array(feature_cols),
    }
    if ts_array is not None:
        save_dict["ts"] = ts_array
    if seed_ts is not None:
        save_dict["seed_ts"] = np.array(seed_ts)

    np.savez_compressed(out_path, **save_dict)

    return {
        "dataset_path": str(out_path),
        "num_windows": int(X.shape[0]),
        "feature_dim": int(X.shape[2]) if X.ndim == 3 else 0,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "base_timeframe": tf,
        "context_timeframes": context_tfs,
        "indicator_set": cfg.indicator_set,
        "dates_used": dates,
        "features_used": feature_cols,
        "features_missing_from_palette": missing_cols,
    }
