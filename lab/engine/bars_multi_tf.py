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
    ]
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
    Y: [N] next-step return over 'horizon' bars, computed from close.
    """
    values = df[feature_cols].values
    closes = df["close"].values.astype(float)

    # next-step return (horizon)
    tgt = (np.roll(closes, -horizon) - closes) / closes

    X_list: List[np.ndarray] = []
    Y_list: List[float] = []

    max_i = len(df) - window - horizon
    for i in range(max_i):
        X_list.append(values[i : i + window])
        Y_list.append(float(tgt[i + window]))

    if not X_list:
        X = np.zeros((0, window, len(feature_cols)), dtype=float)
        Y = np.zeros((0,), dtype=float)
    else:
        X = np.stack(X_list)
        Y = np.array(Y_list, dtype=float)

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

    # load + concat days
    df = join_dates_to_df(out_dir, dates, tf)

    if "close" not in df.columns:
        raise ValueError(f"'close' column is required but missing; df.columns={list(df.columns)}")

    # resolve features against actual columns
    feat_info = resolve_features(df, cfg.indicator_set)
    feature_cols = feat_info["features"]
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
    np.savez_compressed(out_path, X=X, Y=Y)

    return {
        "dataset_path": str(out_path),
        "num_windows": int(X.shape[0]),
        "feature_dim": int(X.shape[2]) if X.ndim == 3 else 0,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "base_timeframe": tf,
        "indicator_set": cfg.indicator_set,
        "dates_used": dates,
        "features_used": feature_cols,
        "features_missing_from_palette": missing_cols,
    }
