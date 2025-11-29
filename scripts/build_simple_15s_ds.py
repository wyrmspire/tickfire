from pathlib import Path
from typing import List, Dict
import json

import numpy as np
import pandas as pd
import pytz

# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "rawprice"
OUT_DIR = ROOT / "out"

SYMBOL_PREFIX = "MES"      # match MESM5, MESU5, etc.
BAR_SECONDS   = 15         # 15-second bars
WINDOW        = 64         # past window length
HORIZON       = 1         # predict next bar
START_DATE    = "20250318"
END_DATE      = "20250325"  # exclusive end (i.e. up to 24th)


# =====================================================================
# HELPERS
# =====================================================================

def load_trades_in_range(raw_dir: Path, start: str, end: str, symbol_prefix: str) -> pd.DataFrame:
    """
    Load trades between [start, end) for all symbols starting with symbol_prefix.
    Expects files like *YYYYMMDD.trades.json in RAW_DIR.
    """
    rows: List[tuple] = []

    dt_range = pd.date_range(start=start, end=end, freq="D", inclusive="left")
    for dt in dt_range:
        day_str = dt.strftime("%Y%m%d")
        # Be a little flexible with filenames
        candidates = list(raw_dir.glob(f"*{day_str}*.trades.json"))
        if not candidates:
            print(f"[load] No files found for {day_str} in {raw_dir}")
            continue

        for f in candidates:
            print(f"[load] Reading {f}")
            with f.open() as fp:
                for line in fp:
                    obj = json.loads(line)
                    if obj.get("action") != "T":
                        continue

                    symbol = obj.get("symbol", "")
                    if not symbol.startswith(symbol_prefix):
                        continue

                    ts = pd.to_datetime(obj["hd"]["ts_event"], utc=True)
                    price = float(obj["price"])
                    size = float(obj["size"])
                    rows.append((ts, price, size, symbol))

    if not rows:
        raise SystemExit("[load] No trades loaded. Check RAW_DIR, filenames, and SYMBOL_PREFIX.")

    df = pd.DataFrame(rows, columns=["ts", "price", "size", "symbol"])
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"[load] Loaded {len(df)} trades from {start} to {end} for prefix {symbol_prefix}")
    return df


def resample_15s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample tick trades into 15-second OHLCV bars per symbol.
    """
    bars = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        ohlc = g["price"].resample(f"{BAR_SECONDS}S").ohlc()
        vol  = g["size"].resample(f"{BAR_SECONDS}S").sum()
        out = pd.concat([ohlc, vol], axis=1)
        out.columns = ["open", "high", "low", "close", "volume"]
        out["symbol"] = sym
        bars.append(out)

    out = pd.concat(bars).sort_index()
    out = out.reset_index()  # ts back to column
    print(f"[bars] Built {len(out)} bars of {BAR_SECONDS}s")
    return out


def back_adjust_additive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple additive back-adjustment for continuous contract.
    When symbol changes:
        shift_total += (prev_close - new_open)
    Then all subsequent prices are shifted by shift_total.
    """
    df = df.sort_values(["ts"]).reset_index(drop=True)
    adj = df.copy()
    shift_total = 0.0
    last_symbol = None

    for i in range(len(df)):
        sym = df.loc[i, "symbol"]
        if last_symbol is None:
            last_symbol = sym
        elif sym != last_symbol:
            prev_close = df.loc[i - 1, "close"]
            new_open = df.loc[i, "open"]
            shift_total += (prev_close - new_open)
            last_symbol = sym

        for col in ["open", "high", "low", "close"]:
            adj.loc[i, col] = df.loc[i, col] + shift_total

    print(f"[adj] Back-adjusted {len(adj)} rows")
    return adj


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candle geometry, time-of-day, RTH flag, and volume z-score.
    """
    out = df.copy()

    # Candle geometry
    out["range"] = out["high"] - out["low"]
    out["body"]  = out["close"] - out["open"]
    upper_body = np.maximum(out["open"], out["close"])
    lower_body = np.minimum(out["open"], out["close"])
    out["upper_wick"] = out["high"] - upper_body
    out["lower_wick"] = lower_body - out["low"]

    # Time features (Chicago)
    tz = pytz.timezone("America/Chicago")
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    local = out["ts"].dt.tz_convert(tz)

    out["minutes_since_midnight"] = (
        local.dt.hour * 60 + local.dt.minute + local.dt.second / 60.0
    )
    ang = 2 * np.pi * (out["minutes_since_midnight"] / (24 * 60))
    out["tod_sin"] = np.sin(ang)
    out["tod_cos"] = np.cos(ang)
    out["dow"] = local.dt.dayofweek

    hour = local.dt.hour
    minute = local.dt.minute
    out["is_rth"] = (
        ((hour > 8) | ((hour == 8) & (minute >= 30))) &
        ((hour < 15) | ((hour == 15) & (minute <= 15)))
    ).astype(int)

    # Volume stats (60-bar rolling)
    v = out["volume"].fillna(0)
    out["vol_mean_w"] = v.rolling(60, min_periods=1).mean()
    out["vol_std_w"]  = v.rolling(60, min_periods=1).std().fillna(0)
    out["vol_z_w"] = (v - out["vol_mean_w"]) / out["vol_std_w"].replace(0, np.nan)
    out["vol_z_w"] = out["vol_z_w"].fillna(0)

    print(f"[feat] Added features on {len(out)} bars")
    return out


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5-dim target:
        ro, rh, rl, rc = returns based on previous close
        rv             = vol_z_w (z-scored volume)
    """
    out = df.copy()
    prev = out["close"].shift(1)
    out["ro"] = (out["open"]  - prev) / prev
    out["rh"] = (out["high"]  - out["open"]) / prev
    out["rl"] = (out["low"]   - out["open"]) / prev
    out["rc"] = (out["close"] - out["open"]) / prev
    out["rv"] = out["vol_z_w"]
    print(f"[tgt] Computed return targets")
    return out


def build_windows(df: pd.DataFrame, window: int, horizon: int) -> Dict[str, np.ndarray]:
    """
    Build sliding windows:
        X: [N, window, feat_dim]
        Y: [N, horizon, 5]
    """
    feats = [
        "open", "high", "low", "close", "volume",
        "minutes_since_midnight", "tod_sin", "tod_cos", "dow", "is_rth",
        "range", "body", "upper_wick", "lower_wick",
        "vol_mean_w", "vol_std_w", "vol_z_w",
    ]
    tgts = ["ro", "rh", "rl", "rc", "rv"]

    data = df.dropna(subset=feats + tgts).reset_index(drop=True)
    F = data[feats].to_numpy(np.float32)
    T = data[tgts].to_numpy(np.float32)
    ts = data["ts"].to_numpy()

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    idx_list: List[np.datetime64] = []

    max_start = len(data) - window - horizon
    if max_start <= 0:
        raise SystemExit(f"[windows] Not enough data ({len(data)}) for window={window}, horizon={horizon}")

    for i in range(max_start):
        X_list.append(F[i : i + window])
        Y_list.append(T[i + window : i + window + horizon])
        idx_list.append(ts[i + window])

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    indices = np.array(idx_list)

    print(f"[windows] Built X: {X.shape}, Y: {Y.shape}")
    return {
        "X": X,
        "Y": Y,
        "indices": indices,
        "feature_cols": np.array(feats),
        "target_cols": np.array(tgts),
    }


def normalize_dataset(ds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize X and Y in a simple z-score way, storing means/stds.
    """
    X = ds["X"]  # [N, T, F]
    Y = ds["Y"]  # [N, H, 5]

    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std  = X.std(axis=(0, 1), keepdims=True)
    X_std[X_std == 0] = 1.0

    Y_mean = Y.mean(axis=(0, 1), keepdims=True)
    Y_std  = Y.std(axis=(0, 1), keepdims=True)
    Y_std[Y_std == 0] = 1.0

    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    out = dict(ds)
    out["X"] = X_norm.astype(np.float32)
    out["Y"] = Y_norm.astype(np.float32)
    out["X_mean"] = X_mean.astype(np.float32)
    out["X_std"] = X_std.astype(np.float32)
    out["Y_mean"] = Y_mean.astype(np.float32)
    out["Y_std"] = Y_std.astype(np.float32)

    print("[norm] Normalized X and Y")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[main] ROOT={ROOT}")
    print(f"[main] RAW_DIR={RAW_DIR}")
    print(f"[main] OUT_DIR={OUT_DIR}")

    df_ticks = load_trades_in_range(RAW_DIR, START_DATE, END_DATE, SYMBOL_PREFIX)
    df_bars  = resample_15s(df_ticks)
    df_adj   = back_adjust_additive(df_bars)
    df_feat  = add_features(df_adj)
    df_ret   = compute_targets(df_feat)

    # Save some bars so we can visually inspect
    raw_bars_path = OUT_DIR / f"bars_simple_15s_raw_{START_DATE}_{END_DATE}.csv"
    feat_bars_path = OUT_DIR / f"bars_simple_15s_feat_{START_DATE}_{END_DATE}.csv"
    df_adj.to_csv(raw_bars_path, index=False)
    df_ret.to_csv(feat_bars_path, index=False)
    print(f"[main] Wrote raw bars to: {raw_bars_path}")
    print(f"[main] Wrote feature bars to: {feat_bars_path}")

    ds = build_windows(df_ret, WINDOW, HORIZON)
    ds_norm = normalize_dataset(ds)

    ds_path = OUT_DIR / f"ds_simple_15s_win{WINDOW}_h{HORIZON}_norm.npz"
    np.savez_compressed(ds_path, **ds_norm)
    print(f"[main] Saved dataset to: {ds_path}")


if __name__ == "__main__":
    main()
