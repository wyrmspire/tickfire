import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import pytz

# ================================================================
#  CONFIG
# ================================================================
RAW_DIR = Path("../data/rawprice")            # folder with .trades.json
OUT_DIR = Path("out")                         # output folder
SYMBOL_PREFIX = "MES"                         # detect contracts like MESM5, MESU5
BAR_SECONDS = 15                              # 15-second bars
WINDOW = 64                                   # model window size
HORIZON = 1                                   # predict next single candle


# ================================================================
#  STEP 1 — Load and combine *all* raw trade files
# ================================================================
def load_all_trades(raw_dir: Path) -> pd.DataFrame:
    rows = []
    files = sorted(raw_dir.glob("*.trades.json"))

    for f in files:
        with f.open() as fp:
            for line in fp:
                obj = json.loads(line)
                if obj.get("action") != "T":
                    continue

                ts = pd.to_datetime(obj["hd"]["ts_event"], utc=True)
                price = float(obj["price"])
                size = float(obj["size"])
                symbol = obj["symbol"]

                rows.append((ts, price, size, symbol))

    df = pd.DataFrame(rows, columns=["ts", "price", "size", "symbol"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ================================================================
#  STEP 2 — Convert ticks → OHLCV by 15-second bars
# ================================================================
def resample_15s(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to fixed 15-second bars for *each symbol*."""
    bars = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        ohlcv = g["price"].resample(f"{BAR_SECONDS}S").ohlc()
        vol = g["size"].resample(f"{BAR_SECONDS}S").sum()
        out = pd.concat([ohlcv, vol], axis=1)
        out.columns = ["open", "high", "low", "close", "volume"]
        out["symbol"] = sym
        bars.append(out)

    out = pd.concat(bars)
    out = out.sort_index()
    return out.reset_index()


# ================================================================
#  STEP 3 — Build back-adjusted continuous contract
# ================================================================
def back_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additive back-adjustment:
        When symbol changes:
            gap = new_open - last_close_of_old
            shift ALL previous prices by +gap
    """
    df = df.copy()
    df["close_fwd"] = df["close"].shift(1)
    adjusted = df.copy()

    shift_total = 0.0
    last_symbol = None

    for i in range(len(df)):
        sym = df.loc[i, "symbol"]

        if last_symbol is None:
            last_symbol = sym
            continue

        if sym != last_symbol:
            # rollover
            new_open = df.loc[i, "open"]
            prev_close = df.loc[i - 1, "close"]
            gap = prev_close - new_open
            shift_total += gap

        # apply shift
        for col in ["open", "high", "low", "close"]:
            adjusted.loc[i, col] = df.loc[i, col] + shift_total

        last_symbol = sym

    return adjusted.drop(columns=["close_fwd"])


# ================================================================
#  STEP 4 — Feature engineering
# ================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Candle geometry
    out["range"] = out["high"] - out["low"]
    out["body"] = out["close"] - out["open"]

    upper_body = np.maximum(out["open"], out["close"])
    lower_body = np.minimum(out["open"], out["close"])
    out["upper_wick"] = out["high"] - upper_body
    out["lower_wick"] = lower_body - out["low"]

    # Time features (America/Chicago)
    tz = pytz.timezone("America/Chicago")
    ts_local = out["ts"].dt.tz_convert(tz)

    out["minutes_since_midnight"] = (
        ts_local.dt.hour * 60 + ts_local.dt.minute + ts_local.dt.second / 60
    )

    angle = 2 * np.pi * (out["minutes_since_midnight"] / (24 * 60))
    out["tod_sin"] = np.sin(angle)
    out["tod_cos"] = np.cos(angle)

    out["dow"] = ts_local.dt.dayofweek

    # RTH flag: 08:30–15:15 CT
    hour = ts_local.dt.hour
    minute = ts_local.dt.minute
    is_after_open = (hour > 8) | ((hour == 8) & (minute >= 30))
    is_before_close = (hour < 15) | ((hour == 15) & (minute <= 15))
    out["is_rth"] = (is_after_open & is_before_close).astype(int)

    # Rolling volume stats
    vol = out["volume"].fillna(0)
    out["vol_mean_w"] = vol.rolling(60, min_periods=1).mean()
    out["vol_std_w"] = vol.rolling(60, min_periods=1).std().fillna(0)
    out["vol_z_w"] = (vol - out["vol_mean_w"]) / out["vol_std_w"].replace(0, np.nan)
    out["vol_z_w"] = out["vol_z_w"].fillna(0)

    return out


# ================================================================
#  STEP 5 — Convert OHLCV → return targets
# ================================================================
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)

    out["ro"] = (out["open"] - prev_close) / prev_close
    out["rh"] = (out["high"] - out["open"]) / prev_close
    out["rl"] = (out["low"] - out["open"]) / prev_close
    out["rc"] = (out["close"] - out["open"]) / prev_close
    out["rv"] = out["vol_z_w"]  # volume target = z-score

    return out


# ================================================================
#  STEP 6 — Build sliding windows for ML
# ================================================================
def build_windows(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    feature_cols = [
        "open","high","low","close","volume",
        "minutes_since_midnight","tod_sin","tod_cos","dow","is_rth",
        "range","body","upper_wick","lower_wick",
        "vol_mean_w","vol_std_w","vol_z_w"
    ]

    target_cols = ["ro","rh","rl","rc","rv"]

    data = df.copy()
    data = data.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    F = data[feature_cols].values.astype(np.float32)
    T = data[target_cols].values.astype(np.float32)
    ts_index = data["ts"].values

    X_list, Y_list, idx_list = [], [], []

    for i in range(len(data) - WINDOW - HORIZON):
        X_list.append(F[i : i + WINDOW])
        Y_list.append(T[i + WINDOW : i + WINDOW + HORIZON])
        idx_list.append(ts_index[i + WINDOW])

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    idx = np.array(idx_list)

    return {
        "X": X,
        "Y": Y,
        "indices": idx,
        "feature_cols": np.array(feature_cols),
        "target_cols": np.array(target_cols)
    }


# ================================================================
#  MAIN
# ================================================================
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading tick data…")
    df_ticks = load_all_trades(RAW_DIR)

    print("Resampling to 15-second bars…")
    df15 = resample_15s(df_ticks)

    print("Building back-adjusted continuous contract…")
    df_adj = back_adjust(df15)

    print("Engineering features…")
    df_feat = add_features(df_adj)

    print("Computing return targets…")
    df_ret = compute_returns(df_feat)

    print("Building ML windows…")
    ds = build_windows(df_ret)

    out_path = OUT_DIR / "ds_MES_continuous_win64_h1.npz"
    np.savez(out_path, **ds)
    print(f"Saved dataset: {out_path}")


if __name__ == "__main__":
    main()
