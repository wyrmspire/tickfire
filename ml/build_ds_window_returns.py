import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

RAW_DIR = Path("../data/rawprice")
OUT_DIR = Path("out")

BAR_SECONDS = 15
WINDOW = 64
HORIZON = 1

def parse_date_from_filename(path: Path):
    import re
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {path}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def load_trades_in_range(raw_dir: Path, start_date: str, end_date: str):
    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    rows = []

    for f in sorted(raw_dir.glob("*.trades.json")):
        file_date = parse_date_from_filename(f)
        if file_date < start_ts or file_date > end_ts:
            continue

        with f.open() as fp:
            for line in fp:
                o = json.loads(line)
                if o.get("action") != "T":
                    continue
                ts = pd.to_datetime(o["hd"]["ts_event"], utc=True)
                rows.append((ts, float(o["price"]), float(o["size"]), o["symbol"]))

    df = pd.DataFrame(rows, columns=["ts","price","size","symbol"])
    return df.sort_values("ts").reset_index(drop=True)

def resample_15s(df):
    bars = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        ohlc = g["price"].resample(f"{BAR_SECONDS}S").ohlc()
        vol  = g["size"].resample(f"{BAR_SECONDS}S").sum()
        out = pd.concat([ohlc, vol], axis=1)
        out.columns = ["open","high","low","close","volume"]
        out["symbol"] = sym
        bars.append(out)
    return pd.concat(bars).sort_index().reset_index()


def back_adjust_additive(df):
    df = df.sort_values("ts").reset_index(drop=True)
    adj = df.copy()
    shift_total = 0.0
    last_symbol = None

    for i in range(len(df)):
        sym = df.loc[i,"symbol"]
        if last_symbol is None:
            last_symbol = sym
        elif sym != last_symbol:
            prev_close = df.loc[i-1,"close"]
            new_open  = df.loc[i,"open"]
            shift_total += (prev_close - new_open)
            last_symbol = sym

        for col in ["open","high","low","close"]:
            adj.loc[i,col] = df.loc[i,col] + shift_total
    return adj

def add_features(df):
    out = df.copy()
    out["range"] = out["high"] - out["low"]
    out["body"]  = out["close"] - out["open"]
    upper_body = np.maximum(out["open"], out["close"])
    lower_body = np.minimum(out["open"], out["close"])
    out["upper_wick"] = out["high"] - upper_body
    out["lower_wick"] = lower_body - out["low"]

    tz = pytz.timezone("America/Chicago")
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    local = out["ts"].dt.tz_convert(tz)

    out["minutes_since_midnight"] = (
        local.dt.hour*60 + local.dt.minute + local.dt.second/60
    )
    ang = 2*np.pi*(out["minutes_since_midnight"]/(24*60))
    out["tod_sin"] = np.sin(ang)
    out["tod_cos"] = np.cos(ang)
    out["dow"] = local.dt.dayofweek

    hour = local.dt.hour
    minute = local.dt.minute
    out["is_rth"] = (
        ((hour > 8) | ((hour == 8) & (minute >= 30))) &
        ((hour < 15) | ((hour == 15) & (minute <= 15)))
    ).astype(int)

    v = out["volume"].fillna(0)
    out["vol_mean_w"] = v.rolling(60, min_periods=1).mean()
    out["vol_std_w"]  = v.rolling(60, min_periods=1).std().fillna(0)
    out["vol_z_w"] = (v - out["vol_mean_w"]) / out["vol_std_w"].replace(0, np.nan)
    out["vol_z_w"] = out["vol_z_w"].fillna(0)
    return out


def compute_targets(df):
    out = df.copy()
    prev = out["close"].shift(1)
    out["ro"] = (out["open"]  - prev)/prev
    out["rh"] = (out["high"]  - out["open"])/prev
    out["rl"] = (out["low"]   - out["open"])/prev
    out["rc"] = (out["close"] - out["open"])/prev
    out["rv"] = out["vol_z_w"]
    return out

def build_windows(df, window, horizon):
    feats = [
        "open","high","low","close","volume",
        "minutes_since_midnight","tod_sin","tod_cos","dow","is_rth",
        "range","body","upper_wick","lower_wick",
        "vol_mean_w","vol_std_w","vol_z_w",
    ]
    tgts = ["ro","rh","rl","rc","rv"]

    data = df.dropna(subset=feats+tgts).reset_index(drop=True)
    F = data[feats].to_numpy(np.float32)
    T = data[tgts].to_numpy(np.float32)
    ts = data["ts"].to_numpy()

    X_list, Y_list, idx_list = [], [], []
    max_start = len(data) - window - horizon
    for i in range(max_start):
        X_list.append(F[i:i+window])
        Y_list.append(T[i+window : i+window+horizon])
        idx_list.append(ts[i+window])

    return {
        "X": np.stack(X_list),
        "Y": np.stack(Y_list),
        "indices": np.array(idx_list),
        "feature_cols": np.array(feats),
        "target_cols": np.array(tgts),
    }

def main():
    start = "20250318"
    end   = "20250405"

    df_ticks = load_trades_in_range(RAW_DIR, start, end)
    df_bars  = resample_15s(df_ticks)
    df_adj   = back_adjust_additive(df_bars)
    df_feat  = add_features(df_adj)
    df_ret   = compute_targets(df_feat)
    ds       = build_windows(df_ret, WINDOW, HORIZON)

    OUT_DIR.mkdir(exist_ok=True)
    outp = OUT_DIR / f"ds_debug_{start}_{end}.npz"
    np.savez_compressed(outp, **ds)
    print("Saved:", outp)

if __name__ == "__main__":
    main()
