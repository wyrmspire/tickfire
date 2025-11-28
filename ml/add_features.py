from pathlib import Path

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df_local = df.copy()
    df_local.index = df_local.index.tz_convert("America/Chicago")

    minutes_since_midnight = (
        df_local.index.hour * 60
        + df_local.index.minute
        + df_local.index.second / 60.0
    )
    df["minutes_since_midnight"] = minutes_since_midnight

    day_minutes = 24 * 60
    df["tod_sin"] = np.sin(2 * np.pi * minutes_since_midnight / day_minutes)
    df["tod_cos"] = np.cos(2 * np.pi * minutes_since_midnight / day_minutes)

    df["dow"] = df_local.index.dayofweek

    hour = df_local.index.hour
    minute = df_local.index.minute
    is_after_open = (hour > 8) | ((hour == 8) & (minute >= 30))
    is_before_close = (hour < 15) | ((hour == 15) & (minute <= 15))
    df["is_rth"] = (is_after_open & is_before_close).astype(int)

    return df


def add_candle_geometry(df: pd.DataFrame) -> pd.DataFrame:
    # Flag rows that actually have trades
    df["has_trade"] = (df["volume"] > 0) & df["open"].notna()

    # Range/body/wicks only meaningful when we have trades
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]

    upper_body = df[["open", "close"]].max(axis=1)
    lower_body = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = df["high"] - upper_body
    df["lower_wick"] = lower_body - df["low"]

    # Returns: leave NaN where we don't have a valid close
    close = df["close"].copy()
    df["ret_close"] = close.pct_change()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_ret_close"] = np.log(close).diff()

    return df


def add_volume_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    vol = df["volume"]
    roll = vol.rolling(window=window, min_periods=1)

    df["vol_mean_w"] = roll.mean()
    df["vol_std_w"] = roll.std()

    std_safe = df["vol_std_w"].replace(0.0, np.nan)
    df["vol_z_w"] = (vol - df["vol_mean_w"]) / std_safe
    df["vol_z_w"] = df["vol_z_w"].fillna(0.0)

    return df


def main():
    symbol = "MESM5"
    date_str = "20250318"  # YYYYMMDD

    in_dir = Path("out")
    in_path = in_dir / f"candles_15s_{symbol}_{date_str}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input candle file not found: {in_path}")

    print(f"Loading candles from: {in_path}")
    df = pd.read_csv(in_path, parse_dates=["ts"])
    df = df.set_index("ts").sort_index()

    df = add_time_features(df)
    df = add_candle_geometry(df)
    df = add_volume_features(df, window=60)

    out_path = in_dir / f"candles_15s_{symbol}_{date_str}_features.csv"
    df.to_csv(out_path, index_label="ts")
    print(f"Wrote feature-enriched candles to: {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
