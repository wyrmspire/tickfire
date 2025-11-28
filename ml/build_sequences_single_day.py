from pathlib import Path

import numpy as np
import pandas as pd


def build_sequences_for_day(
    symbol: str,
    date_str: str,
    window_in: int = 64,
    horizon: int = 1,
):
    """
    Build sliding-window sequences for a single day.

    Inputs (X):  window_in candles worth of features.
    Targets (Y): next `horizon` candles' OHLCV.

    Only windows where all rows have has_trade == True are kept.
    """
    in_dir = Path("out")
    in_path = in_dir / f"candles_15s_{symbol}_{date_str}_features.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Feature file not found: {in_path}")

    print(f"Loading features from: {in_path}")
    df = pd.read_csv(in_path, parse_dates=["ts"])
    df = df.set_index("ts").sort_index()

    # Basic sanity: ensure has_trade exists
    if "has_trade" not in df.columns:
        raise ValueError("Expected 'has_trade' column in features file")

    # Define which columns the model will see as inputs
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "minutes_since_midnight",
        "tod_sin",
        "tod_cos",
        "dow",
        "is_rth",
        "range",
        "body",
        "upper_wick",
        "lower_wick",
        "vol_mean_w",
        "vol_std_w",
        "vol_z_w",
        # You can add returns later if you want:
        # "ret_close",
        # "log_ret_close",
    ]

    target_cols = ["open", "high", "low", "close", "volume"]

    # Only consider rows where we have actual trades and non-null targets
    valid_mask = (
        (df["has_trade"] == True)
        & df["open"].notna()
        & df["close"].notna()
    )

    values_feat = df[feature_cols].to_numpy(dtype=np.float32)
    values_tgt = df[target_cols].to_numpy(dtype=np.float32)
    valid = valid_mask.to_numpy()

    n = len(df)
    sequences_X = []
    sequences_Y = []
    indices = []  # store start timestamps for debugging

    max_start = n - window_in - horizon + 1
    for start in range(max_start):
        end_in = start + window_in
        end_total = end_in + horizon

        # All rows in input+target range must be valid trades
        if not valid[start:end_total].all():
            continue

        X_window = values_feat[start:end_in]
        Y_window = values_tgt[end_in:end_total]

        sequences_X.append(X_window)
        sequences_Y.append(Y_window)
        indices.append(df.index[start])

    if not sequences_X:
        raise ValueError("No valid sequences built; check has_trade and window size.")

    X = np.stack(sequences_X, axis=0)  # [num_seq, window_in, num_features]
    Y = np.stack(sequences_Y, axis=0)  # [num_seq, horizon, 5]
    indices = np.array(indices)

    print(f"Built {X.shape[0]} sequences")
    print(f"X shape: {X.shape} (num_seq, window_in, num_features)")
    print(f"Y shape: {Y.shape} (num_seq, horizon, 5)")

    out_dir = in_dir
    out_path = out_dir / f"ds_{symbol}_{date_str}_win{window_in}_h{horizon}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        Y=Y,
        indices=indices.astype("datetime64[ns]"),
        feature_cols=np.array(feature_cols),
        target_cols=np.array(target_cols),
    )
    print(f"Wrote dataset to: {out_path}")


def main():
    symbol = "MESM5"
    date_str = "20250318"  # YYYYMMDD
    window_in = 64
    horizon = 1

    build_sequences_for_day(symbol, date_str, window_in=window_in, horizon=horizon)


if __name__ == "__main__":
    main()
