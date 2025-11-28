import json
from pathlib import Path

import pandas as pd


def iter_trades_from_json_lines(path: Path):
    """
    Stream trades from a JSON-lines MDP3 file.

    Each line is a JSON object like:
    {
      "ts_recv": "...Z",
      "hd": {"ts_event": "...Z", ...},
      "action": "T",
      "side": "B",
      "price": "5731.750000000",
      "size": 1,
      "symbol": "MESM5",
      ...
    }
    """
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # We only care about trades for now
            if obj.get("action") != "T":
                continue
            yield obj


def load_trades_for_symbol(path: Path, symbol: str) -> pd.DataFrame:
    """
    Load trades for a single symbol from a JSON-lines file into a DataFrame.
    """
    records = []
    for trade in iter_trades_from_json_lines(path):
        if trade.get("symbol") != symbol:
            continue
        hd = trade.get("hd", {})
        ts_event_str = hd.get("ts_event")
        if ts_event_str is None:
            continue

        records.append(
            {
                "ts_event": ts_event_str,
                "price": float(trade["price"]),
                "size": int(trade["size"]),
                "side": trade.get("side"),
            }
        )

    if not records:
        return pd.DataFrame(columns=["ts_event", "price", "size", "side"])

    df = pd.DataFrame.from_records(records)
    # Convert timestamp string to pandas datetime and use as index
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("ts_event").sort_index()
    return df


def trades_to_15s_candles(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tick trades into 15-second OHLCV candles.
    """
    if trades.empty:
        return trades

    # Use the last traded price as our "price" series for OHLC
    price = trades["price"]
    volume = trades["size"]

    ohlc = price.resample("15S").ohlc()
    vol = volume.resample("15S").sum()

    candles = ohlc.copy()
    candles["volume"] = vol.fillna(0)
    # Drop periods with no trades at all if desired, or keep them as NaN/open=high=low=close
    # For now, keep them; it will make later resampling easier.
    return candles


def main():
    # Adjust these as needed
    raw_dir = Path("../data/rawprice").resolve()
    symbol = "MESM5"
    date_str = "20250318"  # YYYYMMDD, just a starting example

    filename = f"glbx-mdp3-2025{date_str[4:6]}{date_str[6:8]}.trades.json"
    # The filenames you showed look like glbx-mdp3-20250318.trades.json already,
    # so let's construct it directly:
    filename = f"glbx-mdp3-{date_str[:4]}{date_str[4:6]}{date_str[6:8]}.trades.json"

    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Cannot find raw file at {path}")

    print(f"Loading trades from: {path}")
    trades = load_trades_for_symbol(path, symbol)
    print(f"Loaded {len(trades)} {symbol} trades")

    candles_15s = trades_to_15s_candles(trades)
    print(f"Built {len(candles_15s)} 15s candles")

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"candles_15s_{symbol}_{date_str}.csv"
    candles_15s.to_csv(out_path, index_label="ts")
    print(f"Wrote 15s candles to: {out_path}")


if __name__ == "__main__":
    main()
