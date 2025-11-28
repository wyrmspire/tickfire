"""
Visualization logic for model maker results.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def resample_candles(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a new timeframe.
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'is_generated': 'last' # Keep track if any part of this candle was generated
    }
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    resampled = df.resample(rule).agg(agg_dict).dropna()
    return resampled

def plot_candlestick(ax, df, width=0.6, colorup='green', colordown='red'):
    """
    Draw candlestick chart on a given axes.
    """
    # Convert dates to matplotlib format
    dates = mdates.date2num(df.index)
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Determine colors
    colors = [colorup if c >= o else colordown for o, c in zip(opens, closes)]
    
    # Plot wicks
    ax.vlines(dates, lows, highs, colors=colors, linewidth=1)
    
    # Plot bodies
    # We use bar plot for bodies
    # Width needs to be in days for date2num. 
    # Approx width: width * (min_diff)
    if len(dates) > 1:
        min_diff = np.min(np.diff(dates))
        w = width * min_diff
    else:
        w = width * 1.0/1440.0 # Default small width
        
    for i, (d, o, c, color) in enumerate(zip(dates, opens, closes, colors)):
        lower = min(o, c)
        height = abs(c - o)
        ax.add_patch(plt.Rectangle((d - w/2, lower), w, height, color=color))

    ax.xaxis_date()
    ax.autoscale_view()

def plot_model_result(run_dir: Path):
    """
    Load generated data, resample if needed, and plot.
    """
    csv_path = run_dir / "generated_sequence.csv"
    if not csv_path.exists():
        print(f"No generated data found in {run_dir}")
        return

    df = pd.read_csv(csv_path, parse_dates=['ts'], index_col='ts')
    
    # Auto-resample logic
    # We want < 100 candles
    # Available rules: 15S -> 1T -> 5T -> 15T -> 1H -> 4H
    rules = ['15s', '1min', '5min', '15min', '1h', '4h']
    current_rule_idx = 0
    
    plot_df = df.copy()
    current_rule = rules[0]
    
    while len(plot_df) > 100 and current_rule_idx < len(rules) - 1:
        current_rule_idx += 1
        current_rule = rules[current_rule_idx]
        plot_df = resample_candles(df, current_rule)
        
    print(f"Plotting {len(plot_df)} candles at {current_rule} timeframe for {run_dir.name}")
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot candles
    plot_candlestick(ax, plot_df)
    
    # Highlight generated region
    # Find where is_generated switches to 1
    gen_start = plot_df[plot_df['is_generated'] == 1].index.min()
    if pd.notnull(gen_start):
        # Draw a vertical line
        ax.axvline(x=gen_start, color='blue', linestyle='--', alpha=0.7, label='Generation Start')
        
        # Shade the generated area
        ax.axvspan(gen_start, plot_df.index.max(), color='blue', alpha=0.1)
    
    ax.set_title(f"Model: {run_dir.name} | Timeframe: {current_rule}")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    
    out_path = run_dir / "chart.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved chart to {out_path}")

if __name__ == "__main__":
    # Test with a dummy file if run directly
    pass
