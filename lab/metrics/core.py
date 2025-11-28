"""
Core metrics for evaluating synthetic vs real price series.

To add later:
- return-based metrics (mean, std, skew, kurtosis, tails)
- volatility clustering metrics
- OHLC shape metrics (range, wicks, body distributions)
- swing structure metrics
- level interaction metrics (VWAP, prior highs/lows, etc.)

These functions will write metrics.json into each run's eval/ folder.
"""
