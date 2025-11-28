"""
Feature palette definitions.

Here we will define small, named sets of features, e.g.:

- F0: OHLCV only
- F1: OHLCV + time-of-day + day-of-week
- F2: F1 + volatility features
- F3: F2 + indicator features (EMA20/50/200, VWAP, ADR, etc.)
- F4: F3 + multi-timeframe context
- F5: macro-conditioned variants

Functions to add later:
- build_feature_frame(df_bars, palette_name) -> np.ndarray or DataFrame
- describe_palette(palette_name) -> human-readable description
"""
