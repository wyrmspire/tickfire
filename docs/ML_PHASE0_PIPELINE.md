# ML Phase-0 Pipeline (`ml/`)

This describes the current working GRU-based 15s price generation pipeline living under `ml/`.  
We will **not** delete it; instead we will treat it as a reference implementation and gradually migrate the logic into `lab/`.

## Files

### 1. `build_candles.py`

- Reads raw MDP3 trade JSON files from `data/rawprice/` (e.g. `glbx-mdp3-20250318.trades.json`).
- Filters for trades of a single symbol (MES contract, e.g. `MESM5`).
- Aggregates tick trades into **15-second OHLCV candles** using pandas resampling.
- Writes to:
  - `out/candles_15s_{symbol}_{date}.csv`
- This is effectively the **tick → 15s OHLCV** step. :contentReference[oaicite:1]{index=1}

### 2. `add_features.py`

- Loads a 15s candle CSV (from `build_candles.py`).
- Adds **time-of-day features** in America/Chicago:
  - `minutes_since_midnight`
  - `tod_sin`, `tod_cos` (sin/cos encoding of time-of-day)
  - `dow` (day-of-week)
  - `is_rth` (approx 08:30–15:15 CT)
- Adds **candle geometry features**:
  - `range`, `body`, `upper_wick`, `lower_wick`
- Adds **volume features** with a rolling window:
  - `vol_mean_w`, `vol_std_w`, `vol_z_w`
- Writes to:
  - `out/candles_15s_{symbol}_{date}_features.csv` :contentReference[oaicite:2]{index=2}

### 3. `build_sequences_single_day.py`

- Loads the feature-enriched 15s candles for one day.
- Builds sliding-window datasets:
  - Inputs X: `window_in` (e.g. 64) timesteps of features
  - Targets Y: next `horizon` timesteps (OHLCV)
- Only keeps windows where `has_trade == True` and OHLC data are valid.
- Writes compressed NPZ dataset:
  - `out/ds_{symbol}_{date}_win{window}_h{horizon}.npz`
- This NPZ contains:
  - `X`, `Y`, `indices`, `feature_cols`, `target_cols`. :contentReference[oaicite:3]{index=3}

### 4. `model_gru.py`

- Defines `PriceGenGRU`, a small GRU-based model:

  - Input: `[batch, seq_len, input_dim]` (e.g. `[N, 64, 17]`)
  - Output: `[batch, horizon, 5]` (OHLCV for next candles)

- Uses a standard stacked GRU + linear head over the final hidden state.
- Horizon is currently 1 but shaped as `[batch, 1, 5]` for future flexibility. :contentReference[oaicite:4]{index=4}

### 5. `train_gru.py`

- Wraps the NPZ dataset into a `PriceSeqDataset` (PyTorch `Dataset`).
- Splits into train/val (e.g. 90/10).
- Trains `PriceGenGRU` with MSE loss to predict next OHLCV.
- Tracks best validation loss and saves **best checkpoint** with:
  - model state dict
  - input_dim, hidden_dim, num_layers, horizon
  - feature_cols and target_cols
- Saves to:
  - `out/gru_pricegen_{symbol}_{date}.pt` :contentReference[oaicite:5]{index=5}

### 6. `generate_gru.py`

- Loads:
  - dataset NPZ (`ds_{symbol}_{date}_win{window}_h{horizon}.npz`)
  - trained model checkpoint (`gru_pricegen_{symbol}_{date}.pt`)
- Picks a seed window `X[seed_index]` (default: last window).
- Reconstructs the last timestamp of the seed window and then:
  - repeatedly predicts the next OHLCV candle
  - repairs OHLC consistency (ensures high/low contain open/close)
  - recomputes time features (`minutes_since_midnight`, `tod_sin`, `tod_cos`, `dow`, `is_rth`)
  - recomputes candle geometry (`range`, `body`, wicks)
  - recomputes rolling volume stats (`vol_mean_w`, `vol_std_w`, `vol_z_w`)
  - slides the window and repeats
- Writes generated 15s OHLCV candles to:
  - `out/gen_15s_{symbol}_{date}.csv` :contentReference[oaicite:6]{index=6}

---

## How this maps into the new lab

These scripts together implement a **Phase-0 pipeline** for:

1. Raw trades → 15s OHLCV candles
2. 15s candles → enriched feature set
3. Features → windowed dataset (X, Y)
4. Dataset → trained small GRU
5. GRU → synthetic 15s candles

In the future, `lab/engine/run_experiment.py` will orchestrate a similar pipeline **per experiment run**, but with:

- more flexible dataset definitions
- multiple feature palettes (F0–F5)
- logging of config + timings + metrics into `runs/sweeps/...`

For now, `ml/` remains the working reference implementation of “a single GRU experiment for one day”.
