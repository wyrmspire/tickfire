# Price Generator System - Architecture & Usage

## Overview

We've built a **tick-based MES price generator** that serves as the physics engine for synthetic market data generation. This is designed to be the **core Layer 0** of our ML pipeline, with all intelligence layered *around* it rather than embedded within it.

## Core Philosophy

> **The generator knows physics. ML knows patterns.**

- **Generator**: Understands what states *mean* (via configs) and how ticks move
- **Markov/ML**: Decides *when* states change and *which* patterns to create
- **Pattern Recognition**: Learns to detect these structures in real data

## Architecture

### Layer 0: Physics Engine (`PriceGenerator`)

**Location**: `lab/generators/price_generator.py`

**What it does**:
- Generates OHLCV bars from tick-level movements
- Respects MES tick size (0.25)
- Applies configurable market states (RANGING, RALLY, ZOMBIE, etc.)
- Incorporates session effects (Asian, London, RTH, etc.)
- Applies day-of-week multipliers

**Key Features**:
1. **Tick-based output** - All prices + integer tick columns for ML
2. **Segment-based state control** - States change at segment boundaries (e.g., 15-min)
3. **Macro regime labeling** - Day-level classification (TREND_DAY, CHOP_DAY, etc.)
4. **Explicit state tracking** - Every bar labeled with state, session, segment_id

### Layer 1: Fractal State Manager (Future Integration)

**Location**: `lab/generators/fractal_states.py`

**What it does**:
- Manages hierarchical states across timeframes
- Day state → Hour state → Minute state
- Proper transition probabilities between levels
- Combined parameter calculation

**Status**: Built but not yet integrated into main generator. Next step is to wire this into `generate_day()` as an optional state driver.

### Layer 2: Markov/ML State Drivers (External)

**Not in generator** - These sit outside and control the generator:

```python
# Markov decides state sequence
state_sequence = markov_model.generate_sequence(...)

# Generator executes the physics
df = gen.generate_day(start_date, state_sequence=state_sequence)
```

### Layer 3: Pattern Recognition (Future)

Train models on synthetic archetypes:
1. Generate clean patterns (RANGING → BREAKOUT, etc.)
2. Extract tick-based features
3. Train encoder/classifier
4. Apply to real MES data to find similar structures

## Output Schema

Every generated bar includes:

### Price Columns (floats)
- `time`, `open`, `high`, `low`, `close`, `volume`

### Tick Columns (integers) - ML-Ready
- `open_ticks`, `high_ticks`, `low_ticks`, `close_ticks`
- `delta_ticks` - Change from previous close (in ticks)
- `range_ticks` - High - Low (in ticks)
- `body_ticks` - |Close - Open| (in ticks)
- `upper_wick_ticks`, `lower_wick_ticks`

### Label Columns
- `state` - Minute-level state (ranging, rally, etc.)
- `session` - Trading session (asian, london, rth, etc.)
- `segment_id` - Segment number (if using segment_length)
- `macro_regime` - Day-level regime (TREND_DAY, CHOP_DAY, etc.)

## Configuration System

### Market States (Minute-level)

Defined in `STATE_CONFIGS`:

- **RANGING**: Choppy, mean-reverting (12 ticks/bar avg, low persistence)
- **FLAT**: Very quiet (4 ticks/bar avg, minimal movement)
- **ZOMBIE**: Slow grind (6 ticks/bar, high persistence, slight bias)
- **RALLY**: Strong directional (18 ticks/bar, 70% up probability)
- **IMPULSIVE**: High volatility (25 ticks/bar, large jumps)
- **BREAKDOWN**: Sharp down move (20 ticks/bar, 75% down probability)
- **BREAKOUT**: Sharp up move (22 ticks/bar, 75% up probability)

### Session Effects

Defined in `SESSION_CONFIGS`:

- **ASIAN**: Low volume (0.4x), low volatility (0.6x)
- **LONDON**: Building activity (0.8x vol, 1.1x volatility)
- **PREMARKET**: Moderate (0.6x vol, 0.9x volatility)
- **RTH**: Peak activity (1.5x vol, 1.3x volatility)
- **AFTERHOURS**: Declining (0.5x vol, 0.7x volatility)

### Day-of-Week Effects

Defined in `DOW_CONFIGS`:

- **Monday**: Higher volatility (1.2x)
- **Friday**: Higher volume (1.2x)
- **Weekend**: Reduced activity (0.3-0.4x)

## Usage Examples

### Basic Day Generation

```python
from lab.generators import PriceGenerator
from datetime import datetime

gen = PriceGenerator(initial_price=5000.0, seed=42)
start_date = datetime(2025, 11, 29, 0, 0, 0)

# Auto state transitions
df = gen.generate_day(start_date, auto_transition=True)
```

### Controlled State Sequence

```python
from lab.generators import MarketState

# Define specific pattern
state_sequence = [
    (0, MarketState.RANGING),
    (60, MarketState.RALLY),
    (180, MarketState.RANGING),
]

df = gen.generate_day(
    start_date,
    state_sequence=state_sequence,
    auto_transition=False
)
```

### Segment-Based Generation (Meso-scale)

```python
# States only change every 15 minutes
df = gen.generate_day(
    start_date,
    auto_transition=True,
    segment_length=15,  # 15-minute segments
    macro_regime='TREND_DAY'
)

# Now you can group by segment_id for meso features
segment_stats = df.groupby('segment_id').agg({
    'delta_ticks': 'sum',
    'range_ticks': 'sum',
    'volume': 'mean',
})
```

### Custom State Configuration

```python
from lab.generators import StateConfig

# Create custom state
custom_state = StateConfig(
    name="mega_volatile",
    avg_ticks_per_bar=40.0,
    volatility_multiplier=3.0,
    max_tick_jump=15,
)

# Generate with custom state
bar = gen.generate_bar(timestamp, custom_state_config=custom_state)
```

## Analysis Utilities

### Summarize Day

```python
from lab.generators.utils import summarize_day, print_summary

summary = summarize_day(df)
print_summary(summary, verbose=True)

# Returns:
# - Overall stats (net move, range, avg delta, etc.)
# - Per-state stats (avg delta, range, up/down bars)
# - Per-session stats
# - Per-segment stats (if segments used)
```

### Compare States

```python
from lab.generators.utils import compare_states

comparison = compare_states(df)
print(comparison)

# Shows avg_delta_ticks, avg_range_ticks, up_pct, etc. per state
```

## Visualization

### Quick Chart

```python
from lab.visualizers import quick_chart

quick_chart(
    df,
    title="MES Simulation",
    save_path="chart.png",
    show_state_changes=True,
    show_session_changes=True,
)
```

### Custom Chart Config

```python
from lab.visualizers import ChartVisualizer, ChartConfig

config = ChartConfig(
    figsize=(20, 10),
    color_up='#26a69a',
    color_down='#ef5350',
    show_volume=True,
    show_state_changes=True,
    show_session_changes=True,
)

viz = ChartVisualizer(config)
viz.create_chart(df, save_path="custom_chart.png")
```

## Next Steps

### 1. Generate Synthetic Archetypes

Create clean pattern libraries:

```python
# Clean RALLY day
rally_day = gen.generate_day(
    start_date,
    state_sequence=[(0, MarketState.RALLY)],
    auto_transition=False,
    macro_regime='TREND_DAY'
)

# RANGING → BREAKOUT pattern
breakout_pattern = gen.generate_day(
    start_date,
    state_sequence=[
        (0, MarketState.RANGING),
        (60, MarketState.BREAKOUT),
        (120, MarketState.RALLY),
    ],
    auto_transition=False,
    segment_length=15,
)
```

### 2. Extract Features for ML

```python
# Tick-based features are already in the DataFrame
features = df[[
    'delta_ticks',
    'range_ticks',
    'body_ticks',
    'upper_wick_ticks',
    'lower_wick_ticks',
    'volume',
]]

# Add derived features
df['tick_direction'] = np.sign(df['delta_ticks'])
df['range_to_body_ratio'] = df['range_ticks'] / (df['body_ticks'] + 1)
```

### 3. Train Pattern Recognizers

```python
# Pretrain on synthetic archetypes
from sklearn.ensemble import RandomForestClassifier

X = extract_features(synthetic_df)
y = synthetic_df['state']

rf = RandomForestClassifier()
rf.fit(X, y)

# Apply to real MES data
real_features = extract_features(real_mes_df)
predicted_states = rf.predict(real_features)
```

### 4. Find High-Probability Zones

```python
# Label good entry points in synthetic data
def label_high_prob_zones(df):
    # After compression in RANGING, before BREAKOUT
    # First pullback in RALLY
    # etc.
    return labels

# Train model to detect these conditions
# Apply to real data to find similar setups
```

## Integration with Fractal States (TODO)

Wire up the fractal state manager:

```python
from lab.generators import FractalStateManager, DayState

# Initialize fractal manager
fractal = FractalStateManager(seed=42)
fractal.initialize_day(DayState.TREND_DAY)

# Use it to drive state sequence
state_sequence = []
for minute in range(1440):
    timestamp = start_date + timedelta(minutes=minute)
    day_state, hour_state, minute_state = fractal.update(timestamp)
    
    # Convert minute_state to MarketState and add to sequence
    # ...

df = gen.generate_day(start_date, state_sequence=state_sequence)
```

## File Structure

```
lab/
├── generators/
│   ├── __init__.py
│   ├── price_generator.py      # Main generator
│   ├── fractal_states.py       # Hierarchical state manager
│   └── utils.py                # Analysis utilities
└── visualizers/
    ├── __init__.py
    └── chart_viz.py            # Chart creation

scripts/
├── demo_price_generation.py    # Basic demos
└── demo_enhanced_features.py   # ML-ready feature demos

out/
└── charts/                     # Generated charts
```

## Key Design Decisions

1. **Tick-based from the start** - Everything in integer ticks, no float rounding issues
2. **States are configs, not behaviors** - Easy to add new states without code changes
3. **Segment-based control** - Aligns with meso-scale pattern thinking
4. **Labels everywhere** - Every bar tagged with state/session/segment/regime
5. **Generator is pure** - No ML inside, just physics
6. **Fractal architecture** - States at multiple timescales (day/hour/minute)

## Performance Notes

- Generating 1 day (1440 bars): ~50ms
- Generating 30 days: ~1.5s
- Chart rendering: ~500ms per chart

All fast enough for rapid iteration and large synthetic dataset generation.
