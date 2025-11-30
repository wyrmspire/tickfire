"""
Demo: Generate and visualize synthetic MES price data

This script demonstrates the price generator and chart visualizer working together.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.price_generator import (
    PriceGenerator,
    MarketState,
    StateConfig,
    STATE_CONFIGS,
)
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig, quick_chart


def demo_basic_generation():
    """Generate a day of data with automatic state transitions"""
    print("=" * 60)
    print("DEMO 1: Basic Day Generation with Auto State Transitions")
    print("=" * 60)
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    
    # Generate a full day
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    df = gen.generate_day(start_date, auto_transition=True)
    
    print(f"\nGenerated {len(df)} bars")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    
    # Show state distribution
    print("\nState distribution:")
    print(df['state'].value_counts())
    
    # Create chart
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ChartConfig(
        title="MES Simulation - Auto State Transitions",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=True,
        show_session_changes=True,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo1_auto_states.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_controlled_states():
    """Generate data with manually controlled state sequence"""
    print("\n" + "=" * 60)
    print("DEMO 2: Controlled State Sequence")
    print("=" * 60)
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=123)
    
    # Define a specific state sequence
    # Format: (bar_index, state)
    state_sequence = [
        (0, MarketState.FLAT),           # Start flat (midnight)
        (180, MarketState.ZOMBIE),       # 3am - slow grind starts
        (390, MarketState.RANGING),      # 6:30am - choppy
        (570, MarketState.RALLY),        # 9:30am - RTH open, rally
        (690, MarketState.IMPULSIVE),    # 11:30am - high volatility
        (810, MarketState.BREAKDOWN),    # 1:30pm - sharp drop
        (900, MarketState.RANGING),      # 3pm - settle into range
        (960, MarketState.FLAT),         # 4pm - afterhours quiet
    ]
    
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    df = gen.generate_day(start_date, state_sequence=state_sequence, auto_transition=False)
    
    print(f"\nGenerated {len(df)} bars with controlled states")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    
    config = ChartConfig(
        title="MES Simulation - Controlled State Sequence",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=True,
        show_session_changes=True,
        major_tick_interval_minutes=120,  # Every 2 hours
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo2_controlled_states.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_custom_state():
    """Generate data with a custom state configuration"""
    print("\n" + "=" * 60)
    print("DEMO 3: Custom State Configuration")
    print("=" * 60)
    
    # Create a custom "mega volatile" state
    custom_state = StateConfig(
        name="mega_volatile",
        avg_ticks_per_bar=40.0,
        ticks_per_bar_std=20.0,
        up_probability=0.5,
        trend_persistence=0.4,  # Low persistence = very choppy
        avg_tick_size=3.0,
        tick_size_std=2.0,
        max_tick_jump=15,
        volatility_multiplier=3.0,
        wick_probability=0.6,
        wick_extension_avg=5.0,
    )
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=456)
    
    # Generate just a few hours with this custom state
    start_date = datetime(2025, 11, 29, 9, 30, 0)  # RTH open
    bars = []
    
    for minute in range(240):  # 4 hours
        timestamp = start_date + timedelta(minutes=minute)
        bar = gen.generate_bar(timestamp, custom_state_config=custom_state)
        bars.append(bar)
    
    import pandas as pd
    df = pd.DataFrame(bars)
    
    print(f"\nGenerated {len(df)} bars with custom 'mega_volatile' state")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    print(f"Max bar range: {(df['high'] - df['low']).max():.2f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    
    config = ChartConfig(
        title="MES Simulation - Custom Mega Volatile State",
        figsize=(16, 9),
        show_volume=True,
        show_state_changes=False,
        show_session_changes=True,
        major_tick_interval_minutes=30,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo3_custom_state.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_state_comparison():
    """Generate samples of each state for comparison"""
    print("\n" + "=" * 60)
    print("DEMO 4: State Comparison")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    states_to_compare = [
        MarketState.FLAT,
        MarketState.RANGING,
        MarketState.ZOMBIE,
        MarketState.RALLY,
        MarketState.IMPULSIVE,
    ]
    
    fig, axes = plt.subplots(len(states_to_compare), 1, figsize=(16, 12))
    
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    for idx, state in enumerate(states_to_compare):
        # Generate 2 hours of data for this state
        gen = PriceGenerator(initial_price=5000.0, seed=42 + idx)
        
        bars = []
        for minute in range(120):
            timestamp = start_date + timedelta(minutes=minute)
            bar = gen.generate_bar(timestamp, state=state)
            bars.append(bar)
        
        import pandas as pd
        df = pd.DataFrame(bars)
        
        # Plot on subplot
        ax = axes[idx]
        
        config = ChartConfig(
            title=f"{state.value.upper()} State",
            show_volume=False,
            show_state_changes=False,
            show_session_changes=False,
            figsize=(16, 3),
        )
        
        viz = ChartVisualizer(config)
        viz.plot_candlestick(ax, df)
        
        ax.set_title(f"{state.value.upper()} State", fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        
        # Add stats
        price_range = df['high'].max() - df['low'].min()
        avg_bar_range = (df['high'] - df['low']).mean()
        
        stats_text = f"Range: {price_range:.2f} | Avg Bar: {avg_bar_range:.2f}"
        ax.text(
            0.02, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "demo4_state_comparison.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison chart saved to: {chart_path}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("MES PRICE GENERATOR DEMO")
    print("=" * 60)
    
    # Run demos
    demo_basic_generation()
    demo_controlled_states()
    demo_custom_state()
    demo_state_comparison()
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print(f"\nCharts saved to: {Path(__file__).resolve().parents[1] / 'out' / 'charts'}")


if __name__ == "__main__":
    main()
