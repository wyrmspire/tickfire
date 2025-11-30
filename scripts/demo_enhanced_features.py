"""
Demo: Enhanced features - segments, tick columns, macro regimes

Shows the new ML-ready features added to the price generator.
"""

from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators import PriceGenerator, MarketState
from lab.generators.utils import summarize_day, print_summary, compare_states
from lab.visualizers import quick_chart


def demo_segment_based_generation():
    """Generate data with 15-minute segment-based state transitions"""
    print("\n" + "=" * 70)
    print("DEMO: Segment-Based State Transitions (15-min segments)")
    print("=" * 70)
    
    gen = PriceGenerator(initial_price=5000.0, seed=789)
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    
    # Generate with 15-minute segments
    # States will only change at segment boundaries (0, 15, 30, 45, etc.)
    df = gen.generate_day(
        start_date,
        auto_transition=True,
        segment_length=15,  # 15-minute segments
        macro_regime='TREND_DAY',  # Manual macro label
    )
    
    print(f"\nGenerated {len(df)} bars with 15-min segments")
    print(f"Unique segments: {df['segment_id'].nunique()}")
    print(f"Macro regime: {df['macro_regime'].iloc[0]}")
    
    # Show first few rows with new columns
    print("\nSample data (first 5 bars):")
    cols_to_show = ['time', 'close', 'close_ticks', 'delta_ticks', 'range_ticks', 
                    'body_ticks', 'state', 'session', 'segment_id', 'macro_regime']
    print(df[cols_to_show].head().to_string(index=False))
    
    # Summarize
    summary = summarize_day(df)
    print_summary(summary, verbose=True)
    
    # Show segment-level stats
    if 'by_segment' in summary:
        print("\nSegment-level stats (first 10 segments):")
        print(f"  {'Seg':>3} {'State':<12} {'Bars':>5} {'NetΔ':>7} {'Range':>7} {'AvgVol':>8}")
        print("  " + "-" * 50)
        for seg_id in sorted(summary['by_segment'].keys())[:10]:
            stats = summary['by_segment'][seg_id]
            print(f"  {seg_id:>3} {stats['state']:<12} {stats['count']:>5} "
                  f"{stats['net_move_ticks']:>7} {stats['range_ticks']:>7} "
                  f"{stats['avg_volume']:>8.0f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    chart_path = output_dir / "demo_segments.png"
    quick_chart(
        df,
        title="MES Simulation - 15-Minute Segment-Based States",
        save_path=chart_path,
        show=False,
        figsize=(20, 10),
        show_state_changes=True,
        show_session_changes=True,
    )
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_tick_based_features():
    """Demonstrate tick-based columns for ML"""
    print("\n" + "=" * 70)
    print("DEMO: Tick-Based Features for ML Training")
    print("=" * 70)
    
    gen = PriceGenerator(initial_price=5000.0, seed=999)
    start_date = datetime(2025, 11, 29, 9, 30, 0)  # RTH open
    
    # Generate just 2 hours
    bars = []
    for minute in range(120):
        from datetime import timedelta
        timestamp = start_date + timedelta(minutes=minute)
        bar = gen.generate_bar(timestamp, MarketState.IMPULSIVE)
        bars.append(bar)
    
    df = pd.DataFrame(bars)
    
    print(f"\nGenerated {len(df)} bars in IMPULSIVE state")
    
    # Show tick-based columns
    print("\nTick-based columns (sample):")
    tick_cols = ['time', 'close', 'close_ticks', 'delta_ticks', 'range_ticks', 
                 'body_ticks', 'upper_wick_ticks', 'lower_wick_ticks']
    print(df[tick_cols].head(10).to_string(index=False))
    
    # Compute some ML-ready features
    print("\nML-Ready Feature Statistics:")
    print(f"  Delta ticks - Mean: {df['delta_ticks'].mean():.2f}, Std: {df['delta_ticks'].std():.2f}")
    print(f"  Range ticks - Mean: {df['range_ticks'].mean():.2f}, Std: {df['range_ticks'].std():.2f}")
    print(f"  Body ticks  - Mean: {df['body_ticks'].mean():.2f}, Std: {df['body_ticks'].std():.2f}")
    print(f"  Upper wick  - Mean: {df['upper_wick_ticks'].mean():.2f}")
    print(f"  Lower wick  - Mean: {df['lower_wick_ticks'].mean():.2f}")
    
    # Show distribution
    print("\nDelta ticks distribution:")
    print(df['delta_ticks'].value_counts().sort_index().head(15))
    
    return df


def demo_macro_regimes():
    """Generate multiple days with different macro regimes"""
    print("\n" + "=" * 70)
    print("DEMO: Macro Regime Labeling")
    print("=" * 70)
    
    regimes_to_test = [
        ('TREND_DAY', [(0, MarketState.ZOMBIE), (300, MarketState.RALLY), (900, MarketState.ZOMBIE)]),
        ('CHOP_DAY', [(0, MarketState.RANGING), (360, MarketState.IMPULSIVE), (720, MarketState.RANGING)]),
        ('QUIET_DAY', [(0, MarketState.FLAT), (480, MarketState.RANGING), (960, MarketState.FLAT)]),
    ]
    
    all_summaries = []
    
    for regime_name, state_seq in regimes_to_test:
        gen = PriceGenerator(initial_price=5000.0, seed=hash(regime_name) % 10000)
        start_date = datetime(2025, 11, 29, 0, 0, 0)
        
        df = gen.generate_day(
            start_date,
            state_sequence=state_seq,
            auto_transition=False,
            macro_regime=regime_name,
        )
        
        summary = summarize_day(df)
        all_summaries.append({
            'regime': regime_name,
            'net_move_ticks': summary['overall']['net_move_ticks'],
            'total_range_ticks': summary['overall']['total_range_ticks'],
            'avg_range_ticks': summary['overall']['avg_range_ticks'],
            'avg_delta_ticks': summary['overall']['avg_delta_ticks'],
            'std_delta_ticks': summary['overall']['std_delta_ticks'],
        })
    
    # Print comparison
    print("\nMacro Regime Comparison:")
    comparison_df = pd.DataFrame(all_summaries)
    print(comparison_df.to_string(index=False))
    
    print("\nThese macro labels can be used for:")
    print("  - Training regime classifiers")
    print("  - Filtering training data by market condition")
    print("  - Evaluating model performance across different regimes")
    print("  - Building regime-conditional strategies")


def demo_state_comparison():
    """Compare characteristics of different states"""
    print("\n" + "=" * 70)
    print("DEMO: State Characteristic Comparison")
    print("=" * 70)
    
    # Generate a day with all states represented
    gen = PriceGenerator(initial_price=5000.0, seed=555)
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    
    state_seq = [
        (0, MarketState.FLAT),
        (120, MarketState.RANGING),
        (240, MarketState.ZOMBIE),
        (360, MarketState.RALLY),
        (480, MarketState.IMPULSIVE),
        (600, MarketState.BREAKDOWN),
        (720, MarketState.BREAKOUT),
        (840, MarketState.RANGING),
    ]
    
    df = gen.generate_day(start_date, state_sequence=state_seq, auto_transition=False)
    
    # Compare states
    comparison = compare_states(df)
    
    print("\nState Characteristics:")
    print(comparison.to_string(index=False))
    
    print("\nKey Insights:")
    print("  - Each state has distinct tick movement patterns")
    print("  - avg_delta_ticks shows directional bias")
    print("  - avg_range_ticks shows volatility")
    print("  - up_pct shows trend strength")
    print("  - These patterns are what ML models will learn to recognize")


def main():
    """Run all enhanced feature demos"""
    print("\n" + "=" * 70)
    print("ENHANCED PRICE GENERATOR FEATURES")
    print("=" * 70)
    
    demo_segment_based_generation()
    demo_tick_based_features()
    demo_macro_regimes()
    demo_state_comparison()
    
    print("\n" + "=" * 70)
    print("All enhanced demos complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  ✓ Tick-based columns ready for ML training")
    print("  ✓ Segment-based state control for meso-scale patterns")
    print("  ✓ Macro regime labels for day-level classification")
    print("  ✓ Comprehensive stats for sanity checking")
    print("\nNext steps:")
    print("  1. Generate synthetic archetype datasets")
    print("  2. Train pattern recognition models")
    print("  3. Apply to real MES data")
    print("=" * 70)


if __name__ == "__main__":
    main()
