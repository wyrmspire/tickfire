"""
Utilities for analyzing synthetic price data

Helper functions to sanity-check generator output and compute statistics.
"""

import pandas as pd
from typing import Dict, Any


def summarize_day(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a day of generated data.
    
    Args:
        df: DataFrame from PriceGenerator.generate_day()
    
    Returns:
        Dictionary with overall stats, per-state stats, and per-session stats
    """
    summary = {}
    
    # Overall day stats
    summary['overall'] = {
        'num_bars': len(df),
        'start_price': float(df['open'].iloc[0]),
        'end_price': float(df['close'].iloc[-1]),
        'net_move': float(df['close'].iloc[-1] - df['open'].iloc[0]),
        'net_move_ticks': int(df['close_ticks'].iloc[-1] - df['open_ticks'].iloc[0]),
        'high': float(df['high'].max()),
        'low': float(df['low'].min()),
        'total_range': float(df['high'].max() - df['low'].min()),
        'total_range_ticks': int((df['high'].max() - df['low'].min()) / 0.25),
        'avg_volume': float(df['volume'].mean()),
        'total_volume': int(df['volume'].sum()),
        'avg_range_ticks': float(df['range_ticks'].mean()),
        'avg_body_ticks': float(df['body_ticks'].mean()),
        'avg_delta_ticks': float(df['delta_ticks'].mean()),
        'std_delta_ticks': float(df['delta_ticks'].std()),
    }
    
    # Macro regime if present
    if 'macro_regime' in df.columns:
        summary['overall']['macro_regime'] = df['macro_regime'].iloc[0]
    
    # Per-state statistics
    if 'state' in df.columns:
        summary['by_state'] = {}
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            summary['by_state'][state] = {
                'count': len(state_df),
                'pct_of_day': float(len(state_df) / len(df) * 100),
                'avg_delta_ticks': float(state_df['delta_ticks'].mean()),
                'std_delta_ticks': float(state_df['delta_ticks'].std()),
                'avg_range_ticks': float(state_df['range_ticks'].mean()),
                'avg_body_ticks': float(state_df['body_ticks'].mean()),
                'avg_volume': float(state_df['volume'].mean()),
                'net_move_ticks': int(state_df['delta_ticks'].sum()),
                'up_bars': int((state_df['delta_ticks'] > 0).sum()),
                'down_bars': int((state_df['delta_ticks'] < 0).sum()),
                'flat_bars': int((state_df['delta_ticks'] == 0).sum()),
            }
    
    # Per-session statistics
    if 'session' in df.columns:
        summary['by_session'] = {}
        for session in df['session'].unique():
            session_df = df[df['session'] == session]
            summary['by_session'][session] = {
                'count': len(session_df),
                'pct_of_day': float(len(session_df) / len(df) * 100),
                'avg_delta_ticks': float(session_df['delta_ticks'].mean()),
                'avg_range_ticks': float(session_df['range_ticks'].mean()),
                'avg_volume': float(session_df['volume'].mean()),
                'net_move_ticks': int(session_df['delta_ticks'].sum()),
            }
    
    # Per-segment statistics if segments exist
    if 'segment_id' in df.columns:
        summary['by_segment'] = {}
        for seg_id in df['segment_id'].unique():
            seg_df = df[df['segment_id'] == seg_id]
            summary['by_segment'][int(seg_id)] = {
                'count': len(seg_df),
                'state': seg_df['state'].iloc[0] if len(seg_df) > 0 else None,
                'net_move_ticks': int(seg_df['delta_ticks'].sum()),
                'range_ticks': int(seg_df['range_ticks'].sum()),
                'avg_volume': float(seg_df['volume'].mean()),
            }
    
    return summary


def print_summary(summary: Dict[str, Any], verbose: bool = True) -> None:
    """
    Pretty-print a summary dictionary.
    
    Args:
        summary: Output from summarize_day()
        verbose: If True, print detailed per-state and per-session stats
    """
    print("\n" + "=" * 60)
    print("DAY SUMMARY")
    print("=" * 60)
    
    # Overall stats
    overall = summary['overall']
    print(f"\nOverall:")
    print(f"  Bars: {overall['num_bars']}")
    print(f"  Price: {overall['start_price']:.2f} → {overall['end_price']:.2f}")
    print(f"  Net Move: {overall['net_move']:.2f} ({overall['net_move_ticks']:+d} ticks)")
    print(f"  Range: {overall['low']:.2f} - {overall['high']:.2f} ({overall['total_range_ticks']} ticks)")
    print(f"  Avg Bar Range: {overall['avg_range_ticks']:.1f} ticks")
    print(f"  Avg Bar Body: {overall['avg_body_ticks']:.1f} ticks")
    print(f"  Avg Delta: {overall['avg_delta_ticks']:.2f} ± {overall['std_delta_ticks']:.2f} ticks")
    print(f"  Total Volume: {overall['total_volume']:,}")
    
    if 'macro_regime' in overall:
        print(f"  Macro Regime: {overall['macro_regime']}")
    
    if verbose and 'by_state' in summary:
        print(f"\nBy State:")
        print(f"  {'State':<15} {'Count':>6} {'%':>6} {'AvgΔ':>8} {'AvgRng':>8} {'NetΔ':>8} {'Up/Dn':>10}")
        print("  " + "-" * 70)
        for state, stats in summary['by_state'].items():
            print(f"  {state:<15} {stats['count']:>6} {stats['pct_of_day']:>5.1f}% "
                  f"{stats['avg_delta_ticks']:>7.2f} {stats['avg_range_ticks']:>7.1f} "
                  f"{stats['net_move_ticks']:>7d} "
                  f"{stats['up_bars']:>4}/{stats['down_bars']:<4}")
    
    if verbose and 'by_session' in summary:
        print(f"\nBy Session:")
        print(f"  {'Session':<15} {'Count':>6} {'%':>6} {'AvgΔ':>8} {'AvgRng':>8} {'NetΔ':>8}")
        print("  " + "-" * 60)
        for session, stats in summary['by_session'].items():
            print(f"  {session:<15} {stats['count']:>6} {stats['pct_of_day']:>5.1f}% "
                  f"{stats['avg_delta_ticks']:>7.2f} {stats['avg_range_ticks']:>7.1f} "
                  f"{stats['net_move_ticks']:>7d}")
    
    print("=" * 60)


def compare_states(df: pd.DataFrame, states_to_compare: list = None) -> pd.DataFrame:
    """
    Create a comparison table of different states.
    
    Args:
        df: DataFrame from generator
        states_to_compare: List of states to compare, or None for all
    
    Returns:
        DataFrame with comparison metrics
    """
    if states_to_compare is None:
        states_to_compare = df['state'].unique()
    
    comparison = []
    
    for state in states_to_compare:
        state_df = df[df['state'] == state]
        if len(state_df) == 0:
            continue
        
        comparison.append({
            'state': state,
            'count': len(state_df),
            'avg_delta_ticks': state_df['delta_ticks'].mean(),
            'std_delta_ticks': state_df['delta_ticks'].std(),
            'avg_range_ticks': state_df['range_ticks'].mean(),
            'avg_body_ticks': state_df['body_ticks'].mean(),
            'avg_upper_wick': state_df['upper_wick_ticks'].mean(),
            'avg_lower_wick': state_df['lower_wick_ticks'].mean(),
            'avg_volume': state_df['volume'].mean(),
            'up_pct': (state_df['delta_ticks'] > 0).sum() / len(state_df) * 100,
        })
    
    return pd.DataFrame(comparison)
