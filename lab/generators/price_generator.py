"""
MES Price Generator with Configurable Market States

This module generates synthetic 1-minute MES candles with realistic tick-based price action.
All parameters are exposed as "knobs" for testing different market conditions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum


class MarketState(Enum):
    """Market behavior states that affect price generation"""
    RANGING = "ranging"          # Tight, choppy, mean-reverting
    FLAT = "flat"                # Very low volatility, minimal movement
    ZOMBIE = "zombie"            # Slow grind in one direction
    RALLY = "rally"              # Strong directional move
    IMPULSIVE = "impulsive"      # High volatility, large swings
    BREAKDOWN = "breakdown"      # Sharp downward move
    BREAKOUT = "breakout"        # Sharp upward move


class Session(Enum):
    """Trading sessions with different characteristics"""
    ASIAN = "asian"              # 18:00-03:00 CT (typically lower volume)
    LONDON = "london"            # 03:00-08:30 CT (increasing activity)
    PREMARKET = "premarket"      # 08:30-09:30 CT (building momentum)
    RTH = "rth"                  # 09:30-16:00 CT (Regular Trading Hours - highest volume)
    AFTERHOURS = "afterhours"    # 16:00-18:00 CT (declining activity)


@dataclass
class StateConfig:
    """Configuration for a specific market state"""
    name: str
    
    # Tick movement parameters
    avg_ticks_per_bar: float = 8.0          # Average number of ticks per 1m bar
    ticks_per_bar_std: float = 4.0          # Std dev of ticks per bar
    
    # Directional bias
    up_probability: float = 0.5             # Probability of upward tick (0.5 = neutral)
    trend_persistence: float = 0.5          # How likely to continue previous direction (0-1)
    
    # Tick size distribution
    avg_tick_size: float = 1.0              # Average ticks per move (1.0 = single tick)
    tick_size_std: float = 0.5              # Std dev of tick size
    max_tick_jump: int = 8                  # Maximum ticks in single move
    
    # Volatility
    volatility_multiplier: float = 1.0      # Overall volatility scaling
    
    # Wick characteristics
    wick_probability: float = 0.3           # Probability of extended wicks
    wick_extension_avg: float = 2.0         # Average wick extension in ticks


# Predefined state configurations
STATE_CONFIGS = {
    MarketState.RANGING: StateConfig(
        name="ranging",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.5,
        trend_persistence=0.3,  # Low persistence = choppy
        avg_tick_size=1.2,
        tick_size_std=0.8,
        max_tick_jump=4,
        volatility_multiplier=1.0,
        wick_probability=0.4,
        wick_extension_avg=2.5,
    ),
    MarketState.FLAT: StateConfig(
        name="flat",
        avg_ticks_per_bar=4.0,
        ticks_per_bar_std=2.0,
        up_probability=0.5,
        trend_persistence=0.4,
        avg_tick_size=1.0,
        tick_size_std=0.3,
        max_tick_jump=2,
        volatility_multiplier=0.3,
        wick_probability=0.2,
        wick_extension_avg=1.0,
    ),
    MarketState.ZOMBIE: StateConfig(
        name="zombie",
        avg_ticks_per_bar=6.0,
        ticks_per_bar_std=3.0,
        up_probability=0.55,  # Slight upward bias
        trend_persistence=0.7,  # High persistence = grind
        avg_tick_size=1.0,
        tick_size_std=0.4,
        max_tick_jump=3,
        volatility_multiplier=0.6,
        wick_probability=0.25,
        wick_extension_avg=1.5,
    ),
    MarketState.RALLY: StateConfig(
        name="rally",
        avg_ticks_per_bar=18.0,
        ticks_per_bar_std=8.0,
        up_probability=0.7,  # Strong upward bias
        trend_persistence=0.8,
        avg_tick_size=1.5,
        tick_size_std=1.0,
        max_tick_jump=6,
        volatility_multiplier=1.5,
        wick_probability=0.3,
        wick_extension_avg=3.0,
    ),
    MarketState.IMPULSIVE: StateConfig(
        name="impulsive",
        avg_ticks_per_bar=25.0,
        ticks_per_bar_std=12.0,
        up_probability=0.5,
        trend_persistence=0.6,
        avg_tick_size=2.0,
        tick_size_std=1.5,
        max_tick_jump=10,
        volatility_multiplier=2.0,
        wick_probability=0.5,
        wick_extension_avg=4.0,
    ),
    MarketState.BREAKDOWN: StateConfig(
        name="breakdown",
        avg_ticks_per_bar=20.0,
        ticks_per_bar_std=10.0,
        up_probability=0.25,  # Strong downward bias
        trend_persistence=0.85,
        avg_tick_size=1.8,
        tick_size_std=1.2,
        max_tick_jump=8,
        volatility_multiplier=1.8,
        wick_probability=0.35,
        wick_extension_avg=3.5,
    ),
    MarketState.BREAKOUT: StateConfig(
        name="breakout",
        avg_ticks_per_bar=22.0,
        ticks_per_bar_std=10.0,
        up_probability=0.75,  # Strong upward bias
        trend_persistence=0.85,
        avg_tick_size=1.8,
        tick_size_std=1.2,
        max_tick_jump=8,
        volatility_multiplier=1.8,
        wick_probability=0.35,
        wick_extension_avg=3.5,
    ),
}


@dataclass
class SessionConfig:
    """Configuration for session-based effects"""
    name: str
    volume_multiplier: float = 1.0          # Relative volume level
    volatility_multiplier: float = 1.0      # Relative volatility
    state_transition_prob: float = 0.05     # Probability of state change per bar


SESSION_CONFIGS = {
    Session.ASIAN: SessionConfig(
        name="asian",
        volume_multiplier=0.4,
        volatility_multiplier=0.6,
        state_transition_prob=0.02,
    ),
    Session.LONDON: SessionConfig(
        name="london",
        volume_multiplier=0.8,
        volatility_multiplier=1.1,
        state_transition_prob=0.05,
    ),
    Session.PREMARKET: SessionConfig(
        name="premarket",
        volume_multiplier=0.6,
        volatility_multiplier=0.9,
        state_transition_prob=0.08,
    ),
    Session.RTH: SessionConfig(
        name="rth",
        volume_multiplier=1.5,
        volatility_multiplier=1.3,
        state_transition_prob=0.06,
    ),
    Session.AFTERHOURS: SessionConfig(
        name="afterhours",
        volume_multiplier=0.5,
        volatility_multiplier=0.7,
        state_transition_prob=0.03,
    ),
}


@dataclass
class DayOfWeekConfig:
    """Day of week effects"""
    name: str
    volume_multiplier: float = 1.0
    volatility_multiplier: float = 1.0


DOW_CONFIGS = {
    0: DayOfWeekConfig("monday", volume_multiplier=1.1, volatility_multiplier=1.2),
    1: DayOfWeekConfig("tuesday", volume_multiplier=1.0, volatility_multiplier=1.0),
    2: DayOfWeekConfig("wednesday", volume_multiplier=1.0, volatility_multiplier=1.0),
    3: DayOfWeekConfig("thursday", volume_multiplier=1.0, volatility_multiplier=1.0),
    4: DayOfWeekConfig("friday", volume_multiplier=1.2, volatility_multiplier=1.1),
    5: DayOfWeekConfig("saturday", volume_multiplier=0.3, volatility_multiplier=0.5),
    6: DayOfWeekConfig("sunday", volume_multiplier=0.4, volatility_multiplier=0.6),
}


class PriceGenerator:
    """
    Generate synthetic MES price bars with configurable market dynamics.
    
    All ticks are 0.25 (MES tick size).
    """
    
    TICK_SIZE = 0.25
    
    def __init__(
        self,
        initial_price: float = 5000.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the price generator.
        
        Args:
            initial_price: Starting price level
            seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.current_price = initial_price
        self.last_direction = 0  # -1, 0, or 1
        self.prev_close_ticks = int(initial_price / self.TICK_SIZE)  # Track for delta_ticks
        
        if seed is not None:
            np.random.seed(seed)
        
        self.rng = np.random.default_rng(seed)
    
    def get_session(self, dt: datetime) -> Session:
        """Determine trading session based on time (Chicago time)"""
        hour = dt.hour
        
        if 18 <= hour or hour < 3:
            return Session.ASIAN
        elif 3 <= hour < 8 or (hour == 8 and dt.minute < 30):
            return Session.LONDON
        elif (hour == 8 and dt.minute >= 30) or hour == 9 and dt.minute < 30:
            return Session.PREMARKET
        elif (hour == 9 and dt.minute >= 30) or (10 <= hour < 15) or (hour == 15 and dt.minute <= 15):
            return Session.RTH
        else:
            return Session.AFTERHOURS
    
    def generate_tick_movement(
        self,
        state_config: StateConfig,
        session_config: SessionConfig,
        dow_config: DayOfWeekConfig,
    ) -> Tuple[int, int]:
        """
        Generate a single tick movement.
        
        Returns:
            (direction, num_ticks) where direction is -1 or 1, num_ticks is the size
        """
        # Determine direction
        if self.rng.random() < state_config.trend_persistence and self.last_direction != 0:
            # Continue previous direction
            direction = self.last_direction
        else:
            # New direction based on state bias
            direction = 1 if self.rng.random() < state_config.up_probability else -1
        
        # Determine tick size
        volatility = (
            state_config.volatility_multiplier *
            session_config.volatility_multiplier *
            dow_config.volatility_multiplier
        )
        
        tick_size = max(
            1,
            int(self.rng.normal(
                state_config.avg_tick_size * volatility,
                state_config.tick_size_std * volatility
            ))
        )
        tick_size = min(tick_size, state_config.max_tick_jump)
        
        self.last_direction = direction
        return direction, tick_size
    
    def generate_bar(
        self,
        timestamp: datetime,
        state: MarketState = MarketState.RANGING,
        custom_state_config: Optional[StateConfig] = None,
    ) -> dict:
        """
        Generate a single 1-minute OHLCV bar.
        
        Args:
            timestamp: Bar timestamp
            state: Market state to use
            custom_state_config: Optional custom state configuration (overrides state)
        
        Returns:
            Dictionary with keys: time, open, high, low, close, volume
        """
        # Get configurations
        state_config = custom_state_config or STATE_CONFIGS[state]
        session = self.get_session(timestamp)
        session_config = SESSION_CONFIGS[session]
        dow_config = DOW_CONFIGS[timestamp.weekday()]
        
        # Determine number of ticks for this bar
        num_ticks = max(
            1,
            int(self.rng.normal(
                state_config.avg_ticks_per_bar,
                state_config.ticks_per_bar_std
            ))
        )
        
        # Generate tick-by-tick price action
        open_price = self.current_price
        high_price = open_price
        low_price = open_price
        current = open_price
        
        for _ in range(num_ticks):
            direction, tick_size = self.generate_tick_movement(
                state_config, session_config, dow_config
            )
            
            # Move price
            price_change = direction * tick_size * self.TICK_SIZE
            current += price_change
            
            # Update high/low
            high_price = max(high_price, current)
            low_price = min(low_price, current)
        
        close_price = current
        
        # Add wicks (extended highs/lows that don't close there)
        if self.rng.random() < state_config.wick_probability:
            # Upper wick
            wick_ticks = max(1, int(self.rng.normal(
                state_config.wick_extension_avg,
                state_config.wick_extension_avg * 0.5
            )))
            high_price += wick_ticks * self.TICK_SIZE
        
        if self.rng.random() < state_config.wick_probability:
            # Lower wick
            wick_ticks = max(1, int(self.rng.normal(
                state_config.wick_extension_avg,
                state_config.wick_extension_avg * 0.5
            )))
            low_price -= wick_ticks * self.TICK_SIZE
        
        # Generate volume (scaled by session and day)
        base_volume = max(
            10,
            int(self.rng.normal(100, 50))
        )
        volume = int(
            base_volume *
            session_config.volume_multiplier *
            dow_config.volume_multiplier *
            state_config.volatility_multiplier
        )
        
        # Update current price for next bar
        self.current_price = close_price
        
        # Round prices to tick size
        open_price = round(open_price / self.TICK_SIZE) * self.TICK_SIZE
        high_price = round(high_price / self.TICK_SIZE) * self.TICK_SIZE
        low_price = round(low_price / self.TICK_SIZE) * self.TICK_SIZE
        close_price = round(close_price / self.TICK_SIZE) * self.TICK_SIZE
        
        # Convert to tick units (integer ticks from zero)
        open_ticks = int(open_price / self.TICK_SIZE)
        high_ticks = int(high_price / self.TICK_SIZE)
        low_ticks = int(low_price / self.TICK_SIZE)
        close_ticks = int(close_price / self.TICK_SIZE)
        
        # Compute tick-based deltas and features
        delta_ticks = close_ticks - self.prev_close_ticks
        range_ticks = high_ticks - low_ticks
        body_ticks = abs(close_ticks - open_ticks)
        
        # Wick calculations in ticks
        upper_body = max(open_ticks, close_ticks)
        lower_body = min(open_ticks, close_ticks)
        upper_wick_ticks = high_ticks - upper_body
        lower_wick_ticks = lower_body - low_ticks
        
        # Update prev_close for next bar
        self.prev_close_ticks = close_ticks
        
        return {
            'time': timestamp,
            # Price columns (floats)
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            # Tick columns (integers) - ML-friendly
            'open_ticks': open_ticks,
            'high_ticks': high_ticks,
            'low_ticks': low_ticks,
            'close_ticks': close_ticks,
            'delta_ticks': delta_ticks,
            'range_ticks': range_ticks,
            'body_ticks': body_ticks,
            'upper_wick_ticks': upper_wick_ticks,
            'lower_wick_ticks': lower_wick_ticks,
            # State labels
            'state': state.value,
            'session': session.value,
        }
    
    def generate_day(
        self,
        start_date: datetime,
        state_sequence: Optional[List[Tuple[int, MarketState]]] = None,
        auto_transition: bool = True,
        segment_length: Optional[int] = None,
        macro_regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a full day of 1-minute bars (1440 bars).
        
        Args:
            start_date: Starting datetime (should be midnight)
            state_sequence: Optional list of (bar_index, state) tuples to control states
            auto_transition: If True, randomly transition between states based on session
            segment_length: If set, only allow state transitions at segment boundaries (e.g., 15 for 15-min segments)
            macro_regime: Optional day-level regime label (e.g., 'UP_DAY', 'DOWN_DAY', 'CHOP_DAY')
        
        Returns:
            DataFrame with OHLCV, tick columns, state labels, and optional macro_regime
        """
        bars = []
        current_state = MarketState.RANGING
        current_segment_id = 0
        
        # Build state map if provided
        state_map = {}
        if state_sequence:
            for bar_idx, state in state_sequence:
                state_map[bar_idx] = state
        
        for minute in range(1440):  # 24 hours * 60 minutes
            timestamp = start_date + timedelta(minutes=minute)
            
            # Update segment ID if using segments
            if segment_length:
                current_segment_id = minute // segment_length
            
            # Check for manual state transition
            if minute in state_map:
                current_state = state_map[minute]
            elif auto_transition:
                # Only transition at segment boundaries if segment_length is set
                can_transition = True
                if segment_length:
                    can_transition = (minute % segment_length == 0)
                
                if can_transition:
                    # Random state transition based on session
                    session = self.get_session(timestamp)
                    session_config = SESSION_CONFIGS[session]
                    
                    if self.rng.random() < session_config.state_transition_prob:
                        # Transition to a new state
                        current_state = self.rng.choice(list(MarketState))
            
            bar = self.generate_bar(timestamp, current_state)
            
            # Add segment ID if using segments
            if segment_length:
                bar['segment_id'] = current_segment_id
            
            bars.append(bar)
        
        df = pd.DataFrame(bars)
        
        # Add macro regime label if provided
        if macro_regime:
            df['macro_regime'] = macro_regime
        else:
            # Infer simple macro regime from net movement
            net_move = df['close'].iloc[-1] - df['open'].iloc[0]
            total_range = df['high'].max() - df['low'].min()
            
            if abs(net_move) > total_range * 0.3:
                df['macro_regime'] = 'UP_DAY' if net_move > 0 else 'DOWN_DAY'
            elif total_range < df['close'].iloc[0] * 0.01:  # Less than 1% range
                df['macro_regime'] = 'QUIET_DAY'
            else:
                df['macro_regime'] = 'CHOP_DAY'
        
        return df
