"""
Fractal State Manager - Hierarchical market states across timeframes

Implements nested states where larger timeframes influence smaller ones:
- Day-level state (e.g., trending day, range day, breakout day)
- Hour-level states within the day state
- Minute-level states within the hour state

This creates realistic multi-timeframe market behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class DayState(Enum):
    """Day-level market states (highest timeframe)"""
    TREND_DAY = "trend_day"              # Strong directional day
    RANGE_DAY = "range_day"              # Choppy, bounded day
    BREAKOUT_DAY = "breakout_day"        # Breaks out of range
    REVERSAL_DAY = "reversal_day"        # V-shaped or inverse V
    QUIET_DAY = "quiet_day"              # Low volatility, minimal movement
    VOLATILE_DAY = "volatile_day"        # High volatility, no clear direction


class HourState(Enum):
    """Hour-level market states (medium timeframe)"""
    IMPULSE = "impulse"                  # Strong directional move
    CONSOLIDATION = "consolidation"      # Tight range
    RETRACEMENT = "retracement"          # Pullback within larger trend
    CONTINUATION = "continuation"        # Resuming previous direction
    REVERSAL = "reversal"                # Changing direction
    CHOPPY = "choppy"                    # No clear direction


class MinuteState(Enum):
    """Minute-level market states (lowest timeframe) - maps to existing MarketState"""
    RALLY = "rally"
    BREAKDOWN = "breakdown"
    RANGING = "ranging"
    FLAT = "flat"
    ZOMBIE = "zombie"
    IMPULSIVE = "impulsive"
    BREAKOUT = "breakout"


@dataclass
class FractalStateConfig:
    """Configuration for how states influence each other across timeframes"""
    
    # Required fields (no defaults)
    day_state: DayState
    hour_state: HourState
    minute_state: MinuteState
    
    # Optional fields (with defaults)
    day_directional_bias: float = 0.5    # 0=down, 0.5=neutral, 1=up
    day_volatility_mult: float = 1.0
    day_trend_strength: float = 0.5      # How strongly day state influences hours
    hour_directional_bias: float = 0.5
    hour_volatility_mult: float = 1.0
    hour_trend_strength: float = 0.5     # How strongly hour state influences minutes
    hour_transition_prob: float = 0.05   # Probability of hour state change per minute
    minute_transition_prob: float = 0.1  # Probability of minute state change per bar


# Define how day states influence hour state probabilities
DAY_TO_HOUR_TRANSITIONS: Dict[DayState, Dict[HourState, float]] = {
    DayState.TREND_DAY: {
        HourState.IMPULSE: 0.3,
        HourState.CONTINUATION: 0.3,
        HourState.RETRACEMENT: 0.2,
        HourState.CONSOLIDATION: 0.15,
        HourState.REVERSAL: 0.03,
        HourState.CHOPPY: 0.02,
    },
    DayState.RANGE_DAY: {
        HourState.CONSOLIDATION: 0.35,
        HourState.CHOPPY: 0.25,
        HourState.IMPULSE: 0.15,
        HourState.RETRACEMENT: 0.15,
        HourState.CONTINUATION: 0.05,
        HourState.REVERSAL: 0.05,
    },
    DayState.BREAKOUT_DAY: {
        HourState.IMPULSE: 0.4,
        HourState.CONTINUATION: 0.25,
        HourState.CONSOLIDATION: 0.2,
        HourState.RETRACEMENT: 0.1,
        HourState.CHOPPY: 0.03,
        HourState.REVERSAL: 0.02,
    },
    DayState.REVERSAL_DAY: {
        HourState.REVERSAL: 0.3,
        HourState.IMPULSE: 0.25,
        HourState.RETRACEMENT: 0.2,
        HourState.CONSOLIDATION: 0.15,
        HourState.CONTINUATION: 0.05,
        HourState.CHOPPY: 0.05,
    },
    DayState.QUIET_DAY: {
        HourState.CONSOLIDATION: 0.5,
        HourState.CHOPPY: 0.2,
        HourState.CONTINUATION: 0.15,
        HourState.IMPULSE: 0.1,
        HourState.RETRACEMENT: 0.03,
        HourState.REVERSAL: 0.02,
    },
    DayState.VOLATILE_DAY: {
        HourState.CHOPPY: 0.3,
        HourState.IMPULSE: 0.25,
        HourState.REVERSAL: 0.2,
        HourState.RETRACEMENT: 0.15,
        HourState.CONSOLIDATION: 0.05,
        HourState.CONTINUATION: 0.05,
    },
}


# Define how hour states influence minute state probabilities
HOUR_TO_MINUTE_TRANSITIONS: Dict[HourState, Dict[MinuteState, float]] = {
    HourState.IMPULSE: {
        MinuteState.RALLY: 0.4,
        MinuteState.BREAKOUT: 0.2,
        MinuteState.IMPULSIVE: 0.2,
        MinuteState.ZOMBIE: 0.1,
        MinuteState.RANGING: 0.05,
        MinuteState.FLAT: 0.03,
        MinuteState.BREAKDOWN: 0.02,
    },
    HourState.CONSOLIDATION: {
        MinuteState.RANGING: 0.4,
        MinuteState.FLAT: 0.3,
        MinuteState.ZOMBIE: 0.15,
        MinuteState.RALLY: 0.05,
        MinuteState.BREAKDOWN: 0.05,
        MinuteState.IMPULSIVE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
    HourState.RETRACEMENT: {
        MinuteState.BREAKDOWN: 0.3,
        MinuteState.RANGING: 0.25,
        MinuteState.ZOMBIE: 0.2,
        MinuteState.FLAT: 0.15,
        MinuteState.RALLY: 0.05,
        MinuteState.IMPULSIVE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
    HourState.CONTINUATION: {
        MinuteState.ZOMBIE: 0.35,
        MinuteState.RALLY: 0.25,
        MinuteState.RANGING: 0.2,
        MinuteState.FLAT: 0.1,
        MinuteState.IMPULSIVE: 0.05,
        MinuteState.BREAKOUT: 0.03,
        MinuteState.BREAKDOWN: 0.02,
    },
    HourState.REVERSAL: {
        MinuteState.IMPULSIVE: 0.3,
        MinuteState.BREAKDOWN: 0.25,
        MinuteState.RALLY: 0.2,
        MinuteState.RANGING: 0.15,
        MinuteState.BREAKOUT: 0.05,
        MinuteState.ZOMBIE: 0.03,
        MinuteState.FLAT: 0.02,
    },
    HourState.CHOPPY: {
        MinuteState.RANGING: 0.35,
        MinuteState.IMPULSIVE: 0.25,
        MinuteState.RALLY: 0.15,
        MinuteState.BREAKDOWN: 0.15,
        MinuteState.FLAT: 0.05,
        MinuteState.ZOMBIE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
}


# Day state characteristics
DAY_STATE_PARAMS = {
    DayState.TREND_DAY: {
        'directional_bias': 0.65,  # Upward bias (can be flipped for down trend)
        'volatility_mult': 1.3,
        'trend_strength': 0.8,
    },
    DayState.RANGE_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 0.9,
        'trend_strength': 0.3,
    },
    DayState.BREAKOUT_DAY: {
        'directional_bias': 0.7,
        'volatility_mult': 1.6,
        'trend_strength': 0.85,
    },
    DayState.REVERSAL_DAY: {
        'directional_bias': 0.5,  # Changes during day
        'volatility_mult': 1.4,
        'trend_strength': 0.6,
    },
    DayState.QUIET_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 0.5,
        'trend_strength': 0.2,
    },
    DayState.VOLATILE_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 2.0,
        'trend_strength': 0.4,
    },
}


# Hour state characteristics
HOUR_STATE_PARAMS = {
    HourState.IMPULSE: {
        'directional_bias': 0.7,
        'volatility_mult': 1.5,
        'trend_strength': 0.8,
    },
    HourState.CONSOLIDATION: {
        'directional_bias': 0.5,
        'volatility_mult': 0.6,
        'trend_strength': 0.3,
    },
    HourState.RETRACEMENT: {
        'directional_bias': 0.35,  # Against main trend
        'volatility_mult': 1.0,
        'trend_strength': 0.6,
    },
    HourState.CONTINUATION: {
        'directional_bias': 0.6,
        'volatility_mult': 1.1,
        'trend_strength': 0.7,
    },
    HourState.REVERSAL: {
        'directional_bias': 0.5,  # Flips during hour
        'volatility_mult': 1.4,
        'trend_strength': 0.7,
    },
    HourState.CHOPPY: {
        'directional_bias': 0.5,
        'volatility_mult': 1.2,
        'trend_strength': 0.2,
    },
}


class FractalStateManager:
    """
    Manages hierarchical market states across timeframes.
    
    Day state influences hour states, which influence minute states.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed"""
        self.rng = np.random.default_rng(seed)
        
        # Current states
        self.day_state: Optional[DayState] = None
        self.hour_state: Optional[HourState] = None
        self.minute_state: Optional[MinuteState] = None
        
        # State parameters
        self.day_params: Dict = {}
        self.hour_params: Dict = {}
        
        # Tracking
        self.current_hour_start: Optional[datetime] = None
        self.bars_in_current_hour: int = 0
    
    def initialize_day(self, day_state: Optional[DayState] = None) -> DayState:
        """
        Initialize a new day with a day-level state.
        
        Args:
            day_state: Specific day state, or None for random
        
        Returns:
            The selected day state
        """
        if day_state is None:
            # Random day state
            day_state = self.rng.choice(list(DayState))
        
        self.day_state = day_state
        self.day_params = DAY_STATE_PARAMS[day_state].copy()
        
        # Initialize first hour state
        self.hour_state = self._transition_hour_state()
        self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
        
        # Initialize first minute state
        self.minute_state = self._transition_minute_state()
        
        return day_state
    
    def _transition_hour_state(self) -> HourState:
        """Transition to a new hour state based on day state"""
        if self.day_state is None:
            return HourState.CONSOLIDATION
        
        # Get transition probabilities from day state
        probs = DAY_TO_HOUR_TRANSITIONS[self.day_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Choose new state
        new_state = self.rng.choice(states, p=probabilities)
        return new_state
    
    def _transition_minute_state(self) -> MinuteState:
        """Transition to a new minute state based on hour state"""
        if self.hour_state is None:
            return MinuteState.RANGING
        
        # Get transition probabilities from hour state
        probs = HOUR_TO_MINUTE_TRANSITIONS[self.hour_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Choose new state
        new_state = self.rng.choice(states, p=probabilities)
        return new_state
    
    def update(self, timestamp: datetime, force_hour_transition: bool = False) -> Tuple[DayState, HourState, MinuteState]:
        """
        Update states for a new bar.
        
        Args:
            timestamp: Current bar timestamp
            force_hour_transition: Force an hour state transition
        
        Returns:
            (day_state, hour_state, minute_state)
        """
        if self.day_state is None:
            self.initialize_day()
        
        # Track hour boundaries
        if self.current_hour_start is None:
            self.current_hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
            self.bars_in_current_hour = 0
        
        # Check if we've entered a new hour
        current_hour = timestamp.replace(minute=0, second=0, microsecond=0)
        if current_hour != self.current_hour_start or force_hour_transition:
            # New hour - higher chance of hour state transition
            if self.rng.random() < 0.3 or force_hour_transition:  # 30% chance on hour boundary
                self.hour_state = self._transition_hour_state()
                self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
            
            self.current_hour_start = current_hour
            self.bars_in_current_hour = 0
        
        # Check for hour state transition (can happen mid-hour)
        elif self.rng.random() < 0.02:  # 2% chance per bar mid-hour
            self.hour_state = self._transition_hour_state()
            self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
        
        # Check for minute state transition
        if self.rng.random() < 0.1:  # 10% chance per bar
            self.minute_state = self._transition_minute_state()
        
        self.bars_in_current_hour += 1
        
        return self.day_state, self.hour_state, self.minute_state
    
    def get_combined_parameters(self) -> Dict:
        """
        Get combined parameters from all timeframe states.
        
        Returns:
            Dictionary with combined directional_bias, volatility_mult, trend_strength
        """
        # Combine day and hour parameters
        day_bias = self.day_params.get('directional_bias', 0.5)
        hour_bias = self.hour_params.get('directional_bias', 0.5)
        
        # Weight by trend strength
        day_strength = self.day_params.get('trend_strength', 0.5)
        hour_strength = self.hour_params.get('trend_strength', 0.5)
        
        # Combined bias (weighted average)
        combined_bias = (
            day_bias * day_strength * 0.3 +
            hour_bias * hour_strength * 0.7
        )
        
        # Combined volatility (multiplicative)
        combined_volatility = (
            self.day_params.get('volatility_mult', 1.0) *
            self.hour_params.get('volatility_mult', 1.0)
        )
        
        # Combined trend strength (average)
        combined_trend_strength = (
            day_strength * 0.4 +
            hour_strength * 0.6
        )
        
        return {
            'directional_bias': combined_bias,
            'volatility_mult': combined_volatility,
            'trend_strength': combined_trend_strength,
            'day_state': self.day_state.value if self.day_state else None,
            'hour_state': self.hour_state.value if self.hour_state else None,
            'minute_state': self.minute_state.value if self.minute_state else None,
        }
