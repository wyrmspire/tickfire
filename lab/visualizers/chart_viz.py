"""
Flexible Candlestick Chart Visualizer

Configurable matplotlib charting with various display options and knobs.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass


@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior"""
    
    # Figure settings
    figsize: Tuple[int, int] = (16, 9)
    dpi: int = 100
    
    # Candle appearance
    candle_width: float = 0.6
    color_up: str = '#26a69a'      # Teal green
    color_down: str = '#ef5350'    # Red
    wick_linewidth: float = 1.0
    wick_alpha: float = 0.8
    
    # Grid and background
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_color: str = '#cccccc'
    background_color: str = '#ffffff'
    
    # Volume subplot
    show_volume: bool = True
    volume_height_ratio: float = 0.25  # Relative to price chart
    volume_alpha: float = 0.5
    
    # Title and labels
    title: str = "MES 1-Minute Chart"
    title_fontsize: int = 14
    ylabel_price: str = "Price"
    ylabel_volume: str = "Volume"
    label_fontsize: int = 11
    
    # X-axis formatting
    date_format: str = '%H:%M'
    major_tick_interval_minutes: Optional[int] = 60  # None for auto
    
    # Annotations
    show_state_changes: bool = True
    state_change_color: str = '#ff9800'
    state_change_alpha: float = 0.3
    
    show_session_changes: bool = True
    session_colors: dict = None
    
    # Legend
    show_legend: bool = True
    legend_loc: str = 'upper left'
    
    def __post_init__(self):
        if self.session_colors is None:
            self.session_colors = {
                'asian': '#9c27b0',
                'london': '#2196f3',
                'premarket': '#ff9800',
                'rth': '#4caf50',
                'afterhours': '#795548',
            }


class ChartVisualizer:
    """
    Flexible candlestick chart creator with multiple display options.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: ChartConfig instance, uses defaults if None
        """
        self.config = config or ChartConfig()
    
    def plot_candlestick(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Plot candlestick chart on given axes.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with OHLC data
            date_column: Name of the datetime column
        """
        # Ensure datetime index
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        # Convert to matplotlib date format
        dates_num = mdates.date2num(dates)
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Determine colors
        colors = [
            self.config.color_up if c >= o else self.config.color_down
            for o, c in zip(opens, closes)
        ]
        
        # Calculate candle width
        if len(dates_num) > 1:
            min_diff = np.min(np.diff(dates_num))
            width = self.config.candle_width * min_diff
        else:
            width = self.config.candle_width * 1.0 / 1440.0
        
        # Plot wicks
        ax.vlines(
            dates_num,
            lows,
            highs,
            colors=colors,
            linewidth=self.config.wick_linewidth,
            alpha=self.config.wick_alpha,
        )
        
        # Plot bodies
        for d, o, c, color in zip(dates_num, opens, closes, colors):
            lower = min(o, c)
            height = abs(c - o)
            if height == 0:
                height = 0.01  # Tiny height for doji
            
            ax.add_patch(
                plt.Rectangle(
                    (d - width / 2, lower),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5,
                )
            )
        
        # Format x-axis
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(self.config.date_format))
        
        if self.config.major_tick_interval_minutes:
            ax.xaxis.set_major_locator(
                mdates.MinuteLocator(interval=self.config.major_tick_interval_minutes)
            )
        
        ax.autoscale_view()
    
    def plot_volume(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Plot volume bars on given axes.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with volume data
            date_column: Name of the datetime column
        """
        # Ensure datetime index
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        dates_num = mdates.date2num(dates)
        volumes = df['volume'].values
        
        # Color based on price movement
        opens = df['open'].values
        closes = df['close'].values
        colors = [
            self.config.color_up if c >= o else self.config.color_down
            for o, c in zip(opens, closes)
        ]
        
        # Calculate bar width
        if len(dates_num) > 1:
            min_diff = np.min(np.diff(dates_num))
            width = self.config.candle_width * min_diff
        else:
            width = self.config.candle_width * 1.0 / 1440.0
        
        # Plot bars
        ax.bar(
            dates_num,
            volumes,
            width=width,
            color=colors,
            alpha=self.config.volume_alpha,
        )
        
        # Format x-axis
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(self.config.date_format))
        
        if self.config.major_tick_interval_minutes:
            ax.xaxis.set_major_locator(
                mdates.MinuteLocator(interval=self.config.major_tick_interval_minutes)
            )
        
        ax.autoscale_view()
    
    def add_state_annotations(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Add visual indicators for state changes.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with 'state' column
            date_column: Name of the datetime column
        """
        if 'state' not in df.columns:
            return
        
        # Ensure datetime
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        dates_num = mdates.date2num(dates)
        
        # Find state changes
        states = df['state'].values
        changes = []
        
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                changes.append((i, states[i]))
        
        # Draw vertical lines at state changes
        for idx, new_state in changes:
            ax.axvline(
                x=dates_num[idx],
                color=self.config.state_change_color,
                linestyle='--',
                alpha=self.config.state_change_alpha,
                linewidth=1.5,
                label=f'→ {new_state}' if idx == changes[0][0] else '',
            )
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(
                dates_num[idx],
                y_pos,
                new_state.upper(),
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                alpha=0.7,
                color=self.config.state_change_color,
            )
    
    def add_session_backgrounds(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        date_column: str = 'time',
    ) -> None:
        """
        Add colored backgrounds for different trading sessions.
        
        Args:
            ax: Matplotlib axes
            df: DataFrame with 'session' column
            date_column: Name of the datetime column
        """
        if 'session' not in df.columns:
            return
        
        # Ensure datetime
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        dates_num = mdates.date2num(dates)
        sessions = df['session'].values
        
        # Find session blocks
        current_session = sessions[0]
        start_idx = 0
        
        for i in range(1, len(sessions)):
            if sessions[i] != current_session:
                # Draw background for previous session
                color = self.config.session_colors.get(current_session, '#cccccc')
                ax.axvspan(
                    dates_num[start_idx],
                    dates_num[i-1],
                    alpha=0.1,
                    color=color,
                    label=current_session.upper() if start_idx == 0 else '',
                )
                
                current_session = sessions[i]
                start_idx = i
        
        # Draw last session
        color = self.config.session_colors.get(current_session, '#cccccc')
        ax.axvspan(
            dates_num[start_idx],
            dates_num[-1],
            alpha=0.1,
            color=color,
            label=current_session.upper() if start_idx == 0 else '',
        )
    
    def create_chart(
        self,
        df: pd.DataFrame,
        date_column: str = 'time',
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Create complete chart with all configured elements.
        
        Args:
            df: DataFrame with OHLCV data
            date_column: Name of the datetime column
            save_path: Optional path to save the chart
            show: Whether to display the chart
        
        Returns:
            Matplotlib figure
        """
        # Create figure and axes
        if self.config.show_volume:
            fig, (ax_price, ax_volume) = plt.subplots(
                2, 1,
                figsize=self.config.figsize,
                dpi=self.config.dpi,
                gridspec_kw={'height_ratios': [1 - self.config.volume_height_ratio, self.config.volume_height_ratio]},
                sharex=True,
            )
        else:
            fig, ax_price = plt.subplots(
                figsize=self.config.figsize,
                dpi=self.config.dpi,
            )
            ax_volume = None
        
        # Set background
        fig.patch.set_facecolor(self.config.background_color)
        ax_price.set_facecolor(self.config.background_color)
        if ax_volume:
            ax_volume.set_facecolor(self.config.background_color)
        
        # Add session backgrounds first (so they're behind everything)
        if self.config.show_session_changes:
            self.add_session_backgrounds(ax_price, df, date_column)
        
        # Plot candlesticks
        self.plot_candlestick(ax_price, df, date_column)
        
        # Add state change annotations
        if self.config.show_state_changes:
            self.add_state_annotations(ax_price, df, date_column)
        
        # Plot volume
        if self.config.show_volume and ax_volume:
            self.plot_volume(ax_volume, df, date_column)
            ax_volume.set_ylabel(self.config.ylabel_volume, fontsize=self.config.label_fontsize)
            
            if self.config.show_grid:
                ax_volume.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)
        
        # Configure price axis
        ax_price.set_title(self.config.title, fontsize=self.config.title_fontsize, pad=15)
        ax_price.set_ylabel(self.config.ylabel_price, fontsize=self.config.label_fontsize)
        
        if self.config.show_grid:
            ax_price.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)
        
        if self.config.show_legend:
            ax_price.legend(loc=self.config.legend_loc, fontsize=9)
        
        # Format dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig


def quick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    save_path: Optional[Path] = None,
    show: bool = True,
    **config_kwargs,
) -> plt.Figure:
    """
    Quick helper to create a chart with minimal setup.
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        save_path: Optional save path
        show: Whether to display
        **config_kwargs: Additional ChartConfig parameters
    
    Returns:
        Matplotlib figure
    """
    config = ChartConfig(title=title, **config_kwargs)
    viz = ChartVisualizer(config)
    return viz.create_chart(df, save_path=save_path, show=show)
