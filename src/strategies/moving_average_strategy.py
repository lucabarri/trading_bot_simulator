import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base_strategy import BaseStrategy, TradingSignal, Signal


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy

    Strategy Logic:
    - BUY when short MA crosses above long MA
    - SELL when short MA crosses below long MA
    - HOLD otherwise
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 50,
        min_crossover_gap: float = 0.0,  # Minimum gap to avoid noise (0.0%)
    ):
        super().__init__(f"MA_Crossover_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.min_crossover_gap = min_crossover_gap

        # Validate parameters
        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window")
        if short_window < 2 or long_window < 2:
            raise ValueError("Moving average windows must be at least 2")

    def get_required_history(self) -> int:
        """Need at least long_window + 1 points to detect crossovers"""
        return self.long_window + 1

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short and long moving averages

        Args:
            data: DataFrame with OHLCV data

        Returns:
            New DataFrame with added MA columns
        """
        df = data.copy()

        # Calculate moving averages using closing prices
        df[f"MA_{self.short_window}"] = (
            df["Close"].rolling(window=self.short_window).mean()
        )
        df[f"MA_{self.long_window}"] = (
            df["Close"].rolling(window=self.long_window).mean()
        )

        # Calculate the difference (short MA - long MA)
        df["MA_diff"] = df[f"MA_{self.short_window}"] - df[f"MA_{self.long_window}"]

        # Calculate percentage difference for noise filtering
        df["MA_diff_pct"] = df["MA_diff"] / df[f"MA_{self.long_window}"]

        return df

    def detect_crossovers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect crossover events

        Args:
            data: DataFrame with MA columns

        Returns:
            DataFrame with crossover signals
        """
        df = data.copy()

        # Detect when short MA crosses above long MA (bullish crossover)
        df["bullish_crossover"] = (
            (df["MA_diff"].shift(1) <= 0)
            & (df["MA_diff"] > 0)
            & (abs(df["MA_diff_pct"]) > self.min_crossover_gap)
        )

        # Detect when short MA crosses below long MA (bearish crossover)
        df["bearish_crossover"] = (
            (df["MA_diff"].shift(1) >= 0)
            & (df["MA_diff"] < 0)
            & (abs(df["MA_diff_pct"]) > self.min_crossover_gap)
        )

        return df

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on moving average crossovers

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of TradingSignal objects
        """
        if len(data) < self.get_required_history():
            return []

        # Calculate moving averages and crossovers
        df = self.calculate_moving_averages(data)
        df = self.detect_crossovers(df)

        signals = []

        # Optimize: Only process rows with crossovers + always include the latest signal
        bullish_indices = df[df["bullish_crossover"] == True].index
        bearish_indices = df[df["bearish_crossover"] == True].index
        crossover_indices = bullish_indices.union(bearish_indices)

        # Always include the latest row for current state
        if len(df) > 0:
            crossover_indices = crossover_indices.union([df.index[-1]])

        # Process only relevant rows (vectorized where possible)
        for idx in crossover_indices:
            row = df.loc[idx]
            signal_type = Signal.HOLD
            confidence = 0.5  # Default confidence
            metadata = {
                "short_ma": row[f"MA_{self.short_window}"],
                "long_ma": row[f"MA_{self.long_window}"],
                "ma_diff": row["MA_diff"],
                "ma_diff_pct": row["MA_diff_pct"],
            }

            # Check for crossover signals
            if row["bullish_crossover"]:
                signal_type = Signal.BUY
                confidence = min(
                    1.0, abs(row["MA_diff_pct"]) * 10
                )  # Higher confidence for larger gaps
                metadata["crossover_type"] = "bullish"

            elif row["bearish_crossover"]:
                signal_type = Signal.SELL
                confidence = min(1.0, abs(row["MA_diff_pct"]) * 10)
                metadata["crossover_type"] = "bearish"

            # Only add non-HOLD signals or the latest signal
            if signal_type != Signal.HOLD or idx == df.index[-1]:
                signal = TradingSignal(
                    signal=signal_type,
                    timestamp=idx,
                    price=row["Close"],
                    confidence=confidence,
                    metadata=metadata,
                )
                signals.append(signal)

        # Store signals in the strategy
        self.signals.extend(signals)

        return signals

    def get_strategy_state(self, data: pd.DataFrame) -> Dict:
        """
        Get current strategy state and indicators

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with current strategy state
        """
        if len(data) < self.get_required_history():
            return {
                "status": "insufficient_data",
                "required_points": self.get_required_history(),
            }

        df = self.calculate_moving_averages(data)
        if len(df) == 0:
            return {
                "status": "no_data",
                "error": "No data available after MA calculations",
            }
        latest = df.iloc[-1]

        return {
            "status": "active",
            "latest_close": latest["Close"],
            "short_ma": latest[f"MA_{self.short_window}"],
            "long_ma": latest[f"MA_{self.long_window}"],
            "ma_diff": latest["MA_diff"],
            "ma_diff_pct": latest["MA_diff_pct"],
            "trend": "bullish" if latest["MA_diff"] > 0 else "bearish",
            "latest_signal": self.get_latest_signal(),
        }


# Convenience function to create a simple MA strategy
def create_simple_ma_strategy(
    short: int = 10, long: int = 50
) -> MovingAverageCrossoverStrategy:
    """
    Create a simple moving average crossover strategy

    Args:
        short: Short moving average period
        long: Long moving average period

    Returns:
        MovingAverageCrossoverStrategy instance
    """
    return MovingAverageCrossoverStrategy(short_window=short, long_window=long)
