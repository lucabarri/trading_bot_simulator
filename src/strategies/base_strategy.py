from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum


class Signal(Enum):
    """Trading signal types"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal:
    """
    Represents a trading signal with metadata
    """

    def __init__(
        self,
        signal: Signal,
        timestamp: pd.Timestamp,
        price: float,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        self.signal = signal
        self.timestamp = timestamp
        self.price = price
        self.confidence = confidence  # 0.0 to 1.0
        self.metadata = metadata or {}

    def __repr__(self):
        return f"TradingSignal({self.signal.value}, {self.timestamp}, ${self.price:.2f}, conf={self.confidence:.2f})"


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    """

    def __init__(self, name: str):
        self.name = name
        self.signals: List[TradingSignal] = []

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on market data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of TradingSignal objects
        """
        pass

    @abstractmethod
    def get_required_history(self) -> int:
        """
        Return the minimum number of data points required for the strategy

        Returns:
            Number of required historical data points
        """
        pass

    def reset(self):
        """Reset strategy state"""
        self.signals = []

    def get_latest_signal(self) -> Optional[TradingSignal]:
        """Get the most recent signal"""
        return self.signals[-1] if self.signals else None

    def get_signal_history(self) -> List[TradingSignal]:
        """Get all generated signals"""
        return self.signals.copy()
