from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import random

# Import data fetcher with fallback for tests
try:
    from ..data.data_fetcher import DataFetcher
    from ..utils.logger import get_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.data_fetcher import DataFetcher
    from utils.logger import get_logger


class MarketSimulator:
    """
    Simulates realistic market conditions for paper trading
    """
    
    def __init__(
        self,
        bid_ask_spread_bps: float = 5.0,  # 5 basis points spread
        volatility_factor: float = 1.0,   # Volatility multiplier
        liquidity_impact_bps: float = 1.0,  # Price impact per $10k trade
        market_hours_only: bool = True,    # Only allow trades during market hours
        weekend_gaps: bool = True          # Simulate weekend price gaps
    ):
        """
        Initialize market simulator
        
        Args:
            bid_ask_spread_bps: Bid-ask spread in basis points
            volatility_factor: Multiplier for price volatility
            liquidity_impact_bps: Price impact per $10k trade size
            market_hours_only: Whether to enforce market hours
            weekend_gaps: Whether to simulate weekend gaps
        """
        self.logger = get_logger(__name__)
        self.bid_ask_spread_bps = bid_ask_spread_bps / 10000.0
        self.volatility_factor = volatility_factor
        self.liquidity_impact_bps = liquidity_impact_bps / 10000.0
        self.market_hours_only = market_hours_only
        self.weekend_gaps = weekend_gaps
        
        # Market state
        self.current_prices: Dict[str, float] = {}
        self.last_update: Optional[pd.Timestamp] = None
        self.market_data: Dict[str, pd.DataFrame] = {}
    
    def load_market_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Load historical market data for simulation
        
        Args:
            symbols: List of stock symbols
            period: Data period to load
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        fetcher = DataFetcher()
        
        for symbol in symbols:
            try:
                data = fetcher.fetch_stock_data(symbol, period=period, interval="1d")
                if len(data) == 0:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                self.market_data[symbol] = data
                self.current_prices[symbol] = data['Close'].iloc[-1]
                self.logger.info(f"Loaded market data for {symbol}: {len(data)} days")
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
        
        self.last_update = pd.Timestamp.now(tz=timezone.utc)
        return self.market_data
    
    def is_market_open(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if market is open at given timestamp
        
        Args:
            timestamp: UTC timestamp to check
            
        Returns:
            True if market is open
        """
        if not self.market_hours_only:
            return True
        
        # Ensure timestamp is UTC
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        
        # Convert UTC to Eastern Time for market hours check
        et_time = timestamp.tz_convert('America/New_York')
        
        # Simple market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
        weekday = et_time.weekday()  # 0 = Monday, 6 = Sunday
        
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return False
        
        # Market hours check in Eastern Time
        hour = et_time.hour
        minute = et_time.minute
        
        # Market opens at 9:30 AM ET
        if hour < 9 or (hour == 9 and minute < 30):
            return False
        
        # Market closes at 4:00 PM ET
        if hour >= 16:
            return False
        
        return True
    
    def simulate_bid_ask_spread(self, price: float) -> Tuple[float, float]:
        """
        Calculate bid and ask prices with spread
        
        Args:
            price: Mid price
            
        Returns:
            (bid_price, ask_price)
        """
        spread = price * self.bid_ask_spread_bps
        half_spread = spread / 2
        
        bid = price - half_spread
        ask = price + half_spread
        
        return bid, ask
    
    def calculate_liquidity_impact(self, symbol: str, trade_value: float, action: str) -> float:
        """
        Calculate price impact based on trade size
        
        Args:
            symbol: Stock symbol
            trade_value: Dollar value of trade
            action: BUY or SELL
            
        Returns:
            Price impact factor (multiplier)
        """
        # Impact scales with trade size
        impact_units = trade_value / 10000.0  # Per $10k
        impact = impact_units * self.liquidity_impact_bps
        
        # Cap maximum impact at 1%
        impact = min(impact, 0.01)
        
        # Buy orders push price up, sell orders push price down
        if action == "BUY":
            return 1 + impact
        else:  # SELL
            return 1 - impact
    
    def simulate_intraday_volatility(self, base_price: float, symbol: str) -> float:
        """
        Add realistic intraday price volatility
        
        Args:
            base_price: Base price to add volatility to
            symbol: Stock symbol (for symbol-specific volatility)
            
        Returns:
            Price with volatility added
        """
        if symbol not in self.market_data:
            return base_price
        
        # Calculate historical volatility
        data = self.market_data[symbol]
        if len(data) < 20:
            daily_volatility = 0.02  # Default 2% daily volatility
        else:
            returns = data['Close'].pct_change().dropna()
            daily_volatility = returns.std()
        
        # Scale volatility for intraday (rough approximation)
        intraday_volatility = daily_volatility * self.volatility_factor * 0.3
        
        # Generate random price movement
        random_factor = np.random.normal(0, intraday_volatility)
        
        return base_price * (1 + random_factor)
    
    def get_market_price(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        action: str,
        trade_value: float = 0.0
    ) -> Optional[float]:
        """
        Get realistic market price for execution
        
        Args:
            symbol: Stock symbol
            timestamp: UTC execution timestamp
            action: BUY or SELL
            trade_value: Dollar value of trade
            
        Returns:
            Execution price or None if market is closed
        """
        # Ensure timestamp is UTC
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        
        if self.market_hours_only and not self.is_market_open(timestamp):
            return None
        
        if symbol not in self.current_prices:
            return None
        
        base_price = self.current_prices[symbol]
        
        # Add intraday volatility
        volatile_price = self.simulate_intraday_volatility(base_price, symbol)
        
        # Apply liquidity impact
        impact_factor = self.calculate_liquidity_impact(symbol, trade_value, action)
        impacted_price = volatile_price * impact_factor
        
        # Apply bid-ask spread
        bid, ask = self.simulate_bid_ask_spread(impacted_price)
        
        # Return appropriate side of spread
        if action == "BUY":
            return ask  # Buyers pay the ask
        else:  # SELL
            return bid  # Sellers receive the bid
    
    def update_prices(self, timestamp: pd.Timestamp):
        """
        Update current prices based on timestamp
        
        Args:
            timestamp: Current UTC timestamp
        """
        # Ensure timestamp is UTC
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        
        if self.last_update is None:
            self.last_update = timestamp
            return
        
        time_diff = timestamp - self.last_update
        
        # If it's a new day, simulate some price movement
        if time_diff.total_seconds() > 86400:  # More than 24 hours
            for symbol in self.current_prices:
                # Add some random daily movement
                daily_change = np.random.normal(0, 0.015)  # 1.5% daily volatility
                self.current_prices[symbol] *= (1 + daily_change)
                
                # Ensure price stays positive
                self.current_prices[symbol] = max(self.current_prices[symbol], 0.01)
        
        self.last_update = timestamp
    
    def simulate_weekend_gap(self, symbol: str) -> float:
        """
        Simulate weekend price gap
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Gap factor (multiplier)
        """
        if not self.weekend_gaps:
            return 1.0
        
        # Weekend gaps are typically small but can be significant
        # 70% chance of no gap, 30% chance of gap
        if random.random() < 0.7:
            return 1.0
        
        # Gap can be positive or negative, typically 0.5-3%
        gap_size = np.random.uniform(-0.03, 0.03)
        return 1 + gap_size
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current market prices for all symbols"""
        return self.current_prices.copy()
    
    def set_price(self, symbol: str, price: float):
        """Manually set price for a symbol (for testing)"""
        self.current_prices[symbol] = price
    
    def get_market_summary(self) -> Dict:
        """Get summary of market simulation parameters"""
        return {
            'symbols': list(self.current_prices.keys()),
            'current_prices': self.current_prices.copy(),
            'bid_ask_spread_bps': self.bid_ask_spread_bps * 10000,
            'volatility_factor': self.volatility_factor,
            'liquidity_impact_bps': self.liquidity_impact_bps * 10000,
            'market_hours_only': self.market_hours_only,
            'weekend_gaps': self.weekend_gaps,
            'last_update': self.last_update
        }
    
    def __repr__(self):
        return f"MarketSimulator(symbols={len(self.current_prices)}, spread={self.bid_ask_spread_bps*10000:.1f}bps)"