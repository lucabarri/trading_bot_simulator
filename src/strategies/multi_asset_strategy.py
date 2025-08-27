from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import external modules with fallback for tests
try:
    from .base_strategy import BaseStrategy, TradingSignal, Signal
    from .moving_average_strategy import create_simple_ma_strategy
    from ..data.data_fetcher import DataFetcher
    from ..utils.logger import get_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from strategies.base_strategy import BaseStrategy, TradingSignal, Signal
    from strategies.moving_average_strategy import create_simple_ma_strategy
    from data.data_fetcher import DataFetcher
    from utils.logger import get_logger


@dataclass
class AssetConfig:
    """Configuration for individual asset in multi-asset strategy"""
    symbol: str
    short_ma: int
    long_ma: int
    volatility: float
    allocation_pct: float


class MultiAssetMAStrategy(BaseStrategy):
    """
    Multi-asset moving average strategy with volatility-based allocation
    
    Features:
    - Independent MA strategies for each asset
    - Volatility-based allocation (lower volatility = higher allocation)
    - Shared cash pool management
    - Sequential parameter optimization
    """
    
    def __init__(
        self, 
        asset_configs: Dict[str, AssetConfig],
        rebalance_frequency: int = 252  # Rebalance allocation yearly
    ):
        """
        Initialize multi-asset strategy
        
        Args:
            asset_configs: Dict of symbol -> AssetConfig
            rebalance_frequency: How often to rebalance allocations (in trading days)
        """
        super().__init__(name="MultiAsset_MA_Strategy")
        self.logger = get_logger(__name__)
        
        self.asset_configs = asset_configs
        self.symbols = list(asset_configs.keys())
        self.rebalance_frequency = rebalance_frequency
        
        # Create individual strategies for each asset
        self.strategies = {}
        for symbol, config in asset_configs.items():
            strategy = create_simple_ma_strategy(
                short=config.short_ma, 
                long=config.long_ma
            )
            strategy.name = f"MA_{config.short_ma}_{config.long_ma}_{symbol}"
            self.strategies[symbol] = strategy
        
        # Track when we last rebalanced
        self.last_rebalance_date = None
        self.days_since_rebalance = 0
        
    def get_required_history(self) -> int:
        """Return maximum lookback period needed across all strategies"""
        return max(strategy.get_required_history() for strategy in self.strategies.values())
    
    def calculate_volatility_allocations(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        lookback_days: int = 60
    ) -> Dict[str, float]:
        """
        Calculate volatility-based allocations
        
        Args:
            data_dict: Dict of symbol -> price data
            lookback_days: Days to calculate volatility over
            
        Returns:
            Dict of symbol -> allocation percentage
        """
        volatilities = {}
        
        # Calculate volatility for each asset
        for symbol in self.symbols:
            if symbol in data_dict and len(data_dict[symbol]) >= lookback_days:
                returns = data_dict[symbol]['Close'].pct_change().dropna()
                if len(returns) >= lookback_days:
                    vol = returns.tail(lookback_days).std() * np.sqrt(252)  # Annualized
                    volatilities[symbol] = vol
                else:
                    # Fallback to shorter period
                    vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
                    volatilities[symbol] = vol
            else:
                # Default volatility if no data
                volatilities[symbol] = 0.2  # 20% default
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        for symbol, vol in volatilities.items():
            # Use inverse volatility, but avoid division by zero
            inv_vol_weights[symbol] = 1.0 / max(vol, 0.01)  # Minimum 1% volatility
        
        # Normalize to sum to 1.0
        total_weight = sum(inv_vol_weights.values())
        allocations = {
            symbol: weight / total_weight 
            for symbol, weight in inv_vol_weights.items()
        }
        
        return allocations
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate signals for multi-asset strategy
        
        Note: This expects data to be a multi-symbol DataFrame or will fetch data for each symbol
        """
        signals = []
        
        # If single DataFrame passed, we need to fetch individual symbol data
        if isinstance(data, pd.DataFrame) and len(self.symbols) > 1:
            # This is likely a single-symbol DataFrame, fetch data for each symbol
            fetcher = DataFetcher()
            data_dict = {}
            
            for symbol in self.symbols:
                try:
                    # Use unified fetch_market_data with auto-detection
                    symbol_data = fetcher.fetch_market_data(symbol, asset_type="auto", period="2y", interval="1d")
                    data_dict[symbol] = symbol_data
                except Exception as e:
                    self.logger.warning(f"Could not fetch data for {symbol}: {e}")
                    continue
        else:
            # Assume data is already organized per symbol
            data_dict = {symbol: data for symbol in self.symbols}
        
        # Check if we need to rebalance allocations
        current_date = data.index[-1] if hasattr(data, 'index') else pd.Timestamp.now()
        
        if (self.last_rebalance_date is None or 
            self.days_since_rebalance >= self.rebalance_frequency):
            
            # Recalculate volatility-based allocations
            new_allocations = self.calculate_volatility_allocations(data_dict)
            
            # Update asset configs with new allocations
            for symbol, allocation in new_allocations.items():
                if symbol in self.asset_configs:
                    self.asset_configs[symbol].allocation_pct = allocation
            
            self.last_rebalance_date = current_date
            self.days_since_rebalance = 0
        else:
            self.days_since_rebalance += 1
        
        # Generate signals for each asset independently
        for symbol in self.symbols:
            if symbol not in data_dict:
                continue
                
            try:
                symbol_data = data_dict[symbol]
                strategy = self.strategies[symbol]
                config = self.asset_configs[symbol]
                
                # Generate signals for this symbol
                symbol_signals = strategy.generate_signals(symbol_data)
                
                # Modify signals to include allocation and symbol info
                for signal in symbol_signals:
                    # Create new signal with allocation information
                    modified_signal = TradingSignal(
                        signal=signal.signal,
                        timestamp=signal.timestamp,
                        price=signal.price,
                        confidence=signal.confidence,
                        metadata={
                            **signal.metadata,
                            'symbol': symbol,  # Store symbol in metadata
                            'allocation_pct': config.allocation_pct,
                            'volatility': config.volatility,
                            'strategy_name': strategy.name
                        }
                    )
                    signals.append(modified_signal)
                    
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        return signals
    
    def reset(self):
        """Reset all individual strategies"""
        for strategy in self.strategies.values():
            strategy.reset()
        self.last_rebalance_date = None
        self.days_since_rebalance = 0
    
    def get_allocation_summary(self) -> Dict[str, Dict]:
        """Get current allocation summary"""
        summary = {}
        for symbol, config in self.asset_configs.items():
            summary[symbol] = {
                'allocation_pct': config.allocation_pct,
                'volatility': config.volatility,
                'short_ma': config.short_ma,
                'long_ma': config.long_ma
            }
        return summary
    
    def __repr__(self):
        return f"MultiAssetMAStrategy({len(self.symbols)} assets: {', '.join(self.symbols)})"


def create_multi_asset_strategy(
    symbols: List[str],
    optimized_params: Dict[str, Dict[str, int]] = None,
    default_short: int = 10,
    default_long: int = 20
) -> MultiAssetMAStrategy:
    """
    Create a multi-asset strategy with given symbols and parameters
    
    Args:
        symbols: List of symbols to trade
        optimized_params: Dict of symbol -> {"short": int, "long": int}
        default_short: Default short MA period
        default_long: Default long MA period
        
    Returns:
        MultiAssetMAStrategy instance
    """
    # Calculate volatilities for each symbol (simplified for now)
    fetcher = DataFetcher()
    asset_configs = {}
    
    logger = get_logger(__name__)
    logger.info(f"Creating multi-asset strategy for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            # Get parameters for this symbol
            if optimized_params and symbol in optimized_params:
                short_ma = optimized_params[symbol]['short']
                long_ma = optimized_params[symbol]['long']
            else:
                short_ma = default_short
                long_ma = default_long
            
            # Calculate volatility (simplified - using recent data)
            try:
                # Use unified fetch_market_data with auto-detection
                data = fetcher.fetch_market_data(symbol, asset_type="auto", period="6mo", interval="1d")
                if len(data) < 30:  # Need minimum data for reliable volatility
                    logger.warning(f"Insufficient data for {symbol} volatility calculation ({len(data)} days)")
                    volatility = 0.20  # Default fallback
                else:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        # Sanity check - volatility should be reasonable
                        if volatility <= 0 or volatility > 5.0:  # 0% to 500% seems reasonable
                            logger.warning(f"Extreme volatility calculated for {symbol}: {volatility:.1%}, using default")
                            volatility = 0.20
                    else:
                        logger.warning(f"No price returns available for {symbol}")
                        volatility = 0.20
            except (ValueError, KeyError, ConnectionError) as e:
                logger.warning(f"Failed to calculate volatility for {symbol}: {e}")
                volatility = 0.20  # Default 20% volatility
            except Exception as e:
                logger.error(f"Unexpected error calculating volatility for {symbol}: {e}")
                volatility = 0.20  # Default fallback
            
            # Create asset config (allocation will be calculated later)
            config = AssetConfig(
                symbol=symbol,
                short_ma=short_ma,
                long_ma=long_ma,
                volatility=volatility,
                allocation_pct=1.0 / len(symbols)  # Equal weight initially
            )
            
            asset_configs[symbol] = config
            logger.info(f"  {symbol}: MA({short_ma},{long_ma}), Vol={volatility:.2%}")
            
        except Exception as e:
            logger.warning(f"Skipping {symbol} due to error: {e}")
            continue
    
    if not asset_configs:
        raise ValueError("No valid assets could be configured")
    
    return MultiAssetMAStrategy(asset_configs)