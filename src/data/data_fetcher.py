import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal

# Import logger utilities
try:
    from ..utils.logger import get_logger, log_data_fetch, log_error, log_warning
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger, log_data_fetch, log_error, log_warning


class DataFetcher:
    """
    Handles fetching and caching market data from multiple sources
    Supports stocks, crypto, and forex
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.logger = get_logger(__name__)
        os.makedirs(data_dir, exist_ok=True)
        
        # Define common crypto symbols for yfinance (with -USD suffix)
        self.crypto_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD', 
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'MATIC': 'MATIC-USD',
            'DOT': 'DOT-USD',
            'AVAX': 'AVAX-USD',
            'LINK': 'LINK-USD',
            'UNI': 'UNI-USD',
            'ATOM': 'ATOM-USD',
            'LTC': 'LTC-USD',
            'BCH': 'BCH-USD',
            'XRP': 'XRP-USD',
            'DOGE': 'DOGE-USD',
            'SHIB': 'SHIB-USD'
        }
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "2y", 
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: If True, download fresh data even if cached version exists
        
        Returns:
            DataFrame with OHLCV data
        """
        
        # Create filename for caching
        filename = f"{symbol}_{period}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if cached data exists and is recent (less than 1 day old)
        if not force_refresh and os.path.exists(filepath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
            current_time = datetime.now(timezone.utc)
            file_age = current_time - file_mtime
            if file_age < timedelta(days=1):
                self.logger.debug(f"Loading cached data for {symbol}")
                cached_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                # Ensure cached data is also in UTC
                cached_data = self._normalize_timezone(cached_data)
                return cached_data
        
        # Fetch fresh data from yfinance
        self.logger.info(f"Fetching fresh data for {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean the data
            data = self._clean_data(data)
            
            # NORMALIZE TO UTC IMMEDIATELY (canonical time)
            data = self._normalize_timezone(data)
            
            # Cache the data (now in UTC)
            data.to_csv(filepath)
            self.logger.debug(f"Data cached to {filepath}")
            log_data_fetch(symbol, len(data), "yfinance")
            
            return data
            
        except Exception as e:
            log_error("data_fetcher", e, f"fetching data for {symbol}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the fetched data
        
        Args:
            data: Raw data from yfinance
            
        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        # Ensure reasonable price data (no negative prices, High >= Low, etc.)
        data = data[data['Low'] > 0]  # Remove negative prices
        data = data[data['High'] >= data['Low']]  # High should be >= Low
        data = data[data['Volume'] >= 0]  # Volume should be non-negative
        
        return data
    
    def _normalize_timezone(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame index to UTC
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            DataFrame with UTC-normalized index
        """
        try:
            # Check if index is already a DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                # Convert to DatetimeIndex first
                data.index = pd.to_datetime(data.index)
            
            # Normalize index to UTC (assume US market timezone for stocks)
            if data.index.tz is None:
                data.index = data.index.tz_localize('America/New_York')
            data.index = data.index.tz_convert('UTC')
            return data
        except Exception as e:
            self.logger.warning(f"Failed to normalize timezone, using as-is: {e}")
            return data
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest closing price
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice', info.get('previousClose', 0))
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return 0.0
    
    def get_info(self, symbol: str) -> dict:
        """
        Get basic information about a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {e}")
            return {}
    
    def detect_asset_type(self, symbol: str) -> str:
        """
        Auto-detect asset type based on symbol format
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Asset type: 'crypto', 'forex', 'stock'
        """
        symbol_upper = symbol.upper()
        
        # Check if it's a crypto symbol (either BTC or BTC-USD format)
        if symbol_upper in self.crypto_symbols or symbol_upper.endswith('-USD'):
            return 'crypto'
        
        # Check if it's forex (e.g., EURUSD=X or EUR/USD)
        if ('=' in symbol and symbol.endswith('=X')) or '/' in symbol:
            return 'forex'
        
        # Default to stock
        return 'stock'
    
    def normalize_crypto_symbol(self, symbol: str) -> str:
        """
        Convert crypto symbol to yfinance format
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'BTC-USD', 'BTCUSD')
            
        Returns:
            Normalized symbol for yfinance (e.g., 'BTC-USD')
        """
        symbol_upper = symbol.upper()
        
        # If already in yfinance format, return as-is
        if symbol_upper.endswith('-USD'):
            return symbol_upper
        
        # If it's a known crypto symbol, convert it
        if symbol_upper in self.crypto_symbols:
            return self.crypto_symbols[symbol_upper]
        
        # If it looks like BTCUSD format, convert to BTC-USD
        if len(symbol_upper) == 6 and symbol_upper.endswith('USD'):
            base = symbol_upper[:3]
            if base in self.crypto_symbols:
                return self.crypto_symbols[base]
        
        # Default: assume it's already correctly formatted or add -USD
        return f"{symbol_upper}-USD" if not symbol_upper.endswith('-USD') else symbol_upper
    
    def fetch_crypto_data(
        self, 
        symbol: str, 
        period: str = "2y", 
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency OHLCV data
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'BTC-USD')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: If True, download fresh data even if cached version exists
        
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize crypto symbol to yfinance format
        yf_symbol = self.normalize_crypto_symbol(symbol)
        
        # Create filename for caching (use original symbol for clarity)
        filename = f"crypto_{symbol}_{period}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if cached data exists and is recent (less than 1 hour for crypto due to 24/7 nature)
        if not force_refresh and os.path.exists(filepath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
            current_time = datetime.now(timezone.utc)
            file_age = current_time - file_mtime
            # For crypto, use shorter cache time since markets are 24/7
            cache_duration = timedelta(hours=1) if interval in ['1m', '2m', '5m'] else timedelta(hours=6)
            if file_age < cache_duration:
                self.logger.debug(f"Loading cached crypto data for {symbol}")
                cached_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                cached_data = self._normalize_timezone(cached_data)
                return cached_data
        
        # Fetch fresh data from yfinance
        self.logger.info(f"Fetching fresh crypto data for {symbol} ({yf_symbol}) from yfinance...")
        try:
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No crypto data found for symbol {symbol} ({yf_symbol})")
            
            # Clean the data
            data = self._clean_data(data)
            
            # Normalize to UTC (crypto markets are global/24h)
            data = self._normalize_timezone(data)
            
            # Cache the data
            data.to_csv(filepath)
            self.logger.debug(f"Crypto data cached to {filepath}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {symbol}: {e}")
            raise
    
    def fetch_forex_data(
        self, 
        symbol: str, 
        period: str = "2y", 
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch forex OHLCV data
        
        Args:
            symbol: Forex symbol (e.g., 'EURUSD=X', 'EUR/USD')
            period: Data period
            interval: Data interval
            force_refresh: If True, download fresh data even if cached version exists
        
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize forex symbol to yfinance format (add =X if needed)
        if '/' in symbol:
            # Convert EUR/USD to EURUSD=X
            yf_symbol = symbol.replace('/', '') + '=X'
        elif not symbol.endswith('=X'):
            yf_symbol = symbol + '=X'
        else:
            yf_symbol = symbol
        
        # Create filename for caching
        filename = f"forex_{symbol.replace('/', '')}_{period}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Check cache (forex markets are closed weekends, so daily cache is fine)
        if not force_refresh and os.path.exists(filepath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
            current_time = datetime.now(timezone.utc)
            file_age = current_time - file_mtime
            if file_age < timedelta(days=1):
                self.logger.debug(f"Loading cached forex data for {symbol}")
                cached_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                cached_data = self._normalize_timezone(cached_data)
                return cached_data
        
        # Fetch fresh data
        self.logger.info(f"Fetching fresh forex data for {symbol} ({yf_symbol}) from yfinance...")
        try:
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No forex data found for symbol {symbol} ({yf_symbol})")
            
            data = self._clean_data(data)
            data = self._normalize_timezone(data)
            
            data.to_csv(filepath)
            self.logger.debug(f"Forex data cached to {filepath}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching forex data for {symbol}: {e}")
            raise
    
    def fetch_market_data(
        self, 
        symbol: str, 
        asset_type: str = "auto",
        period: str = "2y", 
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Unified method to fetch market data for any asset type
        
        Args:
            symbol: Symbol to fetch (e.g., 'AAPL', 'BTC', 'BTC-USD', 'EUR/USD')
            asset_type: Asset type ('auto', 'stock', 'crypto', 'forex')
            period: Data period
            interval: Data interval  
            force_refresh: If True, download fresh data
        
        Returns:
            DataFrame with OHLCV data
        """
        # Input validation
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if asset_type not in ["auto", "stock", "crypto", "forex"]:
            raise ValueError("Asset type must be one of: auto, stock, crypto, forex")
        
        # Auto-detect asset type if requested
        if asset_type == "auto":
            asset_type = self.detect_asset_type(symbol)
            self.logger.debug(f"Auto-detected asset type for {symbol}: {asset_type}")
        
        # Route to appropriate method
        if asset_type == "crypto":
            return self.fetch_crypto_data(symbol, period, interval, force_refresh)
        elif asset_type == "forex":
            return self.fetch_forex_data(symbol, period, interval, force_refresh)
        elif asset_type == "stock":
            return self.fetch_stock_data(symbol, period, interval, force_refresh)
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")
    
    def get_available_crypto_symbols(self) -> list:
        """
        Get list of available crypto symbols
        
        Returns:
            List of crypto symbols
        """
        return list(self.crypto_symbols.keys())
    
    def get_crypto_info(self, symbol: str) -> dict:
        """
        Get information about a cryptocurrency
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dictionary with crypto information
        """
        try:
            yf_symbol = self.normalize_crypto_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            # Add some crypto-specific metadata
            info['asset_type'] = 'crypto'
            info['original_symbol'] = symbol
            info['yfinance_symbol'] = yf_symbol
            return info
        except Exception as e:
            self.logger.error(f"Error getting crypto info for {symbol}: {e}")
            return {'asset_type': 'crypto', 'original_symbol': symbol}


# Example usage - can be moved to separate test file
if __name__ == "__main__":
    import logging
    
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    logger = fetcher.logger
    
    logger.info("=== Testing DataFetcher ===")
    
    # Test a few basic operations
    try:
        # Test stock data
        aapl_data = fetcher.fetch_stock_data("AAPL", period="1mo", interval="1d")
        logger.info(f"Successfully fetched {len(aapl_data)} days of AAPL data")
        
        # Test crypto data  
        btc_data = fetcher.fetch_crypto_data("BTC", period="1mo", interval="1d")
        logger.info(f"Successfully fetched {len(btc_data)} days of BTC data")
        
        # Test unified interface
        data = fetcher.fetch_market_data("AAPL", asset_type="auto", period="1mo")
        logger.info(f"Unified interface test passed: {len(data)} days")
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise