"""
Centralized logging configuration for the trading bot
Replaces ad-hoc print statements with proper structured logging
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


class TradingBotLogger:
    """
    Centralized logger for the trading bot with different log levels and formatting
    """
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure root logger
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup file handler for all logs
        today = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f'trading_bot_{today}.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup error file handler
        error_handler = logging.FileHandler(
            os.path.join(self.log_dir, f'trading_bot_errors_{today}.log'),
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return self._loggers[name]


# Global logger instance
_trading_logger = TradingBotLogger()


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance for the current module
    
    Args:
        name: Logger name (defaults to caller's __name__)
        
    Returns:
        Configured logger instance
    """
    return _trading_logger.get_logger(name)


# Convenience functions to replace common print patterns
def log_data_fetch(symbol: str, days: int, source: str = "yfinance"):
    """Log data fetching operations"""
    logger = get_logger("data_fetcher")
    logger.info(f"Fetched {days} days of data for {symbol} from {source}")


def log_strategy_signal(symbol: str, signal: str, price: float, confidence: float):
    """Log trading signals"""
    logger = get_logger("strategy")
    logger.info(f"Signal: {signal} {symbol} @ ${price:.2f} (confidence: {confidence:.2f})")


def log_trade_execution(symbol: str, action: str, quantity: float, price: float):
    """Log trade executions"""
    logger = get_logger("execution")
    logger.info(f"Executed: {action} {quantity:.4f} {symbol} @ ${price:.2f}")


def log_portfolio_update(total_value: float, cash: float, positions_count: int):
    """Log portfolio updates"""
    logger = get_logger("portfolio")
    logger.debug(f"Portfolio: ${total_value:.2f} total, ${cash:.2f} cash, {positions_count} positions")


def log_backtest_progress(symbol: str, current_date: str, progress_pct: float):
    """Log backtest progress"""
    logger = get_logger("backtest")
    logger.debug(f"Backtest {symbol}: {current_date} ({progress_pct:.1f}% complete)")


def log_optimization_result(symbol: str, params: dict, metric_value: float, metric_name: str):
    """Log optimization results"""
    logger = get_logger("optimization")
    logger.info(f"Best params for {symbol}: {params} -> {metric_name}={metric_value:.4f}")


def log_error(module: str, error: Exception, context: str = ""):
    """Log errors with context"""
    logger = get_logger(module)
    context_str = f" ({context})" if context else ""
    logger.error(f"Error{context_str}: {type(error).__name__}: {error}")


def log_warning(module: str, message: str):
    """Log warnings"""
    logger = get_logger(module)
    logger.warning(message)


def log_performance_metrics(metrics: dict):
    """Log performance metrics"""
    logger = get_logger("performance")
    logger.info(f"Performance: {metrics}")


# Initialize logging on import
_trading_logger