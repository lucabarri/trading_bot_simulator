from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import logging

from .execution_engine import ExecutionEngine, ExecutionMode, OrderType
from .market_simulator import MarketSimulator
from .portfolio import Portfolio

# Import external modules with fallback for tests
try:
    from ..strategies.base_strategy import BaseStrategy, TradingSignal
    from ..data.data_fetcher import DataFetcher
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from strategies.base_strategy import BaseStrategy, TradingSignal
    from data.data_fetcher import DataFetcher


class PaperTrader:
    """
    Complete paper trading system that combines strategy, execution, and market simulation
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: float = 100000.0,
        allocation_pct: float = 0.1,
        symbols: List[str] = None,
        trading_fees: float = 5.0,
        slippage_bps: float = 2.0,
        bid_ask_spread_bps: float = 5.0,
        enable_market_hours: bool = False  # Disable for backtesting
    ):
        """
        Initialize paper trader
        
        Args:
            strategy: Trading strategy to use
            initial_cash: Starting portfolio cash
            allocation_pct: Percentage of portfolio per trade
            symbols: List of symbols to trade (default: ['AAPL'])
            trading_fees: Trading fees per transaction
            slippage_bps: Slippage in basis points
            bid_ask_spread_bps: Bid-ask spread in basis points
            enable_market_hours: Whether to enforce market hours
        """
        self.strategy = strategy
        self.symbols = symbols or ['AAPL']
        self.allocation_pct = allocation_pct
        
        # Initialize components
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.market_simulator = MarketSimulator(
            bid_ask_spread_bps=bid_ask_spread_bps,
            market_hours_only=enable_market_hours
        )
        self.execution_engine = ExecutionEngine(
            portfolio=self.portfolio,
            mode=ExecutionMode.PAPER,
            base_fee=trading_fees,
            slippage_bps=slippage_bps
        )
        
        # Trading state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        
        # Performance tracking
        self.daily_values: List[Dict] = []
        self.signal_history: List[TradingSignal] = []
        
        # Setup logging
        self.logger = logging.getLogger("PaperTrader")
        logging.basicConfig(level=logging.INFO)
    
    def initialize_market_data(self, period: str = "1y") -> bool:
        """
        Load market data for trading
        
        Args:
            period: Data period to load
            
        Returns:
            True if successful
        """
        try:
            self.market_simulator.load_market_data(self.symbols, period)
            self.logger.info(f"Loaded market data for {len(self.symbols)} symbols")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return False
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        symbol: str = "AAPL"
    ) -> Dict:
        """
        Run backtest over historical period
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Symbol to trade
            
        Returns:
            Backtest results
        """
        self.logger.info(f"Starting backtest: {start_date} to {end_date} for {symbol}")
        
        # Load historical data
        fetcher = DataFetcher()
        try:
            # Fetch data (now always in UTC from DataFetcher)
            data = fetcher.fetch_stock_data(symbol, period="2y", interval="1d")
            
            # Parse user dates to UTC (canonical time)
            start_utc = pd.to_datetime(start_date).tz_localize('America/New_York').tz_convert('UTC')
            end_utc = pd.to_datetime(end_date).tz_localize('America/New_York').tz_convert('UTC')
            
            # Add strategy warm-up period
            lookback_days = self.strategy.get_required_history()
            extended_start_utc = start_utc - timedelta(days=lookback_days + 30)
            
            # Filter data using proper UTC comparison (clean datetime logic)
            mask = (data.index >= extended_start_utc) & (data.index <= end_utc)
            data = data[mask]
            
            if len(data) < lookback_days:
                raise ValueError(f"Insufficient data for backtest. Need {lookback_days} days, got {len(data)}")
            
            self.logger.info(f"Loaded {len(data)} days of data for backtest")
            
        except Exception as e:
            self.logger.error(f"Failed to load backtest data: {e}")
            return {}
        
        # Reset components
        self.portfolio.reset()
        self.execution_engine.reset()
        self.strategy.reset()
        self.daily_values = []
        self.signal_history = []
        
        # Initialize market simulator with historical data
        self.market_simulator.set_price(symbol, data['Close'].iloc[0])
        
        # Run backtest day by day
        backtest_start_idx = None
        for i, (timestamp, row) in enumerate(data.iterrows()):
            self.current_time = timestamp
            current_price = row['Close']
            
            # Update market price
            self.market_simulator.set_price(symbol, current_price)
            self.market_simulator.update_prices(timestamp)
            
            # Start generating signals only after we have enough data
            # Clean UTC comparison (no string conversion needed)
            if i >= lookback_days and timestamp.tz_convert('UTC') >= start_utc:
                if backtest_start_idx is None:
                    backtest_start_idx = i
                    self.start_time = timestamp
                    self.logger.info(f"Backtest execution started at {timestamp.strftime('%Y-%m-%d')}")
                
                # Get historical data window for strategy
                window_data = data.iloc[:i+1]
                
                # Generate signals
                signals = self.strategy.generate_signals(window_data)
                
                # Process only the latest signal
                if signals:
                    latest_signal = signals[-1]
                    self.signal_history.append(latest_signal)
                    
                    # Create and execute order if needed
                    order = self.execution_engine.create_order_from_signal(
                        latest_signal, symbol, self.allocation_pct
                    )
                    
                    if order is not None:
                        market_price = self.market_simulator.get_market_price(
                            symbol, timestamp, order.action, order.quantity * current_price
                        )
                        
                        if market_price is not None:
                            success = self.execution_engine.execute_order(order, market_price, timestamp)
                            if success:
                                action = "BUY" if order.action == "BUY" else "SELL"
                                self.logger.info(f"{timestamp.strftime('%Y-%m-%d')}: {action} {order.quantity} {symbol} @ ${market_price:.2f}")
            
            # Record daily portfolio value
            if backtest_start_idx is not None:
                # Update portfolio with current price for unrealized P&L
                if self.portfolio.has_position(symbol):
                    self.portfolio.update_positions({symbol: current_price})
                
                self.daily_values.append({
                    'date': timestamp,
                    'portfolio_value': self.portfolio.total_value,
                    'cash': self.portfolio.cash,
                    'positions_value': self.portfolio.positions_value,
                    'total_pnl': self.portfolio.total_pnl,
                    'price': current_price
                })
        
        self.end_time = timestamp
        
        # Generate backtest report
        return self.generate_backtest_report()
    
    def run_live_paper_trading(self, duration_minutes: int = 60):
        """
        Run live paper trading for specified duration
        
        Args:
            duration_minutes: How long to run (for demo purposes)
        """
        self.logger.info(f"Starting live paper trading for {duration_minutes} minutes")
        
        if not self.initialize_market_data():
            return
        
        self.is_running = True
        self.start_time = pd.Timestamp.now(tz=timezone.utc)
        end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        try:
            while self.is_running and pd.Timestamp.now(tz=timezone.utc) < end_time:
                current_time = pd.Timestamp.now(tz=timezone.utc)
                
                # In a real implementation, this would fetch live data
                # For demo, we'll simulate price updates
                for symbol in self.symbols:
                    self.market_simulator.update_prices(current_time)
                
                # Get current prices
                current_prices = self.market_simulator.get_current_prices()
                
                # For demo, we'll use the latest historical data as current data
                # In reality, you'd fetch real-time data here
                
                # Sleep for a bit (in real trading, this might be much longer)
                import time
                time.sleep(1)  # 1 second intervals for demo
        
        except KeyboardInterrupt:
            self.logger.info("Paper trading stopped by user")
        
        self.is_running = False
        self.end_time = pd.Timestamp.now(tz=timezone.utc)
    
    def generate_backtest_report(self) -> Dict:
        """
        Generate comprehensive backtest report
        
        Returns:
            Backtest results dictionary
        """
        if not self.daily_values:
            return {}
        
        # Convert daily values to DataFrame (dates already in UTC)
        df = pd.DataFrame(self.daily_values)
        # Ensure all dates are properly UTC
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        initial_value = self.portfolio.initial_cash
        if len(df) == 0:
            raise ValueError("No trading data available to calculate performance metrics")
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Risk metrics
        volatility = df['daily_return'].std() * (252 ** 0.5)  # Annualized
        
        # Calculate max drawdown
        rolling_max = df['portfolio_value'].expanding().max()
        # Avoid division by zero by using maximum of rolling_max and a small value
        safe_rolling_max = rolling_max.clip(lower=1e-8)  # Minimum value to avoid division by zero
        drawdown = (df['portfolio_value'] - rolling_max) / safe_rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplified, assuming 0% risk-free rate)
        if volatility > 0:
            sharpe_ratio = (total_return * 252 / len(df)) / volatility
        else:
            sharpe_ratio = 0
        
        # Trading statistics
        execution_summary = self.execution_engine.get_execution_summary()
        
        # Count winning vs losing trades
        closed_positions = self.portfolio.closed_positions
        winning_trades = sum(1 for pos in closed_positions if pos.realized_pnl > 0)
        losing_trades = sum(1 for pos in closed_positions if pos.realized_pnl < 0)
        
        win_rate = winning_trades / len(closed_positions) if closed_positions else 0
        
        avg_win = np.mean([pos.realized_pnl for pos in closed_positions if pos.realized_pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pos.realized_pnl for pos in closed_positions if pos.realized_pnl < 0]) if losing_trades > 0 else 0
        
        report = {
            'backtest_period': {
                'start_date': self.start_time.strftime('%Y-%m-%d') if self.start_time else None,
                'end_date': self.end_time.strftime('%Y-%m-%d') if self.end_time else None,
                'duration_days': len(df)
            },
            'performance': {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': (final_value / initial_value) ** (252 / len(df)) - 1,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100
            },
            'trading_stats': {
                'total_trades': len(closed_positions),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'total_fees': execution_summary.get('total_fees_paid', 0)
            },
            'execution': execution_summary,
            'daily_values': df,
            'closed_positions': [
                {
                    'symbol': pos.symbol,
                    'entry_date': pos.entry_time.strftime('%Y-%m-%d'),
                    'exit_date': pos.exit_time.strftime('%Y-%m-%d'),
                    'entry_price': pos.entry_price,
                    'exit_price': pos.exit_price,
                    'quantity': pos.quantity,
                    'pnl': pos.realized_pnl,
                    'return_pct': pos.calculate_return_pct(pos.exit_price)
                }
                for pos in closed_positions
            ]
        }
        
        return report
    
    def stop(self):
        """Stop the paper trader"""
        self.is_running = False
    
    def get_current_status(self) -> Dict:
        """Get current trading status"""
        return {
            'is_running': self.is_running,
            'current_time': self.current_time,
            'portfolio': self.portfolio.get_portfolio_summary(),
            'market': self.market_simulator.get_market_summary(),
            'execution': self.execution_engine.get_execution_summary(),
            'signals_generated': len(self.signal_history)
        }
    
    def __repr__(self):
        status = "RUNNING" if self.is_running else "STOPPED"
        return f"PaperTrader({self.strategy.name}, {status}, portfolio=${self.portfolio.total_value:.2f})"