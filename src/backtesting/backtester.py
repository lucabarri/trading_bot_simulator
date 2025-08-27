from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

# Import external modules with fallback for tests
try:
    from ..strategies.base_strategy import BaseStrategy, TradingSignal
    from ..data.data_fetcher import DataFetcher
    from ..trading.portfolio import Portfolio
    from ..trading.execution_engine import ExecutionEngine, ExecutionMode
    from ..trading.market_simulator import MarketSimulator
    from ..trading.position import PositionType
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from strategies.base_strategy import BaseStrategy, TradingSignal
    from data.data_fetcher import DataFetcher
    from trading.portfolio import Portfolio
    from trading.execution_engine import ExecutionEngine, ExecutionMode
    from trading.market_simulator import MarketSimulator
    from trading.position import PositionType


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_cash: float = 100000.0
    allocation_pct: float = 0.1
    trading_fees: float = 5.0
    slippage_bps: float = 2.0
    bid_ask_spread_bps: float = 5.0
    enable_market_hours: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not (0 < self.allocation_pct <= 1):
            raise ValueError("Allocation percentage must be between 0 and 1")
        if self.trading_fees < 0:
            raise ValueError("Trading fees cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("Slippage basis points cannot be negative")
        if self.bid_ask_spread_bps < 0:
            raise ValueError("Bid-ask spread basis points cannot be negative")
        if self.slippage_bps > 1000:  # 10%
            raise ValueError("Slippage basis points seems unreasonably high (>1000 bps)")
        if self.bid_ask_spread_bps > 1000:  # 10%
            raise ValueError("Bid-ask spread basis points seems unreasonably high (>1000 bps)")


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_value: float
    final_value: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_fees: float
    duration_days: int
    daily_values: pd.DataFrame
    closed_positions: List[Dict[str, Any]]


class Backtester:
    """
    Dedicated backtesting framework for testing trading strategies
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtester
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger("Backtester")
        logging.basicConfig(level=logging.INFO)
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: str,
        end_date: str,
        config: BacktestConfig = None
    ) -> BacktestResult:
        """
        Run a backtest for a single strategy and symbol
        
        Args:
            strategy: Trading strategy to test
            symbol: Symbol to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            config: Override default config
            
        Returns:
            BacktestResult object
        """
        # Input validation
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if not start_date or not end_date:
            raise ValueError("Start and end dates must be provided")
        if strategy is None:
            raise ValueError("Strategy cannot be None")
            
        config = config or self.config
        
        self.logger.info(f"Starting backtest: {strategy.name} on {symbol} from {start_date} to {end_date}")
        
        # Load and prepare data
        data = self._prepare_data(symbol, start_date, end_date, strategy.get_required_history())
        if data is None or len(data) == 0:
            raise ValueError("Failed to load backtest data")
        
        # Initialize components
        portfolio = Portfolio(initial_cash=config.initial_cash)
        market_simulator = MarketSimulator(
            bid_ask_spread_bps=config.bid_ask_spread_bps,
            market_hours_only=config.enable_market_hours
        )
        execution_engine = ExecutionEngine(
            portfolio=portfolio,
            mode=ExecutionMode.PAPER,
            base_fee=config.trading_fees,
            slippage_bps=config.slippage_bps
        )
        
        # Reset strategy state
        strategy.reset()
        
        # Track performance
        daily_values = []
        signal_history = []
        
        # Parse dates to UTC
        start_utc = pd.to_datetime(start_date).tz_localize('America/New_York').tz_convert('UTC')
        end_utc = pd.to_datetime(end_date).tz_localize('America/New_York').tz_convert('UTC')
        
        # Determine when to start executing (after warm-up period)
        lookback_days = strategy.get_required_history()
        backtest_start_idx = None
        
        # Run backtest day by day
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            
            # Update market simulator
            market_simulator.set_price(symbol, current_price)
            market_simulator.update_prices(timestamp)
            
            # Start executing after warm-up period and within date range
            if i >= lookback_days and timestamp.tz_convert('UTC') >= start_utc:
                if backtest_start_idx is None:
                    backtest_start_idx = i
                    self.logger.info(f"Backtest execution started at {timestamp.strftime('%Y-%m-%d')}")
                
                # Get data window for strategy
                window_data = data.iloc[:i+1]
                
                # Generate signals
                signals = strategy.generate_signals(window_data)
                
                # Process latest signal
                if signals:
                    latest_signal = signals[-1]
                    signal_history.append(latest_signal)
                    
                    # Create and execute order
                    order = execution_engine.create_order_from_signal(
                        latest_signal, symbol, config.allocation_pct
                    )
                    
                    if order is not None:
                        market_price = market_simulator.get_market_price(
                            symbol, timestamp, order.action, order.quantity * current_price
                        )
                        
                        if market_price is not None:
                            success = execution_engine.execute_order(order, market_price, timestamp)
                            if success:
                                action = "BUY" if order.action == "BUY" else "SELL"
                                self.logger.info(f"{timestamp.strftime('%Y-%m-%d')}: {action} {order.quantity} {symbol} @ ${market_price:.2f}")
            
            # Record daily portfolio value (only during execution period)
            if backtest_start_idx is not None:
                # Update portfolio with current price for unrealized P&L
                if portfolio.has_position(symbol):
                    portfolio.update_positions({symbol: current_price})
                
                daily_values.append({
                    'date': timestamp,
                    'portfolio_value': portfolio.total_value,
                    'cash': portfolio.cash,
                    'positions_value': portfolio.positions_value,
                    'total_pnl': portfolio.total_pnl,
                    'price': current_price
                })
        
        # Generate results
        return self._generate_result(
            strategy, symbol, start_date, end_date, 
            daily_values, portfolio, execution_engine
        )
    
    def run_multi_symbol_backtest(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: BacktestConfig = None
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest across multiple symbols
        
        Args:
            strategy: Trading strategy to test
            symbols: List of symbols to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            config: Override default config
            
        Returns:
            Dictionary of symbol -> BacktestResult
        """
        results = {}
        
        for symbol in symbols:
            try:
                result = self.run_backtest(strategy, symbol, start_date, end_date, config)
                results[symbol] = result
                self.logger.info(f"Completed {symbol}: {result.total_return_pct:.2f}% return")
            except Exception as e:
                self.logger.error(f"Failed to backtest {symbol}: {e}")
                continue
        
        return results
    
    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        symbol: str,
        start_date: str,
        end_date: str,
        config: BacktestConfig = None
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies on the same symbol and period
        
        Args:
            strategies: List of strategies to compare
            symbol: Symbol to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            config: Override default config
            
        Returns:
            Dictionary of strategy_name -> BacktestResult
        """
        results = {}
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, symbol, start_date, end_date, config)
                results[strategy.name] = result
                self.logger.info(f"Completed {strategy.name}: {result.total_return_pct:.2f}% return")
            except Exception as e:
                self.logger.error(f"Failed to backtest {strategy.name}: {e}")
                continue
        
        return results
    
    def optimize_parameters(
        self,
        strategy_factory,
        parameter_ranges: Dict[str, List],
        symbol: str,
        start_date: str,
        end_date: str,
        optimization_metric: str = 'sharpe_ratio',
        config: BacktestConfig = None
    ) -> Tuple[Dict, BacktestResult]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_factory: Function that creates strategy with given parameters
            parameter_ranges: Dict of parameter_name -> list of values to test
            symbol: Symbol to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
            config: Override default config
            
        Returns:
            Tuple of (best_parameters, best_result)
        """
        self.logger.info(f"Starting parameter optimization for {len(parameter_ranges)} parameters")
        
        # Generate parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        best_params = None
        best_result = None
        best_score = float('-inf') if optimization_metric != 'max_drawdown' else float('inf')
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                # Create strategy with these parameters
                strategy = strategy_factory(**params)
                
                # Run backtest
                result = self.run_backtest(strategy, symbol, start_date, end_date, config)
                
                # Evaluate score
                if optimization_metric == 'sharpe_ratio':
                    score = result.sharpe_ratio
                elif optimization_metric == 'total_return':
                    score = result.total_return_pct
                elif optimization_metric == 'max_drawdown':
                    score = -abs(result.max_drawdown_pct)  # Minimize drawdown
                else:
                    raise ValueError(f"Unknown optimization metric: {optimization_metric}")
                
                # Check if this is the best so far
                is_better = (score > best_score) if optimization_metric != 'max_drawdown' else (score > best_score)
                
                if is_better:
                    best_score = score
                    best_params = params
                    best_result = result
                
                self.logger.info(f"Combination {i+1}/{len(combinations)}: {params} -> {optimization_metric}={score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed combination {params}: {e}")
                continue
        
        self.logger.info(f"Optimization complete. Best parameters: {best_params}")
        return best_params, best_result
    
    def optimize_per_asset_parameters(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        parameter_ranges: Dict[str, List] = None,
        optimization_metric: str = 'sharpe_ratio',
        config: BacktestConfig = None,
        use_inverse_strategy: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """
        Optimize parameters for each asset independently (sequential optimization)
        
        Args:
            symbols: List of symbols to optimize
            start_date: Start date for optimization
            end_date: End date for optimization
            parameter_ranges: Parameter ranges to test (default: MA ranges)
            optimization_metric: Metric to optimize
            config: Backtesting configuration
            use_inverse_strategy: Whether to use inverse MA strategy instead of normal MA
            
        Returns:
            Dict of symbol -> best parameters
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'short_ma': [5, 10, 15, 20],
                'long_ma': [20, 30, 40, 50]
            }
        
        config = config or self.config
        optimized_params = {}
        
        strategy_type = "INVERSE" if use_inverse_strategy else "NORMAL"
        self.logger.info(f"Starting per-asset optimization for {len(symbols)} symbols using {strategy_type} MA strategy")
        
        # Strategy factory for MA strategies (normal or inverse)
        def ma_strategy_factory(short_ma, long_ma):
            if use_inverse_strategy:
                from ..strategies.inverse_ma_strategy import create_inverse_ma_strategy
                return create_inverse_ma_strategy(short=short_ma, long=long_ma)
            else:
                from ..strategies.moving_average_strategy import create_simple_ma_strategy
                return create_simple_ma_strategy(short=short_ma, long=long_ma)
        
        # Optimize each symbol independently
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Optimizing {symbol} ({i}/{len(symbols)})...")
            
            try:
                best_params, best_result = self.optimize_parameters(
                    strategy_factory=ma_strategy_factory,
                    parameter_ranges=parameter_ranges,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    optimization_metric=optimization_metric,
                    config=config
                )
                
                optimized_params[symbol] = {
                    'short': best_params['short_ma'],
                    'long': best_params['long_ma']
                }
                
                self.logger.info(f"  {symbol}: MA({best_params['short_ma']},{best_params['long_ma']}) "
                               f"-> {optimization_metric}={getattr(best_result, optimization_metric):.3f}")
                
            except Exception as e:
                self.logger.warning(f"  Failed to optimize {symbol}: {e}")
                # Use default parameters
                optimized_params[symbol] = {'short': 10, 'long': 20}
                continue
        
        self.logger.info(f"Per-asset optimization complete for {len(optimized_params)} symbols")
        return optimized_params
    
    def run_multi_strategy_backtest(
        self,
        multi_strategy,
        start_date: str,
        end_date: str,
        config: BacktestConfig = None
    ) -> BacktestResult:
        """
        Run backtest for multi-asset strategy with shared cash pool
        
        Args:
            multi_strategy: MultiAssetMAStrategy instance
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            config: Override default config
            
        Returns:
            BacktestResult for the combined strategy
        """
        config = config or self.config
        
        self.logger.info(f"Starting multi-strategy backtest: {multi_strategy.name}")
        self.logger.info(f"Assets: {', '.join(multi_strategy.symbols)}")
        self.logger.info(f"Period: {start_date} to {end_date}")
        
        # Load data for all symbols
        fetcher = DataFetcher()
        symbol_data = {}
        
        for symbol in multi_strategy.symbols:
            try:
                # Use unified fetch_market_data with auto-detection
                data = fetcher.fetch_market_data(symbol, asset_type="auto", period="2y", interval="1d")
                symbol_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Failed to load data for {symbol}: {e}")
                continue
        
        if not symbol_data:
            raise ValueError("No market data could be loaded")
        
        # Find common date range across all symbols
        start_utc = pd.to_datetime(start_date).tz_localize('America/New_York').tz_convert('UTC')
        end_utc = pd.to_datetime(end_date).tz_localize('America/New_York').tz_convert('UTC')
        
        # Add warm-up period
        lookback_days = multi_strategy.get_required_history()
        extended_start_utc = start_utc - timedelta(days=lookback_days + 30)
        
        # Filter all data to common date range
        for symbol in symbol_data:
            mask = (symbol_data[symbol].index >= extended_start_utc) & (symbol_data[symbol].index <= end_utc)
            symbol_data[symbol] = symbol_data[symbol][mask]
        
        # Find common timestamps across all symbols
        common_timestamps = None
        for symbol, data in symbol_data.items():
            if common_timestamps is None:
                common_timestamps = set(data.index)
            else:
                common_timestamps = common_timestamps.intersection(set(data.index))
        
        common_timestamps = sorted(list(common_timestamps))
        
        if len(common_timestamps) < lookback_days:
            raise ValueError(f"Insufficient common data. Need {lookback_days} days, got {len(common_timestamps)}")
        
        # Initialize components with shared cash pool
        portfolio = Portfolio(initial_cash=config.initial_cash)
        market_simulator = MarketSimulator(
            bid_ask_spread_bps=config.bid_ask_spread_bps,
            market_hours_only=config.enable_market_hours
        )
        execution_engine = ExecutionEngine(
            portfolio=portfolio,
            mode=ExecutionMode.PAPER,
            base_fee=config.trading_fees,
            slippage_bps=config.slippage_bps
        )
        
        # Reset strategy state
        multi_strategy.reset()
        
        # Track performance
        daily_values = []
        signal_history = []
        
        # Run backtest day by day
        backtest_start_idx = None
        
        for i, timestamp in enumerate(common_timestamps):
            # Update market prices for all symbols
            current_prices = {}
            for symbol in multi_strategy.symbols:
                if symbol in symbol_data and timestamp in symbol_data[symbol].index:
                    price = symbol_data[symbol].loc[timestamp, 'Close']
                    current_prices[symbol] = price
                    market_simulator.set_price(symbol, price)
            
            market_simulator.update_prices(timestamp)
            
            # Start executing after warm-up period and within date range
            if i >= lookback_days and timestamp.tz_convert('UTC') >= start_utc:
                if backtest_start_idx is None:
                    backtest_start_idx = i
                    self.logger.info(f"Multi-strategy execution started at {timestamp.strftime('%Y-%m-%d')}")
                
                # Generate signals using one symbol's data as trigger (multi_strategy will fetch all data)
                primary_symbol = multi_strategy.symbols[0]
                if primary_symbol in symbol_data:
                    window_data = symbol_data[primary_symbol].iloc[:i+1]
                    signals = multi_strategy.generate_signals(window_data)
                    
                    # Process signals for all symbols
                    for signal in signals:
                        # Get symbol from signal metadata
                        signal_symbol = signal.metadata.get('symbol', 'UNKNOWN')
                        
                        if signal_symbol in current_prices:
                            signal_history.append(signal)
                            
                            # Get allocation percentage from signal metadata
                            allocation_pct = signal.metadata.get('allocation_pct', config.allocation_pct)
                            
                            # Create and execute order
                            order = execution_engine.create_order_from_signal(
                                signal, signal_symbol, allocation_pct
                            )
                            
                            if order is not None:
                                market_price = market_simulator.get_market_price(
                                    signal_symbol, timestamp, order.action, order.quantity * current_prices[signal_symbol]
                                )
                                
                                if market_price is not None:
                                    success = execution_engine.execute_order(order, market_price, timestamp)
                                    if success:
                                        action = "BUY" if order.action == "BUY" else "SELL"
                                        self.logger.info(f"{timestamp.strftime('%Y-%m-%d')}: {action} {order.quantity} {signal_symbol} @ ${market_price:.2f}")
            
            # Record daily portfolio value (only during execution period)
            if backtest_start_idx is not None:
                # Update portfolio with current prices for unrealized P&L
                if current_prices:
                    portfolio.update_positions(current_prices)
                
                daily_values.append({
                    'date': timestamp,
                    'portfolio_value': portfolio.total_value,
                    'cash': portfolio.cash,
                    'positions_value': portfolio.positions_value,
                    'total_pnl': portfolio.total_pnl,
                    'prices': current_prices.copy()
                })
        
        # Generate results
        return self._generate_multi_strategy_result(
            multi_strategy, start_date, end_date, 
            daily_values, portfolio, execution_engine
        )
    
    def _generate_multi_strategy_result(
        self,
        multi_strategy,
        start_date: str,
        end_date: str,
        daily_values: List[Dict],
        portfolio: Portfolio,
        execution_engine: ExecutionEngine
    ) -> BacktestResult:
        """
        Generate BacktestResult for multi-strategy backtest
        """
        if not daily_values:
            # Return empty result
            return BacktestResult(
                strategy_name=multi_strategy.name,
                symbol="MULTI_ASSET",
                start_date=start_date,
                end_date=end_date,
                initial_value=0,
                final_value=0,
                total_return=0,
                total_return_pct=0,
                annualized_return=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                total_fees=0,
                duration_days=0,
                daily_values=pd.DataFrame(),
                closed_positions=[]
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(daily_values)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        
        # Calculate metrics
        initial_value = portfolio.initial_cash
        if len(df) == 0:
            raise ValueError("No backtest data available to calculate performance metrics")
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Risk metrics
        volatility = df['daily_return'].std() * (252 ** 0.5)  # Annualized
        
        # Max drawdown (safe calculation to avoid division by zero)
        rolling_max = df['portfolio_value'].expanding().max()
        # Avoid division by zero by using maximum of rolling_max and a small value
        safe_rolling_max = rolling_max.clip(lower=1e-8)  # Minimum value to avoid division by zero
        drawdown = (df['portfolio_value'] - rolling_max) / safe_rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe_ratio = (total_return * 252 / len(df)) / volatility
        else:
            sharpe_ratio = 0
        
        # Trading statistics
        closed_positions = portfolio.closed_positions
        winning_trades = sum(1 for pos in closed_positions if pos.realized_pnl > 0)
        losing_trades = sum(1 for pos in closed_positions if pos.realized_pnl < 0)
        win_rate = winning_trades / len(closed_positions) if closed_positions else 0
        
        avg_win = np.mean([pos.realized_pnl for pos in closed_positions if pos.realized_pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pos.realized_pnl for pos in closed_positions if pos.realized_pnl < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        execution_summary = execution_engine.get_execution_summary()
        total_fees = execution_summary.get('total_fees_paid', 0)
        
        # Format closed positions with symbol information
        formatted_positions = [
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
        
        return BacktestResult(
            strategy_name=multi_strategy.name,
            symbol="MULTI_ASSET",
            start_date=start_date,
            end_date=end_date,
            initial_value=initial_value,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return * 100,
            annualized_return=(final_value / initial_value) ** (252 / len(df)) - 1,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown * 100,
            total_trades=len(closed_positions),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_fees=total_fees,
            duration_days=len(df),
            daily_values=df,
            closed_positions=formatted_positions
        )
    
    def _prepare_data(self, symbol: str, start_date: str, end_date: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Load and prepare data for backtesting
        
        Args:
            symbol: Symbol to load
            start_date: Start date
            end_date: End date
            lookback_days: Additional days needed for strategy warm-up
            
        Returns:
            Prepared DataFrame or None if failed
        """
        try:
            fetcher = DataFetcher()
            # Use unified fetch_market_data with auto-detection
            data = fetcher.fetch_market_data(symbol, asset_type="auto", period="2y", interval="1d")
            
            # Parse dates to UTC
            start_utc = pd.to_datetime(start_date).tz_localize(timezone.utc)
            end_utc = pd.to_datetime(end_date).tz_localize(timezone.utc)
            
            # Add warm-up period
            extended_start_utc = start_utc - timedelta(days=lookback_days + 30)
            
            # Filter data
            mask = (data.index >= extended_start_utc) & (data.index <= end_utc)
            data = data[mask]
            
            if len(data) < lookback_days:
                raise ValueError(f"Insufficient data. Need {lookback_days} days, got {len(data)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data for {symbol}: {e}")
            return None
    
    def _generate_result(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: str,
        end_date: str,
        daily_values: List[Dict],
        portfolio: Portfolio,
        execution_engine: ExecutionEngine
    ) -> BacktestResult:
        """
        Generate BacktestResult from backtest run
        
        Args:
            strategy: Strategy used
            symbol: Symbol traded
            start_date: Start date
            end_date: End date
            daily_values: Daily portfolio values
            portfolio: Final portfolio state
            execution_engine: Execution engine with trade history
            
        Returns:
            BacktestResult object
        """
        if not daily_values:
            # Return empty result
            return BacktestResult(
                strategy_name=strategy.name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_value=0,
                final_value=0,
                total_return=0,
                total_return_pct=0,
                annualized_return=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                total_fees=0,
                duration_days=0,
                daily_values=pd.DataFrame(),
                closed_positions=[]
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(daily_values)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        
        # Calculate metrics
        initial_value = portfolio.initial_cash
        if len(df) == 0:
            raise ValueError("No backtest data available to calculate performance metrics")
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Risk metrics
        volatility = df['daily_return'].std() * (252 ** 0.5)  # Annualized
        
        # Max drawdown (safe calculation to avoid division by zero)
        rolling_max = df['portfolio_value'].expanding().max()
        # Avoid division by zero by using maximum of rolling_max and a small value
        safe_rolling_max = rolling_max.clip(lower=1e-8)  # Minimum value to avoid division by zero
        drawdown = (df['portfolio_value'] - rolling_max) / safe_rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe_ratio = (total_return * 252 / len(df)) / volatility
        else:
            sharpe_ratio = 0
        
        # Trading statistics
        closed_positions = portfolio.closed_positions
        winning_trades = sum(1 for pos in closed_positions if pos.realized_pnl > 0)
        losing_trades = sum(1 for pos in closed_positions if pos.realized_pnl < 0)
        win_rate = winning_trades / len(closed_positions) if closed_positions else 0
        
        avg_win = np.mean([pos.realized_pnl for pos in closed_positions if pos.realized_pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pos.realized_pnl for pos in closed_positions if pos.realized_pnl < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        execution_summary = execution_engine.get_execution_summary()
        total_fees = execution_summary.get('total_fees_paid', 0)
        
        # Format closed positions
        formatted_positions = [
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
        
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_value=initial_value,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return * 100,
            annualized_return=(final_value / initial_value) ** (252 / len(df)) - 1,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown * 100,
            total_trades=len(closed_positions),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_fees=total_fees,
            duration_days=len(df),
            daily_values=df,
            closed_positions=formatted_positions
        )