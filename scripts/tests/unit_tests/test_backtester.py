#!/usr/bin/env python3
"""
Test script for the dedicated backtesting framework
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from backtesting.backtester import Backtester, BacktestConfig
from strategies.moving_average_strategy import create_simple_ma_strategy
# Removed timezone_utils import - using simple date formatting


def test_single_backtest():
    """Test basic single strategy backtest"""
    print("Testing Single Strategy Backtest...")
    print("=" * 50)
    
    # Create backtester with custom config
    config = BacktestConfig(
        initial_cash=50000.0,
        allocation_pct=0.15,  # 15% per trade
        trading_fees=5.0,
        slippage_bps=2.0
    )
    
    backtester = Backtester(config)
    
    # Create strategy
    strategy = create_simple_ma_strategy(short=10, long=20)
    
    # Run backtest
    result = backtester.run_backtest(
        strategy=strategy,
        symbol="AAPL",
        start_date="2024-06-01",
        end_date="2024-12-01",
        config=config
    )
    
    print(f"[OK] Backtest completed for {result.strategy_name}")
    print(f"[OK] Symbol: {result.symbol}")
    print(f"[OK] Period: {result.start_date} to {result.end_date} ({result.duration_days} days)")
    print(f"[OK] Initial value: ${result.initial_value:,.2f}")
    print(f"[OK] Final value: ${result.final_value:,.2f}")
    print(f"[OK] Total return: {result.total_return_pct:.2f}%")
    print(f"[OK] Annualized return: {result.annualized_return*100:.2f}%")
    print(f"[OK] Volatility: {result.volatility*100:.2f}%")
    print(f"[OK] Sharpe ratio: {result.sharpe_ratio:.3f}")
    print(f"[OK] Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"[OK] Total trades: {result.total_trades}")
    print(f"[OK] Win rate: {result.win_rate*100:.1f}%")
    print(f"[OK] Total fees: ${result.total_fees:.2f}")
    
    # Show some recent trades
    if result.closed_positions:
        print(f"\nRecent Trades:")
        for i, trade in enumerate(result.closed_positions[-3:], 1):  # Last 3 trades
            print(f"  {i}. {trade['entry_date']} -> {trade['exit_date']}: "
                  f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} "
                  f"P&L: ${trade['pnl']:.2f} ({trade['return_pct']:.1f}%)")
    
    return result


def test_strategy_comparison():
    """Test comparing multiple strategies"""
    print(f"\nTesting Strategy Comparison...")
    print("-" * 40)
    
    backtester = Backtester()
    
    # Create different strategies to compare
    strategies = [
        create_simple_ma_strategy(short=5, long=15),   # Fast
        create_simple_ma_strategy(short=10, long=20),  # Medium
        create_simple_ma_strategy(short=20, long=50),  # Slow
    ]
    
    # Compare strategies
    results = backtester.compare_strategies(
        strategies=strategies,
        symbol="MSFT",
        start_date="2024-03-01",
        end_date="2024-10-01"
    )
    
    print(f"[OK] Compared {len(results)} strategies on MSFT")
    print(f"\nStrategy Comparison Results:")
    print("Strategy                    Return    Sharpe   Drawdown   Trades")
    print("-" * 65)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:<25} {result.total_return_pct:>6.1f}%  {result.sharpe_ratio:>6.3f}  "
              f"{result.max_drawdown_pct:>7.1f}%  {result.total_trades:>6}")
    
    return results


def test_multi_symbol_backtest():
    """Test backtesting across multiple symbols"""
    print(f"\nTesting Multi-Symbol Backtest...")
    print("-" * 40)
    
    backtester = Backtester()
    strategy = create_simple_ma_strategy(short=10, long=20)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    results = backtester.run_multi_symbol_backtest(
        strategy=strategy,
        symbols=symbols,
        start_date="2024-05-01",
        end_date="2024-11-01"
    )
    
    print(f"[OK] Backtested {len(results)} symbols")
    print(f"\nMulti-Symbol Results:")
    print("Symbol   Return   Sharpe   Drawdown   Trades   Win Rate")
    print("-" * 55)
    
    for symbol, result in results.items():
        print(f"{symbol:<6} {result.total_return_pct:>6.1f}%  {result.sharpe_ratio:>6.3f}  "
              f"{result.max_drawdown_pct:>7.1f}%  {result.total_trades:>6}   {result.win_rate*100:>6.1f}%")
    
    return results


def test_parameter_optimization():
    """Test parameter optimization"""
    print(f"\nTesting Parameter Optimization...")
    print("-" * 40)
    
    backtester = Backtester()
    
    # Define strategy factory
    def ma_strategy_factory(short_ma, long_ma):
        return create_simple_ma_strategy(short=short_ma, long=long_ma)
    
    # Define parameter ranges to test
    parameter_ranges = {
        'short_ma': [5, 10, 15],
        'long_ma': [20, 30, 40]
    }
    
    print(f"[OK] Testing {len(parameter_ranges['short_ma']) * len(parameter_ranges['long_ma'])} parameter combinations")
    
    # Run optimization
    best_params, best_result = backtester.optimize_parameters(
        strategy_factory=ma_strategy_factory,
        parameter_ranges=parameter_ranges,
        symbol="AAPL",
        start_date="2024-04-01",
        end_date="2024-09-01",
        optimization_metric='sharpe_ratio'
    )
    
    print(f"[OK] Optimization completed")
    print(f"[OK] Best parameters: {best_params}")
    print(f"[OK] Best result:")
    print(f"     - Return: {best_result.total_return_pct:.2f}%")
    print(f"     - Sharpe: {best_result.sharpe_ratio:.3f}")
    print(f"     - Drawdown: {best_result.max_drawdown_pct:.2f}%")
    print(f"     - Trades: {best_result.total_trades}")
    
    return best_params, best_result


def main():
    """Run all backtesting tests"""
    try:
        # Test single backtest
        single_result = test_single_backtest()
        
        # Test strategy comparison
        comparison_results = test_strategy_comparison()
        
        # Test multi-symbol backtest
        multi_symbol_results = test_multi_symbol_backtest()
        
        # Test parameter optimization
        best_params, best_result = test_parameter_optimization()
        
        print("\n" + "=" * 50)
        print("[OK] All backtesting tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Single strategy backtesting with detailed metrics")
        print("- Strategy comparison and ranking") 
        print("- Multi-symbol portfolio backtesting")
        print("- Automated parameter optimization")
        print("- Comprehensive performance reporting")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Backtesting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)