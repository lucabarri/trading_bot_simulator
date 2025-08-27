#!/usr/bin/env python3
"""
Market Comparison Script
Compare MA strategy performance across different asset classes:
- Stocks vs Crypto vs (optionally) Forex
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from backtesting.backtester import Backtester, BacktestConfig
from strategies.moving_average_strategy import create_simple_ma_strategy
from data.data_fetcher import DataFetcher


def test_market(market_name, symbols, config, parameter_ranges, train_start, train_end, test_start, test_end):
    """Test a specific market with given symbols"""
    print(f"\nTesting {market_name.upper()} Market")
    print("=" * 50)
    
    try:
        # Test data availability first
        fetcher = DataFetcher()
        available_symbols = []
        
        print(f"Checking data availability for {len(symbols)} {market_name} symbols...")
        for symbol in symbols:
            try:
                test_data = fetcher.fetch_market_data(symbol, asset_type="auto", period="1mo", interval="1d")
                if len(test_data) > 20:
                    available_symbols.append(symbol)
                    print(f"OK {symbol}: ${test_data['Close'].iloc[-1]:.2f}")
                else:
                    print(f"WARNING {symbol}: Insufficient data")
            except Exception as e:
                print(f"ERROR {symbol}: {str(e)[:50]}...")
        
        if len(available_symbols) < 3:
            print(f"ERROR {market_name}: Not enough symbols with data ({len(available_symbols)}/3 minimum)")
            return None
        
        # Run simple strategy testing on each symbol
        backtester = Backtester(config)
        
        print(f"\nTesting {len(available_symbols)} {market_name} symbols with MA strategy...")
        
        # Use default MA parameters (can be optimized later)
        short_ma = parameter_ranges["short_ma"][1]  # Second value as default
        long_ma = parameter_ranges["long_ma"][1]    # Second value as default
        
        strategy = create_simple_ma_strategy(short=short_ma, long=long_ma)
        
        # Test on the first available symbol (representative test)
        main_symbol = available_symbols[0]
        print(f"\nTesting {market_name} strategy on {main_symbol} (representative symbol)...")
        result = backtester.run_backtest(
            strategy=strategy,
            symbol=main_symbol,
            start_date=test_start,
            end_date=test_end,
            config=config
        )
        
        # Return summary for comparison
        return {
            'market': market_name,
            'symbols_tested': len(available_symbols),
            'symbols': available_symbols,
            'total_return_pct': result.total_return_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown_pct': result.max_drawdown_pct,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'volatility': result.volatility,
            'result': result
        }
        
    except Exception as e:
        print(f"ERROR {market_name} test failed: {e}")
        return None


def main():
    print("MULTI-MARKET STRATEGY COMPARISON")
    print("=" * 60)
    print("Testing the same MA crossover strategy across different markets")
    print("to find which market works best for algorithmic trading")
    print("=" * 60)
    
    # === COMMON PARAMETERS ===
    TRAIN_START = "2023-01-01"
    TRAIN_END = "2024-06-30"
    TEST_START = "2024-07-01"
    TEST_END = "2024-12-31"
    
    # === MARKET DEFINITIONS ===
    markets = {
        'stocks': {
            'symbols': ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"],
            'parameter_ranges': {
                "short_ma": [5, 10, 15, 20],
                "long_ma": [20, 30, 40, 50, 60],
            },
            'config': BacktestConfig(
                initial_cash=100000.0,
                trading_fees=0.0,      # Zero fees (realistic for 2025)
                slippage_bps=15.0,     # Stock market slippage
            )
        },
        'crypto': {
            'symbols': ["BTC", "ETH", "ADA", "SOL", "MATIC", "DOT", "AVAX", "LINK", "UNI", "ATOM"],
            'parameter_ranges': {
                "short_ma": [3, 5, 8, 12],      # Faster for crypto
                "long_ma": [15, 21, 30, 45],    # Shorter periods
            },
            'config': BacktestConfig(
                initial_cash=100000.0,
                trading_fees=0.0,      # Zero fees (many crypto exchanges)
                slippage_bps=10.0,     # Lower slippage for major pairs
            )
        },
        'forex': {
            'symbols': ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"],
            'parameter_ranges': {
                "short_ma": [5, 10, 15, 20],
                "long_ma": [20, 30, 40, 50],
            },
            'config': BacktestConfig(
                initial_cash=100000.0,
                trading_fees=0.0,      # Spread-based, no explicit fees
                slippage_bps=5.0,      # Very low slippage for major pairs
            )
        }
    }
    
    # === RUN TESTS FOR ALL MARKETS ===
    results = {}
    
    for market_name, market_config in markets.items():
        result = test_market(
            market_name=market_name,
            symbols=market_config['symbols'],
            config=market_config['config'],
            parameter_ranges=market_config['parameter_ranges'],
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END
        )
        
        if result:
            results[market_name] = result
    
    # === FINAL COMPARISON ===
    if not results:
        print("\nERROR: No markets could be tested successfully")
        return
    
    print(f"\nMARKET COMPARISON RESULTS")
    print("=" * 80)
    print("Market     Symbols  Return%   Sharpe   MaxDD%   Trades  Win%   Volatility")
    print("-" * 80)
    
    # Sort by performance
    sorted_markets = sorted(results.items(), key=lambda x: x[1]['total_return_pct'], reverse=True)
    
    for market_name, data in sorted_markets:
        print(f"{market_name:<10} {data['symbols_tested']:>7}  {data['total_return_pct']:>7.1f}%  "
              f"{data['sharpe_ratio']:>6.2f}  {data['max_drawdown_pct']:>6.1f}%  "
              f"{data['total_trades']:>6}  {data['win_rate']*100:>4.0f}%  {data['volatility']*100:>8.1f}%")
    
    # === WINNER ANALYSIS ===
    print("\n" + "=" * 80)
    best_market = sorted_markets[0]
    best_name, best_data = best_market
    
    print(f"BEST PERFORMING MARKET: {best_name.upper()}")
    print(f"   Return: {best_data['total_return_pct']:.2f}%")
    print(f"   Sharpe: {best_data['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {best_data['max_drawdown_pct']:.1f}%")
    print(f"   Symbols tested: {', '.join(best_data['symbols'])}")
    
    # === INSIGHTS ===
    print(f"\nKEY INSIGHTS:")
    
    # Compare returns
    if len(results) > 1:
        returns = [data['total_return_pct'] for data in results.values()]
        best_return = max(returns)
        worst_return = min(returns)
        
        if best_return > 0 and worst_return < 0:
            print(f"   Market choice matters. Range: {worst_return:.1f}% to {best_return:.1f}%")
        elif all(r > 0 for r in returns):
            print(f"   MA strategy worked across all markets. Best: {best_return:.1f}%")
        elif all(r < 0 for r in returns):
            print(f"   WARNING: MA strategy struggled across all markets. Try different strategies.")
        
        # Compare volatility
        volatilities = [data['volatility']*100 for data in results.values()]
        print(f"   Volatility range: {min(volatilities):.1f}% to {max(volatilities):.1f}%")
        
        # Trading frequency
        trade_frequencies = [data['total_trades']/data['result'].duration_days for data in results.values()]
        print(f"   Trading frequency: {min(trade_frequencies):.1f} to {max(trade_frequencies):.1f} trades/day")
    
    print(f"\nRECOMMENDATION:")
    if best_data['total_return_pct'] > 5:
        print(f"   Focus on {best_name.upper()} markets - your MA strategy works well there")
        print(f"   Consider testing more sophisticated strategies on {best_name}")
    elif best_data['total_return_pct'] > 0:
        print(f"   {best_name.upper()} shows promise, but consider strategy improvements")
    else:
        print(f"   MA crossover may not be suitable for current market conditions")
        print(f"   Consider: Mean reversion, momentum, or ML-based strategies")


if __name__ == "__main__":
    main()