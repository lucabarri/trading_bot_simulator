#!/usr/bin/env python3
"""
Test how sensitive your strategy is to different transaction cost assumptions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from backtesting.backtester import Backtester, BacktestConfig
from strategies.multi_asset_strategy import create_multi_asset_strategy

def test_cost_scenario(scenario_name, trading_fees, slippage_bps):
    """Test strategy with specific cost assumptions"""
    print(f"\nTesting {scenario_name}")
    print(f"   Fees: ${trading_fees}/trade, Slippage: {slippage_bps} bps")
    
    config = BacktestConfig(
        initial_cash=100000.0,
        trading_fees=trading_fees,
        slippage_bps=slippage_bps,
    )
    
    # Quick test with fewer symbols for speed
    symbols = ["BTC", "ETH", "ADA"]
    
    try:
        backtester = Backtester(config)
        
        # Quick optimization
        optimized_params = backtester.optimize_per_asset_parameters(
            symbols=symbols,
            start_date="2024-06-01",
            end_date="2025-01-31", 
            parameter_ranges={"short_ma": [5, 10], "long_ma": [20, 30]},
            optimization_metric="total_return",
            config=config,
        )
        
        strategy = create_multi_asset_strategy(symbols=symbols, optimized_params=optimized_params)
        
        result = backtester.run_multi_strategy_backtest(
            multi_strategy=strategy,
            start_date="2025-02-01",
            end_date="2025-08-14",
            config=config,
        )
        
        return {
            'return_pct': result.total_return_pct,
            'total_fees': result.total_fees,
            'trades': result.total_trades
        }
        
    except Exception as e:
        print(f"   ERROR: Failed: {e}")
        return None

def main():
    print("TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    # Test different cost scenarios
    scenarios = [
        ("Ultra-Low Cost (HFT-like)", 0.0, 2.0),
        ("Current Conservative", 0.0, 15.0),
        ("Current Optimistic", 0.0, 8.0),
        ("Old School (2015 era)", 5.0, 20.0),
        ("Worst Case Retail", 2.0, 30.0),
    ]
    
    results = []
    for name, fees, slippage in scenarios:
        result = test_cost_scenario(name, fees, slippage)
        if result:
            results.append((name, fees, slippage, result))
    
    # Compare results
    print(f"\nCOST SENSITIVITY RESULTS:")
    print("=" * 80)
    print("Scenario                    Fees  Slippage  Return%  Total Fees  Trades")
    print("-" * 80)
    
    for name, fees, slippage, data in results:
        print(f"{name:<25}  ${fees:>4.0f}  {slippage:>6.0f}bps  {data['return_pct']:>6.1f}%  "
              f"${data['total_fees']:>8.0f}  {data['trades']:>6}")
    
    if len(results) > 1:
        best_return = max(r[3]['return_pct'] for r in results)
        worst_return = min(r[3]['return_pct'] for r in results)
        print(f"\nCost impact: {worst_return:.1f}% to {best_return:.1f}% return")
        print(f"   Range: {best_return - worst_return:.1f} percentage points")

if __name__ == "__main__":
    main()