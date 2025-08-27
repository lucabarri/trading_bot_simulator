#!/usr/bin/env python3
"""
Enhanced Strategy Test with Comprehensive Visualizations
Combines strategy testing with automatic visualization generation
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from backtesting.backtester import Backtester, BacktestConfig
from strategies.multi_asset_strategy import create_multi_asset_strategy
from strategies.moving_average_strategy import create_simple_ma_strategy
from utils.strategy_visualizer import StrategyVisualizer


def main():
    print("Enhanced Multi-Asset Strategy Testing with Visualizations")
    print("=" * 70)
    print("This will test strategies AND create comprehensive visualizations")
    print("=" * 70)

    # === PARAMETERS ===
    symbols = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "TSLA",  # Tesla
    ]

    # Date ranges
    TRAIN_START = "2024-01-01"
    TRAIN_END = "2024-12-31"
    TEST_START = "2025-01-01"
    TEST_END = "2025-08-14"

    # Parameter ranges
    parameter_ranges = {
        "short_ma": [5, 10, 15],      # Smaller range for demo speed
        "long_ma": [20, 30, 40],      # Smaller range for demo speed
    }

    # Portfolio settings
    config = BacktestConfig(
        initial_cash=100000.0,
        trading_fees=0.0,
        slippage_bps=15.0,
    )

    optimization_metric = "total_return"

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing period: {TEST_START} to {TEST_END}")
    print(f"Parameter combinations: {len(parameter_ranges['short_ma']) * len(parameter_ranges['long_ma'])}")

    # === INITIALIZE COMPONENTS ===
    backtester = Backtester(config)
    visualizer = StrategyVisualizer("strategy_visualizations")

    # === STEP 1: OPTIMIZE NORMAL STRATEGY ===
    print(f"\nSTEP 1: Optimizing NORMAL MA strategy...")
    
    try:
        normal_optimized_params = backtester.optimize_per_asset_parameters(
            symbols=symbols,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            parameter_ranges=parameter_ranges,
            optimization_metric=optimization_metric,
            config=config,
            use_inverse_strategy=False  # Normal strategy
        )

        print(f"\nNORMAL STRATEGY OPTIMIZATION RESULTS:")
        print("Symbol   Short MA   Long MA   Optimized For")
        print("-" * 45)
        for symbol, params in normal_optimized_params.items():
            print(f"{symbol:<6}   {params['short']:>6}     {params['long']:>6}      {optimization_metric}")

    except Exception as e:
        print(f"ERROR: Normal optimization failed: {e}")
        return

    # === STEP 2: OPTIMIZE INVERSE STRATEGY ===
    print(f"\nSTEP 2: Optimizing INVERSE MA strategy...")
    
    try:
        inverse_optimized_params = backtester.optimize_per_asset_parameters(
            symbols=symbols,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            parameter_ranges=parameter_ranges,
            optimization_metric=optimization_metric,
            config=config,
            use_inverse_strategy=True  # Inverse strategy
        )

        print(f"\nINVERSE STRATEGY OPTIMIZATION RESULTS:")
        print("Symbol   Short MA   Long MA   Optimized For")
        print("-" * 45)
        for symbol, params in inverse_optimized_params.items():
            print(f"{symbol:<6}   {params['short']:>6}     {params['long']:>6}      {optimization_metric}")

    except Exception as e:
        print(f"ERROR: Inverse optimization failed: {e}")
        return

    # === STEP 3: CREATE AND TEST STRATEGIES ===
    print(f"\nSTEP 3: Creating and testing strategies...")
    
    results = {}
    
    # Normal Strategy
    print(f"\nTesting Normal Multi-Asset Strategy...")
    try:
        normal_strategy = create_multi_asset_strategy(
            symbols=symbols, 
            optimized_params=normal_optimized_params
        )
        
        normal_result = backtester.run_multi_strategy_backtest(
            multi_strategy=normal_strategy,
            start_date=TEST_START,
            end_date=TEST_END,
            config=config,
        )
        
        results["Normal_MA_Strategy"] = normal_result
        print(f"Normal Strategy: {normal_result.total_return_pct:.2f}% return")
        
    except Exception as e:
        print(f"ERROR: Normal strategy test failed: {e}")

    # Inverse Strategy
    print(f"\nTesting Inverse Multi-Asset Strategy...")
    try:
        inverse_strategy = create_inverse_multi_asset_strategy(
            symbols=symbols, 
            optimized_params=inverse_optimized_params
        )
        
        inverse_result = backtester.run_multi_strategy_backtest(
            multi_strategy=inverse_strategy,
            start_date=TEST_START,
            end_date=TEST_END,
            config=config,
        )
        
        results["Inverse_MA_Strategy"] = inverse_result
        print(f"Inverse Strategy: {inverse_result.total_return_pct:.2f}% return")
        
    except Exception as e:
        print(f"ERROR: Inverse strategy test failed: {e}")

    if not results:
        print("\nERROR: No strategies succeeded")
        return

    # === STEP 4: SHOW BASIC RESULTS ===
    print(f"\nSTRATEGY COMPARISON RESULTS:")
    print("=" * 70)
    
    for strategy_name, result in results.items():
        print(f"\n{strategy_name}:")
        print(f"   Total Return: {result.total_return_pct:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"   Win Rate: {result.win_rate*100:.1f}%")
        print(f"   Total Trades: {result.total_trades}")

    # === STEP 5: CREATE COMPREHENSIVE VISUALIZATIONS ===
    print(f"\nSTEP 5: Creating Comprehensive Visualizations...")
    print("This may take a moment to generate all charts...")
    
    all_files_created = {}
    
    # Individual strategy visualizations
    for strategy_name, result in results.items():
        print(f"\nCreating visualizations for {strategy_name}...")
        try:
            # Get allocation summary for multi-asset visualization
            if hasattr(result, 'strategy') and hasattr(result.strategy, 'get_allocation_summary'):
                try:
                    allocation_summary = result.strategy.get_allocation_summary()
                    allocations = {k: v['allocation_pct'] for k, v in allocation_summary.items()}
                except:
                    allocations = None
            else:
                allocations = None
            
            if allocations:
                # Multi-asset visualization
                files = visualizer.visualize_multi_asset_strategy(
                    result=result,
                    asset_allocations=allocations,
                    save_plots=True,
                    show_plots=False
                )
            else:
                # Single strategy visualization
                files = visualizer.visualize_single_strategy(
                    result=result,
                    save_plots=True,
                    show_plots=False,
                    include_interactive=True
                )
            
            all_files_created[strategy_name] = files
            print(f"   Created {len(files)} visualizations for {strategy_name}")
            
        except Exception as e:
            print(f"   ERROR: Failed to create visualizations for {strategy_name}: {e}")

    # Strategy comparison visualizations
    if len(results) > 1:
        print(f"\nCreating strategy comparison visualizations...")
        try:
            comparison_files = visualizer.compare_strategies(
                results=results,
                comparison_name="normal_vs_inverse",
                save_plots=True,
                show_plots=False,
                include_interactive=True
            )
            all_files_created["Comparison"] = comparison_files
            print(f"   Created {len(comparison_files)} comparison charts")
        except Exception as e:
            print(f"   ERROR: Failed to create comparison visualizations: {e}")

    # HTML Report
    print(f"\nCreating comprehensive HTML report...")
    try:
        main_result = list(results.values())[0]
        other_results = {k: v for k, v in results.items() if v != main_result}
        
        html_report = visualizer.generate_strategy_report(
            result=main_result,
            strategy_type="multi_asset",
            comparison_results=other_results if other_results else None
        )
        print(f"   HTML report created: {os.path.basename(html_report)}")
    except Exception as e:
        print(f"   ERROR: Failed to create HTML report: {e}")
        html_report = None

    # === FINAL SUMMARY ===
    print(f"\n" + "=" * 70)
    print(f"ENHANCED STRATEGY TESTING COMPLETE")
    print(f"=" * 70)
    
    # Strategy performance summary
    print(f"\nFINAL PERFORMANCE COMPARISON:")
    print("-" * 50)
    best_strategy = None
    best_return = float('-inf')
    
    for strategy_name, result in results.items():
        status = "+" if result.total_return_pct > 0 else "-"
        print(f"{status} {strategy_name:<20} {result.total_return_pct:>8.2f}% return")
        print(f"   Sharpe: {result.sharpe_ratio:>6.3f}  MaxDD: {result.max_drawdown_pct:>6.2f}%  Trades: {result.total_trades}")
        
        if result.total_return_pct > best_return:
            best_return = result.total_return_pct
            best_strategy = strategy_name

    if best_strategy:
        print(f"\nBEST PERFORMER: {best_strategy} ({best_return:.2f}% return)")

    # Files created summary
    print(f"\nVISUALIZATIONS CREATED:")
    total_files = sum(len(files) for files in all_files_created.values())
    print(f"   Total charts created: {total_files}")
    print(f"   Saved to: strategy_visualizations/")
    
    if html_report:
        print(f"   HTML report: {os.path.basename(html_report)}")

    print(f"\nChart Types Created:")
    print(f"   • Performance dashboards")
    print(f"   • Equity curves with benchmarks")
    print(f"   • Drawdown analysis")
    print(f"   • Monthly returns heatmaps") 
    print(f"   • Trade analysis")
    print(f"   • Risk analysis")
    print(f"   • Asset allocation charts")
    print(f"   • Strategy comparisons")
    print(f"   • Interactive charts")

    print(f"\nTO VIEW YOUR CHARTS:")
    print(f"   1. Open the 'strategy_visualizations' folder")
    print(f"   2. Browse subfolders for different chart types")
    if html_report:
        print(f"   3. Open {os.path.basename(html_report)} for the complete report")

    print(f"\nStrategy Insights:")
    normal_result = results.get("Normal_MA_Strategy")
    inverse_result = results.get("Inverse_MA_Strategy")
    
    if normal_result and inverse_result:
        if normal_result.total_return_pct > 0 and inverse_result.total_return_pct < 0:
            print(f"   Normal MA strategy works (+{normal_result.total_return_pct:.1f}% vs {inverse_result.total_return_pct:.1f}%)")
        elif normal_result.total_return_pct < 0 and inverse_result.total_return_pct > 0:
            print(f"   Inverse strategy works better ({inverse_result.total_return_pct:.1f}% vs {normal_result.total_return_pct:.1f}%)")
        elif normal_result.total_return_pct > 0 and inverse_result.total_return_pct > 0:
            print(f"   Both strategies profitable - markets might be trending")
        else:
            print(f"   Both strategies lost money - MA might not work in this period")

    print(f"\nAll visualizations saved and ready to view")


if __name__ == "__main__":
    main()