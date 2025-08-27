#!/usr/bin/env python3
"""
Detailed cost analysis script to understand where money is lost
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from backtesting.backtester import Backtester, BacktestConfig
from strategies.multi_asset_strategy import create_multi_asset_strategy


def analyze_costs_breakdown(result, config):
    """
    Analyze the breakdown of losses into different cost components
    """
    print(f"\nDETAILED COST BREAKDOWN ANALYSIS")
    print("=" * 60)
    
    total_loss = result.initial_value - result.final_value
    total_loss_pct = (total_loss / result.initial_value) * 100
    
    print(f"Total Loss: ${total_loss:,.2f} ({total_loss_pct:.2f}%)")
    print(f"Initial Value: ${result.initial_value:,.2f}")
    print(f"Final Value: ${result.final_value:,.2f}")
    
    # 1. TRADING FEES ANALYSIS
    print(f"\n1. TRADING FEES:")
    print("-" * 30)
    trading_fees = result.total_fees
    fees_pct = (trading_fees / result.initial_value) * 100
    fees_pct_of_loss = (trading_fees / total_loss) * 100 if total_loss > 0 else 0
    
    print(f"Total Trading Fees: ${trading_fees:,.2f}")
    print(f"Fees as % of initial capital: {fees_pct:.2f}%")
    print(f"Fees as % of total loss: {fees_pct_of_loss:.1f}%")
    print(f"Number of trades: {result.total_trades}")
    print(f"Average fee per trade: ${trading_fees/result.total_trades:.2f}" if result.total_trades > 0 else "$0.00")
    
    # 2. SLIPPAGE ANALYSIS (estimate based on trade volume)
    print(f"\n2. SLIPPAGE COSTS:")
    print("-" * 30)
    
    # Estimate slippage costs based on trades
    total_trade_volume = 0
    estimated_slippage = 0
    
    if result.closed_positions:
        for trade in result.closed_positions:
            trade_value = trade['quantity'] * trade['entry_price']
            total_trade_volume += trade_value
            # Slippage = trade_value * slippage_bps / 10000 (both entry and exit)
            estimated_slippage += trade_value * (config.slippage_bps / 10000) * 2  # 2x for entry+exit
    
    slippage_pct = (estimated_slippage / result.initial_value) * 100
    slippage_pct_of_loss = (estimated_slippage / total_loss) * 100 if total_loss > 0 else 0
    
    print(f"Estimated Total Slippage: ${estimated_slippage:,.2f}")
    print(f"Slippage as % of initial capital: {slippage_pct:.2f}%")
    print(f"Slippage as % of total loss: {slippage_pct_of_loss:.1f}%")
    print(f"Total trade volume: ${total_trade_volume:,.2f}")
    print(f"Slippage rate used: {config.slippage_bps} basis points ({config.slippage_bps/100:.3f}%)")
    
    # 3. ACTUAL TRADING LOSSES (Pure P&L from price movements)
    print(f"\n3. ACTUAL TRADING LOSSES:")
    print("-" * 30)
    
    pure_trading_loss = total_loss - trading_fees - estimated_slippage
    pure_trading_pct = (pure_trading_loss / result.initial_value) * 100
    pure_trading_pct_of_loss = (pure_trading_loss / total_loss) * 100 if total_loss > 0 else 0
    
    print(f"Pure Trading Loss: ${pure_trading_loss:,.2f}")
    print(f"Trading loss as % of initial capital: {pure_trading_pct:.2f}%")
    print(f"Trading loss as % of total loss: {pure_trading_pct_of_loss:.1f}%")
    
    # 4. WIN/LOSS TRADE ANALYSIS
    print(f"\n4. WINNING vs LOSING TRADES:")
    print("-" * 30)
    
    if result.closed_positions:
        winning_trades = [t for t in result.closed_positions if t['pnl'] > 0]
        losing_trades = [t for t in result.closed_positions if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = sum(t['pnl'] for t in losing_trades)
        
        print(f"Winning trades: {len(winning_trades)} (avg: ${total_wins/len(winning_trades):.2f})" if winning_trades else "Winning trades: 0")
        print(f"Losing trades: {len(losing_trades)} (avg: ${total_losses/len(losing_trades):.2f})" if losing_trades else "Losing trades: 0")
        print(f"Total from winners: ${total_wins:,.2f}")
        print(f"Total from losers: ${total_losses:,.2f}")
        print(f"Net trading P&L: ${total_wins + total_losses:,.2f}")
    
    # 5. SUMMARY TABLE
    print(f"\n5. LOSS BREAKDOWN SUMMARY:")
    print("=" * 60)
    print("Component                Amount         % of Capital    % of Loss")
    print("-" * 60)
    print(f"Trading Fees            ${trading_fees:>10,.2f}      {fees_pct:>7.2f}%       {fees_pct_of_loss:>6.1f}%")
    print(f"Slippage Costs          ${estimated_slippage:>10,.2f}      {slippage_pct:>7.2f}%       {slippage_pct_of_loss:>6.1f}%")
    print(f"Actual Trading Loss     ${pure_trading_loss:>10,.2f}      {pure_trading_pct:>7.2f}%       {pure_trading_pct_of_loss:>6.1f}%")
    print("-" * 60)
    print(f"TOTAL LOSS              ${total_loss:>10,.2f}      {total_loss_pct:>7.2f}%       100.0%")
    
    # 6. RECOMMENDATIONS
    print(f"\nRECOMMENDATIONS:")
    print("-" * 30)
    
    if fees_pct_of_loss > 30:
        print(f"WARNING - HIGH FEES: {fees_pct_of_loss:.1f}% of losses are from trading fees")
        print(f"   Consider: Reduce trading frequency, increase position sizes, lower fees")
    
    if slippage_pct_of_loss > 20:
        print(f"WARNING - HIGH SLIPPAGE: {slippage_pct_of_loss:.1f}% of losses are from slippage")
        print(f"   Consider: Trade less frequently, use limit orders, avoid volatile stocks")
    
    if pure_trading_pct_of_loss > 50:
        print(f"WARNING - STRATEGY ISSUE: {pure_trading_pct_of_loss:.1f}% of losses are from bad trades")
        print(f"   Consider: Review strategy logic, change parameters, add filters")
    
    print(f"\nTrading Frequency Analysis:")
    if result.duration_days > 0:
        trades_per_day = result.total_trades / result.duration_days
        print(f"   - Trades per day: {trades_per_day:.1f}")
        print(f"   - This strategy trades {'very frequently' if trades_per_day > 10 else 'frequently' if trades_per_day > 5 else 'moderately' if trades_per_day > 1 else 'infrequently'}")
        
        if trades_per_day > 10:
            print(f"   - WARNING: High frequency trading increases costs significantly")


def main():
    print("Cost Breakdown Analysis for Multi-Asset Strategy")
    print("=" * 60)
    
    # Use the same parameters as your test
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"]
    
    TRAIN_START = "2023-06-01"
    TRAIN_END = "2025-01-31"
    TEST_START = "2025-02-01"
    TEST_END = "2025-08-14"
    
    parameter_ranges = {
        "short_ma": [5, 10, 15, 20],
        "long_ma": [20, 30, 40, 50, 60],
    }
    
    config = BacktestConfig(
        initial_cash=100000.0,
        trading_fees=0.0,  # Zero fees (realistic for 2025)
        slippage_bps=15.0,  # Realistic slippage (15 basis points)
    )
    
    optimization_metric = "total_return"
    
    print(f"Running analysis with same parameters as your test...")
    
    # Run the same optimization and test
    backtester = Backtester(config)
    
    print(f"Optimizing parameters...")
    optimized_params = backtester.optimize_per_asset_parameters(
        symbols=symbols,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        parameter_ranges=parameter_ranges,
        optimization_metric=optimization_metric,
        config=config,
    )
    
    print(f"Creating multi-asset strategy...")
    multi_strategy = create_multi_asset_strategy(
        symbols=symbols, optimized_params=optimized_params
    )
    
    print(f"Running out-of-sample test...")
    result = backtester.run_multi_strategy_backtest(
        multi_strategy=multi_strategy,
        start_date=TEST_START,
        end_date=TEST_END,
        config=config,
    )
    
    # Now analyze the costs
    analyze_costs_breakdown(result, config)
    
    # Additional analysis: What if we had different costs?
    print(f"\nSENSITIVITY ANALYSIS:")
    print("=" * 40)
    print("What if we changed the costs?")
    
    total_loss = result.initial_value - result.final_value
    
    print(f"\nScenario 1: No trading fees (fees = $0)")
    no_fees_loss = total_loss - result.total_fees
    no_fees_pct = (no_fees_loss / result.initial_value) * 100
    print(f"   Loss would be: ${no_fees_loss:,.2f} ({no_fees_pct:.1f}%)")
    
    print(f"\nScenario 2: No slippage (slippage = 0 bps)")
    estimated_slippage = 0
    if result.closed_positions:
        for trade in result.closed_positions:
            trade_value = trade['quantity'] * trade['entry_price']
            estimated_slippage += trade_value * (config.slippage_bps / 10000) * 2
    
    no_slippage_loss = total_loss - estimated_slippage
    no_slippage_pct = (no_slippage_loss / result.initial_value) * 100
    print(f"   Loss would be: ${no_slippage_loss:,.2f} ({no_slippage_pct:.1f}%)")
    
    print(f"\nScenario 3: No costs at all (fees=0, slippage=0)")
    no_costs_loss = total_loss - result.total_fees - estimated_slippage
    no_costs_pct = (no_costs_loss / result.initial_value) * 100
    print(f"   Loss would be: ${no_costs_loss:,.2f} ({no_costs_pct:.1f}%)")
    print(f"   This represents the pure strategy performance")


if __name__ == "__main__":
    main()