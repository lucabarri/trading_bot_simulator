#!/usr/bin/env python3
"""
Custom Strategy Testing Script
Modify the parameters below to test different MA strategy configurations
"""

import sys
import os
import pandas as pd
import random

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.data_fetcher import DataFetcher
from strategies.moving_average_strategy import MovingAverageCrossoverStrategy
from trading.portfolio import Portfolio
from trading.position import PositionType

# Top 100 US stocks (sample)
TOP_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "V",
    "JNJ",
    "WMT",
    "XOM",
    "UNH",
    "LLY",
    "JPM",
    "PG",
    "MA",
    "HD",
    "CVX",
    "ABBV",
    "MRK",
    "ORCL",
    "COST",
    "AVGO",
    "BAC",
    "KO",
    "PEP",
    "TMO",
    "NFLX",
    "ACN",
    "MCD",
    "CSCO",
    "LIN",
    "AMD",
    "ABT",
    "DHR",
    "VZ",
    "ADBE",
    "CRM",
    "TXN",
    "NKE",
    "QCOM",
    "BMY",
    "PM",
    "NEE",
    "WFC",
    "HON",
    "RTX",
    "COP",
    "UPS",
    "AMGN",
    "T",
    "SPGI",
    "LOW",
    "IBM",
    "CAT",
    "GS",
    "DE",
    "AXP",
    "GILD",
    "LMT",
    "BKNG",
    "BLK",
    "SYK",
    "MDLZ",
    "AMAT",
    "ELV",
    "ISRG",
    "PLD",
    "REGN",
    "ADP",
    "TJX",
    "CVS",
    "VRTX",
    "ZTS",
    "MMC",
    "TMUS",
    "C",
    "FI",
    "SO",
    "PGR",
    "DUK",
    "BSX",
    "SHW",
    "ITW",
    "NOC",
    "EMR",
    "WM",
    "GD",
    "APD",
    "PYPL",
    "ICE",
    "EQIX",
    "NSC",
    "AON",
    "CL",
    "SLB",
    "USB",
    "GM",
    "F",
]


def test_single_stock(
    symbol,
    short_window=10,
    long_window=20,
    min_gap=0.002,
    initial_cash=100000,
    position_size=0.2,
    fees=5.0,
    data_period="2y",
    verbose=False,
):
    """Test strategy on a single stock - simplified version"""

    try:
        # Create strategy
        strategy = MovingAverageCrossoverStrategy(
            short_window=short_window,
            long_window=long_window,
            min_crossover_gap=min_gap,
        )

        # Load data
        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data(symbol, period=data_period, interval="1d")

        if len(data) < strategy.get_required_history():
            return None

        # Generate signals
        signals = strategy.generate_signals(data)
        trading_signals = [s for s in signals if s.signal.value != "HOLD"]

        if len(trading_signals) == 0:
            return None

        # Simulate trading
        cash = initial_cash
        current_position = None
        completed_trades = []

        for signal in trading_signals:
            price = signal.price
            signal_type = signal.signal.value

            if signal_type == "BUY" and current_position is None:
                trade_value = cash * position_size
                shares = int((trade_value - fees) / price)
                total_cost = shares * price + fees

                if shares > 0 and total_cost <= cash:
                    cash -= total_cost
                    current_position = {"shares": shares, "entry_price": price}

            elif signal_type == "SELL" and current_position is not None:
                shares = current_position["shares"]
                entry_price = current_position["entry_price"]
                total_proceeds = shares * price - fees
                pnl = (price - entry_price) * shares - fees

                cash += total_proceeds
                completed_trades.append(pnl)
                current_position = None

        # Calculate final results
        current_price = data["Close"].iloc[-1]
        if current_position is not None:
            unrealized_pnl = (
                current_price - current_position["entry_price"]
            ) * current_position["shares"]
            final_value = cash + (current_position["shares"] * current_price)
        else:
            unrealized_pnl = 0
            final_value = cash

        total_pnl = final_value - initial_cash
        total_return_pct = (total_pnl / initial_cash) * 100

        # Buy and hold comparison
        start_price = data["Close"].iloc[0]
        buy_hold_shares = int(initial_cash / start_price)
        buy_hold_final = buy_hold_shares * current_price + (
            initial_cash - buy_hold_shares * start_price
        )
        buy_hold_return = ((buy_hold_final - initial_cash) / initial_cash) * 100

        # Trading stats
        winning_trades = [t for t in completed_trades if t > 0]
        losing_trades = [t for t in completed_trades if t < 0]
        win_rate = (
            len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
        )

        result = {
            "symbol": symbol,
            "total_return": total_return_pct,
            "buy_hold_return": buy_hold_return,
            "outperformance": total_return_pct - buy_hold_return,
            "total_trades": len(completed_trades),
            "win_rate": win_rate,
            "final_value": final_value,
            "signals_generated": len(trading_signals),
            "has_open_position": current_position is not None,
            "data_days": len(data),
        }

        if verbose:
            print(f"\n{symbol} Results:")
            print(
                f"  Strategy Return: {total_return_pct:+.1f}% | Buy&Hold: {buy_hold_return:+.1f}% | Outperformance: {total_return_pct - buy_hold_return:+.1f}%"
            )
            print(
                f"  Trades: {len(completed_trades)} | Win Rate: {win_rate:.0f}% | Signals: {len(trading_signals)}"
            )

        return result

    except Exception as e:
        if verbose:
            print(f"  ERROR testing {symbol}: {e}")
        return None


def test_custom_strategy(SYMBOL=None):
    """Test strategy with your custom parameters on a single stock"""

    # ========== CUSTOMIZE THESE PARAMETERS ==========
    # SYMBOL = "AMAT"  # Stock symbol to test
    SHORT_WINDOW = 10  # Short moving average period
    LONG_WINDOW = 20  # Long moving average period
    MIN_CROSSOVER_GAP = 0.002  # Minimum gap to avoid noise (0.2%)
    DATA_PERIOD = "2y"  # Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    # Trading simulation parameters
    INITIAL_CASH = 100000  # Starting portfolio value ($100,000)
    POSITION_SIZE_PCT = 0.2  # Use 20% of portfolio per trade
    TRADING_FEES = 5.0  # $5 per trade
    # ================================================

    print(f"TESTING CUSTOM MA STRATEGY WITH P&L SIMULATION")
    print("=" * 70)
    print(f"Symbol: {SYMBOL}")
    print(f"Short MA: {SHORT_WINDOW} days")
    print(f"Long MA: {LONG_WINDOW} days")
    print(f"Min Gap: {MIN_CROSSOVER_GAP*100:.2f}%")
    print(f"Data Period: {DATA_PERIOD}")
    print(f"Initial Cash: ${INITIAL_CASH:,}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.1f}% per trade")
    print(f"Trading Fees: ${TRADING_FEES} per trade")
    print("=" * 70)

    # Create custom strategy
    strategy = MovingAverageCrossoverStrategy(
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW,
        min_crossover_gap=MIN_CROSSOVER_GAP,
    )

    print(f"[OK] Created strategy: {strategy.name}")
    print(f"[OK] Required history: {strategy.get_required_history()} days")

    # Load data
    fetcher = DataFetcher()
    data = fetcher.fetch_stock_data(SYMBOL, period=DATA_PERIOD, interval="1d")
    print(f"[OK] Loaded {len(data)} days of {SYMBOL} data")
    print(
        f"[OK] Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    )

    # Generate signals
    signals = strategy.generate_signals(data)
    trading_signals = [s for s in signals if s.signal.value != "HOLD"]

    print(
        f"\n[OK] Generated {len(signals)} total signals ({len(trading_signals)} trading signals)"
    )

    # Show current strategy state
    state = strategy.get_strategy_state(data)
    print(f"\nCURRENT STRATEGY STATE:")
    print(f"- Status: {state['status']}")
    print(f"- Latest Close: ${state['latest_close']:.2f}")
    print(f"- Short MA ({SHORT_WINDOW}): ${state['short_ma']:.2f}")
    print(f"- Long MA ({LONG_WINDOW}): ${state['long_ma']:.2f}")
    print(f"- MA Difference: ${state['ma_diff']:.2f} ({state['ma_diff_pct']*100:.2f}%)")
    print(f"- Current Trend: {state['trend']}")

    # Print ALL trading signals
    # ========== SIMULATE PORTFOLIO TRADING ==========
    cash = INITIAL_CASH
    current_position = (
        None  # Will store: {'shares': int, 'entry_price': float, 'entry_date': str}
    )
    trade_log = []

    print(f"\nSIMULATING TRADES WITH PORTFOLIO...")
    print(f"Starting cash: ${cash:,.2f}")

    for signal in trading_signals:
        date_str = signal.timestamp.strftime("%Y-%m-%d")
        price = signal.price
        signal_type = signal.signal.value

        if signal_type == "BUY" and current_position is None:
            # Calculate position size based on current cash
            trade_value = cash * POSITION_SIZE_PCT
            shares = int((trade_value - TRADING_FEES) / price)
            total_cost = shares * price + TRADING_FEES

            if shares > 0 and total_cost <= cash:
                # Execute BUY
                cash -= total_cost
                current_position = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": date_str,
                }

                portfolio_value = cash + (shares * price)

                trade_log.append(
                    {
                        "date": date_str,
                        "action": "BUY",
                        "shares": shares,
                        "price": price,
                        "total_cost": total_cost,
                        "fees": TRADING_FEES,
                        "portfolio_value": portfolio_value,
                        "signal_conf": signal.confidence,
                    }
                )
            else:
                trade_log.append(
                    {
                        "date": date_str,
                        "action": "FAILED_BUY",
                        "reason": f"Insufficient funds: need ${total_cost:.0f}, have ${cash:.0f}",
                        "price": price,
                        "shares_calculated": shares,
                        "portfolio_value": cash,
                    }
                )

        elif signal_type == "SELL" and current_position is not None:
            # Execute SELL
            shares = current_position["shares"]
            entry_price = current_position["entry_price"]
            total_proceeds = shares * price - TRADING_FEES
            pnl = (price - entry_price) * shares - TRADING_FEES

            cash += total_proceeds
            portfolio_value = cash

            trade_log.append(
                {
                    "date": date_str,
                    "action": "SELL",
                    "shares": shares,
                    "price": price,
                    "entry_price": entry_price,
                    "total_proceeds": total_proceeds,
                    "fees": TRADING_FEES,
                    "pnl": pnl,
                    "portfolio_value": portfolio_value,
                    "signal_conf": signal.confidence,
                }
            )

            current_position = None  # Close position

        else:
            # Log skipped signal
            reason = "no_position" if signal_type == "SELL" else "already_have_position"
            trade_log.append(
                {
                    "date": date_str,
                    "action": f"SKIPPED_{signal_type}",
                    "reason": reason,
                    "price": price,
                    "portfolio_value": cash
                    + (current_position["shares"] * price if current_position else 0),
                }
            )

    # Calculate final portfolio value
    current_price = data["Close"].iloc[-1]
    if current_position is not None:
        final_portfolio_value = cash + (current_position["shares"] * current_price)
        unrealized_pnl = (
            current_price - current_position["entry_price"]
        ) * current_position["shares"]
    else:
        final_portfolio_value = cash
        unrealized_pnl = 0

    print(f"\nTRADING RESULTS:")
    print("-" * 100)
    print(
        f"{'Date':<12} {'Action':<6} {'Shares':<7} {'Price':<8} {'Total':<10} {'P&L':<10} {'Portfolio':<12} {'Conf':<6}"
    )
    print("-" * 100)

    for trade in trade_log:
        if trade["action"] == "BUY":
            print(
                f"{trade['date']:<12} {trade['action']:<6} {trade['shares']:<7} ${trade['price']:<7.2f} ${trade['total_cost']:<9.0f} {'--':<10} ${trade['portfolio_value']:<11.0f} {trade['signal_conf']:<6.3f}"
            )
        elif trade["action"] == "SELL":
            pnl_str = f"${trade['pnl']:+.0f}"
            print(
                f"{trade['date']:<12} {trade['action']:<6} {trade['shares']:<7} ${trade['price']:<7.2f} ${trade['total_proceeds']:<9.0f} {pnl_str:<10} ${trade['portfolio_value']:<11.0f} {trade['signal_conf']:<6.3f}"
            )
        else:
            reason = trade.get("reason", "Failed")
            print(
                f"{trade['date']:<12} {trade['action']:<12} {'--':<7} ${trade['price']:<7.2f} {reason[:15]:<15} {'--':<10} ${trade['portfolio_value']:<11.0f} {'--':<6}"
            )

    # print("\nALL TRADING SIGNALS SUMMARY:")
    # print("-" * 90)
    # print(
    #     f"{'Date':<12} {'Signal':<6} {'Price':<10} {'Conf':<6} {'MA Diff':<10} {'Type':<10} {'Executed':<10}"
    # )
    # print("-" * 90)

    executed_dates = {
        trade["date"] for trade in trade_log if not trade["action"].startswith("FAILED")
    }

    for signal in trading_signals:
        date_str = signal.timestamp.strftime("%Y-%m-%d")
        crossover_type = signal.metadata.get("crossover_type", "N/A")
        ma_diff_pct = signal.metadata.get("ma_diff_pct", 0) * 100
        executed = "YES" if date_str in executed_dates else "NO"

        # print(
        #     f"{date_str:<12} {signal.signal.value:<6} ${signal.price:<9.2f} {signal.confidence:<6.3f} {ma_diff_pct:<9.2f}% {crossover_type:<10} {executed:<10}"
        # )

    # ========== COMPREHENSIVE P&L ANALYSIS ==========
    total_pnl = final_portfolio_value - INITIAL_CASH
    total_return_pct = (total_pnl / INITIAL_CASH) * 100

    # Calculate buy and hold comparison
    start_price = data["Close"].iloc[0]
    end_price = current_price
    buy_hold_shares = int(INITIAL_CASH / start_price)
    buy_hold_final_value = buy_hold_shares * end_price + (
        INITIAL_CASH - buy_hold_shares * start_price
    )
    buy_hold_pnl = buy_hold_final_value - INITIAL_CASH
    buy_hold_return_pct = (buy_hold_pnl / INITIAL_CASH) * 100

    # Analyze completed trades
    completed_trades = [t for t in trade_log if t["action"] == "SELL"]
    winning_trades = [t for t in completed_trades if t["pnl"] > 0]
    losing_trades = [t for t in completed_trades if t["pnl"] < 0]

    total_fees_paid = (
        len([t for t in trade_log if t["action"] in ["BUY", "SELL"]]) * TRADING_FEES
    )
    total_realized_pnl = sum(t["pnl"] for t in completed_trades)

    print(f"\n" + "=" * 80)
    print(f"FINAL PROFIT & LOSS ANALYSIS")
    print("=" * 80)

    print(f"\nSTRATEGY PERFORMANCE:")
    print(f"- Initial Portfolio Value:     ${INITIAL_CASH:>12,}")
    print(f"- Final Portfolio Value:       ${final_portfolio_value:>12,.2f}")
    print(
        f"- Total P&L:                   ${total_pnl:>12,.2f} ({total_return_pct:+.2f}%)"
    )
    print(f"- Realized P&L:                ${total_realized_pnl:>12,.2f}")
    print(f"- Unrealized P&L:              ${unrealized_pnl:>12,.2f}")
    print(f"- Total Fees Paid:             ${total_fees_paid:>12,.2f}")
    print(f"- Current Cash:                ${cash:>12,.2f}")

    if current_position:
        print(
            f"- Open Position:               {current_position['shares']} shares @ ${current_position['entry_price']:.2f}"
        )
        print(
            f"- Position Value:              ${current_position['shares'] * current_price:>12,.2f}"
        )

    print(f"\nBUY & HOLD COMPARISON:")
    print(f"- Buy & Hold Final Value:      ${buy_hold_final_value:>12,.2f}")
    print(
        f"- Buy & Hold P&L:              ${buy_hold_pnl:>12,.2f} ({buy_hold_return_pct:+.2f}%)"
    )
    print(
        f"- Strategy vs Buy & Hold:      ${total_pnl - buy_hold_pnl:>12,.2f} ({total_return_pct - buy_hold_return_pct:+.2f}%)"
    )

    if completed_trades:
        print(f"\nTRADING STATISTICS:")
        print(f"- Total Completed Trades:      {len(completed_trades):>12}")
        print(
            f"- Winning Trades:              {len(winning_trades):>12} ({len(winning_trades)/len(completed_trades)*100:.1f}%)"
        )
        print(
            f"- Losing Trades:               {len(losing_trades):>12} ({len(losing_trades)/len(completed_trades)*100:.1f}%)"
        )

        if winning_trades:
            avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades)
            max_win = max(t["pnl"] for t in winning_trades)
            print(f"- Average Winning Trade:       ${avg_win:>12,.2f}")
            print(f"- Best Winning Trade:          ${max_win:>12,.2f}")

        if losing_trades:
            avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades)
            max_loss = min(t["pnl"] for t in losing_trades)
            print(f"- Average Losing Trade:        ${avg_loss:>12,.2f}")
            print(f"- Worst Losing Trade:          ${max_loss:>12,.2f}")

            if avg_loss != 0:
                total_wins = sum(t["pnl"] for t in winning_trades)
                total_losses = abs(sum(t["pnl"] for t in losing_trades))
                profit_factor = (
                    total_wins / total_losses if total_losses > 0 else float("inf")
                )
                print(f"- Profit Factor:               {profit_factor:>12.2f}")

    # Signal analysis
    if trading_signals:
        buy_signals = [s for s in trading_signals if s.signal.value == "BUY"]
        sell_signals = [s for s in trading_signals if s.signal.value == "SELL"]

        print(f"\nSIGNAL ANALYSIS:")
        print(f"- Total Trading Signals:       {len(trading_signals):>12}")
        print(f"- BUY Signals:                 {len(buy_signals):>12}")
        print(f"- SELL Signals:                {len(sell_signals):>12}")
        print(f"- Executed Trades:             {len(trade_log):>12}")
        print(
            f"- Signal Execution Rate:       {len(trade_log)/len(trading_signals)*100:>11.1f}%"
        )

        if buy_signals and sell_signals:
            avg_buy_conf = sum(s.confidence for s in buy_signals) / len(buy_signals)
            avg_sell_conf = sum(s.confidence for s in sell_signals) / len(sell_signals)
            # print(f"- Average BUY Confidence:      {avg_buy_conf:>12.3f}")
            # print(f"- Average SELL Confidence:     {avg_sell_conf:>12.3f}")

    print(f"\nSTRATEGY VERDICT:")
    if total_return_pct > buy_hold_return_pct:
        print(
            f"[WIN] STRATEGY BEATS BUY & HOLD by {total_return_pct - buy_hold_return_pct:+.2f}%"
        )
    else:
        print(
            f"[LOSS] STRATEGY UNDERPERFORMS BUY & HOLD by {buy_hold_return_pct - total_return_pct:+.2f}%"
        )

    if total_pnl > 0:
        print(
            f"[PROFIT] PROFITABLE STRATEGY: ${total_pnl:,.2f} profit ({total_return_pct:+.2f}%)"
        )
    else:
        print(
            f"[LOSS] LOSING STRATEGY: ${total_pnl:,.2f} loss ({total_return_pct:+.2f}%)"
        )

    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)
    print("To modify parameters, edit the CUSTOMIZE section at the top of this script")
    print(
        f"Current settings: {SYMBOL}, MA({SHORT_WINDOW},{LONG_WINDOW}), Gap={MIN_CROSSOVER_GAP*100:.2f}%, Size={POSITION_SIZE_PCT*100:.0f}%"
    )

    return {
        "symbol": SYMBOL,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "outperformance_pct": total_return_pct - buy_hold_return_pct,
        "total_trades": len(completed_trades),
        "win_rate_pct": (
            (len(winning_trades) / len(completed_trades) * 100)
            if completed_trades
            else 0
        ),
        "final_portfolio_value": final_portfolio_value,
        "data_days": len(data),
        "signals_generated": len(trading_signals),
    }


def test_multi_stocks():
    """Test strategy on 10 random stocks from top 100"""

    # Strategy parameters
    SHORT_WINDOW = 10
    LONG_WINDOW = 20
    MIN_GAP = 0.002
    INITIAL_CASH = 100000
    POSITION_SIZE = 0.2
    FEES = 5.0
    DATA_PERIOD = "2y"

    print(f"TESTING MA STRATEGY ON 10 RANDOM STOCKS")
    print("=" * 60)
    print(
        f"Strategy: MA({SHORT_WINDOW},{LONG_WINDOW}), Gap={MIN_GAP*100:.1f}%, Size={POSITION_SIZE*100:.0f}%"
    )
    print(f"Capital: ${INITIAL_CASH:,}, Period: {DATA_PERIOD}")
    print("=" * 60)

    # Select 10 random stocks
    random.seed(42)  # For reproducible results
    selected_stocks = random.sample(TOP_STOCKS, 100)

    results = []

    print(f"\nTesting {len(selected_stocks)} stocks: {', '.join(selected_stocks)}")
    print("-" * 60)

    for i, symbol in enumerate(selected_stocks, 1):
        print(f"[{i:2}/10] Testing {symbol}...", end=" ")

        result = test_single_stock(
            symbol=symbol,
            short_window=SHORT_WINDOW,
            long_window=LONG_WINDOW,
            min_gap=MIN_GAP,
            initial_cash=INITIAL_CASH,
            position_size=POSITION_SIZE,
            fees=FEES,
            data_period=DATA_PERIOD,
            verbose=False,
        )

        if result:
            results.append(result)
            print(
                f"Return: {result['total_return']:+6.1f}% | B&H: {result['buy_hold_return']:+6.1f}% | Out: {result['outperformance']:+6.1f}%"
            )
        else:
            print("FAILED (no signals or data)")

    if not results:
        print("\nNo successful tests. Try different parameters.")
        return

    # Sort by outperformance
    results.sort(key=lambda x: x["outperformance"], reverse=True)

    print(f"\n" + "=" * 90)
    print(f"MULTI-STOCK RESULTS SUMMARY ({len(results)} stocks)")
    print("=" * 90)
    print(
        f"{'Rank':<4} {'Symbol':<6} {'Strategy':<10} {'Buy&Hold':<10} {'Outperf':<10} {'Trades':<7} {'WinRate':<8} {'Signals':<8}"
    )
    print("-" * 90)

    for i, r in enumerate(results, 1):
        print(
            f"{i:>3}. {r['symbol']:<6} {r['total_return']:>8.1f}% {r['buy_hold_return']:>8.1f}% {r['outperformance']:>8.1f}% {r['total_trades']:>6} {r['win_rate']:>6.0f}% {r['signals_generated']:>6}"
        )

    # Calculate summary stats
    avg_strategy_return = sum(r["total_return"] for r in results) / len(results)
    avg_buy_hold_return = sum(r["buy_hold_return"] for r in results) / len(results)
    avg_outperformance = sum(r["outperformance"] for r in results) / len(results)
    beating_count = sum(1 for r in results if r["outperformance"] > 0)
    avg_trades = sum(r["total_trades"] for r in results) / len(results)
    avg_win_rate = sum(r["win_rate"] for r in results) / len(results)

    print("-" * 90)
    print(f"AVERAGES:")
    print(
        f"Strategy Return: {avg_strategy_return:+.1f}% | Buy&Hold: {avg_buy_hold_return:+.1f}% | Outperformance: {avg_outperformance:+.1f}%"
    )
    print(
        f"Beating B&H: {beating_count}/{len(results)} ({beating_count/len(results)*100:.0f}%)"
    )
    print(f"Avg Trades: {avg_trades:.1f} | Avg Win Rate: {avg_win_rate:.0f}%")

    print(f"\nBEST PERFORMERS:")
    for i, r in enumerate(results[:3], 1):
        print(
            f"{i}. {r['symbol']}: Strategy {r['total_return']:+.1f}% vs B&H {r['buy_hold_return']:+.1f}% (outperformed by {r['outperformance']:+.1f}%)"
        )

    if len(results) > 3:
        print(f"\nWORST PERFORMERS:")
        for i, r in enumerate(results[-3:], 1):
            print(
                f"{i}. {r['symbol']}: Strategy {r['total_return']:+.1f}% vs B&H {r['buy_hold_return']:+.1f}% (underperformed by {abs(r['outperformance']):-.1f}%)"
            )

    return results


if __name__ == "__main__":
    try:
        # Run multi-stock test
        test_multi_stocks()

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()
