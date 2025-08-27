#!/usr/bin/env python3
"""
Integration test combining strategy signals with portfolio management
"""

import sys
import os
from datetime import datetime, timezone
import pandas as pd

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from data.data_fetcher import DataFetcher
from strategies.moving_average_strategy import create_simple_ma_strategy
from trading.portfolio import Portfolio
# Removed timezone_utils import - using simple datetime operations
from trading.position import PositionType

def test_strategy_portfolio_integration():
    """Test strategy signals with portfolio execution"""
    print("Testing Strategy-Portfolio Integration...")
    print("=" * 50)
    
    # Initialize components
    fetcher = DataFetcher()
    strategy = create_simple_ma_strategy(short=10, long=20)  # Faster signals
    portfolio = Portfolio(initial_cash=100000.0)
    
    print(f"[OK] Initialized components")
    print(f"[OK] Strategy: {strategy.name}")
    print(f"[OK] Portfolio: {portfolio}")
    
    # Fetch data (6 months for faster testing)
    data = fetcher.fetch_stock_data("AAPL", period="6mo", interval="1d")
    print(f"[OK] Loaded {len(data)} days of AAPL data")
    
    # Generate signals
    signals = strategy.generate_signals(data)
    trading_signals = [s for s in signals if s.signal.value != 'HOLD']
    print(f"[OK] Generated {len(trading_signals)} trading signals")
    
    # Execute trades based on signals
    position_size_pct = 0.2  # Use 20% of portfolio per trade
    trades_executed = 0
    
    print(f"\nExecuting trades based on signals...")
    print("-" * 40)
    
    for signal in trading_signals:
        symbol = "AAPL"  # We're only trading AAPL
        price = signal.price
        timestamp = signal.timestamp
        
        try:
            if signal.signal.value == 'BUY' and not portfolio.has_position(symbol):
                # Calculate position size
                quantity = portfolio.calculate_position_size(symbol, price, position_size_pct)
                
                if quantity > 0:
                    # Open long position
                    position = portfolio.open_position(
                        symbol=symbol,
                        position_type=PositionType.LONG,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        fees=5.0  # $5 trading fee
                    )
                    trades_executed += 1
                    print(f"BUY  {timestamp.strftime('%Y-%m-%d')}: {quantity} shares @ ${price:.2f} (conf: {signal.confidence:.2f})")
                    
            elif signal.signal.value == 'SELL' and portfolio.has_position(symbol):
                # Close position
                closed_position = portfolio.close_position(
                    symbol=symbol,
                    price=price,
                    timestamp=timestamp,
                    fees=5.0  # $5 trading fee
                )
                trades_executed += 1
                pnl = closed_position.realized_pnl
                print(f"SELL {timestamp.strftime('%Y-%m-%d')}: {closed_position.quantity} shares @ ${price:.2f} -> P&L: ${pnl:.2f}")
                
        except Exception as e:
            print(f"SKIP {timestamp.strftime('%Y-%m-%d')}: {signal.signal.value} - {e}")
    
    print(f"\n[OK] Executed {trades_executed} trades")
    
    # Portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"\nFinal Portfolio Summary:")
    print(f"- Initial cash: ${portfolio.initial_cash:,.2f}")
    print(f"- Final cash: ${summary['cash']:,.2f}")
    print(f"- Total value: ${summary['total_value']:,.2f}")
    print(f"- Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"- Total return: {summary['total_return_pct']:.2f}%")
    print(f"- Open positions: {summary['open_positions_count']}")
    print(f"- Closed positions: {summary['closed_positions_count']}")
    
    # Show closed positions if any
    if portfolio.closed_positions:
        print(f"\nClosed Positions:")
        for i, pos in enumerate(portfolio.closed_positions, 1):
            entry_date = pos.entry_time.strftime('%Y-%m-%d')
            exit_date = pos.exit_time.strftime('%Y-%m-%d')
            return_pct = pos.calculate_return_pct(pos.exit_price)
            print(f"  {i}. {entry_date} -> {exit_date}: ${pos.entry_price:.2f} -> ${pos.exit_price:.2f} "
                  f"P&L: ${pos.realized_pnl:.2f} ({return_pct:.1f}%)")
    
    # Show current positions if any
    if portfolio.positions:
        print(f"\nOpen Positions:")
        for symbol, pos in portfolio.positions.items():
            entry_date = pos.entry_time.strftime('%Y-%m-%d')
            current_price = data['Close'].iloc[-1] if len(data) > 0 else 0.0  # Use latest price
            unrealized_pnl = pos.calculate_pnl(current_price)
            return_pct = pos.calculate_return_pct(current_price)
            print(f"  {symbol}: {entry_date} {pos.quantity} shares @ ${pos.entry_price:.2f} "
                  f"(current: ${current_price:.2f}) Unrealized P&L: ${unrealized_pnl:.2f} ({return_pct:.1f}%)")
    
    return True

def test_risk_management():
    """Test basic risk management features"""
    print(f"\nTesting Risk Management...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=10000.0)  # Smaller portfolio
    
    # Test position sizing limits
    max_position_pct = 0.25  # Max 25% per position
    price = 200.0
    
    # Calculate position size
    max_shares = portfolio.calculate_position_size("AAPL", price, max_position_pct)
    max_cost = max_shares * price
    max_allocation_pct = max_cost / portfolio.total_value
    
    print(f"[OK] Max position size: {max_shares} shares (${max_cost:.2f}, {max_allocation_pct*100:.1f}%)")
    
    # Test that we can't over-allocate
    assert max_allocation_pct <= max_position_pct + 0.01  # Small tolerance for rounding
    
    # Test cash management
    total_cash_used = 0
    positions_opened = 0
    
    # Try to open multiple positions
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    for symbol in symbols:
        try:
            quantity = portfolio.calculate_position_size(symbol, price, 0.2)  # 20% each
            if quantity > 0:
                cost = quantity * price + 5.0  # Include fees
                if cost <= portfolio.cash:
                    portfolio.open_position(symbol, PositionType.LONG, quantity, price, timestamp, fees=5.0)
                    total_cash_used += cost
                    positions_opened += 1
                    print(f"[OK] Opened {symbol}: {quantity} shares, cost: ${cost:.2f}")
                else:
                    print(f"[SKIP] {symbol}: Insufficient cash (need ${cost:.2f}, have ${portfolio.cash:.2f})")
            else:
                print(f"[SKIP] {symbol}: Position size too small")
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
    
    print(f"[OK] Opened {positions_opened} positions, used ${total_cash_used:.2f} cash")
    print(f"[OK] Remaining cash: ${portfolio.cash:.2f}")
    print(f"[OK] Cash utilization: {((total_cash_used/10000.0)*100):.1f}%")
    
    return True

def main():
    """Run integration tests"""
    try:
        test_strategy_portfolio_integration()
        test_risk_management()
        
        print("\n" + "=" * 50)
        print("[OK] All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)