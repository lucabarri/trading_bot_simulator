#!/usr/bin/env python3
"""
Test script to verify portfolio management functionality
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from trading.portfolio import Portfolio
from trading.position import Position, PositionType, PositionStatus
# Removed timezone_utils import - using simple datetime operations

def test_basic_portfolio():
    """Test basic portfolio functionality"""
    print("Testing Basic Portfolio Operations...")
    print("=" * 50)
    
    # Create portfolio
    portfolio = Portfolio(initial_cash=100000.0)
    print(f"[OK] Created portfolio: {portfolio}")
    
    # Test initial state
    assert portfolio.cash == 100000.0
    assert portfolio.total_value == 100000.0
    assert portfolio.total_pnl == 0.0
    assert len(portfolio.positions) == 0
    print(f"[OK] Initial state verified")
    
    # Test position size calculation
    shares = portfolio.calculate_position_size("AAPL", 200.0, 0.1)  # 10% allocation
    expected_shares = int((100000 * 0.1) / 200.0)  # Should be 50 shares
    assert shares == expected_shares
    print(f"[OK] Position size calculation: {shares} shares for 10% allocation")
    
    return True

def test_position_lifecycle():
    """Test complete position lifecycle"""
    print("\nTesting Position Lifecycle...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=50000.0)
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Open position
    print("Opening AAPL position...")
    position = portfolio.open_position(
        symbol="AAPL",
        position_type=PositionType.LONG,
        quantity=100,
        price=200.0,
        timestamp=timestamp,
        fees=5.0
    )
    
    # Verify position opened correctly
    assert portfolio.has_position("AAPL")
    assert portfolio.cash == 50000.0 - (100 * 200.0) - 5.0  # $29,995
    assert position.cost_basis == 20005.0  # $20,000 + $5 fees
    print(f"[OK] Position opened: {position}")
    print(f"[OK] Remaining cash: ${portfolio.cash:.2f}")
    
    # Test portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"[OK] Portfolio value: ${summary['total_value']:.2f}")
    print(f"[OK] Positions value: ${summary['positions_value']:.2f}")
    
    # Close position at profit
    print("\nClosing position at profit...")
    close_timestamp = timestamp + timedelta(days=5)
    closed_position = portfolio.close_position(
        symbol="AAPL",
        price=220.0,  # $20 profit per share
        timestamp=close_timestamp,
        fees=5.0
    )
    
    # Verify position closed correctly
    assert not portfolio.has_position("AAPL")
    expected_proceeds = (100 * 220.0) - 5.0  # $21,995
    expected_cash = 29995.0 + expected_proceeds  # $51,990
    assert portfolio.cash == expected_cash
    
    expected_pnl = (220.0 - 200.0) * 100 - 5.0 - 5.0  # $1,990 profit
    assert closed_position.realized_pnl == expected_pnl
    print(f"[OK] Position closed: {closed_position}")
    print(f"[OK] Final cash: ${portfolio.cash:.2f}")
    print(f"[OK] Realized P&L: ${closed_position.realized_pnl:.2f}")
    
    return True

def test_multiple_positions():
    """Test managing multiple positions"""
    print("\nTesting Multiple Positions...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=100000.0)
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Open multiple positions
    positions_data = [
        ("AAPL", 100, 200.0),
        ("MSFT", 50, 300.0),
        ("GOOGL", 25, 150.0)
    ]
    
    for symbol, quantity, price in positions_data:
        portfolio.open_position(
            symbol=symbol,
            position_type=PositionType.LONG,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=5.0
        )
        print(f"[OK] Opened {symbol}: {quantity} shares @ ${price:.2f}")
    
    # Verify all positions
    assert len(portfolio.positions) == 3
    summary = portfolio.get_portfolio_summary()
    print(f"[OK] Total positions: {summary['open_positions_count']}")
    print(f"[OK] Portfolio value: ${summary['total_value']:.2f}")
    
    # Close one position
    portfolio.close_position("MSFT", 320.0, timestamp, fees=5.0)
    assert len(portfolio.positions) == 2
    assert len(portfolio.closed_positions) == 1
    print(f"[OK] Closed MSFT position")
    
    return True

def test_portfolio_tracking():
    """Test portfolio tracking and history"""
    print("\nTesting Portfolio Tracking...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=50000.0)
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Open and close a few positions
    portfolio.open_position("AAPL", PositionType.LONG, 100, 200.0, timestamp, fees=5.0)
    portfolio.close_position("AAPL", 210.0, timestamp + timedelta(days=1), fees=5.0)
    
    portfolio.open_position("MSFT", PositionType.LONG, 50, 300.0, timestamp + timedelta(days=2), fees=5.0)
    portfolio.close_position("MSFT", 290.0, timestamp + timedelta(days=3), fees=5.0)
    
    # Test transaction history
    history_df = portfolio.get_transaction_history_df()
    print(f"[OK] Transaction history: {len(history_df)} records")
    print(history_df[['action', 'symbol', 'quantity', 'price']])
    
    # Test closed positions
    closed_df = portfolio.get_closed_positions_df()
    print(f"\n[OK] Closed positions: {len(closed_df)} positions")
    print(closed_df[['symbol', 'entry_price', 'exit_price', 'realized_pnl']])
    
    # Test final summary
    summary = portfolio.get_portfolio_summary()
    print(f"\n[OK] Final portfolio summary:")
    print(f"    Total value: ${summary['total_value']:.2f}")
    print(f"    Total P&L: ${summary['total_pnl']:.2f}")
    print(f"    Return: {summary['total_return_pct']:.2f}%")
    
    return True

def test_error_handling():
    """Test error handling"""
    print("\nTesting Error Handling...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=1000.0)
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Test insufficient funds
    try:
        portfolio.open_position("AAPL", PositionType.LONG, 100, 200.0, timestamp)
        assert False, "Should have raised ValueError for insufficient funds"
    except ValueError as e:
        print(f"[OK] Caught insufficient funds error: {e}")
    
    # Test opening duplicate position
    portfolio.open_position("AAPL", PositionType.LONG, 5, 200.0, timestamp)
    try:
        portfolio.open_position("AAPL", PositionType.LONG, 1, 200.0, timestamp)
        assert False, "Should have raised ValueError for duplicate position"
    except ValueError as e:
        print(f"[OK] Caught duplicate position error: {e}")
    
    # Test closing non-existent position
    try:
        portfolio.close_position("MSFT", 300.0, timestamp)
        assert False, "Should have raised ValueError for non-existent position"
    except ValueError as e:
        print(f"[OK] Caught non-existent position error: {e}")
    
    return True

def main():
    """Run all portfolio tests"""
    try:
        test_basic_portfolio()
        test_position_lifecycle()
        test_multiple_positions()
        test_portfolio_tracking()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("[OK] All portfolio tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Portfolio test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)