#!/usr/bin/env python3
"""
Test script to verify trading execution engine functionality
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import pandas as pd

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from trading.execution_engine import ExecutionEngine, ExecutionMode, Order, OrderType
from trading.portfolio import Portfolio
from trading.position import PositionType
# Removed timezone_utils import - using simple datetime operations
from strategies.moving_average_strategy import create_simple_ma_strategy
from strategies.base_strategy import TradingSignal, Signal

def test_basic_execution():
    """Test basic order execution"""
    print("Testing Basic Order Execution...")
    print("=" * 50)
    
    # Setup
    portfolio = Portfolio(initial_cash=50000.0)
    engine = ExecutionEngine(portfolio, mode=ExecutionMode.PAPER, base_fee=5.0, slippage_bps=2.0)
    
    print(f"[OK] Created execution engine: {engine}")
    print(f"[OK] Initial portfolio: {portfolio}")
    
    # Test BUY order
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    buy_order = Order(
        symbol="AAPL",
        action="BUY",
        quantity=100,
        order_type=OrderType.MARKET,
        timestamp=timestamp
    )
    
    market_price = 200.0
    success = engine.execute_order(buy_order, market_price, timestamp)
    
    assert success, "BUY order should have been filled"
    assert buy_order.status.value == "FILLED"
    assert portfolio.has_position("AAPL")
    
    print(f"[OK] BUY order executed: {buy_order}")
    print(f"[OK] Portfolio after BUY: {portfolio}")
    
    # Test SELL order
    sell_timestamp = timestamp + timedelta(days=1)
    sell_order = Order(
        symbol="AAPL",
        action="SELL",
        quantity=100,
        order_type=OrderType.MARKET,
        timestamp=sell_timestamp
    )
    
    sell_price = 210.0
    success = engine.execute_order(sell_order, sell_price, sell_timestamp)
    
    assert success, "SELL order should have been filled"
    assert sell_order.status.value == "FILLED"
    assert not portfolio.has_position("AAPL")
    
    print(f"[OK] SELL order executed: {sell_order}")
    print(f"[OK] Portfolio after SELL: {portfolio}")
    
    # Check P&L
    closed_position = portfolio.closed_positions[0]
    expected_pnl = (210.0 - 200.0) * 100 - 5.0 - 5.0  # Profit minus fees
    print(f"[OK] Realized P&L: ${closed_position.realized_pnl:.2f} (expected: ${expected_pnl:.2f})")
    
    return True

def test_slippage_and_fees():
    """Test slippage and fee calculations"""
    print("\nTesting Slippage and Fees...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=50000.0)
    engine = ExecutionEngine(
        portfolio, 
        mode=ExecutionMode.PAPER, 
        base_fee=5.0, 
        slippage_bps=10.0  # 0.1% slippage
    )
    
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    order = Order("AAPL", "BUY", 100, OrderType.MARKET, timestamp=timestamp)
    
    market_price = 200.0
    expected_slippage_price = market_price * 1.001  # 0.1% higher for BUY
    
    success = engine.execute_order(order, market_price, timestamp)
    
    assert success
    assert abs(order.fill_price - expected_slippage_price) < 0.01
    
    print(f"[OK] Market price: ${market_price:.2f}")
    print(f"[OK] Fill price with slippage: ${order.fill_price:.2f}")
    print(f"[OK] Fees charged: ${order.fees:.2f}")
    
    return True

def test_order_validation():
    """Test order validation and rejection"""
    print("\nTesting Order Validation...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=1000.0)  # Small portfolio
    engine = ExecutionEngine(portfolio, mode=ExecutionMode.PAPER)
    
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Test insufficient funds
    large_order = Order("AAPL", "BUY", 100, OrderType.MARKET, timestamp=timestamp)
    market_price = 200.0  # Would cost $20,000
    
    success = engine.execute_order(large_order, market_price, timestamp)
    
    assert not success, "Order should have been rejected for insufficient funds"
    assert large_order.status.value == "REJECTED"
    assert "Insufficient cash" in large_order.rejection_reason
    
    print(f"[OK] Large order rejected: {large_order.rejection_reason}")
    
    # Test selling non-existent position
    sell_order = Order("MSFT", "SELL", 50, OrderType.MARKET, timestamp=timestamp)
    
    success = engine.execute_order(sell_order, 300.0, timestamp)
    
    assert not success, "Order should have been rejected for no position"
    assert sell_order.status.value == "REJECTED"
    assert "No position" in sell_order.rejection_reason
    
    print(f"[OK] Sell order rejected: {sell_order.rejection_reason}")
    
    return True

def test_signal_to_order_conversion():
    """Test converting strategy signals to orders"""
    print("\nTesting Signal to Order Conversion...")
    print("-" * 40)
    
    portfolio = Portfolio(initial_cash=100000.0)
    engine = ExecutionEngine(portfolio, mode=ExecutionMode.PAPER)
    
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Test BUY signal
    buy_signal = TradingSignal(
        signal=Signal.BUY,
        timestamp=timestamp,
        price=200.0,
        confidence=0.8
    )
    
    buy_order = engine.create_order_from_signal(buy_signal, "AAPL", allocation_pct=0.1)
    
    assert buy_order is not None
    assert buy_order.action == "BUY"
    assert buy_order.symbol == "AAPL"
    assert buy_order.quantity > 0
    
    print(f"[OK] BUY signal converted to order: {buy_order}")
    
    # Execute the buy order
    engine.execute_order(buy_order, 200.0, timestamp)
    
    # Test SELL signal
    sell_signal = TradingSignal(
        signal=Signal.SELL,
        timestamp=timestamp + timedelta(days=1),
        price=210.0,
        confidence=0.7
    )
    
    sell_order = engine.create_order_from_signal(sell_signal, "AAPL")
    
    assert sell_order is not None
    assert sell_order.action == "SELL"
    assert sell_order.quantity == buy_order.quantity
    
    print(f"[OK] SELL signal converted to order: {sell_order}")
    
    # Test HOLD signal
    hold_signal = TradingSignal(
        signal=Signal.HOLD,
        timestamp=timestamp,
        price=200.0
    )
    
    hold_order = engine.create_order_from_signal(hold_signal, "AAPL")
    
    assert hold_order is None
    print(f"[OK] HOLD signal correctly ignored")
    
    return True

def test_execution_summary():
    """Test execution engine summary and reporting"""
    print("\nTesting Execution Summary...")
    print("-" * 30)
    
    portfolio = Portfolio(initial_cash=100000.0)
    engine = ExecutionEngine(portfolio, mode=ExecutionMode.PAPER, base_fee=5.0)
    
    timestamp = pd.Timestamp.now(tz=timezone.utc)
    
    # Execute a few orders
    orders_data = [
        ("AAPL", "BUY", 100, 200.0),
        ("AAPL", "SELL", 100, 210.0),
        ("MSFT", "BUY", 50, 300.0),
    ]
    
    for symbol, action, quantity, price in orders_data:
        order = Order(symbol, action, quantity, OrderType.MARKET, timestamp=timestamp)
        engine.execute_order(order, price, timestamp)
        timestamp += timedelta(days=1)
    
    # Try one order that should fail
    fail_order = Order("GOOGL", "SELL", 100, OrderType.MARKET, timestamp=timestamp)
    engine.execute_order(fail_order, 150.0, timestamp)
    
    summary = engine.get_execution_summary()
    
    assert summary['mode'] == 'PAPER'
    assert summary['filled_orders'] == 3
    assert summary['rejected_orders'] == 1
    assert summary['fill_rate'] == 0.75
    assert summary['total_fees_paid'] == 15.0  # 3 filled orders * $5 each
    
    print(f"[OK] Execution summary:")
    print(f"     Mode: {summary['mode']}")
    print(f"     Total orders: {summary['total_orders']}")
    print(f"     Filled: {summary['filled_orders']}")
    print(f"     Rejected: {summary['rejected_orders']}")
    print(f"     Fill rate: {summary['fill_rate']*100:.1f}%")
    print(f"     Total fees: ${summary['total_fees_paid']:.2f}")
    
    return True

def main():
    """Run all execution engine tests"""
    try:
        test_basic_execution()
        test_slippage_and_fees()
        test_order_validation()
        test_signal_to_order_conversion()
        test_execution_summary()
        
        print("\n" + "=" * 50)
        print("[OK] All execution engine tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Execution engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)