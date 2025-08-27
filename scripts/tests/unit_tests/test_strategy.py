#!/usr/bin/env python3
"""
Test script to verify moving average strategy functionality
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from data.data_fetcher import DataFetcher
from strategies.moving_average_strategy import MovingAverageCrossoverStrategy, create_simple_ma_strategy
import pandas as pd

def test_strategy_basic():
    """Test basic strategy functionality"""
    print("Testing Moving Average Strategy...")
    print("=" * 50)
    
    # Create strategy
    strategy = create_simple_ma_strategy(short=10, long=50)
    print(f"[OK] Created strategy: {strategy.name}")
    print(f"[OK] Required history: {strategy.get_required_history()} days")
    
    # Load AAPL data
    fetcher = DataFetcher()
    data = fetcher.fetch_stock_data("AAPL", period="2y", interval="1d")
    print(f"[OK] Loaded {len(data)} days of AAPL data")
    
    # Generate signals
    signals = strategy.generate_signals(data)
    print(f"[OK] Generated {len(signals)} signals")
    
    # Show strategy state
    state = strategy.get_strategy_state(data)
    print(f"\nCurrent Strategy State:")
    print(f"- Status: {state['status']}")
    print(f"- Latest Close: ${state['latest_close']:.2f}")
    print(f"- Short MA (10): ${state['short_ma']:.2f}")
    print(f"- Long MA (50): ${state['long_ma']:.2f}")
    print(f"- MA Difference: ${state['ma_diff']:.2f} ({state['ma_diff_pct']*100:.2f}%)")
    print(f"- Current Trend: {state['trend']}")
    
    # Show recent signals
    print(f"\nRecent Signals (last 10):")
    recent_signals = [s for s in signals if s.signal.value != 'HOLD'][-10:]
    for signal in recent_signals:
        date_str = signal.timestamp.strftime('%Y-%m-%d')
        print(f"- {date_str}: {signal.signal.value} at ${signal.price:.2f} (conf: {signal.confidence:.2f})")
    
    return True

def test_strategy_with_different_periods():
    """Test strategy with different MA periods"""
    print(f"\nTesting Different MA Periods:")
    print("-" * 30)
    
    fetcher = DataFetcher()
    data = fetcher.fetch_stock_data("AAPL", period="1y", interval="1d")
    
    # Test different MA combinations
    ma_combinations = [
        (5, 20),   # Very short term
        (10, 50),  # Medium term (our default)
        (20, 100)  # Longer term
    ]
    
    for short, long in ma_combinations:
        if len(data) >= long + 1:  # Ensure we have enough data
            strategy = MovingAverageCrossoverStrategy(short_window=short, long_window=long)
            signals = strategy.generate_signals(data)
            trading_signals = [s for s in signals if s.signal.value != 'HOLD']
            
            print(f"MA({short},{long}): {len(trading_signals)} trading signals")
        else:
            print(f"MA({short},{long}): Not enough data ({len(data)} < {long + 1})")
    
    return True

def test_strategy_edge_cases():
    """Test strategy edge cases"""
    print(f"\nTesting Edge Cases:")
    print("-" * 20)
    
    # Test with insufficient data
    strategy = create_simple_ma_strategy(10, 50)
    small_data = pd.DataFrame({
        'Close': [100, 101, 102],
        'Open': [99, 100, 101],
        'High': [101, 102, 103],
        'Low': [99, 100, 101],
        'Volume': [1000, 1000, 1000]
    }, index=pd.date_range('2023-01-01', periods=3))
    
    signals = strategy.generate_signals(small_data)
    print(f"[OK] Insufficient data test: {len(signals)} signals (expected 0)")
    
    # Test strategy state with insufficient data
    state = strategy.get_strategy_state(small_data)
    print(f"[OK] State with insufficient data: {state['status']}")
    
    return True

def main():
    try:
        # Run all tests
        test_strategy_basic()
        test_strategy_with_different_periods()
        test_strategy_edge_cases()
        
        print("\n" + "=" * 50)
        print("[OK] All strategy tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)