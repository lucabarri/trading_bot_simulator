#!/usr/bin/env python3
"""
Test script to verify data fetching functionality
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from data.data_fetcher import DataFetcher

def main():
    print("Testing Data Fetcher for AAPL...")
    print("=" * 50)
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    try:
        # Fetch AAPL data for 2 years
        print("Fetching AAPL data (2 years, daily)...")
        aapl_data = fetcher.fetch_stock_data("AAPL", period="2y", interval="1d")
        
        print(f"[OK] Successfully fetched {len(aapl_data)} days of AAPL data")
        # Display date range
        start_date = aapl_data.index[0].strftime('%Y-%m-%d')
        end_date = aapl_data.index[-1].strftime('%Y-%m-%d')
        print(f"[OK] Date range: {start_date} to {end_date}")
        print(f"[OK] Data shape: {aapl_data.shape}")
        print(f"[OK] Columns: {list(aapl_data.columns)}")
        
        print("\nFirst 5 rows:")
        print(aapl_data.head())
        
        print("\nLast 5 rows:")
        print(aapl_data.tail())
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"[OK] Average Close Price: ${aapl_data['Close'].mean():.2f}")
        print(f"[OK] Min Close Price: ${aapl_data['Close'].min():.2f}")
        print(f"[OK] Max Close Price: ${aapl_data['Close'].max():.2f}")
        print(f"[OK] Average Volume: {aapl_data['Volume'].mean():,.0f}")
        
        # Test latest price
        print("\nTesting latest price fetch...")
        latest_price = fetcher.get_latest_price("AAPL")
        print(f"[OK] Latest AAPL price: ${latest_price:.2f}")
        
        print("\n" + "=" * 50)
        print("[OK] All tests passed! Data fetcher is working correctly.")
        
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)