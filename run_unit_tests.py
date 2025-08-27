#!/usr/bin/env python3
"""
Quick Unit Test Runner - Run just the core unit tests
For faster feedback during development
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_test(test_file, description):
    """Run a single test file"""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"{'='*50}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            f"python {test_file}", 
            shell=True, 
            cwd=Path(__file__).parent,
            timeout=120  # 2 minute timeout for unit tests
        )
        end_time = time.time()
        
        duration = end_time - start_time
        success = result.returncode == 0
        
        if success:
            print(f"[PASS] ({duration:.2f}s)")
        else:
            print(f"[FAIL] ({duration:.2f}s)")
        
        return success
        
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] (>2 minutes)")
        return False
    except Exception as e:
        print(f"[ERROR]: {e}")
        return False

def main():
    """Run all unit tests quickly"""
    print("TRADING BOT - UNIT TESTS ONLY")
    print("="*50)
    
    # Unit tests in order of dependency
    unit_tests = [
        ("scripts/tests/unit_tests/test_data_fetcher.py", "Data Fetcher"),
        ("scripts/tests/unit_tests/test_strategy.py", "Moving Average Strategy"),
        ("scripts/tests/unit_tests/test_portfolio.py", "Portfolio Management"), 
        ("scripts/tests/unit_tests/test_execution_engine.py", "Execution Engine"),
        ("scripts/tests/unit_tests/test_backtester.py", "Backtesting Framework"),
        ("scripts/tests/unit_tests/test_integration.py", "Component Integration"),
    ]
    
    results = []
    start_time = time.time()
    
    # Run each unit test
    for test_file, description in unit_tests:
        success = run_test(test_file, description)
        results.append((description, success))
        time.sleep(1)  # Brief pause
    
    # Summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print("UNIT TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Duration: {total_duration:.2f}s")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"[PASS] Passed: {passed}/{total}")
    print(f"[FAIL] Failed: {total - passed}/{total}")
    
    # Show results
    for description, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {description}")
    
    if passed == total:
        print(f"\nAll unit tests passed!")
        return True
    else:
        print(f"\n{total - passed} unit test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)