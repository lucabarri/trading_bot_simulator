#!/usr/bin/env python3
"""
Master Test Runner - Run all tests in the trading bot project
Executes all unit tests, specialized tests, and the main strategy test
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description, timeout=300):
    """Run a command and capture its output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        end_time = time.time()
        
        print(f"Exit Code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f}s")
        
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Test exceeded {timeout} seconds")
        return False, "", "Test timed out"
    except Exception as e:
        print(f"ERROR: Failed to run test: {e}")
        return False, "", str(e)

def main():
    """Run all tests in order"""
    print("TRADING BOT - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Starting test run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test configuration
    tests = [
        # Unit tests (core components)
        ("python scripts/tests/unit_tests/test_data_fetcher.py", "Data Fetcher Unit Test", 60),
        ("python scripts/tests/unit_tests/test_strategy.py", "Strategy Unit Test", 120),
        ("python scripts/tests/unit_tests/test_portfolio.py", "Portfolio Unit Test", 60),
        ("python scripts/tests/unit_tests/test_execution_engine.py", "Execution Engine Unit Test", 60),
        ("python scripts/tests/unit_tests/test_backtester.py", "Backtester Unit Test", 180),
        ("python scripts/tests/unit_tests/test_integration.py", "Integration Test", 180),
        
        # Specialized tests
        ("python scripts/tests/test_cost_sensitivity.py", "Cost Sensitivity Analysis", 300),
        
        # Main comprehensive test
        ("python test_my_strategy.py", "Main Strategy Test (Multi-Stock)", 600),
    ]
    
    # Results tracking
    results = []
    total_start = time.time()
    
    # Run each test
    for cmd, description, timeout in tests:
        success, stdout, stderr = run_command(cmd, description, timeout)
        results.append({
            'test': description,
            'command': cmd,
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        })
        
        # Give a brief pause between tests
        time.sleep(2)
    
    # Summary report
    total_end = time.time()
    total_duration = total_end - total_start
    
    print(f"\n\n{'='*80}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    print(f"Tests Run: {len(results)}")
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"[PASS] PASSED: {passed}")
    print(f"[FAIL] FAILED: {failed}")
    print(f"Success Rate: {passed/len(results)*100:.1f}%")
    
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "[PASS]" if result['success'] else "[FAIL]"
        print(f"{i:2d}. {status} - {result['test']}")
    
    # Show failed tests
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print(f"\n[FAIL] FAILED TESTS DETAILS:")
        print("-" * 50)
        for test in failed_tests:
            print(f"* {test['test']}")
            print(f"  Command: {test['command']}")
            if test['stderr']:
                print(f"  Error: {test['stderr'][:200]}...")
            print()
    
    # Final status
    if failed == 0:
        print(f"\nALL TESTS PASSED! Your trading bot is ready to go!")
        return True
    else:
        print(f"\n{failed} test(s) failed. Check the details above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)