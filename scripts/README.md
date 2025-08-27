# Scripts Directory

This directory contains analysis and testing scripts for the trading bot project.

## Structure

### `/analysis/`
Scripts for analyzing strategy performance and market comparison:

- **`analyze_costs.py`** - Detailed cost breakdown analysis for trading strategies
  - Analyzes trading fees, slippage costs, and pure trading performance
  - Provides cost sensitivity analysis and recommendations
  - Shows what-if scenarios with different cost structures

- **`compare_markets.py`** - Compare strategy performance across different asset classes
  - Tests MA strategy on stocks, cryptocurrency, and forex markets
  - Provides market comparison with performance rankings
  - Identifies optimal markets for algorithmic trading

### `/tests/`
Test scripts for specialized analysis and validation:

- **`test_cost_sensitivity.py`** - Transaction cost sensitivity analysis
  - Tests strategy performance under different fee and slippage scenarios
  - Determines break-even points and optimal trading frequencies
  
- **`test_strategy_with_visualization.py`** - Strategy testing with visual output
  - Comprehensive strategy testing with chart generation
  - Performance visualization and analysis reports

### `/tests/unit_tests/`
Unit tests for all core components:

- **`test_data_fetcher.py`** - Data fetching and validation tests
- **`test_strategy.py`** - Moving average strategy logic tests  
- **`test_portfolio.py`** - Portfolio management and P&L tests
- **`test_execution_engine.py`** - Order execution and slippage tests
- **`test_backtester.py`** - Backtesting framework validation
- **`test_integration.py`** - End-to-end system integration tests

## Usage

All scripts should be run from the project root directory:

```bash
# Analysis scripts
python scripts/analysis/analyze_costs.py        # Cost breakdown analysis
python scripts/analysis/compare_markets.py      # Multi-market comparison

# Specialized tests
python scripts/tests/test_cost_sensitivity.py   # Cost sensitivity analysis

# Unit tests (individual components)
python scripts/tests/unit_tests/test_data_fetcher.py
python scripts/tests/unit_tests/test_strategy.py
python scripts/tests/unit_tests/test_portfolio.py
python scripts/tests/unit_tests/test_execution_engine.py
python scripts/tests/unit_tests/test_backtester.py
python scripts/tests/unit_tests/test_integration.py

# Quick test runners (from project root)
python run_unit_tests.py        # Run all unit tests quickly
python run_all_tests.py         # Comprehensive test suite
python test_my_strategy.py      # Main strategy test (100 stocks)
```

## Output

Scripts generate various outputs:

- **Analysis Reports**: Console output with detailed metrics and recommendations
- **Visualizations**: Charts and graphs saved to `/outputs/visualizations/` (when applicable)
- **Log Files**: Execution logs saved to `/logs/` directory
- **Data Cache**: Market data cached in `/data/raw/` for faster subsequent runs

## Test Coverage

The unit test suite provides 100% coverage of core components:
- ✅ Data fetching and market data validation
- ✅ Strategy signal generation and logic
- ✅ Portfolio management and position tracking  
- ✅ Order execution engine with realistic costs
- ✅ Backtesting framework with performance metrics
- ✅ End-to-end system integration