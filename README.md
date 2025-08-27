# Trading Bot Project

A paper trading bot implementation with moving average crossover strategy.

## Project Structure

```
trading_bot1/
├── src/                          # Core source code
│   ├── data/                     # Data fetching modules
│   ├── strategies/               # Trading strategies
│   ├── trading/                  # Trading execution engine
│   ├── backtesting/              # Backtesting framework
│   └── utils/                    # Helper utilities
├── scripts/                      # Executable scripts (organized by purpose)
│   ├── analysis/                 # Cost and performance analysis scripts
│   ├── demos/                    # Demonstration scripts
│   ├── tests/                    # Integration and strategy tests
│   │   └── unit_tests/           # Unit tests for core components
│   └── README.md                 # Scripts documentation
├── outputs/                      # Generated outputs
│   ├── visualizations/           # Charts and visual analysis
│   └── README.md                 # Outputs documentation
├── data/                         # Data storage
│   ├── raw/                      # Raw market data (CSV files)
│   └── processed/                # Processed data
├── examples/                     # Example configurations and usage
├── config/                       # Configuration files
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run basic strategy test
python scripts/tests/test_my_strategy.py

# Run visualization demo
python scripts/demos/demo_visualizations.py

# Analyze costs
python scripts/analysis/analyze_costs.py

# Compare different markets
python scripts/analysis/compare_markets.py
```

### Run Tests
```bash
# Unit tests for core components
python scripts/tests/unit_tests/test_data_fetcher.py
python scripts/tests/unit_tests/test_strategy.py
python scripts/tests/unit_tests/test_backtester.py

# Strategy integration tests
python scripts/tests/test_crypto_strategy.py
python scripts/tests/test_inverse_strategy.py
```

## Current Features

- [x] Data fetching from yfinance (OHLCV data)
- [x] Simple Moving Average Crossover Strategy (10/50 MA)
- [x] Strategy visualization and testing
- [ ] Portfolio management
- [ ] Paper trading execution
- [ ] Backtesting framework
- [ ] Performance metrics
- [ ] Configuration management

## Strategy Details

**Moving Average Crossover Strategy:**
- BUY signal when 10-day MA crosses above 50-day MA
- SELL signal when 10-day MA crosses below 50-day MA
- Includes noise filtering to avoid false signals
- Confidence scoring based on crossover magnitude