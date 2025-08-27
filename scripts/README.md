# Scripts Directory

This directory contains all the executable scripts for the trading bot project, organized by category:

## Structure

### `/analysis/`
Scripts for analyzing strategy performance and costs:
- `analyze_costs.py` - Detailed cost breakdown analysis for trading strategies
- `analyze_inverse_costs.py` - Cost analysis specifically for inverse strategies
- `compare_markets.py` - Compare strategy performance across different markets (stocks, crypto, forex)

### `/demos/`
Demonstration scripts showing various features:
- `demo_visualizations.py` - Basic visualization demo with multiple strategies
- `demo_visualizations_simple.py` - Simplified version of visualization demo
- `demo_micro_trades.py` - Ultra-conservative trading demo with micro positions
- `demo_small_trades.py` - Small position trading demo

### `/tests/`
Test scripts for validating strategies and components:
- `test_my_strategy.py` - Basic multi-asset strategy testing
- `test_crypto_strategy.py` - Cryptocurrency market strategy testing
- `test_inverse_strategy.py` - Inverse MA strategy testing
- `test_cost_sensitivity.py` - Transaction cost sensitivity analysis
- `test_strategy_with_visualization.py` - Comprehensive strategy testing with visualizations
- `test_visualizer_imports.py` - Test script for visualization imports
- `validate_fixes.py` - General validation script

## Usage

All scripts can be run from the project root directory:

```bash
# Run analysis scripts
python scripts/analysis/analyze_costs.py
python scripts/analysis/compare_markets.py

# Run demo scripts
python scripts/demos/demo_visualizations.py

# Run test scripts
python scripts/tests/test_my_strategy.py
```

## Output

Scripts generate outputs in the `/outputs/` directory:
- Visualizations are saved to `/outputs/visualizations/`
- Other outputs go to `/outputs/`