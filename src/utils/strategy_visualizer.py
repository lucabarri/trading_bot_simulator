"""
Strategy Performance Visualization System
Comprehensive visualization tools for analyzing trading strategy performance
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
import os
from pathlib import Path

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Plotly is optional - we'll log this after logger is set up

# Import from our codebase
try:
    from ..backtesting.backtester import BacktestResult
    from .logger import get_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from backtesting.backtester import BacktestResult
    from utils.logger import get_logger


class StrategyVisualizer:
    """
    Comprehensive visualization system for trading strategy analysis
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
        
        # Set style for matplotlib (compatible with different versions)
        try:
            if 'seaborn-v0_8' in plt.style.available:
                plt.style.use('seaborn-v0_8')
            elif 'seaborn' in plt.style.available:
                plt.style.use('seaborn')
            else:
                plt.style.use('default')
            sns.set_palette("husl")
        except Exception:
            # Fallback to basic settings if seaborn unavailable
            plt.style.use('default')
        
        # Create subdirectories
        (self.output_dir / "single_strategy").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "interactive").mkdir(exist_ok=True)
        
        self.logger.info(f"StrategyVisualizer initialized. Output directory: {self.output_dir}")
    
    def visualize_single_strategy(
        self, 
        result: BacktestResult, 
        save_plots: bool = True,
        show_plots: bool = False,
        include_interactive: bool = True
    ) -> Dict[str, str]:
        """
        Create comprehensive visualizations for a single strategy
        
        Args:
            result: BacktestResult object
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots interactively
            include_interactive: Whether to create interactive plotly charts
            
        Returns:
            Dictionary of plot filenames created
        """
        self.logger.info(f"Creating visualizations for strategy: {result.strategy_name}")
        
        files_created = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.strategy_name}_{result.symbol}_{timestamp}"
        
        # 1. Portfolio Performance Dashboard
        files_created['dashboard'] = self._create_performance_dashboard(
            result, base_filename, save_plots, show_plots
        )
        
        # 2. Detailed Equity Curve
        files_created['equity_curve'] = self._create_equity_curve(
            result, base_filename, save_plots, show_plots
        )
        
        # 3. Drawdown Analysis
        files_created['drawdown'] = self._create_drawdown_chart(
            result, base_filename, save_plots, show_plots
        )
        
        # 4. Monthly Returns Heatmap
        files_created['monthly_returns'] = self._create_monthly_returns_heatmap(
            result, base_filename, save_plots, show_plots
        )
        
        # 5. Trade Analysis
        if result.closed_positions:
            files_created['trade_analysis'] = self._create_trade_analysis(
                result, base_filename, save_plots, show_plots
            )
        
        # 6. Risk Analysis
        files_created['risk_analysis'] = self._create_risk_analysis(
            result, base_filename, save_plots, show_plots
        )
        
        # 7. Interactive Dashboard (if plotly available)
        if include_interactive and PLOTLY_AVAILABLE:
            files_created['interactive_dashboard'] = self._create_interactive_dashboard(
                result, base_filename
            )
        
        self.logger.info(f"Created {len(files_created)} visualizations for {result.strategy_name}")
        return files_created
    
    def compare_strategies(
        self,
        results: Dict[str, BacktestResult],
        comparison_name: str = "strategy_comparison",
        save_plots: bool = True,
        show_plots: bool = False,
        include_interactive: bool = True
    ) -> Dict[str, str]:
        """
        Create comparison visualizations for multiple strategies
        
        Args:
            results: Dictionary of strategy_name -> BacktestResult
            comparison_name: Name for the comparison set
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots interactively
            include_interactive: Whether to create interactive plotly charts
            
        Returns:
            Dictionary of plot filenames created
        """
        self.logger.info(f"Creating strategy comparison visualizations for {len(results)} strategies")
        
        files_created = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{comparison_name}_{timestamp}"
        
        # 1. Performance Comparison Dashboard
        files_created['comparison_dashboard'] = self._create_comparison_dashboard(
            results, base_filename, save_plots, show_plots
        )
        
        # 2. Equity Curves Comparison
        files_created['equity_comparison'] = self._create_equity_comparison(
            results, base_filename, save_plots, show_plots
        )
        
        # 3. Risk-Return Scatter Plot
        files_created['risk_return'] = self._create_risk_return_scatter(
            results, base_filename, save_plots, show_plots
        )
        
        # 4. Performance Metrics Table
        files_created['metrics_table'] = self._create_metrics_comparison_table(
            results, base_filename, save_plots, show_plots
        )
        
        # 5. Drawdown Comparison
        files_created['drawdown_comparison'] = self._create_drawdown_comparison(
            results, base_filename, save_plots, show_plots
        )
        
        # 6. Interactive Comparison (if plotly available)
        if include_interactive and PLOTLY_AVAILABLE:
            files_created['interactive_comparison'] = self._create_interactive_comparison(
                results, base_filename
            )
        
        self.logger.info(f"Created {len(files_created)} comparison visualizations")
        return files_created
    
    def visualize_multi_asset_strategy(
        self,
        result: BacktestResult,
        asset_allocations: Optional[Dict[str, float]] = None,
        save_plots: bool = True,
        show_plots: bool = False
    ) -> Dict[str, str]:
        """
        Specialized visualizations for multi-asset strategies
        
        Args:
            result: BacktestResult for multi-asset strategy
            asset_allocations: Dictionary of asset -> allocation percentage
            save_plots: Whether to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of plot filenames created
        """
        self.logger.info(f"Creating multi-asset visualizations for: {result.strategy_name}")
        
        files_created = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"multi_asset_{result.strategy_name}_{timestamp}"
        
        # 1. Asset Allocation Pie Chart
        if asset_allocations:
            files_created['allocation_pie'] = self._create_allocation_pie_chart(
                asset_allocations, base_filename, save_plots, show_plots
            )
        
        # 2. Per-Asset Performance Analysis
        if result.closed_positions:
            files_created['asset_performance'] = self._create_per_asset_performance(
                result, base_filename, save_plots, show_plots
            )
        
        # 3. Standard single strategy visualizations
        single_viz = self.visualize_single_strategy(
            result, save_plots, show_plots, include_interactive=False
        )
        files_created.update(single_viz)
        
        return files_created
    
    def _create_performance_dashboard(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create a comprehensive performance dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance Dashboard: {result.strategy_name} ({result.symbol})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Equity Curve (top-left)
        ax1 = axes[0, 0]
        if len(result.daily_values) > 0:
            ax1.plot(result.daily_values.index, result.daily_values['portfolio_value'], 
                    linewidth=2, color='darkblue')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # 2. Drawdown (top-right)
        ax2 = axes[0, 1]
        if len(result.daily_values) > 0:
            rolling_max = result.daily_values['portfolio_value'].expanding().max()
            drawdown = (result.daily_values['portfolio_value'] - rolling_max) / rolling_max * 100
            ax2.fill_between(result.daily_values.index, drawdown, 0, alpha=0.3, color='red')
            ax2.plot(result.daily_values.index, drawdown, color='darkred', linewidth=1)
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Key Metrics (bottom-left)
        ax3 = axes[1, 0]
        metrics = [
            f"Total Return: {result.total_return_pct:.2f}%",
            f"Annualized Return: {result.annualized_return*100:.2f}%",
            f"Sharpe Ratio: {result.sharpe_ratio:.3f}",
            f"Max Drawdown: {result.max_drawdown_pct:.2f}%",
            f"Volatility: {result.volatility*100:.2f}%",
            f"Win Rate: {result.win_rate*100:.1f}%",
            f"Total Trades: {result.total_trades}",
            f"Profit Factor: {result.profit_factor:.2f}"
        ]
        ax3.text(0.05, 0.95, '\n'.join(metrics), transform=ax3.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Key Performance Metrics')
        
        # 4. Monthly Returns (bottom-right)
        ax4 = axes[1, 1]
        if len(result.daily_values) > 0:
            monthly_returns = result.daily_values['portfolio_value'].resample('ME').last().pct_change()
            monthly_returns = monthly_returns.dropna()
            if len(monthly_returns) > 0:
                colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                bars = ax4.bar(range(len(monthly_returns)), monthly_returns * 100, color=colors, alpha=0.7)
                ax4.set_title('Monthly Returns (%)')
                ax4.set_ylabel('Return (%)')
                ax4.set_xlabel('Month')
                ax4.grid(True, alpha=0.3)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Saved dashboard: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_equity_curve(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create detailed equity curve chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        fig.suptitle(f'Detailed Equity Curve: {result.strategy_name} ({result.symbol})', 
                    fontsize=14, fontweight='bold')
        
        if len(result.daily_values) > 0:
            # Main equity curve
            ax1.plot(result.daily_values.index, result.daily_values['portfolio_value'], 
                    linewidth=2, color='darkblue', label='Portfolio Value')
            
            # Add buy/hold benchmark if we have price data
            if 'price' in result.daily_values.columns:
                initial_price = result.daily_values['price'].iloc[0]
                benchmark = result.initial_value * (result.daily_values['price'] / initial_price)
                ax1.plot(result.daily_values.index, benchmark, 
                        linewidth=1, color='gray', linestyle='--', alpha=0.7, label='Buy & Hold')
            
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Daily returns in subplot
            daily_returns = result.daily_values['portfolio_value'].pct_change()
            ax2.bar(result.daily_values.index, daily_returns * 100, 
                   color=['green' if x > 0 else 'red' for x in daily_returns], alpha=0.6)
            ax2.set_title('Daily Returns (%)')
            ax2.set_ylabel('Return (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_equity_curve.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_drawdown_chart(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create detailed drawdown analysis chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Drawdown Analysis: {result.strategy_name} ({result.symbol})', 
                    fontsize=14, fontweight='bold')
        
        if len(result.daily_values) > 0:
            portfolio_values = result.daily_values['portfolio_value']
            rolling_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - rolling_max) / rolling_max * 100
            
            # Drawdown chart
            ax1.fill_between(result.daily_values.index, drawdown, 0, alpha=0.3, color='red')
            ax1.plot(result.daily_values.index, drawdown, color='darkred', linewidth=1)
            ax1.set_title('Drawdown Over Time')
            ax1.set_ylabel('Drawdown (%)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=result.max_drawdown_pct, color='red', linestyle='--', 
                       label=f'Max Drawdown: {result.max_drawdown_pct:.2f}%')
            ax1.legend()
            
            # Underwater plot
            underwater = (portfolio_values / rolling_max - 1) * 100
            ax2.fill_between(result.daily_values.index, underwater, 0, 
                           where=(underwater < 0), alpha=0.3, color='red', 
                           label='Underwater Periods')
            ax2.plot(result.daily_values.index, underwater, color='darkred', linewidth=1)
            ax2.set_title('Underwater Plot (% Below Peak)')
            ax2.set_ylabel('% Below Peak')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.legend()
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_drawdown.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_monthly_returns_heatmap(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create monthly returns heatmap"""
        if len(result.daily_values) == 0:
            self.logger.warning(f"No daily values data for {result.strategy_name}")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate monthly returns
        monthly_values = result.daily_values['portfolio_value'].resample('ME').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        if len(monthly_returns) > 0:
            # Create pivot table for heatmap
            monthly_returns_df = monthly_returns.to_frame('returns')
            monthly_returns_df['year'] = monthly_returns_df.index.year
            monthly_returns_df['month'] = monthly_returns_df.index.month
            
            pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='returns')
            pivot_table = pivot_table * 100  # Convert to percentage
            
            # Create heatmap
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Monthly Return (%)'}, ax=ax)
            
            ax.set_title(f'Monthly Returns Heatmap: {result.strategy_name} ({result.symbol})')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
            # Set month labels safely
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            # Only set labels if we have the right number of columns
            if pivot_table.shape[1] <= 12:
                current_labels = [month_labels[i-1] for i in pivot_table.columns if 1 <= i <= 12]
                if len(current_labels) == len(ax.get_xticklabels()):
                    ax.set_xticklabels(current_labels)
                else:
                    # Fallback to default month numbers if mismatch
                    ax.set_xlabel('Month (1=Jan, 12=Dec)')
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_monthly_heatmap.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_trade_analysis(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create trade analysis charts"""
        if not result.closed_positions:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Trade Analysis: {result.strategy_name} ({result.symbol})', 
                    fontsize=14, fontweight='bold')
        
        # Extract trade data
        trades_df = pd.DataFrame(result.closed_positions)
        
        # 1. P&L Distribution (top-left)
        ax1 = axes[0, 0]
        ax1.hist(trades_df['pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(trades_df['pnl'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${trades_df["pnl"].mean():.2f}')
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Winning vs Losing Trades (top-right)
        ax2 = axes[0, 1]
        win_loss_counts = [
            len(trades_df[trades_df['pnl'] > 0]),
            len(trades_df[trades_df['pnl'] <= 0])
        ]
        colors = ['green', 'red']
        ax2.pie(win_loss_counts, labels=['Winning', 'Losing'], colors=colors, autopct='%1.1f%%')
        ax2.set_title('Win/Loss Ratio')
        
        # 3. Trade Returns % (bottom-left)
        ax3 = axes[1, 0]
        if 'return_pct' in trades_df.columns:
            ax3.hist(trades_df['return_pct'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.axvline(trades_df['return_pct'].mean(), color='red', linestyle='--',
                       label=f'Mean: {trades_df["return_pct"].mean():.2f}%')
            ax3.set_title('Trade Return % Distribution')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative P&L (bottom-right)
        ax4 = axes[1, 1]
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax4.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='darkblue')
        ax4.set_title('Cumulative P&L by Trade')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative P&L ($)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_trade_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_risk_analysis(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create risk analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Risk Analysis: {result.strategy_name} ({result.symbol})', 
                    fontsize=14, fontweight='bold')
        
        if len(result.daily_values) > 0:
            daily_returns = result.daily_values['portfolio_value'].pct_change().dropna()
            
            # 1. Return Distribution (top-left)
            ax1 = axes[0, 0]
            ax1.hist(daily_returns * 100, bins=30, alpha=0.7, color='skyblue', 
                    density=True, edgecolor='black')
            ax1.axvline(daily_returns.mean() * 100, color='red', linestyle='--',
                       label=f'Mean: {daily_returns.mean()*100:.3f}%')
            ax1.set_title('Daily Returns Distribution')
            ax1.set_xlabel('Daily Return (%)')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Rolling Volatility (top-right)
            ax2 = axes[0, 1]
            rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
            ax2.plot(rolling_vol.index, rolling_vol, linewidth=2, color='orange')
            ax2.axhline(result.volatility * 100, color='red', linestyle='--',
                       label=f'Average: {result.volatility*100:.1f}%')
            ax2.set_title('30-Day Rolling Volatility')
            ax2.set_ylabel('Annualized Volatility (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Risk Metrics (bottom-left)
            ax3 = axes[1, 0]
            risk_metrics = [
                f"Volatility: {result.volatility*100:.2f}%",
                f"Sharpe Ratio: {result.sharpe_ratio:.3f}",
                f"Max Drawdown: {result.max_drawdown_pct:.2f}%",
                f"VaR (95%): {np.percentile(daily_returns*100, 5):.2f}%",
                f"CVaR (95%): {daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean()*100:.2f}%",
                f"Skewness: {daily_returns.skew():.3f}",
                f"Kurtosis: {daily_returns.kurtosis():.3f}",
                f"Best Day: {daily_returns.max()*100:.2f}%",
                f"Worst Day: {daily_returns.min()*100:.2f}%"
            ]
            ax3.text(0.05, 0.95, '\n'.join(risk_metrics), transform=ax3.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title('Risk Metrics')
            
            # 4. Rolling Sharpe Ratio (bottom-right)
            ax4 = axes[1, 1]
            rolling_sharpe = (daily_returns.rolling(window=60).mean() / 
                            daily_returns.rolling(window=60).std()) * np.sqrt(252)
            ax4.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='purple')
            ax4.axhline(result.sharpe_ratio, color='red', linestyle='--',
                       label=f'Overall: {result.sharpe_ratio:.3f}')
            ax4.set_title('60-Day Rolling Sharpe Ratio')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.set_xlabel('Date')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_risk_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_comparison_dashboard(
        self, results: Dict[str, BacktestResult], base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create strategy comparison dashboard"""
        n_strategies = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strategy Comparison Dashboard ({n_strategies} Strategies)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Performance Metrics Comparison (top-left)
        ax1 = axes[0, 0]
        metrics_data = {
            'Strategy': list(results.keys()),
            'Total Return (%)': [r.total_return_pct for r in results.values()],
            'Sharpe Ratio': [r.sharpe_ratio for r in results.values()],
            'Max Drawdown (%)': [r.max_drawdown_pct for r in results.values()]
        }
        
        x = np.arange(len(metrics_data['Strategy']))
        width = 0.25
        
        ax1.bar(x - width, metrics_data['Total Return (%)'], width, label='Total Return (%)', alpha=0.8)
        ax1.bar(x, [s * 10 for s in metrics_data['Sharpe Ratio']], width, label='Sharpe Ratio (×10)', alpha=0.8)
        ax1.bar(x + width, [-d for d in metrics_data['Max Drawdown (%)']], width, label='Max Drawdown (%) [inverted]', alpha=0.8)
        
        ax1.set_title('Key Performance Metrics')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Value')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_data['Strategy'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 2. Risk-Return Scatter (top-right)
        ax2 = axes[0, 1]
        returns = [r.total_return_pct for r in results.values()]
        risks = [r.volatility * 100 for r in results.values()]
        
        scatter = ax2.scatter(risks, returns, s=100, alpha=0.7, c=range(len(results)), cmap='viridis')
        for i, (name, result) in enumerate(results.items()):
            ax2.annotate(name, (risks[i], returns[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax2.set_title('Risk-Return Profile')
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Total Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Win Rates Comparison (bottom-left)
        ax3 = axes[1, 0]
        win_rates = [r.win_rate * 100 for r in results.values()]
        colors = plt.cm.RdYlGn([rate/100 for rate in win_rates])
        bars = ax3.bar(range(len(results)), win_rates, color=colors, alpha=0.8)
        ax3.set_title('Win Rates Comparison')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_xticks(range(len(results)))
        ax3.set_xticklabels(list(results.keys()), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax3.legend()
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Summary Statistics Table (bottom-right)
        ax4 = axes[1, 1]
        table_data = []
        headers = ['Strategy', 'Return%', 'Sharpe', 'MaxDD%', 'Trades', 'WinRate%']
        
        for name, result in results.items():
            table_data.append([
                name[:12] + '...' if len(name) > 12 else name,
                f"{result.total_return_pct:.1f}",
                f"{result.sharpe_ratio:.2f}",
                f"{result.max_drawdown_pct:.1f}",
                f"{result.total_trades}",
                f"{result.win_rate*100:.1f}"
            ])
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.axis('off')
        ax4.set_title('Performance Summary Table')
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "comparisons" / f"{base_filename}_comparison_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_equity_comparison(
        self, results: Dict[str, BacktestResult], base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create equity curves comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, (name, result) in enumerate(results.items()):
            if len(result.daily_values) > 0:
                # Normalize to initial value for comparison
                normalized_values = result.daily_values['portfolio_value'] / result.initial_value
                ax.plot(result.daily_values.index, normalized_values, 
                       linewidth=2, color=colors[i], label=name, alpha=0.8)
        
        ax.set_title('Equity Curves Comparison (Normalized to 1.0)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Portfolio Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Initial Value')
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "comparisons" / f"{base_filename}_equity_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_risk_return_scatter(
        self, results: Dict[str, BacktestResult], base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create risk-return scatter plot with annotations"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        returns = [r.total_return_pct for r in results.values()]
        risks = [r.volatility * 100 for r in results.values()]
        sharpes = [r.sharpe_ratio for r in results.values()]
        
        # Color by Sharpe ratio
        scatter = ax.scatter(risks, returns, s=150, alpha=0.7, c=sharpes, 
                           cmap='RdYlGn', edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        # Annotate points
        for i, (name, result) in enumerate(results.items()):
            ax.annotate(name, (risks[i], returns[i]), xytext=(10, 10), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title('Risk-Return Analysis with Sharpe Ratio')
        ax.set_xlabel('Volatility (% per year)')
        ax.set_ylabel('Total Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add efficient frontier line (if multiple points)
        if len(results) > 2:
            from scipy.optimize import curve_fit
            try:
                def hyperbola(x, a, b, c):
                    return a * np.sqrt(x**2 + b) + c
                
                popt, _ = curve_fit(hyperbola, risks, returns)
                x_smooth = np.linspace(min(risks), max(risks), 100)
                y_smooth = hyperbola(x_smooth, *popt)
                ax.plot(x_smooth, y_smooth, '--', color='red', alpha=0.5, 
                       label='Trend Line')
                ax.legend()
            except:
                pass  # Skip if curve fitting fails
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "comparisons" / f"{base_filename}_risk_return.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_metrics_comparison_table(
        self, results: Dict[str, BacktestResult], base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create detailed metrics comparison table"""
        # Create DataFrame for easy handling
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Strategy': name,
                'Total Return (%)': result.total_return_pct,
                'Annualized Return (%)': result.annualized_return * 100,
                'Volatility (%)': result.volatility * 100,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown (%)': result.max_drawdown_pct,
                'Win Rate (%)': result.win_rate * 100,
                'Total Trades': result.total_trades,
                'Profit Factor': result.profit_factor,
                'Avg Win ($)': result.avg_win,
                'Avg Loss ($)': result.avg_loss
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = []
        headers = list(df.columns)
        
        for _, row in df.iterrows():
            formatted_row = []
            for col in headers:
                if col == 'Strategy':
                    formatted_row.append(row[col])
                elif 'Return' in col or 'Volatility' in col or 'Drawdown' in col or 'Win Rate' in col:
                    formatted_row.append(f"{row[col]:.2f}")
                elif col == 'Sharpe Ratio' or col == 'Profit Factor':
                    formatted_row.append(f"{row[col]:.3f}")
                elif 'Avg' in col:
                    formatted_row.append(f"${row[col]:.2f}")
                else:
                    formatted_row.append(f"{row[col]}")
            table_data.append(formatted_row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Color code best performers
        for i in range(1, len(headers)):  # Skip strategy name column
            col_data = [float(row[i].replace('$', '').replace('%', '')) for row in table_data]
            if 'Drawdown' in headers[i] or 'Loss' in headers[i]:
                best_idx = col_data.index(min(col_data))  # Lower is better
            else:
                best_idx = col_data.index(max(col_data))  # Higher is better
            
            table[(best_idx + 1, i)].set_facecolor('lightgreen')
        
        ax.set_title('Detailed Performance Metrics Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "comparisons" / f"{base_filename}_metrics_table.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_drawdown_comparison(
        self, results: Dict[str, BacktestResult], base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create drawdown comparison chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, (name, result) in enumerate(results.items()):
            if len(result.daily_values) > 0:
                portfolio_values = result.daily_values['portfolio_value']
                rolling_max = portfolio_values.expanding().max()
                drawdown = (portfolio_values - rolling_max) / rolling_max * 100
                
                ax.fill_between(result.daily_values.index, drawdown, 0, 
                               alpha=0.3, color=colors[i])
                ax.plot(result.daily_values.index, drawdown, 
                       color=colors[i], linewidth=2, label=f'{name} (Max: {result.max_drawdown_pct:.1f}%)')
        
        ax.set_title('Drawdown Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "comparisons" / f"{base_filename}_drawdown_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_allocation_pie_chart(
        self, allocations: Dict[str, float], base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create asset allocation pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        assets = list(allocations.keys())
        sizes = list(allocations.values())
        
        # Create pie chart with custom colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
        wedges, texts, autotexts = ax.pie(sizes, labels=assets, autopct='%1.1f%%',
                                         colors=colors, startangle=90, pctdistance=0.85)
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Asset Allocation', fontsize=14, fontweight='bold')
        
        # Add a table with exact values
        table_data = [[asset, f"{alloc:.2%}"] for asset, alloc in allocations.items()]
        table = ax.table(cellText=table_data, colLabels=['Asset', 'Allocation'],
                        cellLoc='center', loc='center right', bbox=[1.1, 0.5, 0.3, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_allocation.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_per_asset_performance(
        self, result: BacktestResult, base_filename: str, save: bool, show: bool
    ) -> Optional[str]:
        """Create per-asset performance analysis"""
        if not result.closed_positions:
            return None
        
        # Group trades by symbol
        trades_df = pd.DataFrame(result.closed_positions)
        asset_performance = trades_df.groupby('symbol').agg({
            'pnl': ['sum', 'count', 'mean'],
            'return_pct': 'mean'
        }).round(2)
        
        # Flatten column names
        asset_performance.columns = ['Total_PnL', 'Trades', 'Avg_PnL', 'Avg_Return_Pct']
        asset_performance = asset_performance.reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Per-Asset Performance: {result.strategy_name}', 
                    fontsize=14, fontweight='bold')
        
        # 1. Total P&L by Asset (top-left)
        ax1 = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in asset_performance['Total_PnL']]
        bars1 = ax1.bar(asset_performance['symbol'], asset_performance['Total_PnL'], 
                       color=colors, alpha=0.7)
        ax1.set_title('Total P&L by Asset')
        ax1.set_ylabel('Total P&L ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars1, asset_performance['Total_PnL']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (height*0.05 if height > 0 else height*0.05),
                    f'${value:.0f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 2. Number of Trades by Asset (top-right)
        ax2 = axes[0, 1]
        ax2.bar(asset_performance['symbol'], asset_performance['Trades'], 
               color='skyblue', alpha=0.7)
        ax2.set_title('Number of Trades by Asset')
        ax2.set_ylabel('Number of Trades')
        ax2.grid(True, alpha=0.3)
        
        # 3. Average Return % by Asset (bottom-left)
        ax3 = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in asset_performance['Avg_Return_Pct']]
        bars3 = ax3.bar(asset_performance['symbol'], asset_performance['Avg_Return_Pct'], 
                       color=colors, alpha=0.7)
        ax3.set_title('Average Return % by Asset')
        ax3.set_ylabel('Average Return (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Performance Summary Table (bottom-right)
        ax4 = axes[1, 1]
        table_data = asset_performance.values.tolist()
        headers = ['Asset', 'Total P&L ($)', 'Trades', 'Avg P&L ($)', 'Avg Return (%)']
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.axis('off')
        ax4.set_title('Asset Performance Summary')
        
        plt.tight_layout()
        
        filename = None
        if save:
            filename = self.output_dir / "single_strategy" / f"{base_filename}_asset_performance.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filename) if filename else None
    
    def _create_interactive_dashboard(self, result: BacktestResult, base_filename: str) -> Optional[str]:
        """Create interactive dashboard using Plotly"""
        if not PLOTLY_AVAILABLE or len(result.daily_values) == 0:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value Over Time', 'Drawdown', 
                           'Daily Returns Distribution', 'Monthly Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Portfolio value
        fig.add_trace(
            go.Scatter(x=result.daily_values.index, 
                      y=result.daily_values['portfolio_value'],
                      mode='lines', name='Portfolio Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Drawdown
        portfolio_values = result.daily_values['portfolio_value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(x=result.daily_values.index, y=drawdown,
                      fill='tonexty', mode='lines', name='Drawdown',
                      line=dict(color='red', width=1)),
            row=1, col=2
        )
        
        # 3. Daily returns histogram
        daily_returns = result.daily_values['portfolio_value'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(x=daily_returns * 100, nbinsx=30, name='Daily Returns',
                        marker_color='skyblue', opacity=0.7),
            row=2, col=1
        )
        
        # 4. Monthly returns
        monthly_values = result.daily_values['portfolio_value'].resample('ME').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns * 100,
                  marker_color=colors, name='Monthly Returns', opacity=0.7),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Dashboard: {result.strategy_name} ({result.symbol})',
            height=600,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_xaxes(title_text="Daily Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)
        
        filename = self.output_dir / "interactive" / f"{base_filename}_interactive.html"
        pyo.plot(fig, filename=str(filename), auto_open=False)
        
        return str(filename)
    
    def _create_interactive_comparison(
        self, results: Dict[str, BacktestResult], base_filename: str
    ) -> Optional[str]:
        """Create interactive comparison dashboard"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curves Comparison', 'Risk-Return Scatter',
                           'Drawdown Comparison', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1[:len(results)]
        
        # 1. Equity curves comparison
        for i, (name, result) in enumerate(results.items()):
            if len(result.daily_values) > 0:
                normalized_values = result.daily_values['portfolio_value'] / result.initial_value
                fig.add_trace(
                    go.Scatter(x=result.daily_values.index, y=normalized_values,
                              mode='lines', name=name, line=dict(color=colors[i], width=2)),
                    row=1, col=1
                )
        
        # 2. Risk-return scatter
        returns = [r.total_return_pct for r in results.values()]
        risks = [r.volatility * 100 for r in results.values()]
        names = list(results.keys())
        
        fig.add_trace(
            go.Scatter(x=risks, y=returns, mode='markers+text', text=names,
                      textposition="top center", marker=dict(size=10, color=colors),
                      name='Strategies'),
            row=1, col=2
        )
        
        # 3. Drawdown comparison
        for i, (name, result) in enumerate(results.items()):
            if len(result.daily_values) > 0:
                portfolio_values = result.daily_values['portfolio_value']
                rolling_max = portfolio_values.expanding().max()
                drawdown = (portfolio_values - rolling_max) / rolling_max * 100
                
                fig.add_trace(
                    go.Scatter(x=result.daily_values.index, y=drawdown,
                              mode='lines', name=f'{name} Drawdown',
                              line=dict(color=colors[i], width=1)),
                    row=2, col=1
                )
        
        # 4. Performance metrics bar chart
        metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        for j, metric in enumerate(metrics):
            if metric == 'Total Return (%)':
                values = [r.total_return_pct for r in results.values()]
            elif metric == 'Sharpe Ratio':
                values = [r.sharpe_ratio for r in results.values()]
            else:  # Max Drawdown
                values = [-r.max_drawdown_pct for r in results.values()]  # Negative for better visualization
            
            fig.add_trace(
                go.Bar(x=list(results.keys()), y=values, name=metric,
                      marker_color=colors[j % len(colors)], opacity=0.7),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Strategy Comparison Dashboard',
            height=700,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
        fig.update_yaxes(title_text="Total Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Strategy", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        filename = self.output_dir / "interactive" / f"{base_filename}_interactive_comparison.html"
        pyo.plot(fig, filename=str(filename), auto_open=False)
        
        return str(filename)
    
    def create_market_comparison_report(
        self,
        stock_results: Dict[str, BacktestResult],
        crypto_results: Dict[str, BacktestResult],
        forex_results: Optional[Dict[str, BacktestResult]] = None,
        save_plots: bool = True,
        show_plots: bool = False
    ) -> Dict[str, str]:
        """
        Create comprehensive market comparison report
        
        Args:
            stock_results: Results from stock market strategies
            crypto_results: Results from crypto market strategies  
            forex_results: Results from forex market strategies (optional)
            save_plots: Whether to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of created file paths
        """
        self.logger.info("Creating comprehensive market comparison report")
        
        all_results = {}
        all_results.update({f"Stock_{k}": v for k, v in stock_results.items()})
        all_results.update({f"Crypto_{k}": v for k, v in crypto_results.items()})
        if forex_results:
            all_results.update({f"Forex_{k}": v for k, v in forex_results.items()})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create overall comparison
        comparison_files = self.compare_strategies(
            all_results, 
            f"market_comparison_{timestamp}",
            save_plots, 
            show_plots,
            include_interactive=True
        )
        
        self.logger.info(f"Market comparison report created with {len(comparison_files)} visualizations")
        return comparison_files
    
    def generate_strategy_report(
        self,
        result: BacktestResult,
        strategy_type: str = "single",
        asset_allocations: Optional[Dict[str, float]] = None,
        comparison_results: Optional[Dict[str, BacktestResult]] = None
    ) -> str:
        """
        Generate a comprehensive HTML report for strategy analysis
        
        Args:
            result: Main strategy result
            strategy_type: Type of strategy ("single", "multi_asset", "comparison")
            asset_allocations: Asset allocations for multi-asset strategies
            comparison_results: Other strategies to compare against
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = self.output_dir / f"strategy_report_{result.strategy_name}_{timestamp}.html"
        
        # Generate all visualizations
        if strategy_type == "multi_asset":
            files_created = self.visualize_multi_asset_strategy(
                result, asset_allocations, save_plots=True, show_plots=False
            )
        else:
            files_created = self.visualize_single_strategy(
                result, save_plots=True, show_plots=False, include_interactive=True
            )
        
        if comparison_results:
            all_results = {result.strategy_name: result}
            all_results.update(comparison_results)
            comparison_files = self.compare_strategies(
                all_results, f"comparison_{timestamp}", save_plots=True, show_plots=False
            )
            files_created.update(comparison_files)
        
        # Create HTML report
        html_content = self._generate_html_report(result, files_created, strategy_type)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Strategy report generated: {report_filename}")
        return str(report_filename)
    
    def _generate_html_report(
        self, result: BacktestResult, files_created: Dict[str, str], strategy_type: str
    ) -> str:
        """Generate HTML content for strategy report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Analysis Report: {result.strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .chart img {{ max-width: 100%; height: auto; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .neutral {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Strategy Analysis Report</h1>
                <h2>{result.strategy_name} ({result.symbol})</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Test Period: {result.start_date} to {result.end_date} ({result.duration_days} days)</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Total Return</h3>
                    <p class="{'positive' if result.total_return_pct > 0 else 'negative'}">{result.total_return_pct:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Sharpe Ratio</h3>
                    <p class="{'positive' if result.sharpe_ratio > 0 else 'negative'}">{result.sharpe_ratio:.3f}</p>
                </div>
                <div class="metric">
                    <h3>Max Drawdown</h3>
                    <p class="negative">{result.max_drawdown_pct:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <p class="{'positive' if result.win_rate > 0.5 else 'negative'}">{result.win_rate*100:.1f}%</p>
                </div>
            </div>
        """
        
        # Add charts
        for chart_name, file_path in files_created.items():
            if file_path and file_path.endswith('.png'):
                relative_path = os.path.relpath(file_path, self.output_dir)
                html += f"""
                <div class="chart">
                    <h3>{chart_name.replace('_', ' ').title()}</h3>
                    <img src="{relative_path}" alt="{chart_name}">
                </div>
                """
        
        # Add interactive charts links
        for chart_name, file_path in files_created.items():
            if file_path and file_path.endswith('.html'):
                relative_path = os.path.relpath(file_path, self.output_dir)
                html += f"""
                <div class="chart">
                    <h3>Interactive {chart_name.replace('_', ' ').title()}</h3>
                    <p><a href="{relative_path}" target="_blank">Open Interactive Chart</a></p>
                </div>
                """
        
        html += """
            </body>
            </html>
        """
        
        return html


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = get_logger(__name__)
    
    # This would be used with actual BacktestResult objects
    logger.info("StrategyVisualizer module loaded successfully")
    logger.info(f"Plotly available: {PLOTLY_AVAILABLE}")
    
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Install with: pip install plotly")
    
    # Create visualizer
    visualizer = StrategyVisualizer("test_visualizations")
    logger.info(f"Visualizer created. Output directory: {visualizer.output_dir}")