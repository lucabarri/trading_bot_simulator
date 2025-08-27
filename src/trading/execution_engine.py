from typing import Dict, List, Optional, Callable
from datetime import datetime, timezone
from enum import Enum
import pandas as pd
import logging

from .portfolio import Portfolio
from .position import PositionType

# Import strategy components when needed to avoid circular imports
try:
    from ..strategies.base_strategy import BaseStrategy, TradingSignal, Signal
except ImportError:
    # Handle when imported from tests
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from strategies.base_strategy import BaseStrategy, TradingSignal, Signal


class ExecutionMode(Enum):
    """Execution modes for the trading engine"""
    PAPER = "PAPER"          # Paper trading simulation
    BACKTEST = "BACKTEST"    # Historical backtesting
    LIVE = "LIVE"            # Live trading (not implemented)


class OrderType(Enum):
    """Order types supported by the execution engine"""
    MARKET = "MARKET"        # Execute at current market price
    LIMIT = "LIMIT"          # Execute at specified price or better
    STOP = "STOP"            # Stop loss order


class OrderStatus(Enum):
    """Order status types"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class Order:
    """
    Represents a trading order
    """
    def __init__(
        self,
        symbol: str,
        action: str,  # BUY or SELL
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.order_id = self._generate_order_id()
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.timestamp = timestamp or pd.Timestamp.now(tz=timezone.utc)
        self.status = OrderStatus.PENDING
        self.fill_price: Optional[float] = None
        self.fill_time: Optional[datetime] = None
        self.fees: float = 0.0
        self.rejection_reason: Optional[str] = None
    
    @staticmethod
    def _generate_order_id() -> str:
        """Generate unique order ID"""
        import uuid
        return f"ORD_{uuid.uuid4().hex[:8].upper()}"
    
    def fill(self, fill_price: float, fill_time: datetime, fees: float = 0.0):
        """Mark order as filled"""
        self.status = OrderStatus.FILLED
        self.fill_price = fill_price
        self.fill_time = fill_time
        self.fees = fees
    
    def reject(self, reason: str):
        """Mark order as rejected"""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
    
    def cancel(self):
        """Cancel the order"""
        self.status = OrderStatus.CANCELLED
    
    def __repr__(self):
        return f"Order({self.order_id}: {self.action} {self.quantity} {self.symbol} @ {self.order_type.value} - {self.status.value})"


class ExecutionEngine:
    """
    Trading execution engine for paper trading and backtesting
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        mode: ExecutionMode = ExecutionMode.PAPER,
        base_fee: float = 5.0,
        fee_per_share: float = 0.0,
        min_fee: float = 1.0,
        max_fee: float = 50.0,
        slippage_bps: float = 2.0  # basis points (0.02%)
    ):
        """
        Initialize execution engine
        
        Args:
            portfolio: Portfolio to manage
            mode: Execution mode (PAPER, BACKTEST, LIVE)
            base_fee: Base trading fee per transaction
            fee_per_share: Additional fee per share
            min_fee: Minimum fee per transaction
            max_fee: Maximum fee per transaction
            slippage_bps: Slippage in basis points
        """
        self.portfolio = portfolio
        self.mode = mode
        self.base_fee = base_fee
        self.fee_per_share = fee_per_share
        self.min_fee = min_fee
        self.max_fee = max_fee
        self.slippage_bps = slippage_bps / 10000.0  # Convert basis points to decimal
        
        # Order tracking
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.rejected_orders: List[Order] = []
        
        # Hooks for custom logic
        self.pre_trade_hooks: List[Callable] = []
        self.post_trade_hooks: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"ExecutionEngine_{mode.value}")
    
    def calculate_fees(self, quantity: int, price: float) -> float:
        """
        Calculate trading fees
        
        Args:
            quantity: Number of shares
            price: Price per share
            
        Returns:
            Total fees for the transaction
        """
        fee = self.base_fee + (quantity * self.fee_per_share)
        return max(self.min_fee, min(fee, self.max_fee))
    
    def apply_slippage(self, price: float, action: str) -> float:
        """
        Apply slippage to execution price
        
        Args:
            price: Original price
            action: BUY or SELL
            
        Returns:
            Price with slippage applied
        """
        if self.slippage_bps == 0:
            return price
        
        # For BUY orders, slippage increases price (unfavorable)
        # For SELL orders, slippage decreases price (unfavorable)
        if action == "BUY":
            return price * (1 + self.slippage_bps)
        else:  # SELL
            return price * (1 - self.slippage_bps)
    
    def validate_order(self, order: Order, current_price: float) -> tuple[bool, Optional[str]]:
        """
        Validate order before execution
        
        Args:
            order: Order to validate
            current_price: Current market price
            
        Returns:
            (is_valid, rejection_reason)
        """
        # Check basic order parameters
        if order.quantity <= 0:
            return False, "Invalid quantity"
        
        if current_price <= 0:
            return False, "Invalid market price"
        
        # Check portfolio constraints for BUY orders
        if order.action == "BUY":
            estimated_cost = (order.quantity * current_price) + self.calculate_fees(order.quantity, current_price)
            if estimated_cost > self.portfolio.cash:
                return False, f"Insufficient cash: need ${estimated_cost:.2f}, have ${self.portfolio.cash:.2f}"
        
        # Check position exists for SELL orders
        elif order.action == "SELL":
            if not self.portfolio.has_position(order.symbol):
                return False, f"No position to sell for {order.symbol}"
            
            position = self.portfolio.get_position(order.symbol)
            if position.quantity < order.quantity:
                return False, f"Insufficient shares: trying to sell {order.quantity}, have {position.quantity}"
        
        # Validate limit orders
        if order.order_type == OrderType.LIMIT and order.price is not None:
            if order.action == "BUY" and current_price > order.price:
                return False, f"Limit buy price ${order.price:.2f} below market ${current_price:.2f}"
            elif order.action == "SELL" and current_price < order.price:
                return False, f"Limit sell price ${order.price:.2f} above market ${current_price:.2f}"
        
        return True, None
    
    def execute_order(self, order: Order, current_price: float, timestamp: datetime) -> bool:
        """
        Execute a trading order
        
        Args:
            order: Order to execute
            current_price: Current market price
            timestamp: Execution timestamp (must be UTC)
            
        Returns:
            True if order was filled, False if rejected
        """
        # Ensure timestamp is UTC (canonical time)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        
        # Validate order
        is_valid, rejection_reason = self.validate_order(order, current_price)
        if not is_valid:
            order.reject(rejection_reason)
            self.rejected_orders.append(order)
            self.logger.warning(f"Order rejected: {order} - {rejection_reason}")
            return False
        
        # Run pre-trade hooks
        for hook in self.pre_trade_hooks:
            try:
                hook(order, current_price, timestamp)
            except Exception as e:
                self.logger.error(f"Pre-trade hook failed: {e}")
        
        # Determine execution price
        if order.order_type == OrderType.MARKET:
            execution_price = self.apply_slippage(current_price, order.action)
        elif order.order_type == OrderType.LIMIT:
            execution_price = order.price if order.price is not None else current_price
        else:
            execution_price = current_price
        
        # Calculate fees
        fees = self.calculate_fees(order.quantity, execution_price)
        
        # Execute the trade in portfolio
        try:
            if order.action == "BUY":
                self.portfolio.open_position(
                    symbol=order.symbol,
                    position_type=PositionType.LONG,
                    quantity=order.quantity,
                    price=execution_price,
                    timestamp=timestamp,
                    fees=fees
                )
            elif order.action == "SELL":
                self.portfolio.close_position(
                    symbol=order.symbol,
                    price=execution_price,
                    timestamp=timestamp,
                    fees=fees
                )
            
            # Mark order as filled
            order.fill(execution_price, timestamp, fees)
            self.filled_orders.append(order)
            
            self.logger.info(f"Order filled: {order} at ${execution_price:.2f} (fees: ${fees:.2f})")
            
            # Run post-trade hooks
            for hook in self.post_trade_hooks:
                try:
                    hook(order, execution_price, timestamp)
                except Exception as e:
                    self.logger.error(f"Post-trade hook failed: {e}")
            
            return True
            
        except Exception as e:
            order.reject(f"Execution failed: {str(e)}")
            self.rejected_orders.append(order)
            self.logger.error(f"Order execution failed: {order} - {e}")
            return False
    
    def create_order_from_signal(
        self,
        signal: TradingSignal,
        symbol: str,
        allocation_pct: float = 0.1,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[Order]:
        """
        Create an order from a trading signal
        
        Args:
            signal: Trading signal
            symbol: Stock symbol
            allocation_pct: Portfolio allocation percentage
            order_type: Type of order to create
            
        Returns:
            Order object or None if signal doesn't warrant an order
        """
        if signal.signal == Signal.HOLD:
            return None
        
        if signal.signal == Signal.BUY and not self.portfolio.has_position(symbol):
            # Calculate position size
            quantity = self.portfolio.calculate_position_size(symbol, signal.price, allocation_pct)
            if quantity > 0:
                return Order(
                    symbol=symbol,
                    action="BUY",
                    quantity=quantity,
                    order_type=order_type,
                    price=signal.price if order_type == OrderType.LIMIT else None,
                    timestamp=signal.timestamp
                )
        
        elif signal.signal == Signal.SELL and self.portfolio.has_position(symbol):
            position = self.portfolio.get_position(symbol)
            return Order(
                symbol=symbol,
                action="SELL",
                quantity=position.quantity,
                order_type=order_type,
                price=signal.price if order_type == OrderType.LIMIT else None,
                timestamp=signal.timestamp
            )
        
        return None
    
    def process_signals(
        self,
        signals: List[TradingSignal],
        current_prices: Dict[str, float],
        allocation_pct: float = 0.1
    ) -> List[Order]:
        """
        Process multiple trading signals and execute resulting orders
        
        Args:
            signals: List of trading signals
            current_prices: Current market prices by symbol
            allocation_pct: Portfolio allocation percentage per trade
            
        Returns:
            List of executed orders
        """
        executed_orders = []
        
        for signal in signals:
            if signal.signal == Signal.HOLD:
                continue
            
            # Assume signal is for the symbol we have price data for
            symbol = None
            current_price = None
            
            # Find matching symbol and price
            for sym, price in current_prices.items():
                symbol = sym
                current_price = price
                break
            
            if symbol is None or current_price is None:
                self.logger.warning(f"No price data for signal: {signal}")
                continue
            
            # Create order from signal
            order = self.create_order_from_signal(signal, symbol, allocation_pct)
            if order is None:
                continue
            
            # Execute order
            if self.execute_order(order, current_price, signal.timestamp):
                executed_orders.append(order)
        
        return executed_orders
    
    def add_pre_trade_hook(self, hook: Callable):
        """Add a pre-trade hook function"""
        self.pre_trade_hooks.append(hook)
    
    def add_post_trade_hook(self, hook: Callable):
        """Add a post-trade hook function"""
        self.post_trade_hooks.append(hook)
    
    def get_execution_summary(self) -> Dict:
        """Get summary of execution engine performance"""
        total_orders = len(self.filled_orders) + len(self.rejected_orders)
        fill_rate = len(self.filled_orders) / total_orders if total_orders > 0 else 0.0
        
        total_fees = sum(order.fees for order in self.filled_orders)
        
        return {
            'mode': self.mode.value,
            'total_orders': total_orders,
            'filled_orders': len(self.filled_orders),
            'rejected_orders': len(self.rejected_orders),
            'fill_rate': fill_rate,
            'total_fees_paid': total_fees,
            'avg_slippage_bps': self.slippage_bps * 10000,
            'orders': {
                'filled': [order.__dict__ for order in self.filled_orders],
                'rejected': [order.__dict__ for order in self.rejected_orders]
            }
        }
    
    def reset(self):
        """Reset execution engine state"""
        self.pending_orders = []
        self.filled_orders = []
        self.rejected_orders = []
    
    def __repr__(self):
        return f"ExecutionEngine({self.mode.value}, filled={len(self.filled_orders)}, rejected={len(self.rejected_orders)})"