from typing import Dict, List, Optional
from datetime import datetime
from .position import Position, PositionType, PositionStatus
import pandas as pd


class Portfolio:
    """
    Manages trading portfolio including cash, positions, and P&L tracking
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize portfolio
        
        Args:
            initial_cash: Starting cash amount
        """
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []
        self.transaction_history: List[Dict] = []
        
    @property
    def total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions_value = sum(pos.current_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def positions_value(self) -> float:
        """Get total value of all open positions"""
        return sum(pos.current_value for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.total_value - self.initial_cash
    
    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L from closed positions"""
        return sum(pos.realized_pnl for pos in self.closed_positions)
    
    @property
    def unrealized_pnl(self) -> float:
        """Get total unrealized P&L from open positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_return_pct(self) -> float:
        """Get total return percentage"""
        return (self.total_pnl / self.initial_cash) * 100
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol"""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has open position for symbol"""
        return symbol in self.positions
    
    def calculate_position_size(self, symbol: str, price: float, allocation_pct: float = 0.1) -> int:
        """
        Calculate position size based on allocation percentage
        
        Args:
            symbol: Stock symbol
            price: Current stock price
            allocation_pct: Percentage of portfolio to allocate (0.0 to 1.0)
            
        Returns:
            Number of shares to buy
        """
        if not (0.0 <= allocation_pct <= 1.0):
            raise ValueError("Allocation percentage must be between 0 and 1")
        
        allocation_amount = self.total_value * allocation_pct
        max_shares = int(allocation_amount / price)
        
        # Ensure we don't exceed available cash
        cost_per_share = price  # Simplified, fees not included here
        max_affordable = int(self.cash / cost_per_share)
        
        return min(max_shares, max_affordable)
    
    def open_position(
        self, 
        symbol: str, 
        position_type: PositionType, 
        quantity: int, 
        price: float, 
        timestamp: datetime,
        fees: float = 0.0
    ) -> Position:
        """
        Open a new position
        
        Args:
            symbol: Stock symbol
            position_type: LONG or SHORT
            quantity: Number of shares
            price: Entry price per share
            timestamp: When position was opened
            fees: Transaction fees
            
        Returns:
            Created Position object
        """
        if self.has_position(symbol):
            raise ValueError(f"Position already exists for {symbol}")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if price <= 0:
            raise ValueError("Price must be positive")
        
        # Calculate total cost
        total_cost = (quantity * price) + fees
        
        # Check if we have enough cash
        if total_cost > self.cash:
            raise ValueError(f"Insufficient cash. Need ${total_cost:.2f}, have ${self.cash:.2f}")
        
        # Create position
        position = Position(
            symbol=symbol,
            position_type=position_type,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            entry_fees=fees
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash -= total_cost
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': timestamp,
            'action': 'OPEN',
            'symbol': symbol,
            'type': position_type.value,
            'quantity': quantity,
            'price': price,
            'fees': fees,
            'total_cost': total_cost,
            'cash_after': self.cash
        })
        
        return position
    
    def close_position(
        self, 
        symbol: str, 
        price: float, 
        timestamp: datetime,
        fees: float = 0.0
    ) -> Position:
        """
        Close an existing position
        
        Args:
            symbol: Stock symbol
            price: Exit price per share
            timestamp: When position was closed
            fees: Transaction fees
            
        Returns:
            Closed Position object
        """
        if not self.has_position(symbol):
            raise ValueError(f"No open position for {symbol}")
        
        position = self.positions[symbol]
        
        # Calculate proceeds
        proceeds = (position.quantity * price) - fees
        
        # Close the position
        position.close_position(price, timestamp, fees)
        
        # Update portfolio
        self.cash += proceeds
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Record transaction
        self.transaction_history.append({
            'timestamp': timestamp,
            'action': 'CLOSE',
            'symbol': symbol,
            'type': position.position_type.value,
            'quantity': position.quantity,
            'price': price,
            'fees': fees,
            'proceeds': proceeds,
            'pnl': position.realized_pnl,
            'cash_after': self.cash
        })
        
        return position
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update portfolio with current market prices
        
        Args:
            current_prices: Dict of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                # Update unrealized P&L (this doesn't change the position object,
                # just calculates current P&L)
                current_price = current_prices[symbol]
                position.update_unrealized_pnl(current_price)
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        return {
            'cash': self.cash,
            'positions_value': self.positions_value,
            'total_value': self.total_value,
            'initial_cash': self.initial_cash,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return_pct': self.total_return_pct,
            'open_positions_count': len(self.positions),
            'closed_positions_count': len(self.closed_positions),
            'open_positions': {symbol: {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_value': pos.current_value,
                'unrealized_pnl': pos.unrealized_pnl
            } for symbol, pos in self.positions.items()}
        }
    
    def get_transaction_history_df(self) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        if not self.transaction_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.transaction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    
    def get_closed_positions_df(self) -> pd.DataFrame:
        """Get closed positions as DataFrame"""
        if not self.closed_positions:
            return pd.DataFrame()
        
        data = []
        for pos in self.closed_positions:
            data.append({
                'symbol': pos.symbol,
                'type': pos.position_type.value,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'exit_price': pos.exit_price,
                'entry_time': pos.entry_time,
                'exit_time': pos.exit_time,
                'entry_fees': pos.entry_fees,
                'exit_fees': pos.exit_fees,
                'realized_pnl': pos.realized_pnl,
                'return_pct': pos.calculate_return_pct(pos.exit_price)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_cash
        self.positions = {}
        self.closed_positions = []
        self.transaction_history = []
    
    def __repr__(self):
        return f"Portfolio(cash=${self.cash:.2f}, positions={len(self.positions)}, total_value=${self.total_value:.2f}, pnl=${self.total_pnl:.2f})"