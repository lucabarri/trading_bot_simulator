from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class PositionType(Enum):
    """Position types"""
    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(Enum):
    """Position status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """
    Represents a trading position (long or short)
    """
    symbol: str
    position_type: PositionType
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_fees: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_fees: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    
    def __post_init__(self):
        """Validate position data"""
        if self.quantity <= 0:
            raise ValueError("Position quantity must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.entry_fees < 0:
            raise ValueError("Entry fees cannot be negative")
    
    @property
    def current_value(self) -> float:
        """Get current market value of the position"""
        current_price = self.exit_price if self.status == PositionStatus.CLOSED else self.entry_price
        return self.quantity * current_price
    
    @property
    def cost_basis(self) -> float:
        """Get total cost basis including fees"""
        return (self.quantity * self.entry_price) + self.entry_fees
    
    @property
    def unrealized_pnl(self) -> float:
        """Get unrealized P&L (for open positions)"""
        if self.status == PositionStatus.CLOSED:
            return 0.0
        
        # For this calculation, we'll use entry_price as current price
        # In real implementation, this would use current market price
        return self.calculate_pnl(self.entry_price)
    
    @property
    def realized_pnl(self) -> float:
        """Get realized P&L (for closed positions)"""
        if self.status == PositionStatus.OPEN or self.exit_price is None:
            return 0.0
        
        return self.calculate_pnl(self.exit_price)
    
    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate P&L at given price
        
        Args:
            current_price: Price to calculate P&L at
            
        Returns:
            P&L amount including all fees
        """
        if self.position_type == PositionType.LONG:
            gross_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            gross_pnl = (self.entry_price - current_price) * self.quantity
        
        total_fees = self.entry_fees + self.exit_fees
        return gross_pnl - total_fees
    
    def calculate_return_pct(self, current_price: float) -> float:
        """
        Calculate percentage return at given price
        
        Args:
            current_price: Price to calculate return at
            
        Returns:
            Return percentage
        """
        pnl = self.calculate_pnl(current_price)
        return (pnl / self.cost_basis) * 100
    
    def close_position(self, exit_price: float, exit_time: datetime, exit_fees: float = 0.0):
        """
        Close the position
        
        Args:
            exit_price: Price at which position was closed
            exit_time: Time when position was closed
            exit_fees: Fees for closing the position
        """
        if self.status == PositionStatus.CLOSED:
            raise ValueError("Position is already closed")
        
        if exit_price <= 0:
            raise ValueError("Exit price must be positive")
        
        if exit_fees < 0:
            raise ValueError("Exit fees cannot be negative")
        
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_fees = exit_fees
        self.status = PositionStatus.CLOSED
    
    def update_unrealized_pnl(self, current_price: float) -> float:
        """
        Update and return unrealized P&L with current market price
        
        Args:
            current_price: Current market price
            
        Returns:
            Updated unrealized P&L
        """
        if self.status == PositionStatus.CLOSED:
            return 0.0
        
        return self.calculate_pnl(current_price)
    
    def __repr__(self):
        status_str = f"{self.status.value}"
        if self.status == PositionStatus.OPEN:
            return f"Position({self.symbol} {self.position_type.value} {self.quantity}@${self.entry_price:.2f} - {status_str})"
        else:
            pnl = self.realized_pnl
            return f"Position({self.symbol} {self.position_type.value} {self.quantity}@${self.entry_price:.2f}->${self.exit_price:.2f} P&L:${pnl:.2f} - {status_str})"