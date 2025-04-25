"""Data models for whale detection."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from decimal import Decimal

@dataclass
class TokenInfo:
    """Token information for whale detection."""
    
    mint: str
    symbol: str
    decimals: int
    price_usd: Decimal
    total_supply: Optional[Decimal] = None
    
    @property
    def formatted_price(self) -> str:
        """Get formatted price string.
        
        Returns:
            Formatted price string
        """
        return f"${float(self.price_usd):.6f}"


@dataclass
class TokenBalance:
    """Token balance information."""
    
    mint: str
    token: str  # Short name or symbol
    amount: Decimal
    decimals: int = 0
    

@dataclass
class TokenValue:
    """Token with value information."""
    
    mint: str
    token: str  # Short name or symbol
    amount: float
    value_usd: float


@dataclass
class WalletValue:
    """Wallet value information."""
    
    total_value_usd: float
    target_token_value_usd: float
    tokens: List[TokenValue] = field(default_factory=list)


@dataclass
class WhaleWallet:
    """Information about a whale wallet."""
    
    wallet: str
    target_token_amount: float
    target_token_value_usd: float
    target_token_supply_percentage: float
    total_value_usd: float
    token_count: int
    top_tokens: List[TokenValue] = field(default_factory=list)


@dataclass
class WhaleDetectionResult:
    """Result of whale detection."""
    
    token_address: str
    token_symbol: str
    token_price_usd: float
    whale_threshold_usd: float
    whale_count: int
    whales: List[WhaleWallet] = field(default_factory=list)
    
    @classmethod
    def error(cls, message: str) -> Dict[str, Any]:
        """Create an error result.
        
        Args:
            message: Error message
            
        Returns:
            Error dictionary
        """
        return {"error": message} 