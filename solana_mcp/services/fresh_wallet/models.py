"""Data models for fresh wallet detection."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from decimal import Decimal

@dataclass
class TokenInfo:
    """Token information for fresh wallet detection."""
    
    mint: str
    symbol: str
    decimals: int
    price_usd: Decimal
    
    @property
    def formatted_price(self) -> str:
        """Get formatted price string.
        
        Returns:
            Formatted price string
        """
        return f"${float(self.price_usd):.6f}"


@dataclass
class FreshWallet:
    """Information about a fresh wallet."""
    
    wallet: str
    is_fresh: bool
    token_count: int
    non_dust_token_count: int
    token_tx_ratio: float
    wallet_age_days: Optional[int]
    target_token_amount: float
    target_token_value_usd: float
    freshness_score: float


@dataclass
class FreshWalletDetectionResult:
    """Result of fresh wallet detection."""
    
    token_address: str
    token_symbol: str
    token_price_usd: float
    fresh_wallet_count: int
    total_analyzed_wallets: int
    fresh_wallets: List[FreshWallet] = field(default_factory=list)
    
    @classmethod
    def error(cls, message: str) -> Dict[str, Any]:
        """Create an error result.
        
        Args:
            message: Error message
            
        Returns:
            Error dictionary
        """
        return {"error": message} 