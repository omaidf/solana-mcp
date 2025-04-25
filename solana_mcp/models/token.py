"""
Token data models for the Solana MCP.

This module defines Pydantic models for token-related data structures, including
token information, metadata, and price data.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, field_validator


class TokenMetadata(BaseModel):
    """
    Model for token metadata information.
    """
    name: Optional[str] = None
    symbol: Optional[str] = None
    logo: Optional[str] = None
    description: Optional[str] = None
    external_url: Optional[str] = None
    image: Optional[str] = None
    is_nft: bool = False


class TokenPrice(BaseModel):
    """
    Model for token price information.
    """
    price_usd: Optional[float] = None
    price_sol: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    last_updated: Optional[int] = None

    @property
    def formatted_price_usd(self) -> str:
        """Get the price formatted as a USD string."""
        if self.price_usd is None:
            return "Unknown"
        
        if self.price_usd >= 10000:
            return f"${self.price_usd:,.2f}"
        elif self.price_usd >= 1:
            return f"${self.price_usd:.2f}"
        elif self.price_usd >= 0.01:
            return f"${self.price_usd:.4f}"
        else:
            return f"${self.price_usd:.8f}"
    
    @property
    def formatted_change(self) -> str:
        """Get the price change formatted as a percentage."""
        if self.price_change_24h is None:
            return "Unknown"
        
        if self.price_change_24h > 0:
            return f"+{self.price_change_24h:.2f}%"
        else:
            return f"{self.price_change_24h:.2f}%"
    
    @property
    def last_updated_datetime(self) -> Optional[datetime]:
        """Get the last updated time as a datetime."""
        if self.last_updated is None:
            return None
        
        return datetime.fromtimestamp(self.last_updated)


class TokenInfo(BaseModel):
    """
    Model for token information.
    
    Contains basic token data as well as references to
    metadata and price information.
    """
    address: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    decimals: int = 0
    total_supply: Optional[str] = None
    metadata: Optional[TokenMetadata] = None
    price: Optional[TokenPrice] = None
    is_nft: bool = False

    @property
    def display_name(self) -> str:
        """Get a display name for the token."""
        if self.name:
            return self.name
        
        return f"Unknown Token ({self.address[:8]}...)"
    
    @property
    def formatted_supply(self) -> str:
        """Get the total supply formatted with decimals."""
        if not self.total_supply:
            return "Unknown"
        
        try:
            raw_supply = int(self.total_supply)
            if self.decimals > 0:
                adjusted_supply = raw_supply / (10 ** self.decimals)
                if adjusted_supply >= 1_000_000_000:
                    return f"{adjusted_supply / 1_000_000_000:.2f}B"
                elif adjusted_supply >= 1_000_000:
                    return f"{adjusted_supply / 1_000_000:.2f}M"
                elif adjusted_supply >= 1_000:
                    return f"{adjusted_supply / 1_000:.2f}K"
                else:
                    return f"{adjusted_supply:,.2f}"
            else:
                return str(raw_supply)
        except ValueError:
            return self.total_supply


class TokenList(BaseModel):
    """
    Model for a list of tokens.
    
    Used for API responses when returning multiple tokens.
    """
    tokens: List[TokenInfo]
    count: int
    total: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "tokens": [
                    {
                        "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "name": "USD Coin",
                        "symbol": "USDC",
                        "decimals": 6,
                        "total_supply": "5034233953603116",
                        "is_nft": False
                    }
                ],
                "count": 1,
                "total": 100,
                "page": 1,
                "page_size": 10
            }
        }
    }


class TokenSupply(BaseModel):
    """Model for token supply information."""
    
    mint_address: str = Field(..., description="The token's mint address")
    total_supply: str = Field(..., description="Total supply as a string")
    circulating_supply: Optional[str] = Field(None, description="Circulating supply as a string")
    decimals: int = Field(..., description="The token's decimals")
    total_holders: Optional[int] = Field(None, description="Total number of token holders")
    max_supply: Optional[str] = Field(None, description="Maximum possible supply as a string")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mint_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "total_supply": "10000000000000000",
                "circulating_supply": "9500000000000000",
                "decimals": 6,
                "total_holders": 1000000,
                "max_supply": "10000000000000000"
            }
        }
    }


class TokenHolder(BaseModel):
    """Model for token holder information."""
    
    address: str
    amount: str
    amount_decimal: Optional[float] = None
    percentage: Optional[float] = None
    
    @property
    def formatted_amount(self) -> str:
        """Get the token amount formatted with proper decimal places."""
        if self.amount_decimal is not None:
            if self.amount_decimal >= 1_000_000_000:
                return f"{self.amount_decimal / 1_000_000_000:.2f}B"
            elif self.amount_decimal >= 1_000_000:
                return f"{self.amount_decimal / 1_000_000:.2f}M"
            elif self.amount_decimal >= 1_000:
                return f"{self.amount_decimal / 1_000:.2f}K"
            else:
                return f"{self.amount_decimal:,.2f}"
        
        return self.amount
    
    @property
    def formatted_percentage(self) -> str:
        """Get the percentage formatted as a string."""
        if self.percentage is None:
            return "Unknown"
        
        return f"{self.percentage:.2f}%"


class TokenListResponse(BaseModel):
    """Model representing a paginated response of tokens."""
    items: List[TokenInfo] = Field(..., description="List of tokens")
    total: int = Field(..., description="Total number of tokens")
    limit: int = Field(..., description="Number of tokens per page")
    offset: int = Field(..., description="Pagination offset")


class TokenHoldersResponse(BaseModel):
    """Model representing a paginated response of token holders."""
    mint_address: str = Field(..., description="Token mint address")
    items: List[TokenHolder] = Field(..., description="List of token holders")
    total: int = Field(..., description="Total number of token holders")
    limit: int = Field(..., description="Number of holders per page")
    offset: int = Field(..., description="Pagination offset") 