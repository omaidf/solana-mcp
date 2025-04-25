"""Response models for the API.

This module defines Pydantic models for standardized API responses.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class TokenMetadataResponse(BaseModel):
    """Model for token metadata responses."""
    
    mint: str = Field(..., description="Token mint address")
    name: str = Field(..., description="Token name")
    symbol: str = Field(..., description="Token symbol")
    uri: Optional[str] = Field(None, description="Token metadata URI")
    category: Optional[str] = Field(None, description="Token category")
    is_meme_token: Optional[bool] = Field(None, description="Whether the token is a meme token")
    decimals: Optional[int] = Field(None, description="Token decimals")
    
class TokenPriceResponse(BaseModel):
    """Model for token price responses."""
    
    price_usd: float = Field(..., description="Token price in USD")
    price_sol: float = Field(..., description="Token price in SOL")
    liquidity_usd: Optional[float] = Field(None, description="Token liquidity in USD")
    market_cap_usd: Optional[float] = Field(None, description="Token market cap in USD")
    volume_24h_usd: Optional[float] = Field(None, description="24-hour trading volume in USD")
    change_24h_percent: Optional[float] = Field(None, description="24-hour price change percentage")
    source: Optional[str] = Field(None, description="Price data source")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")

class TokenFullDetailsResponse(BaseModel):
    """Model for comprehensive token details responses."""
    
    mint: str = Field(..., description="Token mint address")
    metadata: TokenMetadataResponse = Field(..., description="Token metadata")
    price_data: Optional[TokenPriceResponse] = Field(None, description="Token price data")
    supply: Dict[str, Any] = Field(..., description="Token supply information")
    holders_count: Optional[int] = Field(None, description="Number of token holders")
    largest_holder_percentage: Optional[float] = Field(None, description="Percentage held by largest holder")
    launch_date: Optional[str] = Field(None, description="Token launch date")
    age_days: Optional[int] = Field(None, description="Token age in days")
    
class AccountBalanceResponse(BaseModel):
    """Model for account balance responses."""
    
    address: str = Field(..., description="Account address")
    lamports: int = Field(..., description="Balance in lamports")
    sol: float = Field(..., description="Balance in SOL")
    formatted: str = Field(..., description="Formatted balance string")
    
class TransactionResponse(BaseModel):
    """Model for transaction responses."""
    
    signature: str = Field(..., description="Transaction signature")
    slot: int = Field(..., description="Slot in which the transaction was processed")
    block_time: Optional[int] = Field(None, description="Block time (Unix timestamp)")
    confirmations: Optional[int] = Field(None, description="Number of confirmations")
    is_confirmed: bool = Field(..., description="Whether the transaction is confirmed")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information if transaction failed")
    
class TransactionHistoryResponse(BaseModel):
    """Model for transaction history responses."""
    
    address: str = Field(..., description="Account address")
    transactions: List[TransactionResponse] = Field(..., description="List of transactions")
    count: int = Field(..., description="Number of transactions returned")
    
class NaturalLanguageQueryResponse(BaseModel):
    """Model for natural language query responses."""
    
    result: Dict[str, Any] = Field(..., description="Query result")
    session_id: str = Field(..., description="Session ID")
    query_count: int = Field(..., description="Number of queries in the session")
    
class TokenHolderResponse(BaseModel):
    """Model for token holder responses."""
    
    owner: str = Field(..., description="Account address of the holder")
    address: str = Field(..., description="Token account address")
    amount: str = Field(..., description="Token amount as string")
    decimals: int = Field(..., description="Token decimals")
    ui_amount: float = Field(..., description="Token amount as a float")
    percentage: float = Field(..., description="Percentage of total supply")
    
class TokenHoldersResponse(BaseModel):
    """Model for token holders responses."""
    
    mint: str = Field(..., description="Token mint address")
    holders: List[TokenHolderResponse] = Field(..., description="List of token holders")
    total_holders: int = Field(..., description="Total number of token holders")
    holders_shown: int = Field(..., description="Number of holders shown") 