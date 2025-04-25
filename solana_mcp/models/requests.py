"""Request validation models for the API.

This module defines Pydantic models for validating API request data.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, constr

# Regex for Solana public keys
SOLANA_PUBLIC_KEY_REGEX = r"^[1-9A-HJ-NP-Za-km-z]{32,44}$"

class AddressRequest(BaseModel):
    """Model for requests that include a Solana address."""
    
    address: constr(regex=SOLANA_PUBLIC_KEY_REGEX) = Field(
        ...,
        description="A Solana account address"
    )

class TokenRequest(BaseModel):
    """Model for requests that include a token mint address."""
    
    mint: constr(regex=SOLANA_PUBLIC_KEY_REGEX) = Field(
        ...,
        description="A Solana token mint address"
    )

class TransactionHistoryRequest(BaseModel):
    """Model for transaction history requests."""
    
    address: constr(regex=SOLANA_PUBLIC_KEY_REGEX) = Field(
        ...,
        description="The Solana account address"
    )
    limit: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of transactions to return"
    )
    before: Optional[str] = Field(
        None,
        description="Transaction signature to search backwards from"
    )
    search: Optional[str] = Field(
        None,
        description="Optional semantic search query"
    )

class TokenHoldersRequest(BaseModel):
    """Model for token holders requests."""
    
    mint: constr(regex=SOLANA_PUBLIC_KEY_REGEX) = Field(
        ...,
        description="A Solana token mint address"
    )
    limit: Optional[int] = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of holders to return"
    )
    min_balance: Optional[float] = Field(
        None,
        ge=0,
        description="Minimum token balance to include"
    )

class NaturalLanguageQueryRequest(BaseModel):
    """Model for natural language query requests."""
    
    query: str = Field(
        ...,
        min_length=1,
        description="The natural language query string"
    )
    format_level: Optional[str] = Field(
        "auto",
        description="Response format level (minimal, standard, detailed, auto)"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for context preservation"
    )

class ChainAnalysisRequest(BaseModel):
    """Model for chain analysis requests."""
    
    type: str = Field(
        ...,
        description="Analysis type (token_flow, activity_pattern)"
    )
    address: constr(regex=SOLANA_PUBLIC_KEY_REGEX) = Field(
        ...,
        description="The Solana account address to analyze"
    ) 