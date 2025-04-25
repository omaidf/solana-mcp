"""NLP routes for the Solana MCP API.

This module handles natural language processing endpoints for the Solana MCP API.
"""

from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, Body, Query
from pydantic import BaseModel, Field

from solana_mcp.models.api import ApiResponse
from solana_mcp.services.nlp_service import NLPService
from solana_mcp.services.account_service import AccountService
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.analysis_service import AnalysisService
from solana_mcp.services.market_service import MarketService
from solana_mcp.services.cache_service import CacheService
# Update to use the new client structure
from solana_mcp.clients import SolanaClient
from solana_mcp.utils.api import handle_api_errors

# Import dependency providers
from solana_mcp.dependencies import (
    get_solana_client,
    get_cache_service,
    get_account_service,
    get_token_service,
    get_transaction_service,
    get_analysis_service,
    get_market_service
)

# Models
class NLPQuery(BaseModel):
    """Natural language query request model."""
    
    query: str = Field(..., description="The natural language query to process")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    format_level: str = Field("auto", description="Response format detail level (minimal, standard, detailed, auto)")


# Router
nlp_router = APIRouter(prefix="/nlp", tags=["nlp"])


# Dependency for NLP service
async def get_nlp_service(
    solana_client: SolanaClient = Depends(get_solana_client),
    account_service: AccountService = Depends(get_account_service),
    token_service: TokenService = Depends(get_token_service),
    transaction_service: TransactionService = Depends(get_transaction_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    market_service: MarketService = Depends(get_market_service),
    cache_service: Optional[CacheService] = Depends(get_cache_service)
) -> NLPService:
    """Get NLPService instance with injected dependencies.
    
    Args:
        solana_client: The Solana client
        account_service: Service for account operations
        token_service: Service for token operations
        transaction_service: Service for transaction operations
        analysis_service: Service for analysis operations
        market_service: Service for market data operations
        cache_service: Optional cache service
        
    Returns:
        NLPService instance
    """
    return NLPService(
        solana_client=solana_client,
        account_service=account_service,
        token_service=token_service,
        transaction_service=transaction_service,
        analysis_service=analysis_service,
        market_service=market_service,
        cache_service=cache_service
    )


@nlp_router.post("/process", response_model=ApiResponse[Dict[str, Any]])
@handle_api_errors
async def process_query(
    query_data: NLPQuery = Body(...),
    nlp_service: NLPService = Depends(get_nlp_service)
) -> ApiResponse[Dict[str, Any]]:
    """Process a natural language query.
    
    Process free-form text queries about Solana blockchain data.
    The API interprets the query intent and retrieves relevant information.
    
    Examples:
    - "What's the balance of address 4Zc...U9m?"
    - "Show me recent transactions for 4Zc...U9m"
    - "What's the price of SOL token?"
    
    Args:
        query_data: The query data containing the natural language query
        nlp_service: The NLP service
        
    Returns:
        API response with the query result
    """
    result = await nlp_service.process_query(
        query=query_data.query,
        session_id=query_data.session_id,
        format_level=query_data.format_level
    )
    
    return ApiResponse(
        success=True,
        data=result,
        message="Query processed successfully"
    )


@nlp_router.get("/suggest", response_model=ApiResponse[List[str]])
@handle_api_errors
async def get_suggestions(
    input: str = Query(..., description="User input text to generate suggestions from"),
    limit: int = Query(3, description="Maximum number of suggestions to return"),
    nlp_service: NLPService = Depends(get_nlp_service)
) -> ApiResponse[List[str]]:
    """Get query suggestions based on user input.
    
    Generate relevant query suggestions based on provided text input.
    
    Args:
        input: User input text
        limit: Maximum number of suggestions
        nlp_service: The NLP service
        
    Returns:
        API response with suggested queries
    """
    suggestions = await nlp_service.suggest_queries(input, limit)
    
    return ApiResponse(
        success=True,
        data=suggestions,
        message=f"Generated {len(suggestions)} query suggestions"
    ) 