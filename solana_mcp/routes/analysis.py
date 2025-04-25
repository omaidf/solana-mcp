"""Analysis-related API routes.

This module defines routes for analyzing Solana blockchain data.
"""

from fastapi import APIRouter, Depends, Path, Query
from typing import Optional

from solana_mcp.services.analysis_service import AnalysisService
from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.decorators import handle_api_errors
from solana_mcp.utils.responses import ApiResponse

# Create router
router = APIRouter(tags=["analysis"])

# Get the dependencies from the main app
async def get_solana_client() -> SolanaClient:
    """Dependency to get the Solana client."""
    from app import get_solana_client_from_app
    return await get_solana_client_from_app()

async def get_cache_service() -> CacheService:
    """Dependency to get the cache service."""
    from app import get_cache_service_from_app
    return await get_cache_service_from_app()

async def get_transaction_service(
    solana_client: SolanaClient = Depends(get_solana_client),
    cache_service: CacheService = Depends(get_cache_service)
) -> TransactionService:
    """Dependency to get a transaction service."""
    return TransactionService(solana_client, cache_service)

async def get_token_service(
    solana_client: SolanaClient = Depends(get_solana_client),
    cache_service: CacheService = Depends(get_cache_service)
) -> TokenService:
    """Dependency to get a token service."""
    return TokenService(solana_client, cache_service)

async def get_analysis_service(
    solana_client: SolanaClient = Depends(get_solana_client),
    transaction_service: TransactionService = Depends(get_transaction_service),
    token_service: TokenService = Depends(get_token_service),
    cache_service: CacheService = Depends(get_cache_service)
) -> AnalysisService:
    """Dependency to get an analysis service."""
    return AnalysisService(solana_client, transaction_service, token_service, cache_service)

@router.get(
    "/analysis/token-flow/{address}",
    response_model=ApiResponse[dict],
    summary="Analyze token flow",
    description="Analyzes token flow in and out of a Solana account."
)
@handle_api_errors
async def analyze_token_flow(
    address: str = Path(..., description="Account address"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of transactions to analyze"),
    days: Optional[int] = Query(None, description="Number of days to analyze"),
    service: AnalysisService = Depends(get_analysis_service)
) -> ApiResponse[dict]:
    """Analyze token flow for an account.
    
    Args:
        address: The account address
        limit: Maximum number of transactions to analyze
        days: Optional number of days to analyze
        service: The analysis service
        
    Returns:
        Token flow analysis
    """
    result = await service.analyze_token_flow(address, limit, days)
    return ApiResponse.success(result)

@router.get(
    "/analysis/activity-pattern/{address}",
    response_model=ApiResponse[dict],
    summary="Analyze activity pattern",
    description="Analyzes activity patterns for a Solana account."
)
@handle_api_errors
async def analyze_activity_pattern(
    address: str = Path(..., description="Account address"),
    limit: int = Query(200, ge=1, le=1000, description="Maximum number of transactions to analyze"),
    service: AnalysisService = Depends(get_analysis_service)
) -> ApiResponse[dict]:
    """Analyze activity patterns for an account.
    
    Args:
        address: The account address
        limit: Maximum number of transactions to analyze
        service: The analysis service
        
    Returns:
        Activity pattern analysis
    """
    result = await service.analyze_activity_pattern(address, limit)
    return ApiResponse.success(result)

@router.get(
    "/analysis/wallet-profile/{address}",
    response_model=ApiResponse[dict],
    summary="Generate wallet profile",
    description="Generates a profile for a Solana wallet based on its activity."
)
@handle_api_errors
async def get_wallet_profile(
    address: str = Path(..., description="Account address"),
    service: AnalysisService = Depends(get_analysis_service)
) -> ApiResponse[dict]:
    """Generate a profile for a wallet.
    
    Args:
        address: The account address
        service: The analysis service
        
    Returns:
        Wallet profile
    """
    profile = await service.wallet_profile(address)
    return ApiResponse.success(profile) 