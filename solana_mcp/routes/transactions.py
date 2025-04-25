"""Transaction-related API routes.

This module defines routes for working with Solana blockchain transactions.
"""

from fastapi import APIRouter, Depends, Path, Query
from typing import Optional

from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.decorators import handle_api_errors
from solana_mcp.utils.responses import ApiResponse
from solana_mcp.models.responses import TransactionResponse, TransactionHistoryResponse

# Create router
router = APIRouter(tags=["transactions"])

# Get the Solana client dependency from the main app
async def get_solana_client() -> SolanaClient:
    """Dependency to get the Solana client."""
    from app import get_solana_client_from_app
    return await get_solana_client_from_app()

# Get the Cache service dependency from the main app
async def get_cache_service() -> CacheService:
    """Dependency to get the cache service."""
    from app import get_cache_service_from_app
    return await get_cache_service_from_app()

async def get_transaction_service(
    solana_client: SolanaClient = Depends(get_solana_client),
    cache_service: CacheService = Depends(get_cache_service)
) -> TransactionService:
    """Dependency to get a transaction service.
    
    Args:
        solana_client: The Solana client
        cache_service: The cache service
    
    Returns:
        A TransactionService instance
    """
    return TransactionService(solana_client, cache_service)

@router.get(
    "/transaction/{signature}",
    response_model=ApiResponse[dict],
    summary="Get transaction details",
    description="Retrieves detailed information about a Solana transaction by its signature."
)
@handle_api_errors
async def get_transaction(
    signature: str = Path(..., description="Transaction signature"),
    service: TransactionService = Depends(get_transaction_service)
) -> ApiResponse[dict]:
    """Get details of a Solana transaction.
    
    Args:
        signature: The transaction signature
        service: The transaction service
        
    Returns:
        Transaction details
    """
    transaction = await service.get_transaction(signature)
    
    if not transaction:
        return ApiResponse.error(
            f"Transaction with signature {signature} not found",
            error_code="TRANSACTION_NOT_FOUND"
        )
    
    # Parse the transaction for better readability
    parsed_transaction = await service.parse_transaction(transaction)
    
    return ApiResponse.success(parsed_transaction)

@router.get(
    "/account/{address}/transactions",
    response_model=ApiResponse[TransactionHistoryResponse],
    summary="Get transaction history",
    description="Retrieves transaction history for a Solana account with detailed information."
)
@handle_api_errors
async def get_transaction_history(
    address: str = Path(..., description="Account address"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of transactions"),
    before: Optional[str] = Query(None, description="Signature to search backwards from"),
    until: Optional[str] = Query(None, description="Signature to search until"),
    include_details: bool = Query(True, description="Include detailed transaction information"),
    service: TransactionService = Depends(get_transaction_service)
) -> ApiResponse[TransactionHistoryResponse]:
    """Get transaction history for an account.
    
    Args:
        address: The account address
        limit: Maximum number of transactions to return
        before: Signature to search backwards from
        until: Signature to search until (inclusive)
        include_details: Whether to include detailed transaction data
        service: The transaction service
        
    Returns:
        Transaction history
    """
    history = await service.get_transactions_for_address(
        address, 
        limit, 
        before, 
        until, 
        parsed_details=include_details
    )
    
    return ApiResponse.success(history)

@router.get(
    "/transactions/recent",
    response_model=ApiResponse[list],
    summary="Get recent transactions",
    description="Retrieves recent transactions from the Solana blockchain."
)
@handle_api_errors
async def get_recent_transactions(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of transactions"),
    service: TransactionService = Depends(get_transaction_service)
) -> ApiResponse[list]:
    """Get recent transactions from the blockchain.
    
    Args:
        limit: Maximum number of transactions to return
        service: The transaction service
        
    Returns:
        List of recent transactions
    """
    transactions = await service.get_recent_transactions(limit)
    
    # Include parsed transaction data
    parsed_transactions = []
    for tx in transactions:
        parsed_tx = await service.parse_transaction(tx)
        parsed_transactions.append(parsed_tx)
    
    return ApiResponse.success(parsed_transactions) 