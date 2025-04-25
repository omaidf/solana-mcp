"""Account-related API routes.

This module defines routes for working with Solana accounts.
"""

from fastapi import APIRouter, Depends, Path, Query
from typing import Optional

from solana_mcp.services.account_service import AccountService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.decorators import handle_api_errors
from solana_mcp.utils.responses import ApiResponse
from solana_mcp.models.responses import AccountBalanceResponse, TransactionHistoryResponse

# Create router
router = APIRouter(tags=["accounts"])

# Get the Solana client dependency from the main app
async def get_solana_client() -> SolanaClient:
    """Dependency to get the Solana client.
    
    This is replaced during application startup with the actual dependency.
    """
    # This function is replaced by the application's dependency
    # The implementation here is just a placeholder
    from app import get_solana_client_from_app
    return await get_solana_client_from_app()

async def get_account_service(solana_client: SolanaClient = Depends(get_solana_client)) -> AccountService:
    """Dependency to get an account service.
    
    Args:
        solana_client: The Solana client
    
    Returns:
        An AccountService instance
    """
    return AccountService(solana_client)

@router.get(
    "/account/{address}/balance",
    response_model=ApiResponse[AccountBalanceResponse],
    summary="Get account balance",
    description="Retrieves the balance of a Solana account in SOL."
)
@handle_api_errors
async def get_account_balance(
    address: str = Path(..., description="Account address"),
    service: AccountService = Depends(get_account_service)
) -> ApiResponse[AccountBalanceResponse]:
    """Get the balance of a Solana account.
    
    Args:
        address: The account address
        service: The account service
        
    Returns:
        Account balance information
    """
    balance = await service.get_account_balance(address)
    return ApiResponse.success(balance)

@router.get(
    "/account/{address}/info",
    response_model=ApiResponse[dict],
    summary="Get account information",
    description="Retrieves detailed information about a Solana account."
)
@handle_api_errors
async def get_account_info(
    address: str = Path(..., description="Account address"),
    encoding: str = Query("jsonParsed", description="Data encoding format"),
    service: AccountService = Depends(get_account_service)
) -> ApiResponse[dict]:
    """Get information about a Solana account.
    
    Args:
        address: The account address
        encoding: The encoding to use for the account data
        service: The account service
        
    Returns:
        Account information
    """
    info = await service.get_account_info(address, encoding)
    return ApiResponse.success(info)

@router.get(
    "/account/{address}/transactions",
    response_model=ApiResponse[TransactionHistoryResponse],
    summary="Get transaction history",
    description="Retrieves transaction history for a Solana account."
)
@handle_api_errors
async def get_transaction_history(
    address: str = Path(..., description="Account address"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of transactions"),
    before: Optional[str] = Query(None, description="Signature to search backwards from"),
    until: Optional[str] = Query(None, description="Signature to search until"),
    service: AccountService = Depends(get_account_service)
) -> ApiResponse[TransactionHistoryResponse]:
    """Get transaction history for an account.
    
    Args:
        address: The account address
        limit: Maximum number of transactions to return
        before: Signature to search backwards from
        until: Signature to search until (inclusive)
        service: The account service
        
    Returns:
        Transaction history
    """
    history = await service.get_transaction_history(address, limit, before, until)
    return ApiResponse.success(history) 