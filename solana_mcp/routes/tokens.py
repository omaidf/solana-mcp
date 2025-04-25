"""
Token-related API routes for the Solana MCP API.

This module defines the routes for token data, including token metadata, 
account information, and pricing.
"""

from typing import List, Dict, Any

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field

from solana_mcp.services.token_service import TokenService
from solana_mcp.models.token import TokenList
from solana_mcp.utils.api_response import ApiResponse, handle_api_errors
from solana_mcp.utils.dependencies import get_token_service


# Create router for token endpoints
router = APIRouter(prefix="/tokens", tags=["tokens"])


@router.get("", response_model=ApiResponse[TokenList])
@handle_api_errors
async def list_tokens(
    limit: int = Query(20, ge=1, le=100, description="Number of tokens to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    token_service: TokenService = Depends(get_token_service)
):
    """
    List tokens with pagination support.
    
    Returns token basic info including address, name, symbol, and logo.
    """
    # In a real implementation, this would fetch from a database or other source
    tokens = await token_service.list_tokens(limit=limit, offset=offset)
    
    return ApiResponse(
        data=TokenList(tokens=tokens, total=len(tokens)),
        message="Tokens retrieved successfully",
        pagination={
            "limit": limit,
            "offset": offset,
            "total": len(tokens)  # In a real implementation, this would be the total count
        }
    )


@router.get("/{address}", response_model=ApiResponse[Dict[str, Any]])
@handle_api_errors
async def get_token_metadata(
    address: str = Path(..., description="Token mint address"),
    token_service: TokenService = Depends(get_token_service)
):
    """
    Get detailed metadata for a specific token.
    
    Returns name, symbol, description, image URL, and other metadata.
    """
    metadata = await token_service.get_token_metadata(address)
    
    return ApiResponse(
        data=metadata,
        message=f"Metadata for token {address} retrieved successfully"
    )


@router.get("/{address}/supply", response_model=ApiResponse[Dict[str, Any]])
@handle_api_errors
async def get_token_supply(
    address: str = Path(..., description="Token mint address"),
    token_service: TokenService = Depends(get_token_service)
):
    """
    Get the current supply information for a token.
    
    Returns total supply, circulating supply, and decimals.
    """
    supply_data = await token_service.get_token_supply(address)
    
    return ApiResponse(
        data=supply_data,
        message=f"Supply information for token {address} retrieved successfully"
    )


@router.get("/{address}/largest_accounts", response_model=ApiResponse[List[Dict[str, Any]]])
@handle_api_errors
async def get_token_largest_accounts(
    address: str = Path(..., description="Token mint address"),
    limit: int = Query(20, ge=1, le=100, description="Number of accounts to return"),
    token_service: TokenService = Depends(get_token_service)
):
    """
    Get the largest accounts holding a specific token.
    
    Returns a list of accounts sorted by balance in descending order.
    """
    accounts = await token_service.get_token_largest_accounts(address)
    
    # Limit the results based on the query parameter
    limited_accounts = accounts[:limit] if accounts else []
    
    return ApiResponse(
        data=limited_accounts,
        message=f"Largest accounts for token {address} retrieved successfully",
        pagination={
            "limit": limit,
            "total": len(accounts) if accounts else 0
        }
    )


@router.post("/batch", response_model=ApiResponse[Dict[str, Dict[str, Any]]])
@handle_api_errors
async def batch_get_token_metadata(
    token_addresses: List[str],
    token_service: TokenService = Depends(get_token_service)
):
    """
    Get metadata for multiple tokens in a single request.
    
    This endpoint efficiently batches requests to the RPC node.
    """
    metadata_dict = await token_service.get_multiple_token_metadata(token_addresses)
    
    return ApiResponse(
        data=metadata_dict,
        message=f"Metadata for {len(token_addresses)} tokens retrieved successfully"
    ) 