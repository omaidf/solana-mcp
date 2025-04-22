"""API routes for liquidity pool analysis."""

# Standard library imports
import functools
from typing import List, Dict, Any, Optional

# Third-party library imports
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Path, Body

# Internal imports
from solana_mcp.liquidity_analyzer import LiquidityAnalyzer
from solana_mcp.solana_client import get_solana_client, InvalidPublicKeyError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.api_routes.token_analysis import handle_token_exceptions

# Set up logging
logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/liquidity-analysis",
    tags=["liquidity analysis"],
)


@router.get("/pool/{pool_address}")
@handle_token_exceptions
async def analyze_pool(
    request: Request,
    pool_address: str = Path(..., description="The liquidity pool account address"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a specific liquidity pool.
    
    Args:
        request: FastAPI request object
        pool_address: The pool account address
        request_id: Optional request ID for tracing
        
    Returns:
        Pool analysis data
    """
    log_with_context(
        logger,
        "info",
        f"Pool analysis requested for: {pool_address}",
        request_id=request_id,
        pool_address=pool_address
    )
    
    async with get_solana_client() as client:
        analyzer = LiquidityAnalyzer(client)
        result = await analyzer.analyze_pool(pool_address, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Pool analysis completed for: {pool_address}",
            request_id=request_id,
            pool_address=pool_address
        )
        
        return result


@router.get("/user-positions/{wallet_address}")
@handle_token_exceptions
async def get_user_positions(
    request: Request,
    wallet_address: str = Path(..., description="The user's wallet address"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get liquidity positions for a user.
    
    Args:
        request: FastAPI request object
        wallet_address: The user's wallet address
        request_id: Optional request ID for tracing
        
    Returns:
        User liquidity positions
    """
    log_with_context(
        logger,
        "info",
        f"Liquidity positions requested for: {wallet_address}",
        request_id=request_id,
        wallet_address=wallet_address
    )
    
    async with get_solana_client() as client:
        analyzer = LiquidityAnalyzer(client)
        result = await analyzer.get_user_positions(wallet_address, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"User positions fetched for: {wallet_address}",
            request_id=request_id,
            wallet_address=wallet_address,
            position_count=result.get("position_count", 0)
        )
        
        return result


@router.get("/top-pools")
@handle_token_exceptions
async def get_top_pools(
    request: Request,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of pools to return"),
    protocol: Optional[str] = Query(None, description="Filter by protocol (e.g., 'raydium', 'orca')"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get top liquidity pools by TVL.
    
    Args:
        request: FastAPI request object
        limit: Maximum number of pools to return
        protocol: Optional filter by protocol
        request_id: Optional request ID for tracing
        
    Returns:
        Top liquidity pools
    """
    log_with_context(
        logger,
        "info",
        f"Top pools requested (limit: {limit}, protocol: {protocol or 'all'})",
        request_id=request_id,
        limit=limit,
        protocol=protocol
    )
    
    async with get_solana_client() as client:
        analyzer = LiquidityAnalyzer(client)
        result = await analyzer.get_top_pools(limit=limit, protocol=protocol, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Top pools fetched (count: {len(result.get('pools', []))})",
            request_id=request_id,
            pool_count=len(result.get("pools", []))
        )
        
        return result


@router.post("/impermanent-loss")
@handle_token_exceptions
async def calculate_impermanent_loss(
    request: Request,
    token_a_price_change: float = Body(..., ge=0.01, description="Price change ratio for token A (1.0 = no change)"),
    token_b_price_change: float = Body(..., ge=0.01, description="Price change ratio for token B (1.0 = no change)"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate impermanent loss for given price changes.
    
    Args:
        request: FastAPI request object
        token_a_price_change: Price change ratio for token A (1.0 = no change)
        token_b_price_change: Price change ratio for token B (1.0 = no change)
        request_id: Optional request ID for tracing
        
    Returns:
        Impermanent loss calculation
    """
    log_with_context(
        logger,
        "info",
        f"Impermanent loss calculation requested",
        request_id=request_id,
        token_a_price_change=token_a_price_change,
        token_b_price_change=token_b_price_change
    )
    
    async with get_solana_client() as client:
        analyzer = LiquidityAnalyzer(client)
        result = await analyzer.calculate_impermanent_loss(
            token_a_price_change=token_a_price_change,
            token_b_price_change=token_b_price_change,
            request_id=request_id
        )
        
        log_with_context(
            logger,
            "info",
            f"Impermanent loss calculated: {result.get('percentage_loss', 0):.2f}%",
            request_id=request_id,
            percentage_loss=result.get("percentage_loss", 0)
        )
        
        return result 