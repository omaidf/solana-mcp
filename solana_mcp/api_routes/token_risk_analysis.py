"""API routes for token risk analysis."""

# Standard library imports
import functools
from typing import List, Dict, Any, Optional

# Third-party library imports
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Path

# Internal imports
from solana_mcp.token_risk_analyzer import TokenRiskAnalyzer
from solana_mcp.solana_client import get_solana_client, InvalidPublicKeyError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.api_routes.token_analysis import handle_token_exceptions

# Set up logging
logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/token-risk",
    tags=["token risk analysis"],
)


@router.get("/analyze/{mint}")
@handle_token_exceptions
async def analyze_token_risks(
    request: Request,
    mint: str = Path(..., description="The token mint address"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze the risks associated with a token.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token risk analysis data
    """
    log_with_context(
        logger,
        "info",
        f"Token risk analysis requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenRiskAnalyzer(client)
        result = await analyzer.analyze_token_risks(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token risk analysis completed for: {mint}",
            request_id=request_id,
            mint=mint,
            risk_level=result.get("risk_level", "Unknown")
        )
        
        return result


@router.get("/liquidity-locks/{mint}")
@handle_token_exceptions
async def get_liquidity_locks(
    request: Request,
    mint: str = Path(..., description="The token mint address"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get detailed information about liquidity locks for a token.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Liquidity lock information
    """
    log_with_context(
        logger,
        "info",
        f"Liquidity lock info requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenRiskAnalyzer(client)
        # First do complete risk analysis
        full_analysis = await analyzer.analyze_token_risks(mint, request_id=request_id)
        
        # Extract the liquidity information
        liquidity_analysis = full_analysis.get("liquidity_analysis", {})
        
        # Create focused response
        result = {
            "token_mint": mint,
            "token_name": full_analysis.get("name", "Unknown"),
            "token_symbol": full_analysis.get("symbol", "UNKNOWN"),
            "total_liquidity_usd": liquidity_analysis.get("total_liquidity_usd", 0),
            "has_locked_liquidity": liquidity_analysis.get("has_locked_liquidity", False),
            "lock_details": liquidity_analysis.get("lock_details", []),
            "liquidity_risk_score": full_analysis.get("liquidity_risk_score", 0),
            "largest_pool": liquidity_analysis.get("largest_pool", None),
            "liquidity_to_mcap_ratio": liquidity_analysis.get("liquidity_to_mcap_ratio", 0),
            "last_updated": full_analysis.get("last_updated", "")
        }
        
        log_with_context(
            logger,
            "info",
            f"Liquidity lock info completed for: {mint}",
            request_id=request_id,
            mint=mint,
            has_locked_liquidity=result.get("has_locked_liquidity", False)
        )
        
        return result


@router.get("/tokenomics/{mint}")
@handle_token_exceptions
async def get_tokenomics(
    request: Request,
    mint: str = Path(..., description="The token mint address"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get tokenomics information for a token.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Tokenomics information
    """
    log_with_context(
        logger,
        "info",
        f"Tokenomics requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenRiskAnalyzer(client)
        # First do complete risk analysis
        full_analysis = await analyzer.analyze_token_risks(mint, request_id=request_id)
        
        # Get token supply info
        supply_info = await client.get_token_supply(mint)
        
        # Create focused tokenomics response
        result = {
            "token_mint": mint,
            "token_name": full_analysis.get("name", "Unknown"),
            "token_symbol": full_analysis.get("symbol", "UNKNOWN"),
            "supply": {
                "total_supply": float(supply_info.get("value", {}).get("uiAmountString", "0")),
                "decimals": supply_info.get("value", {}).get("decimals", 0),
                "supply_risk_score": full_analysis.get("supply_risk_score", 0)
            },
            "authorities": {
                "has_mint_authority": full_analysis.get("creation_analysis", {}).get("has_mint_authority", False),
                "has_freeze_authority": full_analysis.get("creation_analysis", {}).get("has_freeze_authority", False),
                "authority_risk_score": full_analysis.get("authority_risk_score", 0)
            },
            "creation": full_analysis.get("creation_analysis", {}).get("creation_info", {}),
            "ownership": {
                "total_holders": full_analysis.get("holder_analysis", {}).get("total_holders", 0),
                "top_holder_percentage": full_analysis.get("holder_analysis", {}).get("top_holder_percentage", 0),
                "top_10_percentage": full_analysis.get("holder_analysis", {}).get("top_10_percentage", 0),
                "concentration_index": full_analysis.get("holder_analysis", {}).get("concentration_index", 0),
                "ownership_risk_score": full_analysis.get("ownership_risk_score", 0)
            },
            "liquidity": {
                "total_liquidity_usd": full_analysis.get("liquidity_analysis", {}).get("total_liquidity_usd", 0),
                "liquidity_to_mcap_ratio": full_analysis.get("liquidity_analysis", {}).get("liquidity_to_mcap_ratio", 0),
                "has_locked_liquidity": full_analysis.get("liquidity_analysis", {}).get("has_locked_liquidity", False),
                "liquidity_risk_score": full_analysis.get("liquidity_risk_score", 0)
            },
            "last_updated": full_analysis.get("last_updated", "")
        }
        
        log_with_context(
            logger,
            "info",
            f"Tokenomics completed for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        return result


@router.get("/meme-category/{mint}")
@handle_token_exceptions
async def get_token_category(
    request: Request,
    mint: str = Path(..., description="The token mint address"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get the meme category for a token.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token category information
    """
    log_with_context(
        logger,
        "info",
        f"Token category requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        # Get token metadata
        token_metadata = await client.get_token_metadata(mint)
        name = token_metadata.get("name", "Unknown")
        symbol = token_metadata.get("symbol", "UNKNOWN")
        
        # Use the risk analyzer just for categorization
        analyzer = TokenRiskAnalyzer(client)
        category = analyzer._categorize_token(name, symbol)
        
        # Create response
        result = {
            "token_mint": mint,
            "token_name": name,
            "token_symbol": symbol,
            "category": category,
            "is_meme_token": category in ["Animal", "Food", "Meme"],
            "last_updated": token_metadata.get("last_updated", "")
        }
        
        log_with_context(
            logger,
            "info",
            f"Token category completed for: {mint}, category: {category}",
            request_id=request_id,
            mint=mint,
            category=category
        )
        
        return result


@router.get("/meme-tokens")
@handle_token_exceptions
async def get_meme_tokens(
    request: Request,
    category: Optional[str] = Query(None, description="Filter by token category (Animal, Food, Tech, Meme, Other)"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of tokens to return"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get a list of meme tokens.
    
    Args:
        request: FastAPI request object
        category: Optional category filter
        limit: Maximum number of tokens to return
        request_id: Optional request ID for tracing
        
    Returns:
        List of meme tokens
    """
    log_with_context(
        logger,
        "info",
        f"Meme tokens requested (category: {category or 'all'}, limit: {limit})",
        request_id=request_id,
        category=category,
        limit=limit
    )
    
    # This would be a real implementation with a token database
    # For now, we'll use hardcoded examples
    
    meme_tokens = [
        {
            "mint": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
            "name": "Bonk",
            "symbol": "BONK",
            "category": "Animal",
            "price_usd": 0.000012,
            "market_cap": 750000000,
            "holders": 125000
        },
        {
            "mint": "5tN42n9vMi6ubp67Uy4NnmM5DMZYN8aS8GeB3bEDHr6E",
            "name": "Popcat",
            "symbol": "POPCAT",
            "category": "Animal",
            "price_usd": 0.000003,
            "market_cap": 3000000,
            "holders": 12000
        },
        {
            "mint": "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",
            "name": "Bork",
            "symbol": "BORK",
            "category": "Animal",
            "price_usd": 0.0000025,
            "market_cap": 2500000,
            "holders": 8500
        },
        {
            "mint": "9nEqaUcb16sQ3Tn1psbkWqyhPdLmfHWjKGymREjsAgTE",
            "name": "Mochi",
            "symbol": "MOCHI",
            "category": "Food",
            "price_usd": 0.0000015,
            "market_cap": 1500000,
            "holders": 4200
        },
        {
            "mint": "E1zCt2RzV4qQvXi6XWqQXDKGh7NLwd4nXyfYYKkUXYdE",
            "name": "Wojak",
            "symbol": "WOJAK",
            "category": "Meme",
            "price_usd": 0.000008,
            "market_cap": 8000000,
            "holders": 15000
        },
        {
            "mint": "F4vMh8WONrUKaP3XrivHQT3MQ3F5EMqSYNL5JGzV4q9T",
            "name": "Solana Pizza",
            "symbol": "PIZZA",
            "category": "Food",
            "price_usd": 0.000005,
            "market_cap": 5000000,
            "holders": 9800
        },
        {
            "mint": "7HCXYsQJ1J2rNhz87WQvL1NWz5RZ5N5eLrKoYxL7nkLj",
            "name": "Moonbot",
            "symbol": "MBOT",
            "category": "Tech",
            "price_usd": 0.00019,
            "market_cap": 19000000,
            "holders": 22000
        },
        {
            "mint": "9nt2QCrZbJJnbHWQQe2uKxR1BKMBpGfhNKgHYngKA46W",
            "name": "Crypto Ape",
            "symbol": "CAPE",
            "category": "Animal",
            "price_usd": 0.000045,
            "market_cap": 4500000,
            "holders": 7300
        },
        {
            "mint": "6c4L6fGYDbM5wJdSaeFH2edNdmXnvM6ohnNL2yjrTXEM",
            "name": "SolDoge",
            "symbol": "SOGE",
            "category": "Animal",
            "price_usd": 0.000018,
            "market_cap": 1800000,
            "holders": 5400
        },
        {
            "mint": "8RJgU7hzRbS2kCL98PETR2da121eTvBhZpz3GHLTcRST",
            "name": "Solana Based",
            "symbol": "BASED",
            "category": "Meme",
            "price_usd": 0.0000075,
            "market_cap": 750000,
            "holders": 2100
        }
    ]
    
    # Filter by category if specified
    if category:
        filtered_tokens = [token for token in meme_tokens if token["category"] == category]
    else:
        filtered_tokens = meme_tokens
    
    # Sort by market cap (descending) and limit
    sorted_tokens = sorted(filtered_tokens, key=lambda x: x["market_cap"], reverse=True)[:limit]
    
    result = {
        "tokens": sorted_tokens,
        "total_count": len(filtered_tokens),
        "category": category or "all",
        "limit": limit
    }
    
    log_with_context(
        logger,
        "info",
        f"Returning {len(sorted_tokens)} meme tokens",
        request_id=request_id,
        count=len(sorted_tokens)
    )
    
    return result 