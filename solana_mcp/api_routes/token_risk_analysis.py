"""API routes for token risk analysis."""

# Standard library imports
import functools
from typing import List, Dict, Any, Optional
import base64

# Third-party library imports
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Path

# Internal imports
from solana_mcp.token_risk_analyzer import TokenRiskAnalyzer
from solana_mcp.solana_client import get_solana_client, InvalidPublicKeyError, PublicKey
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
    min_holders: int = Query(0, ge=0, description="Minimum number of holders"),
    min_liquidity: float = Query(0, ge=0, description="Minimum liquidity in USD"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get a list of meme tokens.
    
    Args:
        request: FastAPI request object
        category: Optional category filter
        limit: Maximum number of tokens to return
        min_holders: Minimum number of holders
        min_liquidity: Minimum liquidity in USD
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
    
    async with get_solana_client() as client:
        analyzer = TokenRiskAnalyzer(client)
        
        # Get popular token mints dynamically
        popular_token_mints = []
        
        # Try to get popular tokens from client
        try:
            # Get top tokens by volume, market cap, and social activity
            popular_tokens = await client.get_popular_tokens(limit=30)
            popular_token_mints = [token["mint"] for token in popular_tokens if "mint" in token]
            
            logger.info(f"Retrieved {len(popular_token_mints)} popular tokens from client API")
        except Exception as e:
            logger.warning(f"Error retrieving popular tokens: {str(e)}")
            
            # Fallback list only if client API fails - using well-known tokens
            # This is just a safety measure and should rarely be used
            popular_token_mints = [
                "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",  # BONK
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
                "5tN42n9vMi6ubp67Uy4NnmM5DMZYN8aS8GeB3bEDHr6E",  # POPCAT
                "KiTASSNKG8KQhTj2Go5kJH6yw53poXz8NFMHJmFTGLn",   # DOGE
                "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",  # BORK
                "9nEqaUcb16sQ3Tn1psbkWqyhPdLmfHWjKGymREjsAgTE",  # MOCHI
                "E1zCt2RzV4qQvXi6XWqQXDKGh7NLwd4nXyfYYKkUXYdE",  # WOJAK
                "F4vMh8WONrUKaP3XrivHQT3MQ3F5EMqSYNL5JGzV4q9T",  # PIZZA
                "dogSmMYV9G4Qyd22LkuRKLnuZVBJiKNQw7qgRWfSoZg"    # WIF
            ]
            logger.warning("Using fallback list of popular tokens")
        
        # Analyze each token to collect information
        tokens_data = []
        analyzed_count = 0
        
        for mint in popular_token_mints:
            # Limit the number of tokens to analyze to avoid long processing time
            if analyzed_count >= min(30, limit * 2):
                break
                
            try:
                # Get basic token data
                token_metadata = await client.get_token_metadata(mint)
                token_name = token_metadata.get("metadata", {}).get("name", "Unknown")
                token_symbol = token_metadata.get("metadata", {}).get("symbol", "UNKNOWN")
                
                # Skip tokens with no name or symbol
                if token_name == "Unknown" or token_symbol == "UNKNOWN":
                    continue
                
                # Determine token category
                token_category = analyzer._categorize_token(token_name, token_symbol)
                
                # Skip non-meme tokens if category specified
                if category and token_category != category:
                    continue
                
                # Get token market data
                market_data = await client.get_market_price(mint)
                price_usd = market_data.get("price_data", {}).get("price_usd", 0)
                liquidity = market_data.get("price_data", {}).get("liquidity", {}).get("total_usd", 0)
                
                # Skip tokens with too little liquidity
                if liquidity < min_liquidity:
                    continue
                
                # Get token supply
                supply_info = await client.get_token_supply(mint)
                total_supply = float(supply_info.get("value", {}).get("uiAmountString", "0"))
                
                # Calculate market cap
                market_cap = price_usd * total_supply
                
                # Get holder data (simplified)
                holder_data = {"total_holders": 0}
                try:
                    token_accounts = await client.get_token_largest_accounts(mint)
                    if "value" in token_accounts:
                        holder_data["total_holders"] = len(token_accounts["value"])
                except Exception as e:
                    logger.warning(f"Error getting token holders for {mint}: {str(e)}")
                
                # Skip tokens with too few holders
                if holder_data["total_holders"] < min_holders:
                    continue
                
                # Add token to results
                tokens_data.append({
                    "mint": mint,
                    "name": token_name,
                    "symbol": token_symbol,
                    "category": token_category,
                    "price_usd": price_usd,
                    "market_cap": market_cap,
                    "liquidity_usd": liquidity,
                    "holders": holder_data["total_holders"]
                })
                
                analyzed_count += 1
                
            except Exception as e:
                logger.warning(f"Error analyzing token {mint}: {str(e)}")
        
        # If we couldn't find enough tokens dynamically, use a more dynamic query approach
        if not tokens_data:
            logger.warning("Could not find tokens dynamically, using backup method")
            try:
                # Query top tokens by market cap from client
                top_tokens = await client.get_top_tokens(limit=limit)
                
                for token in top_tokens:
                    mint = token.get("mint")
                    if not mint:
                        continue
                        
                    # Get basic token data
                    token_metadata = await client.get_token_metadata(mint)
                    token_name = token_metadata.get("metadata", {}).get("name", "Unknown")
                    token_symbol = token_metadata.get("metadata", {}).get("symbol", "UNKNOWN")
                    
                    # Determine token category
                    token_category = analyzer._categorize_token(token_name, token_symbol)
                    
                    # Skip non-meme tokens if category specified
                    if category and token_category != category:
                        continue
                    
                    tokens_data.append({
                        "mint": mint,
                        "name": token_name,
                        "symbol": token_symbol,
                        "category": token_category,
                        "price_usd": token.get("price_usd", 0),
                        "market_cap": token.get("market_cap", 0),
                        "liquidity_usd": token.get("liquidity_usd", 0),
                        "holders": token.get("holders", 0)
                    })
            except Exception as e:
                logger.error(f"Error querying top tokens: {str(e)}")
                # As a last resort, return an empty list rather than hardcoded data
                tokens_data = []
        
        # Filter by category if specified
        if category:
            filtered_tokens = [token for token in tokens_data if token["category"] == category]
        else:
            filtered_tokens = tokens_data
        
        # Sort by market cap (descending) and limit
        sorted_tokens = sorted(filtered_tokens, key=lambda x: x.get("market_cap", 0), reverse=True)[:limit]
        
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