"""API routes for token analysis."""

# Standard library imports
import functools
import uuid
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime

# Third-party library imports
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Internal imports
from solana_mcp.token_analyzer import TokenAnalyzer, TokenAnalysis
from solana_mcp.solana_client import get_solana_client, SolanaClient, InvalidPublicKeyError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.config import get_server_config
from solana_mcp.semantic_analysis import TokenQueryAnalyzer

# Set up logging
logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/token-analysis",
    tags=["token analysis"],
)


def handle_token_exceptions(func: Callable) -> Callable:
    """Decorator to handle common exceptions in token analysis endpoints.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with exception handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request if present to get request_id
        request = None
        request_id = None
        
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                # Get request_id from state or generate new one
                request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
                break
                
        if not request_id:
            request_id = str(uuid.uuid4())
            
        log_with_context(
            logger,
            "debug",
            f"Executing endpoint: {func.__name__}",
            request_id=request_id,
            endpoint=func.__name__
        )
        
        try:
            # Add request_id to kwargs
            kwargs["request_id"] = request_id
            
            # Execute the endpoint function
            result = await func(*args, **kwargs)
            
            log_with_context(
                logger,
                "debug",
                f"Endpoint {func.__name__} completed successfully",
                request_id=request_id,
                endpoint=func.__name__
            )
            
            return result
        except InvalidPublicKeyError as e:
            log_with_context(
                logger,
                "error",
                f"Invalid Solana public key: {str(e)}",
                request_id=request_id,
                endpoint=func.__name__,
                error_type="InvalidPublicKeyError",
                error=str(e)
            )
            raise HTTPException(status_code=400, detail=f"Invalid Solana public key: {str(e)}")
        except NotImplementedError as e:
            log_with_context(
                logger,
                "error",
                f"Feature not implemented: {str(e)}",
                request_id=request_id,
                endpoint=func.__name__,
                error_type="NotImplementedError",
                error=str(e)
            )
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            log_with_context(
                logger,
                "error",
                f"Error in endpoint {func.__name__}: {str(e)}",
                request_id=request_id,
                endpoint=func.__name__,
                error_type=type(e).__name__,
                error=str(e)
            )
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    return wrapper


@router.get("/analyze/{mint}")
@handle_token_exceptions
async def analyze_token(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a token by its mint address.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token analysis data
    """
    log_with_context(
        logger,
        "info",
        f"Token analysis requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        analysis = await analyzer.analyze_token(mint, request_id=request_id)
        
        # Convert dataclass to dict for API response
        response = {
            "token_mint": analysis.token_mint,
            "token_name": analysis.token_name,
            "token_symbol": analysis.token_symbol,
            "decimals": analysis.decimals,
            "total_supply": analysis.total_supply,
            "circulation_supply": analysis.circulation_supply,
            "current_price_usd": analysis.current_price_usd,
            "launch_date": analysis.launch_date.isoformat() if analysis.launch_date else None,
            "age_days": analysis.age_days,
            "owner_can_mint": analysis.owner_can_mint,
            "owner_can_freeze": analysis.owner_can_freeze,
            "total_holders": analysis.total_holders,
            "largest_holder_percentage": analysis.largest_holder_percentage,
            "whale_count": analysis.whale_count,
            "whale_percentage": analysis.whale_percentage,
            "whale_holdings_percentage": analysis.whale_holdings_percentage,
            "whale_holdings_usd_total": analysis.whale_holdings_usd_total,
            "fresh_wallet_count": analysis.fresh_wallet_count,
            "fresh_wallet_percentage": analysis.fresh_wallet_percentage,
            "fresh_wallet_holdings_percentage": analysis.fresh_wallet_holdings_percentage,
            "last_updated": analysis.last_updated.isoformat()
        }
        
        log_with_context(
            logger,
            "info",
            f"Token analysis completed for: {mint}",
            request_id=request_id,
            mint=mint,
            token_name=analysis.token_name,
            token_symbol=analysis.token_symbol
        )
        
        return response


@router.get("/metadata/{mint}")
@handle_token_exceptions
async def get_token_metadata(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token metadata.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token metadata
    """
    log_with_context(
        logger,
        "info",
        f"Token metadata requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        metadata = await analyzer.get_token_metadata(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token metadata completed for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        return metadata


@router.get("/supply/{mint}")
@handle_token_exceptions
async def get_token_supply(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token supply information.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token supply data
    """
    log_with_context(
        logger,
        "info",
        f"Token supply requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        supply_info = await analyzer.get_token_supply_and_decimals(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token supply completed for: {mint}",
            request_id=request_id,
            mint=mint,
            supply=supply_info.get("value", {}).get("amount", "0") if "value" in supply_info else "N/A"
        )
        
        return supply_info


@router.get("/price/{mint}")
@handle_token_exceptions
async def get_token_price(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token price information.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token price data
    """
    log_with_context(
        logger,
        "info",
        f"Token price requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        price_data = await analyzer.get_token_price(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token price completed for: {mint}",
            request_id=request_id,
            mint=mint,
            price=price_data.get("price", "N/A")
        )
        
        return price_data


@router.get("/holders/{mint}")
@handle_token_exceptions
async def get_token_holders(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token holder information.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token holder data
    """
    log_with_context(
        logger,
        "info",
        f"Token holders requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        holder_data = await analyzer.get_token_largest_holders(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token holders completed for: {mint}",
            request_id=request_id,
            mint=mint,
            total_holders=holder_data.get("total_holders", 0)
        )
        
        return holder_data


@router.get("/age/{mint}")
@handle_token_exceptions
async def get_token_age(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token age information.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token age data
    """
    log_with_context(
        logger,
        "info",
        f"Token age requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        age_data = await analyzer.get_token_age(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token age completed for: {mint}",
            request_id=request_id,
            mint=mint,
            age_days=age_data.get("age_days", "N/A")
        )
        
        return age_data


@router.get("/authority/{mint}")
@handle_token_exceptions
async def get_token_authority(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token authority information.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Token authority data
    """
    log_with_context(
        logger,
        "info",
        f"Token authority requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        auth_data = await analyzer.get_token_mint_authority(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token authority completed for: {mint}",
            request_id=request_id,
            mint=mint,
            has_mint_authority=auth_data.get("has_mint_authority", False),
            has_freeze_authority=auth_data.get("has_freeze_authority", False)
        )
        
        return auth_data


@router.get("/holders-count/{mint}")
@handle_token_exceptions
async def get_token_holders_count(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get count of token holders.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Number of token holders
    """
    log_with_context(
        logger,
        "info",
        f"Token holders count requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        count = await analyzer.get_token_holders_count(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token holders count completed for: {mint}",
            request_id=request_id,
            mint=mint,
            count=count
        )
        
        return {"count": count}


@router.get("/whales/{mint}")
@handle_token_exceptions
async def get_whale_holders(
    request: Request,
    mint: str,
    threshold_usd: float = Query(50000.0, description="Minimum USD value to consider a holder a whale"),
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token holders with balances over the specified USD threshold (whales).
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        threshold_usd: The minimum USD value to consider a holder a whale (default: $50K)
        request_id: Optional request ID for tracing
        
    Returns:
        Whale holder data
    """
    log_with_context(
        logger,
        "info",
        f"Whale holders requested for: {mint} (threshold: ${threshold_usd})",
        request_id=request_id,
        mint=mint,
        threshold_usd=threshold_usd
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        whale_data = await analyzer.get_whale_holders(mint, threshold_usd=threshold_usd, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Whale holders completed for: {mint}",
            request_id=request_id,
            mint=mint,
            whale_count=whale_data.get("whale_count", 0),
            whale_holdings_percentage=f"{whale_data.get('whale_holdings_percentage', 0):.2f}%"
        )
        
        return whale_data


@router.get("/fresh-wallets/{mint}")
@handle_token_exceptions
async def get_fresh_wallets(
    request: Request,
    mint: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get token holders that only hold this specific token (fresh wallets).
    
    Fresh wallets are wallets that have only bought this specific token and no others,
    which could indicate artificial activity or targeted pump activity.
    
    Args:
        request: FastAPI request object
        mint: The token mint address
        request_id: Optional request ID for tracing
        
    Returns:
        Fresh wallet data
    """
    log_with_context(
        logger,
        "info",
        f"Fresh wallets requested for: {mint}",
        request_id=request_id,
        mint=mint
    )
    
    async with get_solana_client() as client:
        analyzer = TokenAnalyzer(client)
        fresh_wallet_data = await analyzer.get_fresh_wallets(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Fresh wallets completed for: {mint}",
            request_id=request_id,
            mint=mint,
            fresh_wallet_count=fresh_wallet_data.get("fresh_wallet_count", 0),
            fresh_wallet_holdings_percentage=f"{fresh_wallet_data.get('fresh_wallet_holdings_percentage', 0):.2f}%"
        )
        
        return fresh_wallet_data


# Define a model for the semantic query request
class SemanticQueryRequest(BaseModel):
    query: str

@router.post("/query")
@handle_token_exceptions
async def semantic_query(
    request: Request,
    query_request: SemanticQueryRequest,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process a natural language query about tokens.
    
    This endpoint uses NLP to understand the intent of a text query and returns
    the relevant token data based on that intent. It can extract token addresses
    and other parameters directly from the text.
    
    Args:
        request: FastAPI request object
        query_request: The request body containing the query
        request_id: Optional request ID for tracing
        
    Returns:
        Analysis result based on the query intent
    """
    query = query_request.query
    log_with_context(
        logger,
        "info",
        f"Semantic query received: {query}",
        request_id=request_id,
        query=query
    )
    
    async with get_solana_client() as client:
        query_analyzer = TokenQueryAnalyzer(client)
        result = await query_analyzer.analyze_query(query, request_id=request_id)
        
        if result.get("error"):
            log_with_context(
                logger,
                "warning",
                f"Semantic query error: {result['error']}",
                request_id=request_id,
                query=query,
                error=result["error"]
            )
        else:
            log_with_context(
                logger,
                "info",
                f"Semantic query completed successfully",
                request_id=request_id,
                query=query,
                intent=result.get("intent"),
                confidence=result.get("confidence")
            )
        
        return result 