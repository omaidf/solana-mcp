"""REST API endpoints for the Solana MCP server."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from starlette.requests import Request
from starlette.responses import JSONResponse

from solana_mcp.api.error_handling import with_error_handling, with_validation, rate_limit
from solana_mcp.solana_client import SolanaClient


# --- Request/Response models ---

class NlpQueryRequest(BaseModel):
    """Request model for NLP query endpoint."""
    
    query: str = Field(..., min_length=1, description="Natural language query")
    format_level: str = Field("auto", pattern="^(minimal|standard|detailed|auto)$", 
                             description="Response detail level")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""
    
    type: str = Field(..., pattern="^(token_flow|activity_pattern)$", 
                     description="Type of analysis to perform")
    address: str = Field(..., min_length=32, max_length=44, 
                        description="Solana address to analyze")
    limit: Optional[int] = Field(50, ge=1, le=200, 
                                description="Maximum number of items to return")


# --- Endpoints ---

@with_error_handling
@rate_limit(requests_per_minute=60)
async def rest_get_account(request: Request) -> JSONResponse:
    """REST API endpoint for getting account info.
    
    Args:
        request: The HTTP request
        
    Returns:
        Account information as JSON
    """
    address = request.path_params.get("address")
    if not address:
        return JSONResponse({
            "error": "Missing address parameter",
            "error_explanation": "The account address must be provided in the URL path."
        }, status_code=400)
        
    solana_client = request.app.state.solana_client
    
    # The error handling is now done by the decorator
    account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
    return JSONResponse(account_info)


@with_error_handling
@rate_limit(requests_per_minute=60)
async def rest_get_balance(request: Request) -> JSONResponse:
    """REST API endpoint for getting account balance.
    
    Args:
        request: The HTTP request
        
    Returns:
        Account balance as JSON
    """
    address = request.path_params.get("address")
    if not address:
        return JSONResponse({
            "error": "Missing address parameter",
            "error_explanation": "The account address must be provided in the URL path."
        }, status_code=400)
        
    solana_client = request.app.state.solana_client
    
    # The error handling is now done by the decorator
    balance_lamports = await solana_client.get_balance(address)
    balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
    
    return JSONResponse({
        "lamports": balance_lamports,
        "sol": balance_sol,
        "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
    })


@with_error_handling
@with_validation(NlpQueryRequest)
@rate_limit(requests_per_minute=30)  # More restrictive for expensive operations
async def rest_nlp_query(request: Request, validated_data: NlpQueryRequest) -> JSONResponse:
    """Process natural language queries about Solana.
    
    Args:
        request: The HTTP request
        validated_data: Validated request data
        
    Returns:
        Query results as JSON
    """
    # The validation is done by the decorator - validated_data is available
    query = validated_data.query
    format_level = validated_data.format_level
    session_id = validated_data.session_id
    
    # Get session
    from solana_mcp.session import get_or_create_session
    session = get_or_create_session(session_id)
    
    # Process query
    solana_client = request.app.state.solana_client
    from solana_mcp.semantic_search import parse_natural_language_query
    result = await parse_natural_language_query(query, solana_client, session)
    
    # Return result
    return JSONResponse({
        "result": result,
        "session_id": session.id,
        "query_count": len(session.query_history)
    })


@with_error_handling
@with_validation(AnalysisRequest)
@rate_limit(requests_per_minute=20)  # Even more restrictive for analysis
async def rest_chain_analysis(request: Request, validated_data: AnalysisRequest) -> JSONResponse:
    """Analyze patterns in token movements or transactions.
    
    Args:
        request: The HTTP request
        validated_data: Validated request data
        
    Returns:
        Analysis results as JSON
    """
    analysis_type = validated_data.type
    address = validated_data.address
    limit = validated_data.limit or 50
    
    solana_client = request.app.state.solana_client
    
    # Perform analysis based on type
    if analysis_type == "token_flow":
        from solana_mcp.analysis import analyze_token_flow
        result = await analyze_token_flow(address, solana_client, limit)
        return JSONResponse(result)
    
    elif analysis_type == "activity_pattern":
        from solana_mcp.analysis import analyze_activity_pattern
        result = await analyze_activity_pattern(address, solana_client, limit)
        return JSONResponse(result)
    
    else:
        # This should never happen due to validation, but just in case
        return JSONResponse({
            "error": f"Unknown analysis type: {analysis_type}",
            "error_explanation": "Supported analysis types are: token_flow, activity_pattern"
        }, status_code=400) 