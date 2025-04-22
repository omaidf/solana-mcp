"""API routes package for Solana MCP Server."""

# Import routers to make them available for inclusion
from solana_mcp.api_routes.token_analysis import router as token_analysis_router
from solana_mcp.api_routes.liquidity_analysis import router as liquidity_analysis_router
from solana_mcp.api_routes.token_risk_analysis import router as token_risk_router

# List of available routers
__all__ = [
    "token_analysis_router",
    "liquidity_analysis_router", 
    "token_risk_router"
] 