"""Main FastAPI application for Solana MCP Server."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import logging

# Import routers
from solana_mcp.api_routes.token_analysis import router as token_analysis_router
from solana_mcp.api_routes.liquidity_analysis import router as liquidity_analysis_router
from solana_mcp.api_routes.token_risk_analysis import router as token_risk_router

# Import config
from solana_mcp.config import get_server_config, get_solana_config
from solana_mcp.logging_config import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Solana MCP API",
    description="API for Solana token analysis and liquidity tracking",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(token_analysis_router)
app.include_router(liquidity_analysis_router)
app.include_router(token_risk_router)

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint.
    
    Returns:
        Status and version information
    """
    server_config = get_server_config()
    solana_config = get_solana_config()
    
    return {
        "status": "ok",
        "environment": server_config.environment,
        "solana_rpc": solana_config.rpc_url,
        "version": "0.1.0"
    }

# Request middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests.
    
    Args:
        request: The incoming request
        call_next: Next middleware function
        
    Returns:
        Response from the route handler
    """
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    process_time = asyncio.get_event_loop().time() - start_time
    
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.4f}s"
    )
    
    return response

# Run the application if executed directly
if __name__ == "__main__":
    server_config = get_server_config()
    
    # Log configuration info
    logging.info(f"Starting Solana MCP API in {server_config.environment} mode")
    logging.info(f"Listening on {server_config.bind_address}")
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host=server_config.host,
        port=server_config.port,
        reload=server_config.debug
    ) 