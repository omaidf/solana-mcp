"""Main entry point for the Solana MCP Server."""

# Standard library imports
import os
import sys
import argparse
from typing import Dict, Any

# Third-party library imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Internal imports
from solana_mcp.config import get_server_config, AppConfig, get_app_config, get_solana_config
from solana_mcp.logging_config import setup_logging, get_logger, RequestIdMiddleware
from solana_mcp.api_routes.token_analysis import router as token_analysis_router
from solana_mcp.api_routes.liquidity_analysis import router as liquidity_analysis_router
from solana_mcp.api_routes.token_risk_analysis import router as token_risk_router
from solana_mcp.api.error_handling import api_error_handler

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Solana MCP Server",
    description="A comprehensive Solana token analysis server with focus on pumpfun tokens",
    version="1.0.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware
app.add_middleware(RequestIdMiddleware)

# Include routers
app.include_router(token_analysis_router)
app.include_router(liquidity_analysis_router)
app.include_router(token_risk_router)

# Health check endpoint
@app.get("/health")
@api_error_handler
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}

# Version endpoint
@app.get("/version")
@api_error_handler
async def version():
    """Get API version information.
    
    Returns:
        Version information
    """
    return {
        "version": "1.0.0",
        "name": "Solana MCP Server",
    }

# Error handler for uncaught exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception that was raised
        
    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def run_server(port=None):
    """Run the server from command line.
    
    Args:
        port: Optional port override
    
    This function is used as an entry point in setup.py.
    """
    config = get_server_config()
    solana_config = get_solana_config()
    
    # Override port if specified
    if port is not None:
        try:
            config.port = int(port)
        except ValueError:
            logger.error(f"Invalid port number: {port}")
            sys.exit(1)
    
    logger.info(
        f"Starting Solana MCP Server on {config.host}:{config.port} (Environment: {config.environment})"
    )
    logger.info(f"Using Solana RPC URL: {solana_config.rpc_url}")
    
    uvicorn.run(
        "solana_mcp.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    """Run the server directly when script is executed."""
    parser = argparse.ArgumentParser(description="Solana MCP Server")
    parser.add_argument("--port", type=int, help="Server port")
    args = parser.parse_args()
    
    run_server(port=args.port) 