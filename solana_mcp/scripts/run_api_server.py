#!/usr/bin/env python
"""Run the Solana MCP API server with all routes explicitly registered."""

import uvicorn
import argparse
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from solana_mcp.logging_config import setup_logging, RequestIdMiddleware, get_logger
from solana_mcp.api_routes.token_analysis import router as token_analysis_router
from solana_mcp.api_routes.liquidity_analysis import router as liquidity_analysis_router
from solana_mcp.api_routes.token_risk_analysis import router as token_risk_router

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create the FastAPI application
app = FastAPI(
    title="Solana MCP API Server",
    description="Solana token analysis and risk assessment API",
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

# Include routers with explicit order
app.include_router(token_analysis_router)
app.include_router(liquidity_analysis_router)
app.include_router(token_risk_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Version endpoint
@app.get("/version")
async def version():
    """Get API version."""
    return {
        "version": "1.0.0",
        "name": "Solana MCP API Server"
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Solana MCP API server")
    parser.add_argument("--port", type=int, default=None, help="Port to run the server on")
    args = parser.parse_args()
    
    # Determine port from command line args or environment variables
    port = args.port
    if port is None:
        port = int(os.environ.get("PORT", os.environ.get("SOLANA_MCP_PORT", 8000)))
    
    logger.info(f"Starting Solana MCP API Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 