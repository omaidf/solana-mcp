"""
Main application entry point for the Solana MCP API.

This module is a wrapper around the main application defined in solana_mcp.app.
It provides a convenient entry point for running the API server.
"""

import uvicorn
import os
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

# Import the main application
from solana_mcp.app import app as solana_mcp_app
from solana_mcp.config import get_server_config

# Re-export the application
app = solana_mcp_app

if __name__ == "__main__":
    # Get server configuration
    server_config = get_server_config()
    
    # Run the application directly when script is executed
    uvicorn.run(
        "solana_mcp.app:app",
        host=server_config.host,
        port=server_config.port,
        log_level=server_config.log_level.lower(),
        reload=server_config.debug
    ) 