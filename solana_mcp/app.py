"""
Main entry point for the Solana MCP API.

This module initializes the FastAPI application, sets up middleware,
configures routes, and manages the application lifecycle.
"""

import asyncio
import datetime
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Callable

import uvicorn
from fastapi import FastAPI, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from solana_mcp.dependencies import initialize_providers
from solana_mcp.middleware.logging_middleware import LoggingMiddleware
from solana_mcp.middleware.error_handling import handle_api_errors
from solana_mcp.models.api_response import ApiResponse
from solana_mcp.routes import accounts, tokens, transactions, analysis, nlp, tasks
from solana_mcp.utils.background_tasks import get_task_manager
from solana_mcp.utils.config import settings
from solana_mcp.utils.error_handler import register_error_handlers


# API Documentation tags
tags_metadata = [
    {
        "name": "accounts",
        "description": "Operations related to Solana accounts and their information",
    },
    {
        "name": "tokens",
        "description": "Operations related to SPL tokens, token accounts, and metadata",
    },
    {
        "name": "transactions",
        "description": "Operations for retrieving and analyzing Solana transactions",
    },
    {
        "name": "analysis",
        "description": "Advanced analytics and data processing of Solana blockchain data",
    },
    {
        "name": "nlp",
        "description": "Natural language processing for Solana data extraction",
    },
    {
        "name": "tasks",
        "description": "Background task management for long-running operations",
    },
    {
        "name": "system",
        "description": "System-level operations for monitoring and management",
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    
    Args:
        app: The FastAPI application instance
    """
    # Initialize logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize dependency providers
    initialize_providers()
    
    # Initialize background task manager
    task_manager = get_task_manager()
    await task_manager.start()
    app.state.task_manager = task_manager
    
    logging.info("Application initialized successfully")
    
    yield  # Application is running here
    
    # Shutdown procedure
    logging.info("Application shutting down...")
    
    # Cleanup background task manager
    await task_manager.stop()
    
    logging.info("Shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        The configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="Solana MCP API",
        description="""
        # Solana Model Context Protocol API
        
        This API provides access to the Solana blockchain data through a standardized interface,
        enabling efficient data retrieval, analysis, and processing for various use cases.
        
        ## Key Features
        
        - **Account Information**: Query account data, balance, and program ownership
        - **Token Data**: Access SPL token information, metadata, and holder analytics
        - **Transaction Details**: Retrieve transaction information with parsed instructions
        - **Analytics**: Process and analyze blockchain data for accounts and programs
        - **Natural Language Processing**: Extract data using natural language queries
        - **Background Processing**: Handle resource-intensive operations asynchronously
        
        ## Authentication
        
        API keys are required for all endpoints. Provide your API key in the `X-API-Key` header.
        
        ## Rate Limiting
        
        - Standard tier: 5 requests per second
        - Enterprise tier: Contact support for custom limits
        
        ## Best Practices
        
        - Use batch endpoints where possible to reduce API calls
        - Implement caching for frequently accessed data
        - Utilize background processing for heavy operations
        """,
        version="1.0.0",
        openapi_tags=tags_metadata,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        contact={
            "name": "Solana MCP Support",
            "email": "support@example.com",
            "url": "https://example.com/support",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Set to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register API routers
    api_router = APIRouter(prefix="/api")
    
    # Add all route modules
    api_router.include_router(accounts.router, prefix="/accounts", tags=["accounts"])
    api_router.include_router(tokens.router, prefix="/tokens", tags=["tokens"])
    api_router.include_router(transactions.router, prefix="/transactions", tags=["transactions"])
    api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
    api_router.include_router(nlp.router, prefix="/nlp", tags=["nlp"])
    api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    
    # Include the API router in the main app
    app.include_router(api_router)
    
    @app.get("/health", response_model=ApiResponse[Dict[str, Any]], tags=["system"])
    @handle_api_errors
    async def health_check():
        """
        Check the health of the service.
        
        Returns information about the service status, version, and environment.
        This endpoint can be used for monitoring and health checks.
        """
        return ApiResponse(
            success=True,
            data={
                "status": "healthy",
                "version": settings.VERSION,
                "timestamp": datetime.datetime.now().isoformat(),
                "environment": settings.ENVIRONMENT
            }
        )
    
    return app


app = create_application()


if __name__ == "__main__":
    # Run the application directly when script is executed
    uvicorn.run(
        "solana_mcp.app:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG
    ) 