"""Solana MCP server implementation using FastMCP."""

import anyio
import click
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from mcp.server.fastmcp import Context, FastMCP
import mcp.types as types
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from solana_mcp.config import get_app_config
from solana_mcp.solana_client import SolanaClient, get_solana_client
from solana_mcp.session import periodic_session_cleanup
from solana_mcp.api.endpoints import (
    rest_get_account, 
    rest_get_balance, 
    rest_get_token_info,
    rest_get_transactions,
    rest_get_nft_info,
    rest_nlp_query,
    rest_chain_analysis
)


# Set up logging
config = get_app_config()
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Application context for the Solana MCP server."""
    solana_client: SolanaClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context.
    
    Args:
        server: The FastMCP server instance.
        
    Yields:
        The application context.
    """
    # Start session cleanup task
    import asyncio
    cleanup_task = asyncio.create_task(periodic_session_cleanup())
    
    try:
        async with get_solana_client() as solana_client:
            logger.info("Solana client initialized")
            yield AppContext(solana_client=solana_client)
    finally:
        # Cancel cleanup task when shutting down
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Session cleanup task cancelled")


# Create the FastMCP server with Solana lifespan
app = FastMCP(
    "solana-mcp",
    lifespan=app_lifespan,
    dependencies=["solana"]
)


# -------------------------------------------
# Register MCP Resources 
# -------------------------------------------

# Import and register the resources
from solana_mcp.resources.account import register_resources as register_account_resources
register_account_resources(app)

# Other resource categories would be registered here
# register_token_resources(app)
# register_transaction_resources(app)
# register_program_resources(app)
# register_network_resources(app)
# register_nft_resources(app)


# -------------------------------------------
# Register MCP Tools
# -------------------------------------------

@app.tool()
async def get_solana_account(ctx: Context, address: str) -> str:
    """Fetch a Solana account's information.
    
    Args:
        ctx: The request context
        address: The account public key
        
    Returns:
        Formatted account information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    from solana_mcp.semantic_search import get_account_details
    
    result = await get_account_details(address, solana_client, format_level="detailed")
    return str(result)


@app.tool()
async def get_solana_balance(ctx: Context, address: str) -> str:
    """Fetch a Solana account's balance.
    
    Args:
        ctx: The request context
        address: The account public key
        
    Returns:
        Account balance in SOL
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    from solana_mcp.semantic_search import get_account_balance
    
    result = await get_account_balance(address, solana_client)
    return str(result.get("formatted", "Error retrieving balance"))


@app.tool()
async def analyze_address(ctx: Context, address: str, analysis_type: str = "token_flow") -> str:
    """Analyze an address on the Solana blockchain.
    
    Args:
        ctx: The request context
        address: The address to analyze
        analysis_type: Type of analysis (token_flow, activity_pattern, token_holding)
        
    Returns:
        Analysis results
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    
    from solana_mcp.analysis import (
        analyze_token_flow, 
        analyze_activity_pattern,
        analyze_token_holding_distribution
    )
    
    if analysis_type == "token_flow":
        result = await analyze_token_flow(address, solana_client)
    elif analysis_type == "activity_pattern":
        result = await analyze_activity_pattern(address, solana_client)
    elif analysis_type == "token_holding":
        result = await analyze_token_holding_distribution(address, solana_client)
    else:
        return f"Invalid analysis type: {analysis_type}. Supported types: token_flow, activity_pattern, token_holding"
    
    import json
    return json.dumps(result, indent=2)


@app.tool()
async def semantic_search(ctx: Context, address: str, query: str) -> str:
    """Search for transactions semantically.
    
    Args:
        ctx: The request context
        address: The account address
        query: Semantic search query (e.g., "token transfers", "nft sales")
        
    Returns:
        Matching transactions
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    from solana_mcp.semantic_search import semantic_transaction_search
    
    result = await semantic_transaction_search(address, query, solana_client)
    import json
    return json.dumps(result, indent=2)


@app.tool()
async def natural_language_query(ctx: Context, query: str) -> str:
    """Process a natural language query about Solana.
    
    Args:
        ctx: The request context
        query: Natural language query
        
    Returns:
        Query results
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    from solana_mcp.semantic_search import parse_natural_language_query
    
    result = await parse_natural_language_query(query, solana_client)
    import json
    return json.dumps(result, indent=2)


# -------------------------------------------
# Server Runner
# -------------------------------------------

@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    help="Transport type (stdio or sse)",
)
@click.option("--port", type=int, help="Port to listen on for SSE transport")
@click.option("--host", type=str, help="Host to bind to for SSE transport")
@click.option("--config", type=str, help="Path to config file")
def run_server(
    transport: Optional[str] = None,
    port: Optional[int] = None,
    host: Optional[str] = None,
    config: Optional[str] = None,
) -> None:
    """Run the Solana MCP server.
    
    Args:
        transport: Transport type. Defaults to environment setting or "stdio".
        port: Port to listen on for SSE transport. Defaults to environment setting or 8000.
        host: Host to bind to for SSE transport. Defaults to environment setting or "0.0.0.0".
        config: Path to config file.
    """
    # Get server config with environment defaults
    app_config = get_app_config()
    
    # Override with CLI options if provided
    if transport:
        app_config.server.transport = transport
    if port:
        app_config.server.port = port
    if host:
        app_config.server.host = host
    
    # Run server with appropriate transport
    if app_config.server.transport == "sse":
        logger.info(f"Starting server with SSE transport on {app_config.server.host}:{app_config.server.port}")
        
        # Set up SSE transport
        sse = SseServerTransport("/messages/")
        
        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        
        # Set up REST API endpoints alongside SSE
        # Create Starlette app with CORS middleware
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=app_config.api.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            Middleware(
                SessionMiddleware,
                secret_key=app_config.session.secret_key,
                max_age=60 * 60 * 24  # 1 day
            )
        ]
        
        starlette_app = Starlette(
            debug=app_config.server.debug,
            middleware=middleware,
            routes=[
                # SSE routes
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
                
                # REST API endpoints
                Route("/api/account/{address}", rest_get_account),
                Route("/api/balance/{address}", rest_get_balance),
                Route("/api/token/{mint}", rest_get_token_info),
                Route("/api/transactions/{address}", rest_get_transactions),
                Route("/api/nft/{mint}", rest_get_nft_info),
                
                # New LLM-focused endpoints
                Route("/api/nlp/query", rest_nlp_query, methods=["POST"]),
                Route("/api/analysis", rest_chain_analysis, methods=["POST"]),
                
                # Service endpoints
                Route("/health", lambda request: {"status": "healthy"}),
                Route("/api/docs", lambda request: {"message": "API documentation coming soon"}),
            ],
        )
        
        # Set up shared Solana client for the REST endpoints
        @starlette_app.on_event("startup")
        async def startup():
            """Initialize resources on server startup."""
            try:
                # Create a Solana client for the application
                solana_client = await get_solana_client().__aenter__()
                starlette_app.state.solana_client = solana_client
                
                # Start session cleanup task (safe to run in background)
                import asyncio
                asyncio.create_task(periodic_session_cleanup())
                
                logger.info("Server startup completed successfully")
            except Exception as e:
                logger.error(f"Error during server startup: {str(e)}")
                # Re-raise to prevent server from starting with incomplete initialization
                raise
                
        @starlette_app.on_event("shutdown")
        async def shutdown():
            """Clean up resources on server shutdown."""
            # Close the client explicitly if needed
            if hasattr(starlette_app.state, "solana_client"):
                try:
                    await starlette_app.state.solana_client.close()
                except Exception as e:
                    logger.error(f"Error closing Solana client: {str(e)}")
            
            # Clean up sessions
            from solana_mcp.session import clean_expired_sessions
            try:
                clean_expired_sessions()
            except Exception as e:
                logger.error(f"Error clearing sessions: {str(e)}")
            
            logger.info("Server shutdown completed successfully.")
        
        # Start the server
        import uvicorn
        uvicorn.run(
            starlette_app, 
            host=app_config.server.host, 
            port=app_config.server.port
        )
    else:
        # Default to stdio
        logger.info("Starting server with stdio transport")
        from mcp.server.stdio import stdio_server
        
        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
        
        anyio.run(arun)


if __name__ == "__main__":
    run_server() 