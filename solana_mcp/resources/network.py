"""Network resources for the Solana MCP server."""

import json
from typing import Dict, Any, Optional

from mcp.server.fastmcp import Context

from solana_mcp.solana_client import SolanaClient, SolanaRpcError


async def get_network_epoch(*, ctx: Optional[Context] = None) -> str:
    """Get current epoch information.
    
    Args:
        ctx: The request context (injected by MCP)
        
    Returns:
        Epoch information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_network_epoch_impl(solana_client)
    
    # If we got here, we're using the context's client
    return await _get_network_epoch_impl(solana_client)


async def _get_network_epoch_impl(solana_client: SolanaClient) -> str:
    """Implementation for get_network_epoch."""
    try:
        epoch_info = await solana_client.get_epoch_info()
        current_slot = await solana_client.get_slot()
        return json.dumps({
            "epoch_info": epoch_info,
            "current_slot": current_slot
        }, indent=2)
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_network_validators(*, ctx: Optional[Context] = None) -> str:
    """Get information about network validators.
    
    Args:
        ctx: The request context (injected by MCP)
        
    Returns:
        Network node information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_network_validators_impl(solana_client)
    
    # If we got here, we're using the context's client
    return await _get_network_validators_impl(solana_client)


async def _get_network_validators_impl(solana_client: SolanaClient) -> str:
    """Implementation for get_network_validators."""
    try:
        # Get cluster nodes
        cluster_nodes = await solana_client.get_cluster_nodes()
        return json.dumps({"validators": cluster_nodes}, indent=2)
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def register_resources(app):
    """Register network resources with the app.
    
    Args:
        app: The FastMCP app instance
    """
    app.resource("solana://network/epoch")(get_network_epoch)
    app.resource("solana://network/validators")(get_network_validators) 