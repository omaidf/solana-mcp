"""Transaction resources for the Solana MCP server."""

import json
from typing import Dict, Any, Optional

from mcp.server.fastmcp import Context

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError


async def get_transaction_details(signature: str, *, ctx: Optional[Context] = None) -> str:
    """Get transaction details.
    
    Args:
        signature: The transaction signature
        ctx: The request context (injected by MCP)
        
    Returns:
        Transaction details as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_transaction_details_impl(solana_client, signature)
    
    # If we got here, we're using the context's client
    return await _get_transaction_details_impl(solana_client, signature)


async def _get_transaction_details_impl(solana_client: SolanaClient, signature: str) -> str:
    """Implementation for get_transaction_details."""
    try:
        transaction = await solana_client.get_transaction(signature)
        return json.dumps(transaction, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_address_transactions(address: str, limit: int = 20, *, ctx: Optional[Context] = None) -> str:
    """Get transaction history for an address.
    
    Args:
        address: The account address
        limit: Maximum number of transactions to return
        ctx: The request context (injected by MCP)
        
    Returns:
        Transaction history as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_address_transactions_impl(solana_client, address, limit)
    
    # If we got here, we're using the context's client
    return await _get_address_transactions_impl(solana_client, address, limit)


async def _get_address_transactions_impl(solana_client: SolanaClient, address: str, limit: int) -> str:
    """Implementation for get_address_transactions."""
    try:
        signatures = await solana_client.get_signatures_for_address(address, limit=limit)
        return json.dumps({"address": address, "signatures": signatures}, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def register_resources(app):
    """Register transaction resources with the app.
    
    Args:
        app: The FastMCP app instance
    """
    @app.resource("solana://transaction/{signature}")
    async def transaction_details_resource(signature: str) -> str:
        return await get_transaction_details(signature, ctx=None)
        
    @app.resource("solana://transactions/{address}")
    async def address_transactions_resource(address: str) -> str:
        return await get_address_transactions(address, ctx=None) 