"""Token resources for the Solana MCP server."""

import json
from typing import Dict, Any, Optional

from mcp.server.fastmcp import Context

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError, get_solana_client
from solana_mcp.clients.token_client import TokenClient


async def get_token_accounts(owner: str, *, ctx: Optional[Context] = None) -> str:
    """Get token accounts owned by an address.
    
    Args:
        owner: The owner address
        ctx: The request context (injected by MCP)
        
    Returns:
        Token account information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        async with get_solana_client() as solana_client:
            return await _get_token_accounts_impl(solana_client, owner)
    
    # If we got here, we're using the context's client
    return await _get_token_accounts_impl(solana_client, owner)


async def _get_token_accounts_impl(solana_client: SolanaClient, owner: str) -> str:
    """Implementation for get_token_accounts."""
    try:
        token_accounts = await solana_client.get_token_accounts_by_owner(owner)
        return json.dumps(token_accounts, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_token_info(mint: str, *, ctx: Optional[Context] = None) -> str:
    """Get token information for a mint.
    
    Args:
        mint: The token mint address
        ctx: The request context (injected by MCP)
        
    Returns:
        Token information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        async with get_solana_client() as solana_client:
            return await _get_token_info_impl(solana_client, mint)
    
    # If we got here, we're using the context's client
    return await _get_token_info_impl(solana_client, mint)


async def _get_token_info_impl(solana_client: SolanaClient, mint: str) -> str:
    """Implementation for get_token_info."""
    try:
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get token supply
        supply = await solana_client.get_token_supply(mint)
        
        # Get token price if available
        price_data = await solana_client.get_market_price(mint)
        
        # Combine data
        token_info = {
            "mint": mint,
            "metadata": metadata,
            "supply": supply,
            "price": price_data
        }
        
        return json.dumps(token_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_token_holders(mint: str, *, ctx: Optional[Context] = None) -> str:
    """Get token holders for a mint.
    
    Args:
        mint: The token mint address
        ctx: The request context (injected by MCP)
        
    Returns:
        Token holders information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        async with get_solana_client() as solana_client:
            return await _get_token_holders_impl(solana_client, mint)
    
    # If we got here, we're using the context's client
    return await _get_token_holders_impl(solana_client, mint)


async def _get_token_holders_impl(solana_client: SolanaClient, mint: str) -> str:
    """Implementation for get_token_holders."""
    try:
        # Get largest accounts for this token
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # For each account, get the owner
        holders = []
        for account in largest_accounts:
            # Get the account info to find the owner
            account_info = await solana_client.get_account_info(
                account["address"], 
                encoding="jsonParsed"
            )
            
            # Extract owner and balance information
            if "parsed" in account_info.get("data", {}):
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data:
                    info = parsed_data["info"]
                    holders.append({
                        "owner": info.get("owner"),
                        "address": account["address"],
                        "amount": info.get("tokenAmount", {}).get("amount"),
                        "decimals": info.get("tokenAmount", {}).get("decimals"),
                        "uiAmount": info.get("tokenAmount", {}).get("uiAmount")
                    })
        
        return json.dumps({"mint": mint, "holders": holders}, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_all_token_data(mint: str, *, ctx: Optional[Context] = None) -> str:
    """Get comprehensive token data for a mint.
    
    Args:
        mint: The token mint address
        ctx: The request context (injected by MCP)
        
    Returns:
        Complete token data as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        async with get_solana_client() as solana_client:
            return await _get_all_token_data_impl(solana_client, mint)
    
    # If we got here, we're using the context's client
    return await _get_all_token_data_impl(solana_client, mint)


async def _get_all_token_data_impl(solana_client: SolanaClient, mint: str) -> str:
    """Implementation for get_all_token_data."""
    try:
        # Create a token client from the solana client
        token_client = TokenClient(solana_client.config)
        
        # Use token client's get_all_token_data method
        all_token_data = await token_client.get_all_token_data(mint)
        
        return json.dumps(all_token_data, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def register_resources(app):
    """Register token resources with the app.
    
    Args:
        app: The FastMCP app instance
    """
    @app.resource("solana://tokens/{owner}")
    async def tokens_resource(owner: str) -> str:
        return await get_token_accounts(owner, ctx=None)
        
    @app.resource("solana://token/{mint}")
    async def token_info_resource(mint: str) -> str:
        return await get_token_info(mint, ctx=None)
        
    @app.resource("solana://token/{mint}/holders")
    async def token_holders_resource(mint: str) -> str:
        return await get_token_holders(mint, ctx=None)
        
    @app.resource("solana://token/{mint}/all")
    async def token_all_data_resource(mint: str) -> str:
        return await get_all_token_data(mint, ctx=None) 