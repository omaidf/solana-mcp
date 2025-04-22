"""NFT resources for the Solana MCP server."""

import json
from typing import Dict, Any, Optional

from mcp.server.fastmcp import Context

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError


async def get_nft_info(mint: str, *, ctx: Optional[Context] = None) -> str:
    """Get NFT information.
    
    Args:
        mint: The NFT mint address
        ctx: The request context (injected by MCP)
        
    Returns:
        NFT information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_nft_info_impl(solana_client, mint)
    
    # If we got here, we're using the context's client
    return await _get_nft_info_impl(solana_client, mint)


async def _get_nft_info_impl(solana_client: SolanaClient, mint: str) -> str:
    """Implementation for get_nft_info."""
    try:
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Determine if it's an NFT (supply of 1, etc.)
        supply = await solana_client.get_token_supply(mint)
        
        # Check if this is an NFT
        is_nft = False
        if "value" in supply and supply["value"]["amount"] == "1" and supply["value"]["decimals"] == 0:
            is_nft = True
        
        # Return NFT info
        nft_info = {
            "mint": mint,
            "is_nft": is_nft,
            "metadata": metadata,
            "supply": supply
        }
        
        return json.dumps(nft_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def register_resources(app):
    """Register NFT resources with the app.
    
    Args:
        app: The FastMCP app instance
    """
    app.resource("solana://nft/{mint}")(get_nft_info) 