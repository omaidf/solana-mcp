"""Program resources for the Solana MCP server."""

import json
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import Context

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError


async def get_program_info(program_id: str, *, ctx: Optional[Context] = None) -> str:
    """Get program information.
    
    Args:
        program_id: The program ID
        ctx: The request context (injected by MCP)
        
    Returns:
        Program information as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_program_info_impl(solana_client, program_id)
    
    # If we got here, we're using the context's client
    return await _get_program_info_impl(solana_client, program_id)


async def _get_program_info_impl(solana_client: SolanaClient, program_id: str) -> str:
    """Implementation for get_program_info."""
    try:
        # Get the program account itself
        program_account = await solana_client.get_account_info(program_id, encoding="jsonParsed")
        
        # Prepare result
        program_info = {
            "program_id": program_id,
            "account": program_account,
            "is_executable": program_account.get("executable", False),
            "owner": program_account.get("owner"),
            "lamports": program_account.get("lamports", 0)
        }
        
        return json.dumps(program_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_program_account_list(
    program_id: str,
    limit: int = 10,
    offset: int = 0,
    memcmp: Optional[str] = None,
    datasize: Optional[int] = None,
    *, ctx: Optional[Context] = None
) -> str:
    """Get accounts owned by a program.
    
    Args:
        program_id: The program ID
        limit: Maximum number of accounts to return
        offset: Offset to start from
        memcmp: JSON-encoded memcmp filter (e.g. '{"offset":0,"bytes":"base58bytes"}')
        datasize: Filter by exact data size
        ctx: The request context (injected by MCP)
        
    Returns:
        Program accounts as JSON string
    """
    # Get client from context or create new one
    if ctx:
        solana_client = ctx.request_context.lifespan_context.solana_client
    else:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            return await _get_program_account_list_impl(solana_client, program_id, limit, offset, memcmp, datasize)
    
    # If we got here, we're using the context's client
    return await _get_program_account_list_impl(solana_client, program_id, limit, offset, memcmp, datasize)


async def _get_program_account_list_impl(
    solana_client: SolanaClient,
    program_id: str,
    limit: int,
    offset: int,
    memcmp: Optional[str],
    datasize: Optional[int]
) -> str:
    """Implementation for get_program_account_list."""
    try:
        filters = []
        
        # Add memcmp filter if provided
        if memcmp:
            try:
                memcmp_filter = json.loads(memcmp)
                filters.append({"memcmp": memcmp_filter})
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid memcmp JSON format"})
        
        # Add datasize filter if provided
        if datasize is not None:
            filters.append({"dataSize": datasize})
        
        # Get program accounts with pagination
        accounts = await solana_client.get_program_accounts(
            program_id,
            filters=filters if filters else None,
            encoding="jsonParsed",
            limit=limit,
            offset=offset
        )
        
        # Count total found
        account_count = len(accounts)
        
        return json.dumps({
            "program_id": program_id,
            "count": account_count,
            "limit": limit,
            "offset": offset,
            "accounts": accounts
        }, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def register_resources(app):
    """Register program resources with the app.
    
    Args:
        app: The FastMCP app instance
    """
    app.resource("solana://program/{program_id}")(get_program_info)
    app.resource("solana://program/{program_id}/accounts")(get_program_account_list) 