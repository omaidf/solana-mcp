"""Account resources for the Solana MCP server."""

import json
from typing import Dict, Any

from mcp.server.fastmcp import Context

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError


async def get_account(ctx: Context, address: str) -> str:
    """Get Solana account information.
    
    Args:
        ctx: The request context
        address: The account address
        
    Returns:
        Account information as JSON string
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
        return json.dumps(account_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_balance(ctx: Context, address: str) -> str:
    """Get Solana account balance.
    
    Args:
        ctx: The request context
        address: The account address
        
    Returns:
        Account balance in SOL
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        balance_lamports = await solana_client.get_balance(address)
        balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        return json.dumps({
            "lamports": balance_lamports,
            "sol": balance_sol,
            "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
        }, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_account_with_programs(ctx: Context, address: str) -> str:
    """Get Solana account information with program details.
    
    Args:
        ctx: The request context
        address: The account address
        
    Returns:
        Enhanced account information as JSON string
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        # Get basic account info
        account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
        
        # Add program labels if account is owned by a known program
        if "owner" in account_info:
            owner = account_info["owner"]
            # Map common program IDs to names
            program_map = {
                "11111111111111111111111111111111": "System Program",
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "Token Program",
                "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s": "Metaplex Token Metadata Program",
                "Stake11111111111111111111111111111111111111": "Stake Program",
                "Vote111111111111111111111111111111111111111": "Vote Program",
                "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca Program",
                "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "Jupiter Aggregator"
            }
            if owner in program_map:
                account_info["owner_name"] = program_map[owner]
        
        # Check if this is a token account
        if account_info.get("owner") == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
            # Extract token info if available
            if "data" in account_info and "parsed" in account_info["data"]:
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data and "mint" in parsed_data["info"]:
                    mint = parsed_data["info"]["mint"]
                    # Try to get token metadata
                    try:
                        metadata = await solana_client.get_token_metadata(mint)
                        account_info["token_metadata"] = metadata
                    except Exception:
                        pass
        
        return json.dumps(account_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


async def get_associated_accounts(ctx: Context, address: str, account_type: str = "all") -> str:
    """Get accounts associated with an address.
    
    Args:
        ctx: The request context
        address: The account address
        account_type: Type of accounts to return (all, token, nft)
        
    Returns:
        Associated accounts as JSON string
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        result = {"address": address, "associated_accounts": []}
        
        # Get token accounts
        if account_type in ["all", "token", "nft"]:
            token_accounts = await solana_client.get_token_accounts_by_owner(address)
            
            for account in token_accounts:
                account_info = {
                    "account_address": account.get("pubkey"),
                    "type": "token"
                }
                
                # Extract token data if available
                if "account" in account and "data" in account["account"]:
                    data = account["account"]["data"]
                    if "parsed" in data and "info" in data["parsed"]:
                        info = data["parsed"]["info"]
                        
                        # Add mint and token amount
                        if "mint" in info:
                            account_info["mint"] = info["mint"]
                        
                        if "tokenAmount" in info:
                            account_info["token_amount"] = info["tokenAmount"]
                            
                            # Determine if this is likely an NFT
                            decimals = info["tokenAmount"].get("decimals", 0)
                            amount = info["tokenAmount"].get("uiAmount", 0)
                            
                            if decimals == 0 and amount == 1:
                                account_info["type"] = "nft"
                                
                                # Skip if we only want fungible tokens
                                if account_type == "token" and account_info["type"] == "nft":
                                    continue
                                
                                # Skip if we only want NFTs
                                if account_type == "nft" and account_info["type"] != "nft":
                                    continue
                                
                                # For NFTs, get metadata
                                try:
                                    metadata = await solana_client.get_token_metadata(info["mint"])
                                    account_info["metadata"] = metadata
                                except Exception:
                                    pass
                
                result["associated_accounts"].append(account_info)
        
        # Add counts
        result["total_count"] = len(result["associated_accounts"])
        result["nft_count"] = sum(1 for account in result["associated_accounts"] if account.get("type") == "nft")
        result["token_count"] = sum(1 for account in result["associated_accounts"] if account.get("type") == "token")
        
        return json.dumps(result, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def register_resources(app):
    """Register account resources with the app.
    
    Args:
        app: The FastMCP app instance
    """
    app.resource("solana://account/{address}")(get_account)
    app.resource("solana://balance/{address}")(get_balance)
    app.resource("solana://account/{address}/enhanced")(get_account_with_programs)
    app.resource("solana://account/{address}/associated")(get_associated_accounts)
    app.resource("solana://account/{address}/associated/{account_type}")(get_associated_accounts) 