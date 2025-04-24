"""Semantic search and natural language processing for Solana blockchain data."""

import re
import asyncio
import datetime
from typing import Dict, List, Any, Optional

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError
from solana_mcp.session import Session
from solana_mcp.config import get_app_config

# Get configuration
config = get_app_config()

# Basic patterns for NL query understanding
QUERY_PATTERNS = [
    # Balance queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:sol|solana)? ?balance (?:of|for) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_balance",
        "params": lambda match: {"address": match.group(1)}
    },
    # Account info queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:information|info|details) (?:about|for|of) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_account_info",
        "params": lambda match: {"address": match.group(1)}
    },
    # Token queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:token|tokens|token info|token details) (?:of|for|owned by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_accounts",
        "params": lambda match: {"owner": match.group(1)}
    },
    # Token info
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:information|info|details) (?:about|for|of) token (?:with mint )?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_info",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Transaction history
    {
        "pattern": r"(?:what are|get|show|check|find) (?:the )?(?:transactions|tx|transaction history) (?:of|for|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})(?: with limit (\d+))?",
        "intent": "get_transactions",
        "params": lambda match: {"address": match.group(1), "limit": int(match.group(2)) if match.group(2) else 20}
    },
    # NFT queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:nft|nft info|nft details) (?:with mint )?([a-zA-Z0-9]{32,44})",
        "intent": "get_nft_info",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Whale queries - Added new pattern to handle direct questions about whales
    {
        "pattern": r"(?:are there|are there any|do you see|can you find|any) (?:whales|whale|large holder|big investor|big wallet) (?:in|for|holding) (?:this token|this|token|mint)? ?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_whales",
        "params": lambda match: {"mint": match.group(1)}
    },
]

# Define common transaction types and their keywords
TRANSACTION_CATEGORIES = {
    "token_transfer": ["transfer", "send", "receive", "spl-token", "token program"],
    "nft_mint": ["mint", "nft", "metaplex", "metadata", "master edition"],
    "nft_sale": ["sale", "marketplace", "auction", "bid", "offer", "purchase"],
    "swap": ["swap", "exchange", "trade", "jupiter", "orca", "raydium"],
    "stake": ["stake", "delegate", "staking", "validator", "withdraw stake"],
    "system_transfer": ["system program", "sol transfer", "lamports"],
    "vote": ["vote", "voting", "governance"],
    "program_deploy": ["bpf loader", "deploy", "upgrade", "program"],
    "failed": ["failed", "error", "rejected"]
}


async def parse_natural_language_query(query: str, solana_client: SolanaClient, session: Optional[Session] = None) -> Dict[str, Any]:
    """Parse a natural language query into an API call with improved error handling.
    
    Args:
        query: The natural language query
        solana_client: The Solana client
        session: Optional session for context
        
    Returns:
        The query results
    """
    # Input validation
    if not query:
        return {
            "error": "Empty query",
            "error_explanation": "Please provide a query about Solana blockchain data."
        }
    
    # Normalize query
    normalized_query = query.lower().strip()
    
    # Try to match against patterns
    for pattern_info in QUERY_PATTERNS:
        try:
            match = re.search(pattern_info["pattern"], normalized_query)
            if match:
                intent = pattern_info["intent"]
                params = pattern_info["params"](match)
                
                # Execute the intent
                if intent == "get_balance":
                    if not params.get("address"):
                        return {"error": "Missing address", "error_explanation": "A valid Solana address is required."}
                        
                    result = await get_account_balance(params["address"], solana_client, format_level="auto")
                    if session:
                        session.add_query(query, result)
                        session.update_context_for_entity("address", params["address"], {"last_balance_check": datetime.datetime.now().isoformat()})
                    return result
                
                elif intent == "get_account_info":
                    if not params.get("address"):
                        return {"error": "Missing address", "error_explanation": "A valid Solana address is required."}
                        
                    result = await get_account_details(params["address"], solana_client, format_level="auto")
                    if session:
                        session.add_query(query, result)
                        session.update_context_for_entity("address", params["address"], {"last_info_check": datetime.datetime.now().isoformat()})
                    return result
                
                elif intent == "get_token_accounts":
                    if not params.get("owner"):
                        return {"error": "Missing owner address", "error_explanation": "A valid Solana address is required."}
                        
                    result = await get_token_accounts_for_owner(params["owner"], solana_client, format_level="auto")
                    if session:
                        session.add_query(query, result)
                        session.update_context_for_entity("address", params["owner"], {"last_token_check": datetime.datetime.now().isoformat()})
                    return result
                
                elif intent == "get_token_info":
                    if not params.get("mint"):
                        return {"error": "Missing token mint address", "error_explanation": "A valid token mint address is required."}
                        
                    result = await get_token_details(params["mint"], solana_client, format_level="auto")
                    if session:
                        session.add_query(query, result)
                        session.update_context_for_entity("token", params["mint"], {"last_check": datetime.datetime.now().isoformat()})
                    return result
                
                elif intent == "get_token_whales":
                    if not params.get("mint"):
                        return {"error": "Missing token mint address", "error_explanation": "A valid token mint address is required."}
                        
                    # Initialize the TokenAnalyzer to access whale data
                    from solana_mcp.token_analyzer import TokenAnalyzer
                    token_analyzer = TokenAnalyzer(solana_client)
                    
                    # Get whale data with default threshold ($50k)
                    whale_data = await token_analyzer.get_whale_holders(params["mint"], threshold_usd=50000.0)
                    
                    # Get basic token info for context
                    token_metadata = await solana_client.get_token_metadata(params["mint"])
                    token_name = token_metadata.get("name", "Unknown")
                    token_symbol = token_metadata.get("symbol", "UNKNOWN") 
                    
                    # Add token info if not already in the whale data
                    if "token_name" not in whale_data or whale_data["token_name"] == "Unknown":
                        whale_data["token_name"] = token_name
                    if "token_symbol" not in whale_data or whale_data["token_symbol"] == "UNKNOWN":
                        whale_data["token_symbol"] = token_symbol
                        
                    # Get total holders to ensure we have this data
                    holders_data = await token_analyzer.get_token_largest_holders(params["mint"])
                    total_holders = holders_data.get("total_holders", whale_data.get("total_holders_analyzed", 0))
                    whale_data["total_holders"] = total_holders
                    
                    # Add to session if available
                    if session:
                        session.add_query(query, whale_data)
                        session.update_context_for_entity("token", params["mint"], {"last_whale_check": datetime.datetime.now().isoformat()})
                    
                    return whale_data
                
                elif intent == "get_transactions":
                    if not params.get("address"):
                        return {"error": "Missing address", "error_explanation": "A valid Solana address is required."}
                        
                    result = await get_transaction_history_for_address(
                        params["address"], 
                        solana_client, 
                        limit=params.get("limit", 20),
                        format_level="auto"
                    )
                    if session:
                        session.add_query(query, result)
                        session.update_context_for_entity("address", params["address"], {"last_tx_check": datetime.datetime.now().isoformat()})
                    return result
                
                elif intent == "get_nft_info":
                    if not params.get("mint"):
                        return {"error": "Missing NFT mint address", "error_explanation": "A valid NFT mint address is required."}
                        
                    result = await get_nft_details(params["mint"], solana_client, format_level="auto")
                    if session:
                        session.add_query(query, result)
                        session.update_context_for_entity("nft", params["mint"], {"last_check": datetime.datetime.now().isoformat()})
                    return result
        except Exception as e:
            # Handle parsing errors for each pattern
            print(f"Error parsing query with pattern {pattern_info['pattern']}: {str(e)}")
            continue
    
    # If no pattern matched, return error
    return {
        "error": "I couldn't understand your query. Please try rephrasing or use a more specific request.",
        "error_explanation": "The query format wasn't recognized. Try following one of the supported formats.",
        "supported_queries": [
            "Get balance of [address]",
            "Get information about [address]",
            "Get tokens owned by [address]",
            "Get information about token [mint]",
            "Get transactions of [address]",
            "Get NFT info [mint]"
        ]
    }


async def categorize_transaction(tx_data: Dict[str, Any], solana_client: SolanaClient) -> List[str]:
    """Categorize a transaction based on its contents.
    
    Args:
        tx_data: Transaction data
        solana_client: Solana client
        
    Returns:
        List of categories that apply to this transaction
    """
    # Get full transaction data if only signature is provided
    if isinstance(tx_data, str) or (isinstance(tx_data, dict) and len(tx_data.keys()) == 1 and "signature" in tx_data):
        signature = tx_data if isinstance(tx_data, str) else tx_data["signature"]
        try:
            tx_data = await solana_client.get_transaction(signature)
        except Exception:
            return ["unknown"]
    
    # Check for transaction error
    if tx_data.get("meta", {}).get("err"):
        return ["failed"]
    
    categories = []
    
    # Extract program IDs from transaction
    programs = set()
    instructions = []
    
    # Extract program IDs and instructions - fixing the potential array access issue
    if "transaction" in tx_data and "message" in tx_data["transaction"]:
        message = tx_data["transaction"]["message"]
        
        # Get account keys
        account_keys = message.get("accountKeys", [])
        
        # Get instructions
        instructions = message.get("instructions", [])
        
        # Extract program IDs - safely accessing account_keys array
        for instr in instructions:
            program_id_index = instr.get("programId")
            if program_id_index is not None and isinstance(program_id_index, int):
                if 0 <= program_id_index < len(account_keys):
                    program_id = account_keys[program_id_index]
                    programs.add(program_id)
                    
    # For RPC v0 transaction format, handle program IDs differently
    elif "transaction" in tx_data and "message" in tx_data["transaction"] and "programIdIndex" in tx_data["transaction"]["message"]:
        message = tx_data["transaction"]["message"]
        if "accountKeys" in message and "instructions" in message:
            account_keys = message["accountKeys"]
            for instr in message["instructions"]:
                if "programIdIndex" in instr and isinstance(instr["programIdIndex"], int):
                    idx = instr["programIdIndex"]
                    if 0 <= idx < len(account_keys):
                        programs.add(account_keys[idx])
    
    # Check for system transfers
    if "11111111111111111111111111111111" in programs:
        categories.append("system_transfer")
    
    # Check for token transfers
    if "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" in programs:
        categories.append("token_transfer")
    
    # Check for stake operations
    if "Stake11111111111111111111111111111111111111" in programs:
        categories.append("stake")
    
    # Check for vote operations
    if "Vote111111111111111111111111111111111111111" in programs:
        categories.append("vote")
    
    # Check for Metaplex operations
    if "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s" in programs:
        # Could be NFT mint or other metadata operations
        categories.append("nft_mint")
    
    # Check for known marketplaces
    marketplace_programs = [
        "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K",  # Magic Eden
        "hausS13jsjafwWwGqZTUQRmWyvyxn9EQpqMwV1PBBmk",  # Tensor
        "CJsLwbP1iu5DuUikHEJnLfANgKy6stB2uFgvBBHoyxwz"  # Solanart
    ]
    
    for program in marketplace_programs:
        if program in programs:
            categories.append("nft_sale")
            break
    
    # Check for AMM/DEX programs
    swap_programs = [
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",  # Jupiter
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",  # Orca
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"   # Raydium
    ]
    
    for program in swap_programs:
        if program in programs:
            categories.append("swap")
            break
    
    # Additional checks based on token balance changes
    if "meta" in tx_data and "postTokenBalances" in tx_data["meta"] and "preTokenBalances" in tx_data["meta"]:
        pre = tx_data["meta"]["preTokenBalances"]
        post = tx_data["meta"]["postTokenBalances"]
        
        # If a new token account appears in post that wasn't in pre, it might be a mint
        if isinstance(pre, list) and isinstance(post, list):
            pre_accounts = {item.get("accountIndex") for item in pre if isinstance(item, dict)}
            post_accounts = {item.get("accountIndex") for item in post if isinstance(item, dict)}
            
            if post_accounts - pre_accounts:
                # New token accounts were created
                categories.append("token_transfer")
    
    # If no categories identified, mark as unknown
    if not categories:
        categories.append("unknown")
    
    return categories


async def semantic_transaction_search(
    address: str,
    query: str,
    solana_client: SolanaClient,
    limit: int = 20
) -> Dict[str, Any]:
    """Search transactions semantically based on purpose or category.
    
    Args:
        address: Account address
        query: Semantic search query
        solana_client: Solana client
        limit: Maximum number of transactions to return
        
    Returns:
        Matching transactions
    """
    # Input validation
    if not address:
        return {"error": "Address is required", "error_explanation": "A valid Solana address is needed for transaction search"}
    
    if not query:
        return {"error": "Query is required", "error_explanation": "A search query is needed to filter transactions"}
    
    # Normalize query
    query = query.lower().strip()
    
    # Get transaction signatures for the address
    try:
        # Build options dictionary first
        options = {"limit": min(100, limit * 2)}  # Get more than needed to filter down, but cap at 100
        
        signatures = await solana_client.get_signatures_for_address(
            address, 
            options
        )
        
        # Check if no signatures were found
        if not signatures:
            return {
                "address": address,
                "query": query,
                "transactions_found": 0,
                "message": "No transactions found for this address"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "error_explanation": "Error fetching transaction signatures for this address"
        }
    
    # Determine which categories to look for based on query
    target_categories = []
    for category, keywords in TRANSACTION_CATEGORIES.items():
        for keyword in keywords:
            if keyword in query:
                target_categories.append(category)
                break
    
    # If no categories matched, look for specific tokens or amounts
    if not target_categories:
        # For token name searches, NFT name searches, etc.
        # This would require a more complex implementation
        return {
            "error": "Couldn't understand the search query. Try using more specific terms like 'token transfer', 'nft', 'swap', etc."
        }
    
    # Fetch and filter transactions
    matching_transactions = []
    processed_count = 0
    
    # Use a throttled approach to avoid rate limits
    for i, sig_info in enumerate(signatures):
        # Stop if we've reached the limit
        if len(matching_transactions) >= limit:
            break
            
        # Get full transaction
        try:
            tx = await solana_client.get_transaction(sig_info["signature"])
            
            # Categorize transaction
            categories = await categorize_transaction(tx, solana_client)
            
            # Check if any target category matches
            if any(category in categories for category in target_categories):
                # Add explanation
                tx["categories"] = categories
                tx["explanation"] = explain_transaction(tx)
                matching_transactions.append(tx)
        except Exception as e:
            # Log error but continue processing other transactions
            print(f"Error processing transaction {sig_info.get('signature')}: {str(e)}")
            continue
            
        processed_count += 1
        
        # Process at most 50 transactions to avoid taking too long
        if processed_count >= 50:
            break
            
        # Add a small delay every 5 transactions to avoid rate limiting
        if i > 0 and i % 5 == 0:
            await asyncio.sleep(0.2)
    
    # Check if no matching transactions were found after processing
    if not matching_transactions:
        return {
            "address": address,
            "query": query,
            "matching_categories": target_categories,
            "transactions_found": 0,
            "message": f"No transactions matching the categories {', '.join(target_categories)} were found"
        }
    
    return {
        "address": address,
        "query": query,
        "matching_categories": target_categories,
        "transactions_found": len(matching_transactions),
        "transactions": matching_transactions
    }


def explain_transaction(tx_data: Dict[str, Any]) -> str:
    """Create a natural language explanation of a transaction.
    
    Args:
        tx_data: Transaction data
        
    Returns:
        Human-readable explanation
    """
    # Extract key information
    slot = tx_data.get("slot", "unknown")
    confirmations = tx_data.get("confirmations", "unknown")
    signature = tx_data.get("transaction", {}).get("signatures", ["unknown"])[0]
    if isinstance(signature, list) and len(signature) > 0:
        signature = signature[0]
    
    # Create basic explanation
    explanation = f"Transaction {signature[:8]}... occurred at slot {slot}"
    if confirmations != "unknown":
        explanation += f" with {confirmations} confirmations"
    
    # Add status
    if tx_data.get("meta", {}).get("err"):
        explanation += ". It failed with an error."
    else:
        explanation += ". It was successful."
    
    # Try to add more specific details based on categories
    if "categories" in tx_data:
        categories = tx_data["categories"]
        
        if "token_transfer" in categories:
            explanation += " This transaction involves token transfers."
        
        if "nft_sale" in categories:
            explanation += " This appears to be an NFT sale transaction."
            
        if "swap" in categories:
            explanation += " This is a token swap transaction."
    
    return explanation


# -------------------------------------------
# Context-Aware Response Formatting
# -------------------------------------------

def format_response(data: Any, format_level: str = "standard") -> Dict[str, Any]:
    """Format a response based on the requested detail level.
    
    Args:
        data: The data to format
        format_level: The format level (minimal, standard, detailed, auto)
        
    Returns:
        Formatted response
    """
    # Handle auto format level
    if format_level == "auto":
        # Simple heuristic - if data is large, use minimal
        if isinstance(data, dict):
            try:
                import json
                serialized = json.dumps(data)
                if len(serialized) > 1000:
                    format_level = "minimal"
                else:
                    format_level = "standard"
            except (TypeError, OverflowError):
                # If data can't be serialized, default to standard format
                format_level = "standard"
        else:
            format_level = "standard"
    
    # Handle different format levels
    if format_level == "minimal":
        return create_minimal_format(data)
    elif format_level == "detailed":
        return create_detailed_format(data)
    else:  # standard
        return data


def create_minimal_format(data: Any) -> Dict[str, Any]:
    """Create a minimal format of the data.
    
    Args:
        data: The data to format
        
    Returns:
        Minimally formatted data
    """
    if not isinstance(data, dict):
        return data
    
    # Extract key information based on data type
    if "lamports" in data and "sol" in data:
        # Balance data
        return {
            "sol": data.get("sol"),
            "formatted": data.get("formatted")
        }
    elif "mint" in data and "supply" in data:
        # Token data
        return {
            "mint": data.get("mint"),
            "name": data.get("metadata", {}).get("name"),
            "symbol": data.get("metadata", {}).get("symbol"),
            "supply": data.get("supply", {}).get("uiAmount")
        }
    elif "transactions" in data and isinstance(data["transactions"], list):
        # Transaction history
        return {
            "address": data.get("address"),
            "transaction_count": len(data.get("transactions", [])),
            "recent_transactions": [tx.get("signature") for tx in data.get("transactions", [])[:5]]
        }
    
    # Default minimal extraction for any data
    return {k: v for k, v in data.items() if k in ["address", "signature", "error"]}


def create_detailed_format(data: Any) -> Dict[str, Any]:
    """Create a detailed format of the data with additional information.
    
    Args:
        data: The data to format
        
    Returns:
        Detailed formatted data with explanations
    """
    if not isinstance(data, dict):
        return {"data": data, "explanation": "Simple value returned"}
    
    # Add explanations based on data type
    if "lamports" in data and "sol" in data:
        # Balance data
        data["explanation"] = "This shows the account balance in both lamports (smallest unit) and SOL."
        data["context"] = {
            "sol_usd_conversion": "Approximate USD value would require current market data."
        }
        return data
    elif "mint" in data and "supply" in data:
        # Token data
        data["explanation"] = "This shows details about a Solana token, including its supply and metadata."
        return data
    elif "transactions" in data and isinstance(data["transactions"], list):
        # Transaction history
        data["explanation"] = f"Transaction history for address {data.get('address')}."
        if len(data.get("transactions", [])) > 0:
            data["recent_transaction_explanation"] = explain_transaction(data["transactions"][0])
        return data
    
    # Default detailed format
    return {
        "data": data,
        "explanation": "Detailed data structure returned from the Solana blockchain."
    }


# -------------------------------------------
# Utility Functions for Semantic Search
# -------------------------------------------

async def get_account_balance(address: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get account balance with formatting.
    
    Args:
        address: Account address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted balance information
    """
    try:
        balance_lamports = await solana_client.get_balance(address)
        balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        
        data = {
            "lamports": balance_lamports,
            "sol": balance_sol,
            "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
        }
        
        return format_response(data, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_account_details(address: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get account details with formatting.
    
    Args:
        address: Account address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted account information
    """
    try:
        account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
        
        # Add additional information
        if account_info:
            account_info["address"] = address
            
            # Add owner program information if available
            if "owner" in account_info:
                owner = account_info["owner"]
                # You can add a mapping of common program IDs if needed
        
        return format_response(account_info, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_token_accounts_for_owner(owner: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get token accounts owned by an address with formatting.
    
    Args:
        owner: Owner address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted token account information
    """
    try:
        token_accounts = await solana_client.get_token_accounts_by_owner(owner)
        
        # Add additional information
        result = {
            "owner": owner,
            "token_accounts": token_accounts,
            "token_count": len(token_accounts)
        }
        
        return format_response(result, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_token_details(mint: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get token details with formatting.
    
    Args:
        mint: Token mint address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted token information
    """
    try:
        # Get token supply
        supply = await solana_client.get_token_supply(mint)
        
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get largest token accounts
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Get market price data if available
        price_data = await solana_client.get_market_price(mint)
        
        # Compile all information
        token_info = {
            "mint": mint,
            "supply": supply,
            "metadata": metadata,
            "largest_accounts": largest_accounts,
            "price_data": price_data
        }
        
        return format_response(token_info, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The mint address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_transaction_history_for_address(
    address: str, 
    solana_client: SolanaClient, 
    limit: int = 20,
    before: str = None,
    format_level: str = "standard"
) -> Dict[str, Any]:
    """Get transaction history for an address with formatting.
    
    Args:
        address: Account address
        solana_client: Solana client
        limit: Maximum number of transactions
        before: Signature to search backwards from
        format_level: Response format level
        
    Returns:
        Formatted transaction history
    """
    try:
        # Get signatures
        # Build options dictionary first
        options = {}
        if before:
            options["before"] = before
        options["limit"] = limit
        
        signatures = await solana_client.get_signatures_for_address(
            address, 
            options
        )
        
        result = {
            "address": address,
            "transactions": signatures
        }
        
        return format_response(result, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_nft_details(mint: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get NFT details with formatting.
    
    Args:
        mint: NFT mint address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted NFT information
    """
    try:
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get token account to find the owner
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Get the current owner if possible
        owner = None
        if largest_accounts and len(largest_accounts) > 0:
            # Get the account with the highest balance
            largest_account = largest_accounts[0]["address"]
            account_info = await solana_client.get_account_info(largest_account, encoding="jsonParsed")
            
            if "parsed" in account_info.get("data", {}):
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data:
                    owner = parsed_data["info"].get("owner")
        
        # Compile NFT information
        nft_info = {
            "mint": mint,
            "metadata": metadata,
            "owner": owner,
            "token_standard": "Unknown"  # In a real implementation, determine if it's NFT/SFT
        }
        
        return format_response(nft_info, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The mint address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"} 