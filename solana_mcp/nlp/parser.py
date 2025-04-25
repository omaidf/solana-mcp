"""Query parsing logic for natural language processing."""

import re
import logging
from typing import Dict, List, Any, Optional

from solana_mcp.solana_client import SolanaClient
from solana_mcp.nlp.patterns import QUERY_PATTERNS, TRANSACTION_SEARCH_PATTERNS
from solana_mcp.nlp.formatter import format_response

logger = logging.getLogger(__name__)

async def parse_natural_language_query(query: str, 
                                     solana_client: SolanaClient, 
                                     session = None) -> Dict[str, Any]:
    """Parse a natural language query into an API call.
    
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
    
    # Check for semantic transaction search patterns
    for pattern in TRANSACTION_SEARCH_PATTERNS:
        match = re.search(pattern, normalized_query, re.IGNORECASE)
        if match:
            # Extract address and search terms
            groups = match.groups()
            if len(groups) >= 2:
                # For the first pattern
                address = groups[0]
                search_terms = groups[1]
                
                # Check if the pattern is reversed (second pattern)
                if not re.match(r"[a-zA-Z0-9]{32,44}", address):
                    search_terms = groups[0]
                    address = groups[1]
                
                # Perform semantic search
                from solana_mcp.semantic_search import semantic_transaction_search
                result = await semantic_transaction_search(address, search_terms, solana_client)
                
                # Update session if provided
                if session:
                    session.add_query(query, result)
                    session.update_context_for_entity("address", address, {"last_search": search_terms})
                
                return result
    
    # Try to match against patterns
    for pattern_info in QUERY_PATTERNS:
        try:
            match = re.search(pattern_info["pattern"], normalized_query)
            if match:
                intent = pattern_info["intent"]
                params = pattern_info["params"](match)
                
                # Execute the intent
                result = await execute_intent(intent, params, solana_client)
                
                # Update session if provided
                if session:
                    session.add_query(query, result)
                    
                    # Update context based on the executed intent
                    if intent == "get_balance" and "address" in params:
                        session.update_context_for_entity("address", params["address"], 
                                                        {"last_balance_check": True})
                    elif intent == "get_account_info" and "address" in params:
                        session.update_context_for_entity("address", params["address"], 
                                                        {"last_info_check": True})
                    elif intent == "get_token_accounts" and "owner" in params:
                        session.update_context_for_entity("address", params["owner"], 
                                                        {"last_token_check": True})
                    elif intent == "get_token_info" and "mint" in params:
                        session.update_context_for_entity("token", params["mint"], 
                                                        {"last_check": True})
                    elif intent == "get_token_whales" and "mint" in params:
                        session.update_context_for_entity("token", params["mint"], 
                                                        {"last_whale_check": True})
                    elif intent == "get_fresh_wallets" and "mint" in params:
                        session.update_context_for_entity("token", params["mint"], 
                                                        {"last_fresh_wallet_check": True})
                    elif intent == "get_transactions" and "address" in params:
                        session.update_context_for_entity("address", params["address"], 
                                                        {"last_tx_check": True})
                    elif intent == "get_nft_info" and "mint" in params:
                        session.update_context_for_entity("nft", params["mint"], 
                                                        {"last_check": True})
                
                return result
        except Exception as e:
            # Handle parsing errors for each pattern
            logger.error(f"Error parsing query with pattern {pattern_info['pattern']}: {str(e)}")
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
            "Get NFT info [mint]",
            "Find whales for token [mint]",
            "Find new/fresh wallets for token [mint]"
        ]
    }


async def execute_intent(intent: str, params: Dict[str, Any], solana_client: SolanaClient) -> Dict[str, Any]:
    """Execute a specific intent with the given parameters.
    
    Args:
        intent: The intent to execute
        params: The parameters for the intent
        solana_client: The Solana client
        
    Returns:
        The result of the intent execution
    """
    # Import here to avoid circular imports
    from solana_mcp.semantic_search import (
        get_account_balance, get_account_details, get_token_accounts_for_owner,
        get_token_details, get_transaction_history_for_address, get_nft_details
    )
    from solana_mcp.services.whale_detector.detector import detect_whale_wallets
    from solana_mcp.services.fresh_wallet.detector import detect_fresh_wallets
    
    if intent == "get_balance":
        if not params.get("address"):
            return {"error": "Missing address", "error_explanation": "A valid Solana address is required."}
            
        return await get_account_balance(params["address"], solana_client, format_level="auto")
    
    elif intent == "get_account_info":
        if not params.get("address"):
            return {"error": "Missing address", "error_explanation": "A valid Solana address is required."}
            
        return await get_account_details(params["address"], solana_client, format_level="auto")
    
    elif intent == "get_token_accounts":
        if not params.get("owner"):
            return {"error": "Missing owner address", "error_explanation": "A valid Solana address is required."}
            
        return await get_token_accounts_for_owner(params["owner"], solana_client, format_level="auto")
    
    elif intent == "get_token_info":
        if not params.get("mint"):
            return {"error": "Missing token mint address", "error_explanation": "A valid token mint address is required."}
            
        return await get_token_details(params["mint"], solana_client, format_level="auto")
    
    elif intent == "get_token_whales":
        if not params.get("mint"):
            return {"error": "Missing token mint address", "error_explanation": "A valid token mint address is required."}
        
        # Use our whale detector service
        return await detect_whale_wallets(params["mint"], solana_client)
    
    elif intent == "get_fresh_wallets":
        if not params.get("mint"):
            return {"error": "Missing token mint address", "error_explanation": "A valid token mint address is required."}
        
        # Use our fresh wallet detector service
        return await detect_fresh_wallets(params["mint"], solana_client)
    
    elif intent == "get_transactions":
        if not params.get("address"):
            return {"error": "Missing address", "error_explanation": "A valid Solana address is required."}
            
        return await get_transaction_history_for_address(
            params["address"], 
            solana_client, 
            limit=params.get("limit", 20),
            format_level="auto"
        )
    
    elif intent == "get_nft_info":
        if not params.get("mint"):
            return {"error": "Missing NFT mint address", "error_explanation": "A valid NFT mint address is required."}
            
        return await get_nft_details(params["mint"], solana_client, format_level="auto")
    
    else:
        return {"error": f"Unknown intent: {intent}", "error_explanation": "The requested operation is not supported."} 