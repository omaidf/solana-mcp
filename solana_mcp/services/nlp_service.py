"""NLP service for Solana MCP.

This module provides natural language processing capabilities for the Solana MCP API.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import json
from datetime import datetime

from solana_mcp.services.base_service import BaseService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.services.account_service import AccountService
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.analysis_service import AnalysisService
from solana_mcp.services.market_service import MarketService
from solana_mcp.clients import SolanaClient
from solana_mcp.utils.decorators import handle_errors
from solana_mcp.utils.errors import SolanaMCPError, DataProcessingError


class NLPService(BaseService):
    """Service for natural language processing of blockchain queries."""
    
    def __init__(
        self,
        solana_client: SolanaClient,
        account_service: AccountService,
        token_service: TokenService,
        transaction_service: TransactionService,
        analysis_service: AnalysisService,
        market_service: MarketService,
        cache_service: Optional[CacheService] = None
    ):
        """Initialize the NLP service.
        
        Args:
            solana_client: The Solana client
            account_service: Service for account operations
            token_service: Service for token operations
            transaction_service: Service for transaction operations
            analysis_service: Service for analysis operations
            market_service: Service for market data operations
            cache_service: Optional cache service
        """
        super().__init__()
        self.client = solana_client
        self.account_service = account_service
        self.token_service = token_service
        self.transaction_service = transaction_service
        self.analysis_service = analysis_service
        self.market_service = market_service
        self.cache = cache_service
        
        # Intent patterns for query classification
        self.intent_patterns = {
            "account_balance": [
                r"(?:balance|how much).+(?:address|account|wallet).+([a-zA-Z0-9]{32,44})",
                r"(?:how much).+(?:sol|token).+([a-zA-Z0-9]{32,44})",
                r"([a-zA-Z0-9]{32,44}).+(?:balance|have)",
            ],
            "token_price": [
                r"(?:price|value|worth).+(?:token|coin).+([a-zA-Z0-9]{32,44})",
                r"([a-zA-Z0-9]{32,44}).+(?:price|cost|worth|value|trading at)",
                r"how much is.+([a-zA-Z0-9]{32,44})",
            ],
            "transaction_info": [
                r"(?:transaction|tx).+([a-zA-Z0-9]{70,100})",
                r"([a-zA-Z0-9]{70,100})",
            ],
            "account_transactions": [
                r"(?:transactions|tx|transfers).+(?:for|from|to).+([a-zA-Z0-9]{32,44})",
                r"([a-zA-Z0-9]{32,44}).+(?:transactions|activity)",
            ],
            "token_holders": [
                r"(?:holders|who holds).+([a-zA-Z0-9]{32,44})",
                r"([a-zA-Z0-9]{32,44}).+(?:holders|owned by)",
            ],
            "market_overview": [
                r"(?:market|overview|sol price)",
                r"(?:what is).+(?:market|overview)",
            ],
            "token_info": [
                r"(?:token|info|about).+([a-zA-Z0-9]{32,44})",
                r"([a-zA-Z0-9]{32,44}).+(?:token|info)",
            ],
            "wallet_profile": [
                r"(?:profile|analyze).+(?:wallet|account).+([a-zA-Z0-9]{32,44})",
                r"([a-zA-Z0-9]{32,44}).+(?:profile|analyze)",
            ],
        }
    
    @handle_errors
    async def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        format_level: str = "auto"
    ) -> Dict[str, Any]:
        """Process a natural language query.
        
        Args:
            query: The natural language query
            session_id: Optional session ID for continuity
            format_level: Response format detail level (minimal, standard, detailed, auto)
            
        Returns:
            Query result
        """
        self.log_with_context(
            "info",
            f"Processing query: {query}",
            session_id=session_id,
            format_level=format_level
        )
        
        # Check cache for identical query
        cache_key = f"nlp_query:{query}:{format_level}"
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Detect intent and extract entities
        intent, entities = self._classify_intent(query)
        
        # If no intent detected, return generic response
        if not intent:
            return {
                "query": query,
                "intent": "unknown",
                "result": "I couldn't understand your query. Please try rephrasing it.",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id or self._generate_session_id()
            }
        
        # Execute query based on intent
        result = await self._execute_intent(intent, entities, format_level)
        
        # Format the response
        response = {
            "query": query,
            "intent": intent,
            "entities": entities,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or self._generate_session_id()
        }
        
        # Cache the result
        if self.cache:
            self.cache.set(cache_key, response, ttl=60)  # 1 minute TTL
            
        return response
    
    def _classify_intent(self, query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Classify the intent of a query and extract entities.
        
        Args:
            query: The natural language query
            
        Returns:
            Tuple of (intent, entities)
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, normalized_query)
                if match:
                    # Extract entities from the match
                    entities = {}
                    if match.groups():
                        if intent in ["account_balance", "account_transactions", "wallet_profile"]:
                            entities["address"] = match.group(1)
                        elif intent in ["token_price", "token_holders", "token_info"]:
                            entities["mint"] = match.group(1)
                        elif intent == "transaction_info":
                            entities["signature"] = match.group(1)
                    
                    # Extract numeric values (limit, days)
                    limit_match = re.search(r"(\d+).+(?:transactions|tx)", normalized_query)
                    if limit_match:
                        entities["limit"] = int(limit_match.group(1))
                        
                    days_match = re.search(r"(\d+).+(?:days|day)", normalized_query)
                    if days_match:
                        entities["days"] = int(days_match.group(1))
                    
                    return intent, entities
        
        # No intent matched
        return None, {}
    
    @handle_errors
    async def _execute_intent(
        self, 
        intent: str, 
        entities: Dict[str, Any],
        format_level: str
    ) -> Any:
        """Execute a query based on its intent and entities.
        
        Args:
            intent: The query intent
            entities: Extracted entities
            format_level: Response format detail level
            
        Returns:
            Query result
        """
        try:
            if intent == "account_balance" and "address" in entities:
                result = await self.account_service.get_account_balance(entities["address"])
                return self._format_result(result, format_level)
                
            elif intent == "token_price" and "mint" in entities:
                result = await self.market_service.get_token_price(entities["mint"])
                return self._format_result(result, format_level)
                
            elif intent == "transaction_info" and "signature" in entities:
                tx = await self.transaction_service.get_transaction(entities["signature"])
                if tx:
                    parsed_tx = await self.transaction_service.parse_transaction(tx)
                    return self._format_result(parsed_tx, format_level)
                return "Transaction not found"
                
            elif intent == "account_transactions" and "address" in entities:
                limit = entities.get("limit", 10)
                result = await self.transaction_service.get_transactions_for_address(
                    entities["address"],
                    limit=limit,
                    parsed_details=format_level != "minimal"
                )
                return self._format_result(result, format_level)
                
            elif intent == "token_holders" and "mint" in entities:
                result = await self.token_service.get_token_holders_batch(
                    entities["mint"],
                    max_accounts=entities.get("limit", 10)
                )
                return self._format_result(result, format_level)
                
            elif intent == "market_overview":
                result = await self.market_service.get_market_overview()
                return self._format_result(result, format_level)
                
            elif intent == "token_info" and "mint" in entities:
                metadata = await self.token_service.get_token_metadata(entities["mint"])
                supply = await self.token_service.get_token_supply(entities["mint"])
                price = await self.market_service.get_token_price(entities["mint"])
                
                result = {
                    "mint": entities["mint"],
                    "metadata": metadata,
                    "supply": supply,
                    "price": price
                }
                return self._format_result(result, format_level)
                
            elif intent == "wallet_profile" and "address" in entities:
                result = await self.analysis_service.wallet_profile(entities["address"])
                return self._format_result(result, format_level)
                
            else:
                return "I understood your query but couldn't find the right data to answer it."
                
        except Exception as e:
            self.logger.error(f"Error executing intent {intent}: {str(e)}", exc_info=True)
            raise DataProcessingError(f"Error processing query: {str(e)}")
    
    def _format_result(self, result: Any, format_level: str) -> Any:
        """Format the result based on the requested format level.
        
        Args:
            result: The raw result
            format_level: Format detail level
            
        Returns:
            Formatted result
        """
        if isinstance(result, str):
            return result
            
        # Determine format level if auto
        if format_level == "auto":
            if isinstance(result, dict) and len(result) <= 5:
                format_level = "detailed"
            elif isinstance(result, dict) and len(result) <= 10:
                format_level = "standard"
            else:
                format_level = "minimal"
        
        # Apply formatting
        if format_level == "minimal":
            # Return only essential information
            if isinstance(result, dict):
                if "address" in result and "sol" in result:
                    # Account balance
                    return f"{result.get('sol', 0)} SOL"
                elif "price_usd" in result:
                    # Token price
                    return f"${result.get('price_usd', 0)}"
                elif "count" in result and "transactions" in result:
                    # Transaction list
                    return f"{result.get('count', 0)} transactions found"
                elif "signature" in result and "status" in result:
                    # Transaction
                    return f"Transaction {result.get('status', 'unknown')}"
            
            # Default minimal format
            return str(result)
            
        elif format_level == "standard":
            # Return moderate detail
            if isinstance(result, dict):
                # Filter to most important fields
                if "transactions" in result and isinstance(result["transactions"], list):
                    # For transaction lists, limit details
                    result["transactions"] = [
                        {k: v for k, v in tx.items() if k in ["signature", "slot", "block_time", "error"]}
                        for tx in result["transactions"][:5]  # Limit to 5 transactions
                    ]
                
                # Remove very detailed or large nested objects
                for key in list(result.keys()):
                    value = result[key]
                    if isinstance(value, dict) and len(value) > 10:
                        # Simplify large nested dictionaries
                        result[key] = {"summary": f"{len(value)} items"}
                    elif isinstance(value, list) and len(value) > 5:
                        # Truncate long lists
                        result[key] = value[:5] + [f"... {len(value) - 5} more items"]
            
            return result
            
        elif format_level == "detailed":
            # Return full detail
            return result
            
        # Default to returning as is
        return result
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID.
        
        Returns:
            Session ID
        """
        import uuid
        return str(uuid.uuid4())
    
    @handle_errors
    async def suggest_queries(self, input_text: str, limit: int = 3) -> List[str]:
        """Suggest related queries based on user input.
        
        Args:
            input_text: The user input text
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        # List of suggestion templates
        suggestions = []
        
        # Check for addresses
        address_match = re.search(r"([a-zA-Z0-9]{32,44})", input_text)
        if address_match:
            address = address_match.group(1)
            suggestions.extend([
                f"What is the balance of {address}?",
                f"Show me recent transactions for {address}",
                f"Generate a profile for wallet {address}"
            ])
        
        # Check for token-related words
        if re.search(r"token|price|nft|coin", input_text.lower()):
            if address_match:
                address = address_match.group(1)
                suggestions.extend([
                    f"What is the price of token {address}?",
                    f"Show top holders of token {address}",
                    f"Get information about token {address}"
                ])
            else:
                suggestions.extend([
                    "Show me the current market overview",
                    "What are the trending tokens?",
                    "Show recent token activity"
                ])
        
        # Check for transaction-related words
        if re.search(r"transaction|tx|transfer", input_text.lower()):
            if address_match:
                address = address_match.group(1)
                suggestions.append(f"Show me transactions for {address}")
            else:
                suggestions.append("Show me recent transactions on Solana")
        
        # Add general suggestions if we have few specific ones
        general_suggestions = [
            "Show me the SOL price",
            "What are the top NFT collections?",
            "Show trending tokens on Solana",
            "What is the current Solana market overview?"
        ]
        
        # Combine and limit suggestions
        all_suggestions = suggestions + general_suggestions
        return all_suggestions[:limit] 