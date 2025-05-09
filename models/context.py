"""
Model Context generation for Solana blockchain data
"""
import time
import uuid
import json
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from core.solana import SolanaClient

# Setup logger
logger = logging.getLogger(__name__)

class ModelContext:
    """Model Context for Solana blockchain data processing"""
    
    # Available model types
    AVAILABLE_MODELS = [
        "transaction-history",
        "token-holdings"
    ]
    
    def __init__(self, model_type: str, network: str = "mainnet"):
        """Initialize model context generator"""
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Available models: {', '.join(self.AVAILABLE_MODELS)}")
            
        self.model_type = model_type
        self.network = network
        self._solana_client = None
        
    async def get_solana_client(self) -> SolanaClient:
        """
        Get or create a SolanaClient instance with proper lifecycle management
        
        Returns:
            SolanaClient: A properly initialized Solana client
        """
        if self._solana_client is None:
            self._solana_client = SolanaClient(network=self.network)
            await self._solana_client.__aenter__()
        return self._solana_client
        
    async def close(self):
        """Close resources when done"""
        if self._solana_client is not None:
            await self._solana_client.__aexit__(None, None, None)
            self._solana_client = None
            
    async def __aenter__(self):
        """Async context manager entry"""
        self._in_context_manager = True
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self._in_context_manager = False
        await self.close()
        
    async def generate(self, address: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate model context for a given address
        
        Args:
            address: The Solana address to generate context for
            parameters: Optional parameters to customize the context generation
            
        Returns:
            Dict[str, Any]: The generated context data with metadata
            
        Raises:
            ValueError: If the model type is unsupported or if context generation fails
        """
        if parameters is None:
            parameters = {}
            
        # Generate context ID
        context_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        client_created_here = False
        try:
            # Generate data based on model type
            if self.model_type == "transaction-history":
                data = await self._generate_transaction_history(address, parameters)
            elif self.model_type == "token-holdings":
                data = await self._generate_token_holdings(address, parameters)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            # Compile response
            response = {
                "context_id": context_id,
                "address": address,
                "model_type": self.model_type,
                "timestamp": timestamp,
                "data": data
            }
            
            return response
        except Exception as e:
            # Re-raise with context
            raise ValueError(f"Context generation failed: {str(e)}") from e
        finally:
            # Only close the client if we're not in a context manager
            # If this ModelContext instance is used with async with, let __aexit__ handle it
            if not hasattr(self, "_in_context_manager") or not self._in_context_manager:
                await self.close()
        
    async def _generate_transaction_history(self, address: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transaction history context for an address"""
        try:
            # Get account info
            account_info = await self.get_solana_client().get_account_info(address)
            
            # Get recent transactions
            # Note: For a complete implementation, you would need to use a
            # block explorer API or maintain your own transaction index
            # This is a simplified placeholder
            
            # Get basic account balance
            balance = await self.get_solana_client().get_balance(address)
            
            context = {
                "summary": {
                    "address": address,
                    "balance_sol": balance.get("balance_sol", 0),
                    "account_age_days": 0,  # Placeholder - would need block data
                    "transaction_count": 0  # Placeholder - would need indexer
                },
                "recent_transactions": [],
                "activity_metrics": {
                    "daily_average_txs": 0,
                    "weekly_volume_sol": 0,
                    "common_interactions": []
                }
            }
            
            return context
        except Exception as e:
            logger.exception(f"Error generating transaction history for {address}")
            raise ValueError(f"Failed to generate transaction history: {str(e)}")
        
    async def _generate_token_holdings(self, address: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate token holdings context for an address"""
        try:
            # Get SOL balance
            balance = await self.get_solana_client().get_balance(address)
            
            # Get token accounts with proper decimal handling
            token_accounts = await self.get_solana_client().get_token_accounts(address)
            
            # Process token holdings
            tokens = []
            estimated_value_usd = 0

            for account in token_accounts:
                # Check if account has necessary data
                if "mint" not in account or not account["mint"]:
                    logger.warning(f"Skipping token account with missing mint: {account.get('address', 'unknown')}")
                    continue
                
                # Use ui_amount which has proper decimal handling already
                token_amount = account.get("ui_amount", 0)
                
                # Only include tokens with non-zero balance
                if token_amount > 0:
                    token_data = {
                        "mint": account.get("mint"),
                        "amount": token_amount,
                        "raw_amount": account.get("amount", 0),
                        "token_address": account.get("address"),
                        "decimals": account.get("decimals", 0),
                        "symbol": "UNKNOWN",  # Would require token metadata
                        "name": f"Token {account.get('mint', '')[0:8]}..."  # Would require token metadata
                    }
                    
                    # Add token data to the list
                    tokens.append(token_data)
                
            # Sort by amount (largest first)
            tokens.sort(key=lambda x: x["amount"], reverse=True)
                
            context = {
                "sol_balance": balance.get("balance_sol", 0),
                "token_count": len(tokens),
                "tokens": tokens,
                "estimated_value_usd": estimated_value_usd  # Would require price data
            }
            
            return context
        except Exception as e:
            logger.exception(f"Error generating token holdings for {address}")
            raise ValueError(f"Failed to generate token holdings: {str(e)}") 