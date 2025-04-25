"""Transaction service for Solana MCP.

This module provides services for working with Solana blockchain transactions.
"""

from typing import Dict, List, Any, Optional

from solana_mcp.services.base_service import BaseService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.decorators import validate_solana_key

class TransactionService(BaseService):
    """Service for working with Solana transactions."""
    
    def __init__(self, solana_client: SolanaClient, cache_service: Optional[CacheService] = None):
        """Initialize the transaction service.
        
        Args:
            solana_client: The Solana client to use
            cache_service: Optional cache service
        """
        super().__init__()
        self.client = solana_client
        self.cache = cache_service
        
    async def get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get a transaction by signature.
        
        Args:
            signature: The transaction signature
            
        Returns:
            Transaction details
        """
        self.log_with_context(
            "info", 
            f"Getting transaction {signature}"
        )
        
        # Use cache if available
        if self.cache:
            cached_tx = self.cache.get(f"tx:{signature}")
            if cached_tx:
                return cached_tx
        
        # Get transaction details
        tx = await self.execute_with_fallback(
            self.client.get_transaction(signature),
            fallback_value=None,
            error_message=f"Error fetching transaction {signature}"
        )
        
        if tx and self.cache:
            # Cache for a long time since transactions are immutable
            self.cache.set(f"tx:{signature}", tx, ttl=3600)
            
        return tx
    
    @validate_solana_key
    async def get_transactions_for_address(
        self,
        address: str,
        limit: int = 20,
        before: Optional[str] = None,
        until: Optional[str] = None,
        parsed_details: bool = True
    ) -> Dict[str, Any]:
        """Get transaction history for an account with detailed parsing.
        
        Args:
            address: The account address
            limit: Maximum number of transactions to return
            before: Signature to search backwards from
            until: Signature to search until (inclusive)
            parsed_details: Whether to include parsed transaction details
            
        Returns:
            Transaction history with details
        """
        self.log_with_context(
            "info", 
            f"Getting detailed transaction history for account {address}",
            limit=limit,
            before=before,
            until=until
        )
        
        # Get signatures
        signatures = await self.client.get_signatures_for_address(
            address,
            before=before,
            until=until,
            limit=limit
        )
        
        if not signatures:
            return {
                "address": address,
                "transactions": [],
                "count": 0
            }
            
        # If we need parsed details, fetch transactions
        transactions = signatures
        if parsed_details and signatures:
            # Get transaction details for each signature
            # This could be optimized to use batching or concurrent requests
            detailed_transactions = []
            
            for sig_info in signatures:
                signature = sig_info.get("signature")
                # Add basic info from signatures endpoint
                tx_info = {
                    "signature": signature,
                    "slot": sig_info.get("slot"),
                    "block_time": sig_info.get("blockTime"),
                    "memo": sig_info.get("memo"),
                    "error": sig_info.get("err")
                }
                
                # Only fetch full transaction if needed (could be expensive)
                if signature:
                    # Use cache to avoid refetching transactions
                    cached_tx = self.cache.get(f"tx:{signature}") if self.cache else None
                    
                    if cached_tx:
                        tx_info["details"] = cached_tx
                    else:
                        # Fetch transaction details
                        tx_details = await self.execute_with_timeout(
                            self.client.get_transaction(signature),
                            timeout=5.0,
                            fallback_value=None,
                            error_message=f"Timeout fetching transaction {signature}"
                        )
                        
                        if tx_details:
                            tx_info["details"] = tx_details
                            # Cache the transaction details
                            if self.cache:
                                self.cache.set(f"tx:{signature}", tx_details, ttl=3600)
                
                detailed_transactions.append(tx_info)
                
            transactions = detailed_transactions
        
        return {
            "address": address,
            "transactions": transactions,
            "count": len(transactions)
        }
        
    async def get_recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transactions from the blockchain.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of recent transactions
        """
        self.log_with_context(
            "info", 
            f"Getting {limit} recent transactions"
        )
        
        # Get recent transaction signatures
        signatures = await self.execute_with_fallback(
            self.client.get_recent_transaction_signatures(limit),
            fallback_value=[],
            error_message=f"Error fetching recent transactions"
        )
        
        if not signatures:
            return []
            
        # Get transaction details
        transactions = []
        for signature in signatures:
            tx = await self.get_transaction(signature)
            if tx:
                transactions.append(tx)
                
        return transactions
    
    async def parse_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a transaction into a more user-friendly format.
        
        Args:
            transaction: The transaction data from the blockchain
            
        Returns:
            Parsed transaction data
        """
        if not transaction:
            return {}
            
        try:
            # Extract basic transaction info
            result = {
                "signature": transaction.get("transaction", {}).get("signatures", [""])[0],
                "slot": transaction.get("slot"),
                "block_time": transaction.get("blockTime"),
                "recent_block_hash": transaction.get("transaction", {}).get("message", {}).get("recentBlockhash"),
                "fee": transaction.get("meta", {}).get("fee"),
                "status": "success" if not transaction.get("meta", {}).get("err") else "failed",
                "error": transaction.get("meta", {}).get("err"),
            }
            
            # Extract instructions if available
            instructions = []
            if "transaction" in transaction and "message" in transaction["transaction"]:
                message = transaction["transaction"]["message"]
                
                # Get accounts
                accounts = message.get("accountKeys", [])
                result["accounts"] = accounts
                
                # Parse instructions
                for idx, instr in enumerate(message.get("instructions", [])):
                    program_id_idx = instr.get("programIdIndex")
                    program_id = accounts[program_id_idx] if program_id_idx is not None and program_id_idx < len(accounts) else None
                    
                    instruction = {
                        "program_id": program_id,
                        "accounts": [accounts[i] for i in instr.get("accounts", []) if i < len(accounts)],
                        "data": instr.get("data"),
                    }
                    
                    # Add some basic instruction type detection
                    if program_id:
                        # Here we could add specific parsing logic based on program IDs
                        # For common programs like Token Program, System Program, etc.
                        instruction["program_name"] = self._get_program_name(program_id)
                    
                    instructions.append(instruction)
            
            result["instructions"] = instructions
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing transaction: {str(e)}", exc_info=True)
            return {
                "signature": transaction.get("transaction", {}).get("signatures", [""])[0] if transaction else "",
                "error": "Error parsing transaction data"
            }
    
    def _get_program_name(self, program_id: str) -> str:
        """Map program IDs to human-readable names.
        
        Args:
            program_id: The program ID
            
        Returns:
            Human-readable program name
        """
        # Common Solana program IDs
        program_map = {
            "11111111111111111111111111111111": "System Program",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "Token Program",
            "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL": "Token Associated Program",
            "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s": "Token Metadata Program",
            # Add more as needed
        }
        
        return program_map.get(program_id, "Unknown Program") 