"""Account service for Solana MCP.

This module provides services for working with Solana accounts.
"""

from typing import Dict, Any, Optional

from solana_mcp.services.base_service import BaseService
from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError
from solana_mcp.utils.decorators import validate_solana_key

class AccountService(BaseService):
    """Service for working with Solana accounts."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the account service.
        
        Args:
            solana_client: The Solana client to use
        """
        super().__init__()
        self.client = solana_client
    
    @validate_solana_key
    async def get_account_balance(self, address: str) -> Dict[str, Any]:
        """Get the balance of a Solana account.
        
        Args:
            address: The account address
            
        Returns:
            Account balance information
        """
        self.log_with_context(
            "info", 
            f"Getting balance for account {address}"
        )
        
        balance_lamports = await self.client.get_balance(address)
        balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        
        return {
            "address": address,
            "lamports": balance_lamports,
            "sol": balance_sol,
            "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
        }
    
    @validate_solana_key
    async def get_account_info(self, address: str, encoding: str = "jsonParsed") -> Dict[str, Any]:
        """Get information about a Solana account.
        
        Args:
            address: The account address
            encoding: The encoding to use for the account data
            
        Returns:
            Account information
        """
        self.log_with_context(
            "info", 
            f"Getting info for account {address}",
            encoding=encoding
        )
        
        account_info = await self.client.get_account_info(address, encoding)
        
        # Add the address to the response for convenience
        if account_info:
            account_info["address"] = address
        
        return account_info
    
    @validate_solana_key
    async def get_transaction_history(
        self,
        address: str,
        limit: int = 20,
        before: Optional[str] = None,
        until: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get transaction history for an account.
        
        Args:
            address: The account address
            limit: Maximum number of transactions to return
            before: Signature to search backwards from
            until: Signature to search until (inclusive)
            
        Returns:
            Transaction history
        """
        self.log_with_context(
            "info", 
            f"Getting transaction history for account {address}",
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
        
        return {
            "address": address,
            "transactions": signatures,
            "count": len(signatures)
        } 