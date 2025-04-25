"""Account-related Solana RPC client operations.

This module provides specialized client functionality for Solana account operations.
"""

from typing import Dict, List, Any, Optional

from solana_mcp.clients.base_client import BaseSolanaClient, InvalidPublicKeyError, validate_public_key

class AccountClient(BaseSolanaClient):
    """Client for Solana account operations."""
    
    async def get_account_info(self, account: str, encoding: str = "base64") -> Dict[str, Any]:
        """Get account information.
        
        Args:
            account: The account public key
            encoding: The encoding for the account data
            
        Returns:
            Account information
            
        Raises:
            InvalidPublicKeyError: If the account is not a valid Solana public key
        """
        if not validate_public_key(account):
            raise InvalidPublicKeyError(account)
            
        return await self._make_request(
            "getAccountInfo", 
            [account, {"encoding": encoding}]
        )
    
    async def get_balance(self, account: str) -> int:
        """Get account balance.
        
        Args:
            account: The account public key
            
        Returns:
            Account balance in lamports
            
        Raises:
            InvalidPublicKeyError: If the account is not a valid Solana public key
        """
        if not validate_public_key(account):
            raise InvalidPublicKeyError(account)
            
        return await self._make_request("getBalance", [account])
    
    async def get_program_accounts(
        self, 
        program_id: str, 
        filters: Optional[List[Dict[str, Any]]] = None,
        encoding: str = "base64",
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all accounts owned by a program.
        
        Args:
            program_id: The program ID
            filters: Optional filters to apply
            encoding: The encoding for the account data
            limit: Maximum number of accounts to return
            offset: Offset to start from
            
        Returns:
            List of program accounts
            
        Raises:
            InvalidPublicKeyError: If the program_id is not a valid Solana public key
        """
        if not validate_public_key(program_id):
            raise InvalidPublicKeyError(program_id)
            
        params = [program_id, {"encoding": encoding}]
        if filters:
            params[1]["filters"] = filters
            
        # Add pagination if specified
        if limit is not None:
            params[1]["limit"] = limit
        if offset is not None:
            params[1]["offset"] = offset
        
        return await self._make_request("getProgramAccounts", params)
    
    async def get_signatures_for_address(
        self,
        address: str,
        before: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get transaction signatures for an address.
        
        Args:
            address: The account address
            before: Signature to start searching backwards from
            until: Signature to search until (inclusive)
            limit: Maximum number of signatures to return
            
        Returns:
            List of transaction signatures
            
        Raises:
            InvalidPublicKeyError: If the address is not a valid Solana public key
        """
        if not validate_public_key(address):
            raise InvalidPublicKeyError(address)
            
        options: Dict[str, Any] = {"limit": limit}
        if before:
            options["before"] = before
        if until:
            options["until"] = until
            
        return await self._make_request(
            "getSignaturesForAddress",
            [address, options]
        ) 