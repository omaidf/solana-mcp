"""
RPC service for Solana blockchain communication.

This module provides a service for making RPC calls to the Solana blockchain,
with proper error handling, retries, and timeout management.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from solana_mcp.services.base_service import BaseService
from solana_mcp.utils.error_handling import (
    handle_errors,
    ConnectionError,
    RPCError,
    ValidationError,
    SolanaMCPError
)
from solana_mcp.utils.config import get_settings

# Constants for Solana
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

# Configure logger
logger = logging.getLogger(__name__)

class RPCService(BaseService):
    """Service for making RPC requests to the Solana blockchain."""
    
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RPC service.
        
        Args:
            rpc_url: URL of the Solana RPC endpoint
            timeout: Default timeout for operations in seconds
            max_retries: Maximum number of retry attempts for failed requests
            logger: Optional logger instance
        """
        super().__init__(timeout=timeout, logger=logger)
        settings = get_settings()
        self.rpc_url = rpc_url or settings.SOLANA_RPC_URL
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=timeout)
        self.logger.info(f"RPCService initialized with endpoint {self.rpc_url}")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        self.logger.info("RPCService closed")
    
    @handle_errors(retries=3, retry_exceptions=[ConnectionError, httpx.RequestError], retry_delay=1.0)
    async def make_request(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Make an RPC request to the Solana API.
        
        Args:
            method: RPC method name
            params: RPC method parameters
            
        Returns:
            RPC response data
            
        Raises:
            ConnectionError: If there is a connection error
            RPCError: If the RPC request fails
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            async with self.log_timing(f"rpc_request.{method}"):
                response = await self.client.post(
                    self.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                # Raise for HTTP status errors
                response.raise_for_status()
                
                # Parse JSON response
                result = response.json()
                
                # Handle RPC errors
                if "error" in result:
                    error = result["error"]
                    code = error.get("code", 0)
                    message = error.get("message", "Unknown error")
                    data = error.get("data")
                    
                    # Raise standardized RPCError
                    raise RPCError(method, code, message, data)
                
                # Success case - return the result
                return result["result"]
                
        except httpx.HTTPStatusError as e:
            raise RPCError(
                method, 
                e.response.status_code, 
                f"HTTP error: {e.response.status_code}", 
                {"response": e.response.text}
            )
            
        except httpx.RequestError as e:
            raise ConnectionError(f"Connection error during RPC request to {method}: {str(e)}")
            
        except RPCError:
            # Pass through our custom RPCError
            raise
            
        except Exception as e:
            raise RPCError(
                method, 
                -1, 
                f"Unexpected error: {str(e)}"
            )
    
    # Solana RPC methods
    @handle_errors(retries=2)
    async def get_account_info(self, account: str, encoding: str = "base64") -> Dict[str, Any]:
        """
        Get account information.
        
        Args:
            account: The account public key
            encoding: The encoding for the account data
            
        Returns:
            Account information
        """
        if not self._validate_public_key(account):
            raise ValidationError(f"Invalid account address: {account}")
            
        params = [
            account,
            {"encoding": encoding}
        ]
        
        return await self.make_request("getAccountInfo", params)
    
    @handle_errors(retries=2)
    async def get_balance(self, account: str) -> int:
        """
        Get account balance.
        
        Args:
            account: The account public key
            
        Returns:
            Account balance in lamports
        """
        if not self._validate_public_key(account):
            raise ValidationError(f"Invalid account address: {account}")
            
        params = [account]
        result = await self.make_request("getBalance", params)
        return result["value"]
    
    @handle_errors(retries=2)
    async def get_token_accounts_by_owner(
        self, 
        owner: str, 
        mint: Optional[str] = None,
        program_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get token accounts by owner.
        
        Args:
            owner: The owner public key
            mint: Filter by mint. Defaults to None.
            program_id: Filter by token program ID. Defaults to None.
            
        Returns:
            List of token accounts
        """
        if not self._validate_public_key(owner):
            raise ValidationError(f"Invalid owner address: {owner}")
            
        if mint and not self._validate_public_key(mint):
            raise ValidationError(f"Invalid mint address: {mint}")
            
        if program_id and not self._validate_public_key(program_id):
            raise ValidationError(f"Invalid program ID: {program_id}")
            
        if mint:
            params = [owner, {"mint": mint}]
        elif program_id:
            params = [owner, {"programId": program_id}]
        else:
            # Default to SPL Token program
            params = [
                owner, 
                {"programId": TOKEN_PROGRAM_ID}
            ]
        
        params.append({"encoding": "jsonParsed"})
        result = await self.make_request("getTokenAccountsByOwner", params)
        return result["value"]
    
    @handle_errors(retries=2)
    async def get_token_supply(self, mint: str) -> Dict[str, Any]:
        """
        Get token supply.
        
        Args:
            mint: The token mint address
            
        Returns:
            Token supply information
        """
        if not self._validate_public_key(mint):
            raise ValidationError(f"Invalid mint address: {mint}")
        
        params = [mint, {"commitment": "confirmed"}]
        result = await self.make_request("getTokenSupply", params)
        return result["value"]
    
    @handle_errors(retries=2)
    async def get_token_largest_accounts(self, mint: str) -> List[Dict[str, Any]]:
        """
        Get largest token accounts by token mint.
        
        Args:
            mint: The token mint address
            
        Returns:
            List of largest token accounts
        """
        if not self._validate_public_key(mint):
            raise ValidationError(f"Invalid mint address: {mint}")
            
        params = [mint]
        result = await self.make_request("getTokenLargestAccounts", params)
        return result["value"]
    
    @handle_errors(retries=2)
    async def get_token_metadata(self, mint: str) -> Dict[str, Any]:
        """
        Get token metadata using the Metaplex standard.
        
        Args:
            mint: The token mint address
            
        Returns:
            Token metadata
        """
        if not self._validate_public_key(mint):
            raise ValidationError(f"Invalid mint address: {mint}")
        
        # Find the metadata PDA for this mint
        metadata_address = self._find_metadata_pda(mint)
        
        # Get the metadata account info
        account_info = await self.get_account_info(metadata_address, encoding="base64")
        
        if not account_info["value"]:
            raise ValidationError(f"No metadata found for mint: {mint}")
            
        # Here you would normally parse the metadata from the account data
        # This is a simplified version that just returns the raw data
        return {
            "address": metadata_address,
            "raw_data": account_info["value"]
        }
    
    @handle_errors(retries=2)
    async def get_program_accounts(
        self, 
        program_id: str, 
        filters: Optional[List[Dict[str, Any]]] = None,
        encoding: str = "base64",
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get program accounts.
        
        Args:
            program_id: The program ID
            filters: Optional filters
            encoding: The encoding for the account data
            limit: Optional result limit
            offset: Optional result offset
            
        Returns:
            List of program accounts
        """
        if not self._validate_public_key(program_id):
            raise ValidationError(f"Invalid program ID: {program_id}")
            
        config: Dict[str, Any] = {"encoding": encoding}
        
        if filters:
            config["filters"] = filters
            
        if limit is not None:
            config["limit"] = limit
            
        if offset is not None:
            config["offset"] = offset
            
        params = [program_id, config]
        result = await self.make_request("getProgramAccounts", params)
        return result
    
    @handle_errors(retries=2)
    async def get_signatures_for_address(
        self, 
        address: str, 
        before: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get transaction signatures for an address.
        
        Args:
            address: The account address
            before: Signature to search backwards from
            limit: Maximum number of signatures to return
            
        Returns:
            List of transaction signatures
        """
        if not self._validate_public_key(address):
            raise ValidationError(f"Invalid address: {address}")
            
        config: Dict[str, Any] = {"limit": limit}
        
        if before:
            config["before"] = before
            
        params = [address, config]
        return await self.make_request("getSignaturesForAddress", params)
    
    @handle_errors(retries=2)
    async def get_transaction(self, signature: str) -> Dict[str, Any]:
        """
        Get transaction details.
        
        Args:
            signature: The transaction signature
            
        Returns:
            Transaction details
        """
        # Validate signature format (base58 encoded string)
        if not signature or not isinstance(signature, str) or len(signature) < 32:
            raise ValidationError(f"Invalid transaction signature: {signature}")
            
        params = [
            signature, 
            {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
        ]
        
        return await self.make_request("getTransaction", params)
    
    def _validate_public_key(self, pubkey: str) -> bool:
        """
        Validate a Solana public key.
        
        Args:
            pubkey: The public key to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - should be a string of the right length with valid characters
        if not pubkey or not isinstance(pubkey, str) or len(pubkey) < 32:
            return False
            
        # Check if it contains only valid base58 characters
        return all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in pubkey)
    
    def _find_metadata_pda(self, mint_address: str) -> str:
        """
        Find the metadata PDA for a given mint address.
        
        Args:
            mint_address: The mint address to find metadata for
            
        Returns:
            Metadata PDA address
            
        Note:
            This is a simplified version that doesn't actually compute the PDA.
            In a real implementation, you would derive this using the Metaplex SDK.
        """
        # This is a placeholder - in a real implementation, you would derive the PDA
        # using the appropriate seeds and the Metaplex program ID
        return f"metadata_for_{mint_address}"
    
    @handle_errors(retries=2)
    async def get_block(self, slot: int) -> Dict[str, Any]:
        """
        Get block information.
        
        Args:
            slot: The slot number
            
        Returns:
            Block information
        """
        if not isinstance(slot, int) or slot < 0:
            raise ValidationError(f"Invalid slot number: {slot}")
            
        params = [
            slot, 
            {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
        ]
        
        return await self.make_request("getBlock", params)
    
    @handle_errors(retries=2)
    async def get_blocks(self, start_slot: int, end_slot: Optional[int] = None) -> List[int]:
        """
        Get a list of confirmed blocks.
        
        Args:
            start_slot: Start slot
            end_slot: End slot (optional)
            
        Returns:
            List of block slot numbers
        """
        if not isinstance(start_slot, int) or start_slot < 0:
            raise ValidationError(f"Invalid start slot: {start_slot}")
            
        if end_slot is not None and (not isinstance(end_slot, int) or end_slot < start_slot):
            raise ValidationError(f"Invalid end slot: {end_slot}")
            
        params = [start_slot]
        if end_slot is not None:
            params.append(end_slot)
            
        return await self.make_request("getBlocks", params)


# Singleton accessor for the RPCService
def get_rpc_service(
    rpc_url: Optional[str] = None,
    timeout: float = 30.0
) -> RPCService:
    """
    Get a singleton instance of the RPCService.
    
    This function will be replaced by the dependency injection system.
    
    Args:
        rpc_url: Optional RPC URL override
        timeout: Optional timeout override
        
    Returns:
        Singleton RPCService instance
    """
    from solana_mcp.utils.dependency_injection import ServiceProvider
    provider = ServiceProvider.get_instance()
    
    # Check if we already have a service instance
    try:
        service = provider.get(RPCService)
    except KeyError:
        # Create and register a new service instance
        service = RPCService(rpc_url=rpc_url, timeout=timeout)
        provider.register(RPCService, service)
    
    return service 