"""
RPC service for Solana blockchain communication.

This module provides a service for making RPC calls to the Solana blockchain,
with proper error handling, retries, and timeout management.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

import httpx

from solana_mcp.services.base_service import BaseService, handle_errors
from solana_mcp.utils.errors import (
    RpcConnectionError, 
    RpcError, 
    ValidationError,
    ResourceNotFoundError
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
    
    @handle_errors()
    async def make_request(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Make an RPC request to the Solana API.
        
        Args:
            method: RPC method name
            params: RPC method parameters
            
        Returns:
            RPC response data
            
        Raises:
            RpcConnectionError: If there is a connection error
            RpcError: If the RPC request fails
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        # Retry parameters
        initial_retry_delay = 1.0  # starting delay in seconds
        max_retry_delay = 10.0  # maximum delay in seconds
        retriable_status_codes = {408, 429, 500, 502, 503, 504}
        
        for retry_count in range(self.max_retries + 1):
            try:
                async with self.log_timing(f"rpc_request.{method}"):
                    # Log the attempt if it's a retry
                    if retry_count > 0:
                        self.logger.info(f"Retry attempt {retry_count}/{self.max_retries} for {method}")
                    
                    response = await self.client.post(
                        self.rpc_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    # Handle HTTP status errors
                    if response.status_code in retriable_status_codes and retry_count < self.max_retries:
                        wait_time = min(
                            initial_retry_delay * (2 ** retry_count), 
                            max_retry_delay
                        )
                        self.logger.warning(f"HTTP status {response.status_code}, retrying in {wait_time}s: {method}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Raise for other HTTP status errors
                    response.raise_for_status()
                    
                    # Parse JSON response
                    result = response.json()
                    
                    # Handle RPC errors
                    if "error" in result:
                        error = result["error"]
                        message = f"Solana RPC error: {error.get('message', 'Unknown error')}"
                        if "data" in error:
                            message += f" - {json.dumps(error['data'])}"
                        
                        # Check for rate limiting errors and retry if possible
                        if ("rate limited" in message.lower() or 
                            error.get("code") == -32005) and retry_count < self.max_retries:
                            wait_time = min(
                                initial_retry_delay * (2 ** retry_count), 
                                max_retry_delay
                            )
                            self.logger.warning(f"Rate limited, retrying in {wait_time}s: {method}")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # Non-retriable RPC error or max retries reached
                        raise RpcError(
                            message=message,
                            details={"method": method, "error": error}
                        )
                    
                    # Success case - return the result
                    return result["result"]
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code in retriable_status_codes and retry_count < self.max_retries:
                    wait_time = min(
                        initial_retry_delay * (2 ** retry_count), 
                        max_retry_delay
                    )
                    self.logger.warning(f"Request failed with status {e.response.status_code}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                
                raise RpcError(
                    message=f"RPC request failed with status {e.response.status_code}",
                    details={"method": method, "status_code": e.response.status_code}
                )
                
            except httpx.RequestError as e:
                if retry_count < self.max_retries:
                    wait_time = min(
                        initial_retry_delay * (2 ** retry_count), 
                        max_retry_delay
                    )
                    self.logger.warning(f"Connection error, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                
                raise RpcConnectionError(
                    message=f"Connection error during RPC request: {str(e)}",
                    details={"method": method, "error": str(e)}
                )
                
            except Exception as e:
                if retry_count < self.max_retries:
                    wait_time = min(
                        initial_retry_delay * (2 ** retry_count), 
                        max_retry_delay
                    )
                    self.logger.warning(f"Unexpected error, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                
                raise RpcError(
                    message=f"Unexpected error during RPC request: {str(e)}",
                    details={"method": method, "error": str(e)}
                )
        
        # We should never reach here, but just in case
        raise RpcError(
            message=f"RPC request failed after {self.max_retries} retries",
            details={"method": method}
        )
    
    # Solana RPC methods
    @handle_errors()
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
            
        return await self.make_request(
            "getAccountInfo", 
            [account, {"encoding": encoding}]
        )
    
    @handle_errors()
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
            
        return await self.make_request("getBalance", [account])
    
    @handle_errors()
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
        return await self.make_request("getTokenAccountsByOwner", params)
    
    @handle_errors()
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
            
        return await self.make_request("getTokenSupply", [mint])
    
    @handle_errors()
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
            
        return await self.make_request("getTokenLargestAccounts", [mint])
    
    @handle_errors()
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
        
        # This is a simplified implementation that would be replaced with actual Metaplex 
        # metadata account lookup and parsing in a real implementation
        
        # First try to get the metadata account PDA
        metadata_pda = self._find_metadata_pda(mint)
        
        try:
            # Get the metadata account
            metadata_account = await self.get_account_info(metadata_pda, encoding="base64")
            
            # In a real implementation, we would parse the metadata account data
            # Here we'll just return a placeholder
            return {
                "name": f"Token {mint[:8]}",
                "symbol": mint[:4].upper(),
                "uri": f"https://metadata.example.com/{mint}",
                "mint": mint,
                "is_nft": False
            }
        except (RpcError, ResourceNotFoundError):
            # If metadata account not found, return minimal info
            return {
                "name": f"Unknown Token {mint[:8]}",
                "symbol": "UNKN",
                "mint": mint,
                "is_nft": False
            }
    
    @handle_errors()
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
            
        config = {"encoding": encoding}
        if filters:
            config["filters"] = filters
        
        # Implement pagination for large result sets if needed
        if limit is not None and limit > 0:
            config["limit"] = limit
        if offset is not None and offset >= 0:
            config["offset"] = offset
            
        return await self.make_request("getProgramAccounts", [program_id, config])
    
    @handle_errors()
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
            
        config = {"limit": limit}
        if before:
            config["before"] = before
            
        return await self.make_request("getSignaturesForAddress", [address, config])
    
    @handle_errors()
    async def get_transaction(self, signature: str) -> Dict[str, Any]:
        """
        Get transaction details.
        
        Args:
            signature: The transaction signature
            
        Returns:
            Transaction details
        """
        return await self.make_request(
            "getTransaction", 
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
        )
    
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
        
        This is a simplified implementation. In a real system, this would
        compute the actual PDA using the Metaplex metadata program.
        
        Args:
            mint_address: The token's mint address
            
        Returns:
            The metadata account address
        """
        # In a real implementation, you would compute this using:
        # PublicKey.findProgramAddress(
        #     [Buffer.from('metadata'), METADATA_PROGRAM_ID.toBytes(), mint.toBytes()],
        #     METADATA_PROGRAM_ID
        # )
        
        # This is a placeholder that just appends 'metadata' to the mint address
        # In a real implementation, use the proper PDA derivation
        return f"{mint_address}_metadata"


@lru_cache()
def get_rpc_service() -> RPCService:
    """
    Get or create an RPCService instance.
    
    Returns:
        A singleton instance of RPCService
    """
    settings = get_settings()
    return RPCService(
        rpc_url=settings.SOLANA_RPC_URL,
        timeout=settings.RPC_TIMEOUT
    ) 