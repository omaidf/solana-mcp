"""Async Solana client for MCP server."""

# Standard library imports
import base64
import json
import re
import asyncio
import time
from typing import Any, Dict, List, Optional, Union, cast, Tuple
from contextlib import asynccontextmanager
from urllib.parse import urljoin

# Third-party library imports
import httpx
from cachetools import TTLCache, cached

# Internal imports
from solana_mcp.config import SolanaConfig, get_solana_config
from solana_mcp.logging_config import get_logger, log_with_context

# Get logger
logger = get_logger(__name__)

# Solana public key validation pattern (base58 format)
PUBKEY_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

# Default SPL Token Program ID
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

# Metaplex Token Metadata Program ID
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

# Jupiter Aggregator Program ID for DEX data
JUPITER_PROGRAM_ID = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"


def validate_public_key(pubkey: str) -> bool:
    """Validate if a string is a properly formatted Solana public key.
    
    Args:
        pubkey: The public key to validate
        
    Returns:
        True if the public key is valid, False otherwise
    """
    return bool(PUBKEY_PATTERN.match(pubkey))


class SolanaRpcError(Exception):
    """Exception raised for Solana RPC errors."""
    
    def __init__(self, message: str, error_data: Dict[str, Any]):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_data: Error data from the RPC response
        """
        super().__init__(message)
        self.error_data = error_data


class InvalidPublicKeyError(ValueError):
    """Exception raised for invalid Solana public keys."""
    
    def __init__(self, pubkey: str):
        """Initialize the exception.
        
        Args:
            pubkey: The invalid public key
        """
        super().__init__(f"Invalid Solana public key format: {pubkey}")
        self.pubkey = pubkey


class SolanaClient:
    """Async client for interacting with Solana RPC endpoints."""
    
    def __init__(self, config: SolanaConfig = None):
        """Initialize the Solana client.
        
        Args:
            config: Solana configuration. Defaults to environment-based config.
        """
        self.config = config or get_solana_config()
        self.headers = {"Content-Type": "application/json"}
        
        # Set up auth if provided
        self.auth = None
        if self.config.rpc_user and self.config.rpc_password:
            self.auth = (self.config.rpc_user, self.config.rpc_password)
        
        # Shared HTTP client for better performance
        self._http_client = None
    
    async def _make_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make a JSON-RPC request to the Solana node.
        
        Args:
            method: The RPC method to call
            params: The parameters to pass to the method
            
        Returns:
            The JSON-RPC response
        """
        if params is None:
            params = []
            
        # Add commitment if not already in params
        if method not in ["getHealth", "getVersion"]:
            # Check if any param is a dict with 'commitment' key
            has_commitment = any(
                isinstance(p, dict) and "commitment" in p 
                for p in params
            )
            
            # If no commitment in params, add it as the last param
            if not has_commitment:
                # Some methods take commitment as part of an options object
                if method in [
                    "getAccountInfo", "getBalance", "getBlockHeight",
                    "getConfirmedBlock", "getConfirmedTransaction",
                    "getProgramAccounts", "getTransaction", "getTokenAccountBalance"
                ]:
                    # If the last param is a dict, add commitment to it
                    if params and isinstance(params[-1], dict):
                        params[-1]["commitment"] = self.config.commitment
                    else:
                        params.append({"commitment": self.config.commitment})
                else:
                    # Otherwise add commitment as a separate param
                    params.append(self.config.commitment)
        
        # Build JSON-RPC request
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        # Retry parameters
        max_retries = 3
        retry_delay = 1.0  # starting delay in seconds
        
        # Initialize the HTTP client if needed
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                auth=self.auth,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        
        for retry_count in range(max_retries + 1):
            try:
                response = await self._http_client.post(
                    self.config.rpc_url,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                if "error" in result:
                    error = result["error"]
                    message = f"Solana RPC error: {error.get('message', 'Unknown error')}"
                    if "data" in error:
                        message += f" - {json.dumps(error['data'])}"
                    
                    # Check for rate limiting errors and retry if possible
                    if "rate limited" in message.lower() and retry_count < max_retries:
                        wait_time = retry_delay * (2 ** retry_count)  # Exponential backoff
                        logger.warning(f"Rate limited, retrying in {wait_time}s: {method}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    raise SolanaRpcError(message, error)
                
                return result["result"]
                
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                # Handle network and HTTP errors with retries
                if retry_count < max_retries:
                    wait_time = retry_delay * (2 ** retry_count)  # Exponential backoff
                    logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    # Max retries reached, raise the exception
                    logger.error(f"Request failed after {max_retries} retries: {str(e)}")
                    raise
        
        # We should never reach here due to the raise in the loop above
        raise RuntimeError("Unexpected error in _make_request")
    
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
    
    async def get_token_accounts_by_owner(
        self, 
        owner: str, 
        mint: Optional[str] = None,
        program_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get token accounts by owner.
        
        Args:
            owner: The owner public key
            mint: Filter by mint. Defaults to None.
            program_id: Filter by token program ID. Defaults to None.
            
        Returns:
            List of token accounts
            
        Raises:
            InvalidPublicKeyError: If the owner, mint, or program_id is not a valid Solana public key
        """
        if not validate_public_key(owner):
            raise InvalidPublicKeyError(owner)
            
        if mint and not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        if program_id and not validate_public_key(program_id):
            raise InvalidPublicKeyError(program_id)
            
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
        return await self._make_request("getTokenAccountsByOwner", params)
    
    async def get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get transaction details.
        
        Args:
            signature: The transaction signature
            
        Returns:
            Transaction details
        """
        # Validate transaction signature format - base58 encoded signatures
        # should be alphanumeric and typical length is 88 characters, but allow some flexibility
        if not signature or not isinstance(signature, str):
            raise ValueError(f"Transaction signature must be a non-empty string")
        
        # Use a more general validation for base58 encoded data
        if not re.match(r"^[1-9A-HJ-NP-Za-km-z]{43,128}$", signature):
            raise ValueError(f"Invalid transaction signature format: {signature}")
        
        return await self._make_request(
            "getTransaction", 
            [signature, {"encoding": "json"}]
        )
    
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
    
    async def get_recent_blockhash(self) -> Dict[str, Any]:
        """Get a recent blockhash.
        
        Returns:
            Recent blockhash and fee calculator
        """
        return await self._make_request("getLatestBlockhash")
    
    async def get_token_supply(self, mint: str) -> Dict[str, Any]:
        """Get token supply.
        
        Args:
            mint: The mint public key
            
        Returns:
            Token supply information
            
        Raises:
            InvalidPublicKeyError: If the mint is not a valid Solana public key
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        return await self._make_request("getTokenSupply", [mint])
        
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
    
    async def get_token_accounts_by_delegate(
        self,
        delegate: str,
        mint: Optional[str] = None,
        program_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get token accounts by delegate.
        
        Args:
            delegate: The delegate public key
            mint: Filter by mint. Defaults to None.
            program_id: Filter by token program ID. Defaults to None.
            
        Returns:
            List of token accounts
            
        Raises:
            InvalidPublicKeyError: If the delegate, mint, or program_id is not a valid Solana public key
        """
        if not validate_public_key(delegate):
            raise InvalidPublicKeyError(delegate)
            
        if mint and not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        if program_id and not validate_public_key(program_id):
            raise InvalidPublicKeyError(program_id)
            
        if mint:
            params = [delegate, {"mint": mint}]
        elif program_id:
            params = [delegate, {"programId": program_id}]
        else:
            params = [
                delegate, 
                {"programId": TOKEN_PROGRAM_ID}
            ]
        
        params.append({"encoding": "jsonParsed"})
        return await self._make_request("getTokenAccountsByDelegate", params)
    
    async def get_token_largest_accounts(self, mint: str) -> List[Dict[str, Any]]:
        """Get the largest accounts for a token.
        
        Args:
            mint: The mint address
            
        Returns:
            List of token accounts sorted by balance
            
        Raises:
            InvalidPublicKeyError: If the mint is not a valid Solana public key
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        return await self._make_request("getTokenLargestAccounts", [mint])
    
    async def get_block(self, slot: int) -> Dict[str, Any]:
        """Get information about a block.
        
        Args:
            slot: The slot number
            
        Returns:
            Block information
        """
        return await self._make_request(
            "getBlock",
            [slot, {"encoding": "json", "transactionDetails": "full", "rewards": True}]
        )
    
    async def get_blocks(
        self, 
        start_slot: int, 
        end_slot: Optional[int] = None,
        commitment: Optional[str] = None
    ) -> List[int]:
        """Get a list of confirmed blocks.
        
        Args:
            start_slot: Start slot (inclusive)
            end_slot: End slot (inclusive, optional)
            commitment: Commitment level
            
        Returns:
            List of block slots
        """
        params = [start_slot]
        if end_slot is not None:
            params.append(end_slot)
        if commitment:
            params.append(commitment)
        
        return await self._make_request("getBlocks", params)
    
    async def get_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Get information about all the nodes participating in the cluster.
        
        Returns:
            List of node information
        """
        return await self._make_request("getClusterNodes")
    
    async def get_epoch_info(self) -> Dict[str, Any]:
        """Get information about the current epoch.
        
        Returns:
            Epoch information
        """
        return await self._make_request("getEpochInfo")
    
    async def get_inflation_rate(self) -> Dict[str, Any]:
        """Get the specific inflation values for the current epoch.
        
        Returns:
            Inflation rate information
        """
        return await self._make_request("getInflationRate")
    
    async def get_minimum_balance_for_rent_exemption(self, size: int) -> int:
        """Get the minimum balance required for rent exemption.
        
        Args:
            size: The data size
            
        Returns:
            Minimum balance in lamports
        """
        return await self._make_request("getMinimumBalanceForRentExemption", [size])
    
    async def get_slot(self) -> int:
        """Get the current slot.
        
        Returns:
            The current slot
        """
        return await self._make_request("getSlot")
    
    async def get_token_metadata(self, mint: str) -> Dict[str, Any]:
        """Get token metadata from Metaplex token metadata program.
        
        Args:
            mint: The mint address
            
        Returns:
            Token metadata
            
        Raises:
            InvalidPublicKeyError: If the mint is not a valid Solana public key
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
        
        # Instead of trying to derive the metadata PDA here, which requires the Solana SDK,
        # we'll use the getProgramAccounts method with filters to look for metadata accounts
        # associated with this mint
        
        try:
            # Use getProgramAccounts with memcmp filters to find the metadata account
            filters = [
                {
                    "memcmp": {
                        "offset": 33,  # Offset where mint address appears in metadata accounts
                        "bytes": mint   # The mint address to match
                    }
                }
            ]
            
            metadata_accounts = await self.get_program_accounts(
                METADATA_PROGRAM_ID,
                filters=filters,
                encoding="jsonParsed"
            )
            
            # If we found a metadata account, return its data
            if metadata_accounts and len(metadata_accounts) > 0:
                metadata = metadata_accounts[0]
                # Add the mint to the response for reference
                metadata["mint"] = mint
                return metadata
            else:
                # If no metadata found, return a minimal response
                return {
                    "mint": mint,
                    "metadata_found": False,
                    "message": "No metadata account found for this mint"
                }
        except Exception as e:
            # If error occurred, return error information
            return {
                "error": str(e),
                "mint": mint
            }
    
    async def get_market_price(self, token_mint: str) -> Dict[str, Any]:
        """Get market price for a token from a price oracle.
        
        This is a simplified implementation. In a real client, you would integrate
        with Pyth, Jupiter, or another price oracle service.
        
        Args:
            token_mint: The token mint address
            
        Returns:
            Price information
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Use more reliable price API endpoint
        price_api_url = "https://price.jup.ag/v4/price"
        
        # Set a reasonable timeout for external API calls
        timeout = min(10.0, self.config.timeout)
        
        try:
            # Use httpx client with appropriate timeout
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{price_api_url}?ids={token_mint}")
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse the response
                data = response.json()
                
                # Extract the price data for the specified token
                price_data = data.get("data", {}).get(token_mint)
                
                if price_data:
                    return {
                        "mint": token_mint,
                        "price_usd": price_data.get("price", 0),
                        "price_source": "Jupiter Aggregator",
                        "last_updated": data.get("timestamp", "unknown")
                    }
                else:
                    # Token not found in price data
                    return {
                        "mint": token_mint,
                        "price_found": False,
                        "message": "No price data available for this token"
                    }
        except httpx.HTTPStatusError as e:
            # Handle HTTP status errors (4XX, 5XX)
            return {
                "error": f"HTTP error: {e.response.status_code}",
                "mint": token_mint,
                "status_code": e.response.status_code
            }
        except httpx.RequestError as e:
            # Handle request errors (connection, timeout, etc.)
            return {
                "error": f"Request error: {str(e)}",
                "mint": token_mint
            }
        except Exception as e:
            # Handle any other unexpected errors
            return {
                "error": f"Unexpected error: {str(e)}",
                "mint": token_mint
            }
    
    async def __aenter__(self):
        """Async context manager entry point.
        
        Returns:
            Self reference for use in 'async with' statements
        """
        # Initialize any resources if needed
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        # Clean up any resources
        await self.close()
    
    async def close(self):
        """Close and clean up resources used by the client.
        
        This method should be called when the client is no longer needed.
        """
        # Clean up HTTP client if it exists
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


@asynccontextmanager
async def get_solana_client():
    """Get a Solana client as an async context manager.
    
    Yields:
        SolanaClient: An initialized Solana client.
    """
    client = SolanaClient()
    try:
        yield client
    finally:
        # No cleanup needed for now, but could add connection pool management later
        pass 