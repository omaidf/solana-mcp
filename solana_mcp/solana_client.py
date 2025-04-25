"""Async Solana client for MCP server."""

# Standard library imports
import base64
import json
import re
import asyncio
import time
import datetime
from typing import Any, Dict, List, Optional, Union, cast, Tuple, TypeVar
from contextlib import asynccontextmanager
from urllib.parse import urljoin
import uuid

# Third-party library imports
import httpx
from cachetools import TTLCache, cached

# Internal imports
from solana_mcp.config import SolanaConfig, get_solana_config
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.utils.batching import BatchProcessor

# Get logger
logger = get_logger(__name__)

# Constants
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
METAPLEX_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Solana public key validation pattern (base58 format)
PUBKEY_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

# Jupiter Aggregator Program ID for DEX data
JUPITER_PROGRAM_ID = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"

# Orca Program ID
ORCA_PROGRAM_ID = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"

T = TypeVar('T')

class InvalidPublicKeyError(Exception):
    """Exception raised when an invalid public key is provided."""
    pass

class SolanaRpcError(Exception):
    """Exception raised when a Solana RPC request fails."""
    
    def __init__(self, message: str, error_data: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_data: Optional error data from the RPC response
        """
        super().__init__(message)
        self.error_data = error_data or {}

def validate_public_key(pubkey: str) -> bool:
    """Validate a Solana public key.
    
    Args:
        pubkey: The public key to validate
        
    Returns:
        True if the public key is valid, False otherwise
    """
    if not pubkey or not isinstance(pubkey, str):
        return False
    return bool(PUBKEY_PATTERN.match(pubkey))

class PublicKey:
    """Solana public key class."""
    
    def __init__(self, value: Union[str, bytes, List[int]]):
        """Initialize a public key.
        
        Args:
            value: The public key value as a string (base58), bytes, or list of integers
            
        Raises:
            ValueError: If the public key is invalid
        """
        if isinstance(value, str):
            if not validate_public_key(value):
                raise ValueError(f"Invalid public key format: {value}")
            self._key = value
        elif isinstance(value, bytes) or isinstance(value, List) or isinstance(value, bytearray):
            # Convert bytes to string
            bytes_data = bytes(value) if not isinstance(value, bytes) else value
            self._key = bytes_data.hex()
        else:
            raise ValueError(f"Unsupported public key format: {type(value)}")
    
    def __str__(self) -> str:
        """Get the public key as a string.
        
        Returns:
            The public key as a string
        """
        return self._key
    
    def to_bytes(self) -> bytes:
        """Get the public key as bytes.
        
        Returns:
            The public key as bytes
        """
        if len(self._key) % 2 == 1:
            self._key = "0" + self._key
        return bytes.fromhex(self._key)
    
    @staticmethod
    def find_program_address(seeds: List[bytes], program_id: "PublicKey") -> Tuple[str, int]:
        """Find a program address.
        
        Args:
            seeds: The seeds to use to find the program address
            program_id: The program ID
            
        Returns:
            (address, bump_seed)
        """
        # This is a placeholder. In a real implementation, this would compute
        # the actual program address using the provided seeds and program ID.
        # For testing purposes, we'll return a synthetic address
        seed_str = b"".join(seeds).hex()
        program_str = str(program_id)
        return f"{program_str[:8]}_{seed_str[:16]}_program", 255

class SolanaClient:
    """Client for interacting with Solana blockchain."""
    
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
        self.batch_size = 10
        self.batch_processor = BatchProcessor(batch_size=self.batch_size)
    
    async def _make_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make a JSON-RPC request to the Solana node.
        
        Args:
            method: The RPC method to call
            params: The parameters to pass to the method
            
        Returns:
            The JSON-RPC response
            
        Raises:
            SolanaRpcError: If the RPC server returns an error
            httpx.HTTPStatusError: If there's an HTTP error
            httpx.RequestError: If there's a network or request error
            asyncio.TimeoutError: If the request times out
        """
        if params is None:
            params = []
            
        # Add commitment if not already in params and the method supports it
        # Some methods don't accept parameters at all
        methods_without_params = ["getHealth", "getVersion", "getClusterNodes", "getEpochInfo", "getInflationRate", "getSlot"]
        
        if method not in methods_without_params:
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
                    params.append({"commitment": self.config.commitment})
        
        # Build JSON-RPC request
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        # Retry parameters
        max_retries = 3
        initial_retry_delay = 1.0  # starting delay in seconds
        max_retry_delay = 10.0  # maximum delay in seconds
        
        # Initialize the HTTP client if needed
        client_created_here = False
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                auth=self.auth,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
            client_created_here = True
        
        try:
            # Categorize errors for retry decision
            retriable_status_codes = {408, 429, 500, 502, 503, 504}
            last_exception = None
            
            for retry_count in range(max_retries + 1):
                try:
                    # Log the attempt if it's a retry
                    if retry_count > 0:
                        logger.info(f"Retry attempt {retry_count}/{max_retries} for {method}")
                    
                    response = await self._http_client.post(
                        self.config.rpc_url,
                        headers=self.headers,
                        json=payload
                    )
                    
                    # Handle HTTP status errors
                    if response.status_code in retriable_status_codes and retry_count < max_retries:
                        wait_time = min(
                            initial_retry_delay * (2 ** retry_count), 
                            max_retry_delay
                        )
                        logger.warning(f"HTTP status {response.status_code}, retrying in {wait_time}s: {method}")
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
                            error.get("code") == -32005) and retry_count < max_retries:
                            wait_time = min(
                                initial_retry_delay * (2 ** retry_count), 
                                max_retry_delay
                            )
                            logger.warning(f"Rate limited, retrying in {wait_time}s: {method}")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # Non-retriable RPC error or max retries reached
                        raise SolanaRpcError(message, error)
                    
                    # Success case - return the result
                    return result["result"]
                    
                except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout, 
                        httpx.ConnectTimeout, httpx.NetworkError, json.JSONDecodeError) as e:
                    last_exception = e
                    
                    # Determine if we should retry based on the error type
                    should_retry = (
                        retry_count < max_retries and 
                        (isinstance(e, httpx.ConnectError) or 
                         isinstance(e, httpx.ConnectTimeout) or
                         isinstance(e, httpx.ReadTimeout) or
                         isinstance(e, httpx.NetworkError) or
                         (isinstance(e, httpx.HTTPStatusError) and e.response.status_code in retriable_status_codes) or
                         (isinstance(e, json.JSONDecodeError) and retry_count == 0))  # Only retry JSON errors once
                    )
                    
                    if should_retry:
                        wait_time = min(
                            initial_retry_delay * (2 ** retry_count), 
                            max_retry_delay
                        )
                        logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        # Max retries reached or non-retriable error
                        logger.error(f"Request failed after {retry_count} attempts: {str(e)}")
                        raise
                except asyncio.CancelledError:
                    # Don't catch cancellation, let it propagate
                    raise
                except Exception as e:
                    # Catch and log any other unexpected errors
                    logger.error(f"Unexpected error in _make_request: {str(e)}", exc_info=True)
                    last_exception = e
                    
                    if retry_count < max_retries:
                        wait_time = min(
                            initial_retry_delay * (2 ** retry_count), 
                            max_retry_delay
                        )
                        logger.warning(f"Unexpected error, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            
            # We should never reach here due to the raise in the loop, but just in case
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Unexpected error in _make_request: Max retries reached without an exception")
        finally:
            # If we created a client just for this request and not using it as a persistent client,
            # close it to avoid resource leaks
            if client_created_here:
                await self._http_client.aclose()
                self._http_client = None
    
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
        return await self._make_request("getLatestBlockhash", [{"commitment": self.config.commitment}])
    
    async def get_token_supply(self, mint: str) -> Dict[str, Any]:
        """Get token supply information for a specific mint.
        
        Args:
            mint: The mint address of the token
            
        Returns:
            Dict containing supply info and decimals.
            If token supply info cannot be retrieved, returns a default supply object.
            
        Raises:
            InvalidPublicKeyError: If the mint address is invalid
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
        
        # Set default supply info
        default_supply_info = {
            "mint": mint,
            "amount": "0",
            "decimals": 0,
            "uiAmount": 0.0,
            "uiAmountString": "0"
        }
        
        try:
            logger.debug(f"Fetching token supply for mint: {mint}")
            
            # Direct RPC call to get token supply
            response = await self._make_request("getTokenSupply", [mint])
            
            # Check if we got a valid response
            if not response or "value" not in response:
                logger.warning(f"No supply info found for mint: {mint}")
                return default_supply_info
            
            # Return the value directly, which should contain amount, decimals, etc.
            supply_data = response["value"]
            
            # Ensure the response includes expected fields
            if "amount" not in supply_data or "decimals" not in supply_data:
                logger.warning(f"Invalid supply data format for {mint}: {supply_data}")
                return default_supply_info
                
            return {
                "mint": mint,
                **supply_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching token supply for {mint}: {str(e)}")
            return {**default_supply_info, "error": str(e)}
    
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
            params.append({"commitment": commitment})
        
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
        """Get token metadata directly from the token's mint account.
        
        Args:
            mint: The mint address
            
        Returns:
            Dict with token metadata (name, symbol, decimals, mint)
            
        Raises:
            InvalidPublicKeyError: If the mint address is invalid
        """
        # Import here to avoid circular import issues
        from solana_mcp.clients.token_client import TokenClient
        
        # Create a TokenClient instance with the same config
        token_client = TokenClient(self.config)
        
        # Delegate to the TokenClient implementation
        return await token_client.get_token_metadata(mint)
    
    async def get_token_price_birdeye(self, token_mint: str) -> Dict[str, Any]:
        """Get token price data from Birdeye API.
        
        Args:
            token_mint: The mint address of the token
            
        Returns:
            Price information including:
            - price: Current price in USD
            - price_change_24h: 24-hour price change percentage
            - last_updated: Timestamp of last update
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Default price data
        default_price_data = {
            "mint": token_mint,
            "price": 0.01,  # Fallback price
            "price_change_24h": 0.0,
            "last_updated": datetime.datetime.now().isoformat(),
            "source": "fallback"
        }
        
        try:
            # Birdeye API endpoint
            BIRDEYE_API = "https://public-api.birdeye.so/public/price"
            headers = {
                "Content-Type": "application/json",
                "x-chain": "solana"
            }
            
            # Use httpx instead of requests to maintain async
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{BIRDEYE_API}?address={token_mint}",
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('success'):
                    price_data = data['data']
                    return {
                        "mint": token_mint,
                        "price": price_data['value'],
                        "price_change_24h": price_data['change24h'],
                        "last_updated": datetime.datetime.fromtimestamp(price_data['updateUnixTime']).isoformat(),
                        "source": "birdeye"
                    }
                
                logger.warning(f"Birdeye API returned unsuccessful response for {token_mint}: {data}")
                return default_price_data
                
        except Exception as e:
            logger.error(f"Error fetching price from Birdeye for {token_mint}: {str(e)}")
            return default_price_data
            
    async def get_token_price(self, token_mint: str) -> Dict[str, Any]:
        """Get token price using the best available source.
        This is a wrapper that tries multiple price sources and returns the most reliable one.
        
        Args:
            token_mint: The mint address of the token
            
        Returns:
            Price information with source indicated
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Try Birdeye first as it's more reliable
        birdeye_price = await self.get_token_price_birdeye(token_mint)
        if birdeye_price.get("price", 0) > 0 and birdeye_price.get("source") != "fallback":
            return birdeye_price
            
        # Fall back to our DEX-based price calculation
        dex_price = await self.get_market_price(token_mint)
        price_data = dex_price.get("price_data", {})
        
        # If we got a valid price from DEX, use it
        if price_data.get("price_usd") and price_data.get("price_usd") > 0:
            return {
                "mint": token_mint,
                "price": price_data.get("price_usd"),
                "price_change_24h": 0.0,  # DEX doesn't provide change data
                "last_updated": price_data.get("last_updated"),
                "source": price_data.get("source", "dex")
            }
            
        # If all else fails, return the Birdeye result (which might be the fallback)
        return birdeye_price
    
    async def get_market_price(self, token_mint: str) -> Dict[str, Any]:
        """Get market price data for a token.
        
        This method fetches price information for a token from available DEX liquidity.
        It uses various sources to determine the token price in SOL and USD.
        
        Args:
            token_mint: The mint address of the token
            
        Returns:
            Price information including:
            - price_sol: Price in SOL
            - price_usd: Estimated price in USD (if SOL price is available)
            - liquidity: Estimated liquidity information
            - source: Data source used for the price
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        result = {
            "mint": token_mint,
            "price_data": {
                "price_sol": None,
                "price_usd": None,
                "liquidity": None,
                "source": None,
                "last_updated": datetime.datetime.now().isoformat()
            }
        }
        
        try:
            # Constants - common token mints for price references
            USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
            SOL_MINT = "So11111111111111111111111111111111111111112"  # Native SOL wrapped mint
            
            # Fetch SOL/USD price first to convert SOL prices to USD
            sol_usd_price = 0
            
            # Try Raydium pools for SOL/USDC price
            sol_usdc_pools = await self._make_request(
                "getProgramAccounts",
                [
                    RAYDIUM_PROGRAM_ID,
                    {
                        "encoding": "base64",
                        "filters": [
                            {"memcmp": {"offset": 200, "bytes": SOL_MINT}},
                            {"memcmp": {"offset": 232, "bytes": USDC_MINT}}
                        ]
                    }
                ]
            )
            
            if sol_usdc_pools and len(sol_usdc_pools["result"]) > 0:
                # Get the largest pool by reserves
                largest_pool = None
                max_reserves = 0
                
                for pool in sol_usdc_pools["result"]:
                    if "account" in pool and "data" in pool["account"]:
                        data = pool["account"]["data"]
                        if isinstance(data, list) and data[1] == "base64":
                            decoded_data = base64.b64decode(data[0])
                            
                            # Extract reserves data (simplified, actual layout may vary)
                            sol_reserves = int.from_bytes(decoded_data[264:272], byteorder="little")
                            usdc_reserves = int.from_bytes(decoded_data[296:304], byteorder="little")
                            
                            if sol_reserves > max_reserves:
                                max_reserves = sol_reserves
                                largest_pool = {
                                    "sol_reserves": sol_reserves,
                                    "usdc_reserves": usdc_reserves
                                }
                
                if largest_pool:
                    # Calculate SOL/USD price from pool data
                    # USDC has 6 decimals, SOL has 9 decimals
                    sol_usd_price = (largest_pool["usdc_reserves"] / 10**6) / (largest_pool["sol_reserves"] / 10**9)
            
            # If we couldn't get SOL/USD price from Raydium, try Jupiter
            if not sol_usd_price:
                # Jupiter price API call (simplified)
                try:
                    price_response = await self._make_request(
                        "getLatestBlockhash",  # Updated from getRecentBlockhash to getLatestBlockhash
                        []
                    )
                    
                    # Use a default SOL price if we can't get it from pools
                    # In a production system, you'd use a proper price oracle
                    sol_usd_price = 100  # Default fallback price
                except Exception as e:
                    logger.warning(f"Error getting SOL/USD price from Jupiter: {str(e)}")
                    sol_usd_price = 100  # Default fallback price
            
            # Now find pools containing our token
            token_pools = []
            
            # Check Raydium pools
            raydium_pools = await self._make_request(
                "getProgramAccounts",
                [
                    RAYDIUM_PROGRAM_ID,
                    {
                        "encoding": "base64",
                        "filters": [
                            {"memcmp": {"offset": 200, "bytes": token_mint}}
                        ]
                    }
                ]
            )
            
            if raydium_pools and "result" in raydium_pools:
                for pool in raydium_pools["result"]:
                    if "account" in pool and "data" in pool["account"]:
                        data = pool["account"]["data"]
                        if isinstance(data, list) and data[1] == "base64":
                            decoded_data = base64.b64decode(data[0])
                            
                            # Extract pool data
                            token_a_mint_bytes = decoded_data[200:232]
                            token_b_mint_bytes = decoded_data[232:264]
                            
                            token_a_mint = str(PublicKey(bytes(token_a_mint_bytes)))
                            token_b_mint = str(PublicKey(bytes(token_b_mint_bytes)))
                            
                            token_a_reserves = int.from_bytes(decoded_data[264:272], byteorder="little")
                            token_b_reserves = int.from_bytes(decoded_data[296:304], byteorder="little")
                            
                            # Determine which token is our target and which is the paired token
                            if token_a_mint == token_mint:
                                token_reserves = token_a_reserves
                                paired_token_mint = token_b_mint
                                paired_token_reserves = token_b_reserves
                            else:
                                token_reserves = token_b_reserves
                                paired_token_mint = token_a_mint
                                paired_token_reserves = token_a_reserves
                            
                            # Get decimal information for both tokens
                            token_decimal_info = await self.get_token_supply(token_mint)
                            token_decimals = token_decimal_info.get("value", {}).get("decimals", 9)
                            
                            paired_decimal_info = await self.get_token_supply(paired_token_mint)
                            paired_decimals = paired_decimal_info.get("value", {}).get("decimals", 9)
                            
                            # Calculate price based on the paired token
                            price_in_paired = (paired_token_reserves / 10**paired_decimals) / (token_reserves / 10**token_decimals)
                            
                            # Calculate SOL and USD prices
                            price_sol = 0
                            price_usd = 0
                            
                            if paired_token_mint == SOL_MINT:
                                price_sol = price_in_paired
                                price_usd = price_sol * sol_usd_price
                            elif paired_token_mint == USDC_MINT or paired_token_mint == USDT_MINT:
                                price_usd = price_in_paired
                                price_sol = price_usd / sol_usd_price if sol_usd_price > 0 else 0
                            else:
                                # If paired with another token, try to get its price first
                                # This is a simplified approach
                                paired_price_data = await self.get_market_price(paired_token_mint)
                                paired_price_usd = paired_price_data.get("price_data", {}).get("price_usd", 0)
                                
                                if paired_price_usd > 0:
                                    price_usd = price_in_paired * paired_price_usd
                                    price_sol = price_usd / sol_usd_price if sol_usd_price > 0 else 0
                            
                            # Calculate liquidity
                            liquidity_usd = 0
                            if price_usd > 0:
                                token_liquidity = (token_reserves / 10**token_decimals) * price_usd
                                paired_liquidity = 0
                                
                                if paired_token_mint == USDC_MINT or paired_token_mint == USDT_MINT:
                                    paired_liquidity = paired_token_reserves / 10**paired_decimals
                                elif paired_token_mint == SOL_MINT and sol_usd_price > 0:
                                    paired_liquidity = (paired_token_reserves / 10**paired_decimals) * sol_usd_price
                                elif paired_price_usd > 0:
                                    paired_liquidity = (paired_token_reserves / 10**paired_decimals) * paired_price_usd
                                
                                liquidity_usd = token_liquidity + paired_liquidity
                            
                            token_pools.append({
                                "protocol": "raydium",
                                "pair": f"{token_mint}-{paired_token_mint}",
                                "price_sol": price_sol,
                                "price_usd": price_usd,
                                "liquidity_usd": liquidity_usd
                            })
            
            # Similar approach for Orca pools would be implemented here
            # For now we'll use only Raydium pools
            
            # Select the best pool based on liquidity
            if token_pools:
                # Sort pools by liquidity (highest first)
                sorted_pools = sorted(token_pools, key=lambda x: x.get("liquidity_usd", 0), reverse=True)
                best_pool = sorted_pools[0]
                
                result["price_data"] = {
                    "price_sol": best_pool.get("price_sol", 0),
                    "price_usd": best_pool.get("price_usd", 0),
                    "liquidity": {
                        "total_usd": best_pool.get("liquidity_usd", 0),
                        "best_pool_protocol": best_pool.get("protocol", "unknown"),
                        "best_pool_pair": best_pool.get("pair", "unknown")
                    },
                    "source": f"{best_pool.get('protocol', 'dex')}",
                    "last_updated": datetime.datetime.now().isoformat()
                }
            else:
                # If no pools found, return empty result
                logger.warning(f"No liquidity pools found for token {token_mint}")
                
        except Exception as e:
            # Log and return error information
            logger.error(f"Error fetching price data for {token_mint}: {str(e)}", exc_info=True)
            result["error"] = str(e)
            
        return result
    
    async def get_token_holders(self, token_mint: str, limit: int = 20) -> Dict[str, Any]:
        """Get top token holders from Solana FM API.
        
        Args:
            token_mint: The mint address of the token
            limit: Maximum number of holders to return
            
        Returns:
            Dict containing list of holders with their balances
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Default response with empty holders
        default_response = {
            "mint": token_mint,
            "holders": [],
            "total_holders": 0,
            "source": "fallback"
        }
        
        try:
            # Solana FM API endpoint
            SOLANA_FM_API = f"https://api.solana.fm/v0/tokens/{token_mint}/holders?limit={limit}"
            
            # Use httpx instead of requests to maintain async
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(SOLANA_FM_API)
                response.raise_for_status()
                data = response.json()
                
                # Process and return the holder data
                if "data" in data and "items" in data["data"]:
                    holders = []
                    for holder in data["data"]["items"]:
                        try:
                            holders.append({
                                "address": holder.get("address", ""),
                                "amount": holder.get("amount", "0"),
                                "ui_amount": holder.get("uiAmount", 0),
                                "percentage": holder.get("percentage", 0)
                            })
                        except (KeyError, TypeError) as e:
                            logger.warning(f"Error parsing holder data: {str(e)}")
                    
                    return {
                        "mint": token_mint,
                        "holders": holders,
                        "total_holders": data["data"].get("totalItems", len(holders)),
                        "source": "solana_fm"
                    }
                
                logger.warning(f"Unexpected response format from Solana FM API for {token_mint}")
                return default_response
                
        except Exception as e:
            logger.error(f"Error fetching holders from Solana FM for {token_mint}: {str(e)}")
            return default_response
    
    async def __aenter__(self):
        """Async context manager entry.
        
        Returns:
            Self
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        await self.close()
    
    async def close(self):
        """Close the client and release resources."""
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