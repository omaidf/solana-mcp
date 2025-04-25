"""Base Solana RPC client for MCP.

This module provides the core functionality for making RPC requests to Solana nodes.
"""

# Standard library imports
import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Union, TypeVar, cast
from contextlib import asynccontextmanager
import re

# Third-party library imports
import httpx
from cachetools import TTLCache, cached

# Internal imports
from solana_mcp.config import SolanaConfig, get_solana_config
from solana_mcp.logging_config import get_logger
from solana_mcp.utils.validation import validate_public_key, InvalidPublicKeyError

# Type variable for generic functions
T = TypeVar('T')

# Get logger
logger = get_logger(__name__)

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

class BaseSolanaClient:
    """Base client for interacting with Solana blockchain."""
    
    def __init__(self, config: Optional[SolanaConfig] = None):
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
    
    async def get_recent_blockhash(self) -> Dict[str, Any]:
        """Get a recent blockhash.
        
        Returns:
            Recent blockhash and fee calculator
        """
        return await self._make_request("getLatestBlockhash", [{"commitment": self.config.commitment}])
    
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
async def get_base_solana_client():
    """Get a Solana client as an async context manager.
    
    Yields:
        BaseSolanaClient: An initialized Solana client.
    """
    client = BaseSolanaClient()
    try:
        yield client
    finally:
        await client.close() 