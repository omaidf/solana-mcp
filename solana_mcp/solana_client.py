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
import base58
from solana.rpc.api import Pubkey

# Third-party library imports
import httpx
from cachetools import TTLCache, cached

# Internal imports
from solana_mcp.config import SolanaConfig, get_solana_config
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.utils.validation import validate_public_key, InvalidPublicKeyError, validate_transaction_signature
from solana_mcp.constants import (
    TOKEN_PROGRAM_ID, METADATA_PROGRAM_ID, METAPLEX_PROGRAM_ID, 
    RAYDIUM_PROGRAM_ID, JUPITER_PROGRAM_ID, ORCA_PROGRAM_ID,
    USDC_MINT, USDT_MINT, SOL_MINT
)

# Get logger
logger = get_logger(__name__)

T = TypeVar('T')

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
        
        # Store Birdeye API key if available
        self.birdeye_api_key = self.config.birdeye_api_key
        
        # Shared HTTP client for better performance
        self._http_client = None
        self.batch_size = 10
    
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
        # Add temporary logging to trace calls
        log_params = json.dumps(params) if params else "[]"
        if len(log_params) > 200: log_params = log_params[:197] + "..."
        logger.info(f"_make_request called: method={method}, params={log_params}")

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
        
        # Initialize the HTTP client if needed (assuming it's managed externally by __aenter__/__aexit__)
        if self._http_client is None:
            # This should ideally not happen if client is used within context manager
            logger.warning("Re-initializing httpx client within _make_request. Was SolanaClient used outside its context manager?")
            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                auth=self.auth,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        
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
            
        # Logic moved from AccountClient
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
            
        # Logic moved from AccountClient
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
        
        # Use centralized validation utility
        if not validate_transaction_signature(signature):
            raise ValueError(f"Invalid transaction signature format: {signature}")
        
        # Import here to avoid circular import issues
        from solana_mcp.clients.transaction_client import TransactionClient
        
        # Create a TransactionClient instance with the same config
        transaction_client = TransactionClient(self.config)
        
        # Delegate to the TransactionClient implementation
        return await transaction_client.get_transaction(signature)
    
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
            
        # Logic moved from AccountClient
        params = [program_id, {"encoding": encoding}]
        config = {}
        if filters:
            config["filters"] = filters
        if limit is not None:
            config["limit"] = limit # Note: getProgramAccounts might not support limit/offset directly in std RPC
        if offset is not None:
            # Offset might not be supported by standard getProgramAccounts
            # Check RPC documentation if this feature is needed
            logger.warning("Offset parameter might not be supported by standard getProgramAccounts RPC call.")
            # params[1]["offset"] = offset # Assuming offset is part of config dict
            pass # Currently ignoring offset as it's unlikely to be supported this way
        
        # Ensure config dict is added only if it contains something
        if config:
             params.append(config)
        elif len(params) == 1: # Only program_id was provided
            # Add encoding object if no other config exists
             params.append({"encoding": encoding})

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
        
        # Logic moved from TokenClient
        try:
            # Set default supply info structure matching RPC response
            default_rpc_response_structure = {
                 "value": {
                      "amount": "0",
                      "decimals": 0,
                      "uiAmount": 0.0,
                      "uiAmountString": "0"
                 }
             }
            
            # Get token supply using more robust error handling
            logger.debug(f"Fetching token supply for mint: {mint}")
            
            # _make_request already extracts the "result" field which contains the "value" dict
            response_value = await self._make_request("getTokenSupply", [mint]) 
            
            # Check the nested structure: response = {"context":..., "value": {"amount":..., "decimals":...}}
            if response_value and isinstance(response_value.get("value"), dict) and \
               "amount" in response_value["value"] and "decimals" in response_value["value"]:
                    supply_value = response_value["value"] # Extract the inner value dict
                    
                    # Validate supply value structure
                    decimals = int(supply_value.get("decimals", 0))
                    amount = supply_value.get("amount", "0")
                    
                    # Calculate UI amount (human-readable)
                    try:
                        # Use Decimal for precision before converting to float
                        from decimal import Decimal, InvalidOperation
                        ui_amount = float(Decimal(amount) / (Decimal(10) ** decimals)) if decimals >= 0 else float(Decimal(amount))
                        # Format with correct decimals using Decimal
                        ui_amount_string = f"{Decimal(amount) / (Decimal(10) ** decimals) if decimals >=0 else Decimal(amount):f}" 
                    except (InvalidOperation, ValueError, TypeError):
                        logger.warning(f"Could not calculate uiAmount for {mint}", exc_info=True)
                        ui_amount = 0.0
                        ui_amount_string = "0"
                    
                    # Return structure matching RPC response's "value" field
                    # Wrap the validated value in the expected {"value": ...} structure
                    result_to_return = {
                         "value": {
                             "amount": amount,
                             "decimals": decimals,
                             "uiAmount": ui_amount,
                             "uiAmountString": ui_amount_string
                         }
                     }
                    logger.debug(f"Returning successfully parsed token supply for {mint}: {result_to_return}")
                    return result_to_return
            
            # If parsing failed or structure was wrong, log and raise an error instead of returning default
            err_msg = f"Failed to parse token supply data for {mint}. Response: {response_value}"
            logger.warning(err_msg)
            raise SolanaRpcError(err_msg, error_data=response_value)
            
        except SolanaRpcError as rpc_err:
             logger.error(f"RPC Error fetching token supply for {mint}: {rpc_err}", exc_info=True)
             raise # Re-raise RPC errors
        except InvalidPublicKeyError:
             logger.error(f"Invalid public key error for {mint} in get_token_supply", exc_info=True)
             raise # Re-raise validation errors
        except Exception as e:
            # Catch unexpected errors during processing
            logger.error(f"Unexpected error processing token supply for {mint}: {str(e)}", exc_info=True)
            raise SolanaRpcError(f"Unexpected error processing supply for {mint}: {e}") from e
    
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
            
        # Logic moved from AccountClient
        options: Dict[str, Any] = {"limit": limit}
        if before:
            options["before"] = before
        if until:
            options["until"] = until
            
        # Pass options dict as the second parameter
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
    
    async def get_token_largest_accounts(self, mint: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the largest accounts for a token.
        
        Args:
            mint: The mint address
            limit: The maximum number of accounts to return.
            
        Returns:
            List of token accounts sorted by balance (structure from RPC: {address, amount, decimals, uiAmount, uiAmountString})
            
        Raises:
            InvalidPublicKeyError: If the mint is not a valid Solana public key
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
        
        # Logic moved from TokenClient
        # _make_request extracts the 'result', which should be the 'value' array
        # Pass limit as an option
        params = [mint, {"limit": limit}]
        response_value = await self._make_request("getTokenLargestAccounts", params)
        # The RPC response structure is { "context": {...}, "value": [...] } 
        # _make_request returns the value part directly if successful
        if isinstance(response_value, list):
            return response_value
        else:
            logger.warning(f"Unexpected format for getTokenLargestAccounts for {mint}: {response_value}")
            return [] # Return empty list on unexpected format
    
    async def get_block(self, slot: int) -> Dict[str, Any]:
        """Get information about a block.
        
        Args:
            slot: The slot number
            
        Returns:
            Block information
        """
        # Import here to avoid circular import issues
        from solana_mcp.clients.transaction_client import TransactionClient
        
        # Create a TransactionClient instance with the same config
        transaction_client = TransactionClient(self.config)
        
        # Delegate to the TransactionClient implementation
        return await transaction_client.get_block(slot)
    
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
        # Import here to avoid circular import issues
        from solana_mcp.clients.transaction_client import TransactionClient
        
        # Create a TransactionClient instance with the same config
        transaction_client = TransactionClient(self.config)
        
        # Delegate to the TransactionClient implementation
        return await transaction_client.get_blocks(start_slot, end_slot, commitment)
    
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
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        # Logic moved from TokenClient - Including Metaplex parsing and Jupiter fallback
        # Default metadata if metadata doesn't exist
        default_metadata = {
            "name": "Unknown Token",
            "symbol": "UNKNOWN",
            "uri": "",
            "source": "none"
        }
        
        try:
            logger.debug(f"Fetching token metadata for mint: {mint} via Metaplex")
            
            # Find Metaplex metadata account PDA
            # This requires the findProgramAddress logic from solana-py or equivalent
            # Re-implementing basic PDA finding for this specific case:
            try:
                 seeds = [b'metadata', bytes(Pubkey.from_string(METADATA_PROGRAM_ID)), bytes(Pubkey.from_string(mint))]
                 metadata_pda, _ = Pubkey.find_program_address(seeds, Pubkey.from_string(METADATA_PROGRAM_ID))
                 logger.debug(f"Calculated metadata PDA for {mint}: {metadata_pda}")
            except Exception as pda_err:
                 logger.warning(f"Could not calculate metadata PDA for {mint}: {pda_err}")
                 metadata_pda = None

            if metadata_pda:
                 # Fetch account info for the PDA
                 account_data = await self._make_request("getAccountInfo", [str(metadata_pda), {"encoding": "base64"}])

                 if account_data and isinstance(account_data, dict) and account_data.get("value") and account_data["value"].get("data"):
                      data_base64 = account_data["value"]["data"][0]
                      decoded_data = base64.b64decode(data_base64)
                      
                      # Basic parsing based on Metaplex Token Metadata standard structure
                      # WARNING: This manual parsing is fragile and might break with future Metaplex versions.
                      # Using an SDK (like mpl-token-metadata for Python) is recommended for robustness.
                      try:
                          # Simplified parsing - assumes standard layout
                          # Key (1 byte), UpdateAuthority (32), Mint (32), Data struct marker (1) = 66 bytes offset? Check standard.
                          # Let's assume Data struct starts after Mint: 1 + 32 + 32 = 65 bytes offset
                          data_struct_offset = 65 
                          if len(decoded_data) > data_struct_offset + 4: # Need at least name length
                               name_len = int.from_bytes(decoded_data[data_struct_offset : data_struct_offset+4], 'little')
                               name_start = data_struct_offset + 4
                               name_end = name_start + name_len
                               name = decoded_data[name_start:name_end].decode('utf-8').rstrip('\x00')
                               
                               symbol_len_offset = name_end
                               if len(decoded_data) > symbol_len_offset + 4:
                                    symbol_len = int.from_bytes(decoded_data[symbol_len_offset : symbol_len_offset+4], 'little')
                                    symbol_start = symbol_len_offset + 4
                                    symbol_end = symbol_start + symbol_len
                                    symbol = decoded_data[symbol_start:symbol_end].decode('utf-8').rstrip('\x00')

                                    uri_len_offset = symbol_end
                                    if len(decoded_data) > uri_len_offset + 4:
                                         uri_len = int.from_bytes(decoded_data[uri_len_offset : uri_len_offset+4], 'little')
                                         uri_start = uri_len_offset + 4
                                         uri_end = uri_start + uri_len
                                         uri = decoded_data[uri_start:uri_end].decode('utf-8').rstrip('\x00')

                                         metadata = {
                                              "name": name if name else default_metadata["name"],
                                              "symbol": symbol if symbol else default_metadata["symbol"],
                                              "uri": uri if uri else default_metadata["uri"],
                                              "source": "metaplex"
                                         }
                                         logger.debug(f"Successfully parsed Metaplex metadata for {mint}: {metadata}")
                                         return metadata
                      except Exception as parse_err:
                          logger.warning(f"Failed to parse Metaplex metadata for {mint} from PDA {metadata_pda}: {parse_err}", exc_info=True)
                 else:
                      logger.warning(f"Metaplex metadata account not found or empty for PDA {metadata_pda}")
            
            # If Metaplex parsing failed or PDA not found, try Jupiter token list API
            logger.debug(f"Falling back to Jupiter token list for {mint} metadata")
            try:
                # Use httpx directly as this isn't a standard RPC call
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Using the strict list first as it might be smaller/faster
                    for list_url in ["https://token.jup.ag/strict", "https://token.jup.ag/all"]:
                        try:
                             response = await client.get(list_url)
                             if response.status_code == 200:
                                 tokens = response.json()
                                 for token in tokens:
                                     if token.get("address") == mint:
                                         logger.info(f"Found metadata for {mint} in Jupiter list: {list_url}")
                                         return {
                                             "name": token.get("name", default_metadata["name"]),
                                             "symbol": token.get("symbol", default_metadata["symbol"]),
                                             "uri": token.get("logoURI", default_metadata["uri"]),
                                             "source": "jupiter"
                                         }
                        except Exception as http_err:
                             logger.warning(f"Error fetching from Jupiter list {list_url}: {http_err}")
                             if list_url == "https://token.jup.ag/strict": # Don't retry if strict fails badly
                                  await asyncio.sleep(0.5) # Small delay before trying full list
                             else:
                                  raise # Re-raise error from 'all' list
            except Exception as e:
                logger.warning(f"Error fetching metadata from Jupiter API for {mint}: {str(e)}")
                
            # If all methods fail, return the default metadata
            logger.warning(f"Could not find metadata for {mint} from any source, returning default.")
            return default_metadata
            
        except Exception as e:
            logger.error(f"Unexpected error fetching token metadata for {mint}: {str(e)}", exc_info=True)
            return default_metadata
    
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
        
        # Check if API key is available
        if not self.birdeye_api_key:
            logger.warning("Birdeye API key not configured. Cannot fetch price from Birdeye.")
            return default_price_data
            
        try:
            # Correct Birdeye API endpoint for single price
            BIRDEYE_API = "https://public-api.birdeye.so/defi/price"
            headers = {
                "accept": "application/json",
                "x-chain": "solana",
                "X-API-KEY": self.birdeye_api_key # Use the configured API key
            }
            
            # Define query parameters
            params = {"address": token_mint}
            
            # Use httpx instead of requests to maintain async
            # Use the shared client if available, otherwise create a temporary one
            # Ensure client is initialized if None (shouldn't happen with context mgr)
            http_client = self._http_client or httpx.AsyncClient(timeout=self.config.timeout)
            client_was_none = self._http_client is None
            
            try:
                 response = await http_client.get(
                      BIRDEYE_API,
                      headers=headers,
                      params=params # Pass address as query parameter
                 )
                 response.raise_for_status()
                 data = response.json()
            finally:
                 # Close client only if we created it temporarily here
                 if client_was_none:
                      await http_client.aclose()

            if data.get('success'):
                price_data = data['data']
                # Check if price_data is not None before accessing keys
                if price_data:
                    return {
                        "mint": token_mint,
                        "price": price_data.get('value'), # Birdeye uses 'value' for price
                        "price_change_24h": price_data.get('change24h', 0.0), # Add default for robustness
                        "last_updated": datetime.datetime.fromtimestamp(price_data['updateUnixTime']).isoformat() if price_data.get('updateUnixTime') else datetime.datetime.now().isoformat(),
                        "source": "birdeye"
                    }
                else:
                    logger.warning(f"Birdeye API returned success=true but data is null for {token_mint}")
                    return default_price_data
            
            logger.warning(f"Birdeye API returned unsuccessful response for {token_mint}: {data}")
            return default_price_data
            
        except httpx.HTTPStatusError as e:
            # Log specific HTTP errors
            logger.error(f"HTTP error fetching price from Birdeye for {token_mint}: {e.response.status_code} - {e.response.text}", exc_info=True)
            return default_price_data
        except Exception as e:
            logger.error(f"Error fetching price from Birdeye for {token_mint}: {str(e)}", exc_info=True)
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
            
        # If all else fails, return the Birdeye result (which might be the fallback)
        logger.warning(f"Falling back to Birdeye result (potentially default) for {token_mint} price.")
        return birdeye_price
    
    async def get_market_price(self, token_mint: str) -> Dict[str, Any]:
        """(Simplified) Get market price data, primarily focused on SOL/USD.
        
        This method now primarily fetches the SOL/USD price via Birdeye.
        The complex and brittle DEX pool logic has been removed.
        
        Args:
            token_mint: The mint address of the token (largely ignored now, but kept for signature compatibility).
            
        Returns:
            Price information containing SOL/USD price if available.
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        # Validate the input mint, although it's not directly used for SOL price
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        result = {
            "mint": token_mint, # Keep the original mint for context
            "price_data": {
                "price_sol": None, # No longer calculated
                "price_usd": None, # This will hold SOL/USD price
                "liquidity": None, # No longer calculated
                "source": None,
                "last_updated": datetime.datetime.now().isoformat()
            }
        }
        
        try:
            # Fetch SOL/USD price using the Birdeye wrapper
            sol_price_data = await self.get_token_price_birdeye(SOL_MINT)
            
            if sol_price_data and sol_price_data.get("source") != "fallback":
                sol_usd_price = sol_price_data.get("price")
                if sol_usd_price:
                    result["price_data"]["price_usd"] = float(sol_usd_price)
                    result["price_data"]["source"] = f"sol_price_from_{sol_price_data.get('source', 'unknown')}"
                    result["price_data"]["last_updated"] = sol_price_data.get("last_updated", result["price_data"]["last_updated"])
                else:
                     logger.warning(f"Could not extract SOL/USD price from Birdeye response: {sol_price_data}")
            else:
                logger.warning(f"Failed to get SOL/USD price via Birdeye: {sol_price_data}")

        except Exception as e:
            # Log and potentially include error in result
            logger.error(f"Error fetching SOL price data for market context: {str(e)}", exc_info=True)
            result["error"] = f"Failed to get SOL market price context: {str(e)}"
            
        return result
    
    async def get_token_holders(self, token_mint: str, limit: int = 20) -> Dict[str, Any]:
        """Get top token holders from Solana FM API.
        
        Args:
            token_mint: The mint address of the token
            limit: Maximum number of holders to return
            
        Returns:
            Dict containing list of holders with their balances and source information.
            Example format: {"mint": ..., "holders": [{"address": ..., "amount": ..., "ui_amount": ..., "percentage": ...}], "total_holders": ..., "source": "solana_fm"}
            Returns a default structure with source='fallback' on error.
            
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
            
            # Use httpx for the external API call
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(SOLANA_FM_API)
                response.raise_for_status() # Raise HTTP errors
                data = response.json()
                
                # Process and return the holder data
                # Check structure based on SolanaFM documentation/examples
                if data.get("status") == "success" and isinstance(data.get("result"), dict):
                    result_data = data["result"]
                    holders_list = result_data.get("data", [])
                    # Ensure holders_list is actually a list
                    if not isinstance(holders_list, list):
                         logger.warning(f"Unexpected format for Solana FM holders data: {holders_list}")
                         return default_response

                    holders = []
                    for holder in holders_list:
                        # Validate individual holder structure if necessary
                        if isinstance(holder, dict) and "address" in holder and "amount" in holder:
                            try:
                                # Amount is usually string, uiAmount might be float/int
                                holders.append({
                                    "address": holder.get("address", ""),
                                    "amount": holder.get("amount", "0"), # Keep as string from API
                                    "ui_amount": holder.get("uiAmount", 0.0), # Use provided uiAmount
                                    "percentage": holder.get("percentage", 0.0)
                                })
                            except (KeyError, TypeError, ValueError) as e:
                                logger.warning(f"Error parsing individual holder data from SolanaFM: {holder}, Error: {e}")
                        else:
                            logger.warning(f"Skipping invalid holder data item from SolanaFM: {holder}")
                    
                    return {
                        "mint": token_mint,
                        "holders": holders,
                        # Use pagination total if available, otherwise count returned items
                        "total_holders": result_data.get("pagination", {}).get("total", len(holders)), 
                        "source": "solana_fm"
                    }
                else:
                    logger.warning(f"Unexpected response status or structure from Solana FM API for {token_mint}: {data.get('status')}")
                    return default_response
                
        except httpx.HTTPStatusError as e:
             logger.error(f"HTTP error fetching holders from Solana FM for {token_mint}: {e.response.status_code} - {e.response.text}")
             return {**default_response, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Error fetching holders from Solana FM for {token_mint}: {str(e)}", exc_info=True)
            return {**default_response, "error": str(e)}
    
    async def get_all_token_data(self, token_mint: str, holder_limit: int = 10) -> Dict[str, Any]:
        """Get comprehensive token data combining metadata, supply, price, and holder information.
        
        Args:
            token_mint: The mint address of the token
            holder_limit: Maximum number of holders to return. Defaults to 10.
            
        Returns:
            Consolidated token data including metadata, supply, price, and holders
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
        
        # Fetch data concurrently
        results = await asyncio.gather(
            self.get_token_metadata(token_mint),
            self.get_token_supply(token_mint),
            self.get_token_price(token_mint), # Uses Birdeye primarily
            self.get_token_holders(token_mint, limit=holder_limit), # Uses Solana FM
            return_exceptions=True # Return exceptions instead of raising them immediately
        )

        metadata, supply_data, price_data, holders_data = results

        # Process results, checking for exceptions
        errors = []
        final_metadata = default_metadata = {"name": "Unknown", "symbol": "UNKNOWN", "uri": "", "source": "error"}
        final_supply = default_supply = {"amount": "0", "decimals": 0, "ui_amount": 0.0, "ui_amount_string": "0"}
        final_price = default_price = {"current_price_usd": 0.0, "price_change_24h": 0.0, "last_updated": datetime.datetime.now().isoformat(), "source": "error"}
        final_holders = default_holders = {"total_holders": 0, "top_holders": []}

        if isinstance(metadata, Exception):
            errors.append({"metadata_error": str(metadata)})
            logger.error(f"Failed get_token_metadata for {token_mint}: {metadata}")
        else:
            final_metadata = metadata

        if isinstance(supply_data, Exception):
            errors.append({"supply_error": str(supply_data)})
            logger.error(f"Failed get_token_supply for {token_mint}: {supply_data}")
        elif isinstance(supply_data, dict): # Supply returns the value dict directly now
            final_supply = supply_data
        else:
            errors.append({"supply_error": f"Unexpected supply data format: {type(supply_data)}"})
            logger.error(f"Unexpected supply data format for {token_mint}: {type(supply_data)}")

        if isinstance(price_data, Exception):
            errors.append({"price_error": str(price_data)})
            logger.error(f"Failed get_token_price for {token_mint}: {price_data}")
        elif isinstance(price_data, dict):
            # Adapt structure from get_token_price
            final_price = {
                 "current_price_usd": price_data.get("price", 0.0),
                 "price_change_24h": price_data.get("price_change_24h", 0.0),
                 "last_updated": price_data.get("last_updated"),
                 "source": price_data.get("source", "unknown")
             }
        else:
             errors.append({"price_error": f"Unexpected price data format: {type(price_data)}"})
             logger.error(f"Unexpected price data format for {token_mint}: {type(price_data)}")

        if isinstance(holders_data, Exception):
            errors.append({"holders_error": str(holders_data)})
            logger.error(f"Failed get_token_holders for {token_mint}: {holders_data}")
        elif isinstance(holders_data, dict):
             final_holders = {
                 "total_holders": holders_data.get("total_holders", 0),
                 "top_holders": holders_data.get("holders", [])
             }
        else:
             errors.append({"holders_error": f"Unexpected holders data format: {type(holders_data)}"})
             logger.error(f"Unexpected holders data format for {token_mint}: {type(holders_data)}")

        # Combine all data into a comprehensive response
        result = {
            "token_mint": token_mint,
            "name": final_metadata.get("name"),
            "symbol": final_metadata.get("symbol"),
            "decimals": final_supply.get("decimals"),
            "supply": {
                "amount": final_supply.get("amount"),
                "ui_amount": final_supply.get("uiAmount"),
                "ui_amount_string": final_supply.get("uiAmountString")
            },
            "price": final_price,
            "holders": final_holders,
            "metadata": {
                "uri": final_metadata.get("uri"),
                "source": final_metadata.get("source")
            },
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        if errors:
            result["errors"] = errors
            
        return result
    
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

    # New method using Helius specific getTokenAccounts
    async def helius_get_token_accounts_by_mint(
        self, 
        mint_address: str, 
        max_accounts_to_fetch: int = 2000 # Limit total accounts fetched via pagination
    ) -> List[Dict[str, Any]]:
        """Get token accounts for a mint using Helius' enhanced getTokenAccounts method.

        Handles pagination to fetch multiple pages up to `max_accounts_to_fetch`.

        Args:
            mint_address: The token mint address.
            max_accounts_to_fetch: The approximate maximum number of accounts to retrieve via pagination.

        Returns:
            List of token account details [{'address': str, 'mint': str, 'owner': str, 'amount': int, ...}].
            Returns empty list on error or if no accounts found.

        Raises:
            InvalidPublicKeyError: If the mint address is invalid.
            SolanaRpcError: If the underlying RPC call fails after retries.
        """
        if not validate_public_key(mint_address):
            raise InvalidPublicKeyError(mint_address)

        all_token_accounts = []
        page = 1
        limit_per_page = 1000 # Helius limit per page
        
        logger.info(f"Fetching token accounts for {mint_address} using Helius getTokenAccounts (max ~{max_accounts_to_fetch})")

        while True:
            try:
                logger.debug(f"Fetching page {page} for {mint_address}...")
                params = {
                    "page": page,
                    "limit": limit_per_page,
                    "mint": mint_address,
                    "displayOptions": { # Optional: customize displayed fields if needed
                        # "showZeroBalance": False # Default is False
                    }
                }
                
                # Note: _make_request extracts the "result" field automatically
                page_result = await self._make_request("getTokenAccounts", params)

                if not page_result or not isinstance(page_result.get("token_accounts"), list):
                    logger.warning(f"Helius getTokenAccounts returned unexpected result for {mint_address}, page {page}: {page_result}")
                    break # Stop pagination on unexpected result

                token_accounts_on_page = page_result["token_accounts"]
                all_token_accounts.extend(token_accounts_on_page)
                
                logger.debug(f"Fetched {len(token_accounts_on_page)} accounts on page {page} for {mint_address}. Total fetched: {len(all_token_accounts)}")

                # Stop if no more accounts on the page or we've fetched enough
                if len(token_accounts_on_page) < limit_per_page or len(all_token_accounts) >= max_accounts_to_fetch:
                    break

                page += 1
                # Add a small delay between pages to be nice to the API
                await asyncio.sleep(0.1)

            except SolanaRpcError as e:
                logger.error(f"RPC error fetching page {page} for {mint_address} via Helius getTokenAccounts: {e}")
                # Depending on the error, we might stop or continue, for now stop.
                raise # Re-raise the error to be handled by the caller
            except Exception as e:
                logger.error(f"Unexpected error during Helius getTokenAccounts pagination for {mint_address} (page {page}): {e}", exc_info=True)
                # Stop pagination on unexpected errors
                break
        
        logger.info(f"Finished fetching Helius token accounts for {mint_address}. Total found: {len(all_token_accounts)}")
        return all_token_accounts


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
        # Ensure the client's resources are released
        await client.close()