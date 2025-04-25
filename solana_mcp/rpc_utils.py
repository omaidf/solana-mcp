"""
Utilities for optimizing Solana RPC calls - batching, filtering, and connection management.
"""

import asyncio
import logging
import random
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from solana_mcp.solana_client import SolanaClient, SolanaRpcError
from solana_mcp.config import SolanaConfig
from solana_mcp.cache import global_cache

logger = logging.getLogger(__name__)

class RPCConnectionPool:
    """Manage a pool of RPC connections to distribute load."""
    
    def __init__(self, rpc_urls: List[str]):
        """Initialize the connection pool.
        
        Args:
            rpc_urls: List of RPC endpoint URLs
        """
        self.rpc_urls = rpc_urls
        self.clients = []
        self.current_index = 0
        self.is_initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize all clients in the pool."""
        if self.is_initialized:
            return
            
        async with self._lock:
            # Check again inside the lock
            if self.is_initialized:
                return
                
            for url in self.rpc_urls:
                config = SolanaConfig(rpc_url=url)
                client = SolanaClient(config)
                self.clients.append(client)
            
            self.is_initialized = True
            logger.info(f"Initialized RPC connection pool with {len(self.clients)} endpoints")
    
    def get_client(self) -> SolanaClient:
        """Get the next client in the rotation.
        
        Returns:
            A Solana RPC client
        """
        if not self.is_initialized or not self.clients:
            raise RuntimeError("RPC connection pool not initialized")
            
        client = self.clients[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.clients)
        return client
    
    def get_random_client(self) -> SolanaClient:
        """Get a random client from the pool.
        
        Returns:
            A Solana RPC client
        """
        if not self.is_initialized or not self.clients:
            raise RuntimeError("RPC connection pool not initialized")
            
        return random.choice(self.clients)
    
    async def close_all(self):
        """Close all clients in the pool."""
        for client in self.clients:
            await client.close()
        self.clients = []
        self.is_initialized = False
        logger.info("Closed all RPC connections in the pool")


async def get_multiple_accounts(
    client: SolanaClient, 
    addresses: List[str], 
    commitment: str = "confirmed",
    encoding: str = "base64"
) -> Dict[str, Any]:
    """Fetch multiple accounts in a single RPC call.
    
    Args:
        client: Solana RPC client
        addresses: List of account addresses
        commitment: Commitment level
        encoding: Data encoding
        
    Returns:
        Dictionary mapping addresses to account data
    """
    if not addresses:
        return {}
    
    # Process in chunks to stay within RPC limits
    CHUNK_SIZE = 100
    all_results = {}
    
    for i in range(0, len(addresses), CHUNK_SIZE):
        chunk = addresses[i:i+CHUNK_SIZE]
        
        # Use the getMultipleAccounts RPC method
        method = "getMultipleAccounts"
        params = [
            chunk,
            {"commitment": commitment, "encoding": encoding}
        ]
        
        try:
            response = await client._make_request(method, params)
            
            if "result" in response and "value" in response["result"]:
                results = response["result"]["value"]
                for j, result in enumerate(results):
                    # Match results with their addresses
                    if result:
                        all_results[chunk[j]] = result
            
            # Add a small delay between chunks to avoid rate limiting
            if i + CHUNK_SIZE < len(addresses):
                await asyncio.sleep(0.2)
                
        except SolanaRpcError as e:
            logger.error(f"Error fetching accounts batch {i//CHUNK_SIZE}: {str(e)}")
            # Continue with the next batch instead of failing completely
    
    return all_results


async def get_filtered_token_accounts(
    client: SolanaClient,
    mint: str, 
    min_balance: float = 0.0,
    max_accounts: int = 100,
    order: str = "desc"  # 'desc' for largest first
) -> List[Dict[str, Any]]:
    """Get filtered token accounts to optimize RPC queries.
    
    Args:
        client: Solana RPC client
        mint: Token mint address
        min_balance: Minimum token balance to include
        max_accounts: Maximum number of accounts to return
        order: Sort order ('asc' or 'desc')
        
    Returns:
        List of token accounts meeting criteria
    """
    try:
        # Get token accounts
        largest_accounts_result = await client.get_token_largest_accounts(mint)
        
        if "value" not in largest_accounts_result or not largest_accounts_result["value"]:
            return []
        
        # Filter and sort accounts
        accounts = largest_accounts_result["value"]
        filtered_accounts = []
        
        for account in accounts:
            ui_amount = account.get("uiAmount", 0)
            # Handle different types of uiAmount
            if ui_amount is None:
                ui_amount = 0.0
            elif isinstance(ui_amount, str):
                try:
                    ui_amount = float(ui_amount)
                except (ValueError, TypeError):
                    ui_amount = 0.0
            
            # Apply minimum balance filter
            if ui_amount >= min_balance:
                filtered_accounts.append(account)
        
        # Sort accounts by balance
        if order.lower() == "asc":
            filtered_accounts.sort(key=lambda x: float(x.get("uiAmount", 0) or 0))
        else:
            filtered_accounts.sort(key=lambda x: float(x.get("uiAmount", 0) or 0), reverse=True)
        
        # Limit to max_accounts
        return filtered_accounts[:max_accounts]
        
    except Exception as e:
        logger.error(f"Error getting filtered token accounts for {mint}: {str(e)}")
        return []


async def get_token_account_owners(
    client: SolanaClient,
    token_accounts: List[Dict[str, Any]]
) -> Dict[str, str]:
    """Extract token account owners efficiently.
    
    Args:
        client: Solana RPC client
        token_accounts: List of token accounts
        
    Returns:
        Mapping of token account addresses to owner addresses
    """
    if not token_accounts:
        return {}
    
    # Get addresses to query
    addresses = [account.get("address") for account in token_accounts if account.get("address")]
    
    # Get account info for all addresses in one batch
    account_info_map = await get_multiple_accounts(client, addresses)
    
    # Extract owner addresses
    token_account_owners = {}
    token_program_id = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    
    for account_address in addresses:
        account_info = account_info_map.get(account_address)
        
        if account_info:
            try:
                # Check if account is owned by token program
                account_owner = account_info.get("owner")
                
                if account_owner == token_program_id:
                    # Extract owner from account data
                    data = account_info.get("data")
                    
                    if isinstance(data, list) and len(data) >= 2 and data[0] == "base64":
                        import base64
                        import base58
                        
                        # Decode base64 data
                        data_bytes = base64.b64decode(data[1])
                        
                        # Check if data length is sufficient
                        if len(data_bytes) >= 64:
                            # Extract owner pubkey (offset 32 in token account data)
                            owner = base58.b58encode(data_bytes[32:64]).decode('utf-8')
                            token_account_owners[account_address] = owner
            except Exception as e:
                logger.error(f"Error extracting owner for {account_address}: {str(e)}")
    
    return token_account_owners


async def get_wallet_age(
    client: SolanaClient,
    wallet_address: str
) -> Optional[int]:
    """Get wallet age in days based on oldest transaction.
    
    Args:
        client: Solana RPC client
        wallet_address: Wallet address
        
    Returns:
        Age in days or None if unknown
    """
    import datetime
    
    try:
        # Get oldest signature
        response = await client.get_signatures_for_address(wallet_address, limit=1, before=None)
        
        if response and "result" in response and response["result"]:
            # Extract creation timestamp from oldest signature
            block_time = response["result"][0].get("blockTime")
            
            if block_time:
                creation_date = datetime.datetime.fromtimestamp(block_time)
                current_date = datetime.datetime.now()
                age_days = (current_date - creation_date).days
                return age_days
    except Exception as e:
        logger.error(f"Error getting wallet age for {wallet_address}: {str(e)}")
    
    return None


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 8.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: Tuple = (SolanaRpcError,)
):
    """Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_factor: Factor to increase backoff time
        allowed_exceptions: Exceptions that trigger a retry
        
    Returns:
        Function result
    
    Raises:
        The last exception if all retries fail
    """
    retries = 0
    last_exception = None
    backoff = initial_backoff
    
    while retries <= max_retries:
        try:
            return await func()
        except allowed_exceptions as e:
            last_exception = e
            retries += 1
            
            if retries > max_retries:
                break
                
            # Add some jitter to avoid thundering herd
            jitter = random.uniform(0.8, 1.2)
            sleep_time = min(backoff * jitter, max_backoff)
            
            logger.warning(f"Retry {retries}/{max_retries} after {sleep_time:.2f}s: {str(e)}")
            await asyncio.sleep(sleep_time)
            
            # Increase backoff for next retry
            backoff = min(backoff * backoff_factor, max_backoff)
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    
    # This should never happen, but just in case
    raise RuntimeError("Retries failed but no exception was raised") 