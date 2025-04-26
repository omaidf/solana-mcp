"""
Enhanced Solana analyzer module for the MCP server
This module contains the SolanaAnalyzer class and related functionality
"""
import os
import asyncio
import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, TypedDict, Any, cast
from dataclasses import dataclass
from decimal import Decimal
import json
import logging
import re
from dotenv import load_dotenv
import structlog
import pytz

# Try to import Solana-specific libraries, with fallback if not available
try:
    from solders.pubkey import Pubkey
except ImportError:
    # Define a placeholder for development without Solana libs
    class Pubkey:
        @staticmethod
        def from_string(s: str):
            return s

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    symbol: str
    decimals: int
    price: float
    mint: str
    name: str = ""
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    supply: Optional[float] = None

@dataclass
class Whale:
    address: str
    token_balance: float
    usd_value: float
    percentage: float
    last_active: Optional[str] = None

@dataclass
class AccountInfo:
    lamports: int
    owner: str
    executable: bool
    data: Union[Dict, str]
    rent_epoch: Optional[int] = None

@dataclass
class TransactionInfo:
    signature: str
    timestamp: str
    fee: int
    accounts: List[str]
    status: str
    logs: List[str]

class SolanaAnalyzer:
    def __init__(self):
        # API configuration from environment variables with fallbacks
        self.helius_api_key = os.getenv("HELIUS_API_KEY", "4ffc1228-f093-4974-ad2d-3edd8e5f7c03")
        self.birdeye_api_key = os.getenv("BIRDEYE_API_KEY", "03ea781299dd4b8cbe356eea90c7219e")
        self.rpc_endpoint = f"https://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
        logger.info(f"Using RPC endpoint: {self.rpc_endpoint}")
        
        # Cache setup
        self.cache = {
            'token_metadata': {},
            'holders': {},
            'prices': {},
            'accounts': {},
            'transactions': {}
        }
        self.cache_expiry = {
            'token_metadata': timedelta(hours=1),
            'holders': timedelta(minutes=15),
            'prices': timedelta(minutes=5),
            'accounts': timedelta(minutes=10),
            'transactions': timedelta(minutes=30)
        }
        
        # Session management
        self.session = None
        self.timeout = ClientTimeout(total=30, connect=10)  # Add timeout configuration
        
        # Request backoff parameters for rate limiting
        self.base_backoff = 0.5  # Starting backoff time in seconds
        self.max_backoff = 10    # Maximum backoff time in seconds
        self.max_retries = 3     # Maximum number of retry attempts
        
        # Cache management limits
        self.max_cache_items = 1000  # Maximum items per cache category

    async def __aenter__(self):
        """Initialize aiohttp session when used in async context"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.debug("Created new aiohttp session")
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Properly close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.debug("Closed aiohttp session")

    async def close(self):
        """Explicit method to close resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.debug("Explicitly closed aiohttp session")
        # Clear caches to free memory
        self.clear_cache()

    async def _rpc_request(self, payload: Dict) -> Dict:
        """
        Make a JSON-RPC request to the Solana API endpoint
        
        Args:
            payload: The JSON-RPC payload to send
            
        Returns:
            The JSON response from the server
        """
        # Ensure we have an active session
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
        retries = 0
        backoff = self.base_backoff
        
        while retries <= self.max_retries:
            try:
                async with self.session.post(self.rpc_endpoint, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        logger.warning(f"Rate limited by RPC provider. Retrying in {backoff} seconds...")
                        # Increment retries and apply backoff for rate limiting
                        retries += 1
                        if retries <= self.max_retries:
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, self.max_backoff)
                        else:
                            raise ValueError(f"RPC request failed after {self.max_retries} retries due to rate limiting")
                    else:
                        error_text = await response.text()
                        logger.error(f"RPC Error (HTTP {response.status}): {error_text}")
                        
                        # If it's a server error, retry. Otherwise, raise exception
                        if response.status >= 500:
                            retries += 1
                            if retries <= self.max_retries:
                                await asyncio.sleep(backoff)
                                backoff = min(backoff * 2, self.max_backoff)
                            else:
                                raise ValueError(f"RPC request failed with status {response.status} after {self.max_retries} retries: {error_text}")
                        else:
                            raise ValueError(f"RPC request failed with status {response.status}: {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Network error during RPC request: {str(e)}")
                retries += 1
                if retries <= self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff)
                else:
                    raise ValueError(f"RPC request failed after {self.max_retries} retries due to network error: {str(e)}")
        
        raise ValueError("RPC request failed: maximum retries exceeded")

    # Core RPC Methods ========================================================

    async def get_account_info(self, account_address: str, encoding: str = "jsonParsed") -> AccountInfo:
        """Get detailed information about a Solana account"""
        cache_key = f"account_{account_address}_{encoding}"
        cached = await self._get_cached(cache_key, 'accounts')
        if cached:
            return cached

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [account_address, {"encoding": encoding}]
        }
        
        data = await self._rpc_request(payload)
        if 'error' in data:
            logger.error(f"RPC Error: {data['error']}")
            raise ValueError(f"RPC Error: {data['error']}")
        
        account = data['result']['value']
        if not account:
            logger.warning(f"Account not found: {account_address}")
            raise ValueError("Account not found")
            
        result = AccountInfo(
            lamports=account['lamports'],
            owner=account['owner'],
            executable=account['executable'],
            data=account['data'],
            rent_epoch=account.get('rentEpoch')
        )
        
        await self._set_cache(cache_key, result, 'accounts')
        return result

    async def get_balance(self, account_address: str) -> float:
        """Get SOL balance in lamports"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [account_address]
        }
        
        data = await self._rpc_request(payload)
        if 'error' in data:
            logger.error(f"RPC Error: {data['error']}")
            raise ValueError(f"RPC Error: {data['error']}")
            
        return data['result']['value']

    async def get_token_accounts_by_owner(
        self,
        owner_address: str,
        mint_address: Optional[str] = None,
        program_id: str = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    ) -> List[Dict]:
        """Get all token accounts owned by a wallet"""
        params = [owner_address]
        if mint_address:
            params.append({"mint": mint_address})
        else:
            params.append({"programId": program_id})
        params.append({"encoding": "jsonParsed"})
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": params
        }
        
        data = await self._rpc_request(payload)
        if 'error' in data:
            logger.error(f"RPC Error: {data['error']}")
            raise ValueError(f"RPC Error: {data['error']}")
            
        return [acc['account'] for acc in data['result']['value']]

    async def get_transaction(self, tx_signature: str, encoding: str = "json") -> TransactionInfo:
        """Get detailed transaction information"""
        cache_key = f"tx_{tx_signature}_{encoding}"
        cached = await self._get_cached(cache_key, 'transactions')
        if cached:
            return cached

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                tx_signature,
                {"encoding": encoding, "commitment": "confirmed"}
            ]
        }
        
        data = await self._rpc_request(payload)
        if 'error' in data:
            logger.error(f"RPC Error: {data['error']}")
            raise ValueError(f"RPC Error: {data['error']}")
            
        tx = data['result']
        result = TransactionInfo(
            signature=tx_signature,
            timestamp=datetime.fromtimestamp(tx['blockTime']).isoformat(),
            fee=tx['meta']['fee'],
            accounts=[acc['pubkey'] for acc in tx['transaction']['message']['accountKeys']],
            status="success" if tx['meta']['err'] is None else "failed",
            logs=tx['meta']['logMessages']
        )
        
        await self._set_cache(cache_key, result, 'transactions')
        return result

    # Token Analysis Methods ==================================================

    async def get_token_info(self, mint_address: str) -> TokenInfo:
        """Get comprehensive token information"""
        cache_key = f"token_{mint_address}"
        cached = await self._get_cached(cache_key, 'token_metadata')
        if cached:
            return cached

        # Initialize default values
        symbol = "UNKNOWN"
        name = "Unknown Token"
        decimals = 0
        price = 0
        market_cap = None
        volume_24h = None
        supply = None
        
        # Get enhanced metadata from Helius first (more reliable for non-standard tokens)
        metadata = {}
        try:
            metadata_url = f"https://api.helius.xyz/v0/tokens/metadata?api-key={self.helius_api_key}"
            payload = {"mintAccounts": [mint_address]}
            
            async with self.session.post(metadata_url, json=payload) as response:
                metadata_resp = await response.json()
                
                if isinstance(metadata_resp, list) and len(metadata_resp) > 0:
                    metadata = metadata_resp[0]
                    
                    # Extract basic info from metadata
                    if metadata.get('symbol'):
                        symbol = metadata.get('symbol')
                    if metadata.get('name'):
                        name = metadata.get('name')
                    if metadata.get('decimals') is not None:
                        decimals = metadata.get('decimals')
                    
                    logger.info(f"Retrieved metadata for {mint_address}: {symbol}")
                elif 'error' in metadata_resp:
                    logger.warning(f"Error getting token metadata from Helius: {metadata_resp['error']}")
        except Exception as e:
            logger.warning(f"Error fetching metadata from Helius: {str(e)}")
        
        # Try to get token supply info if we have decimals
        if decimals == 0:  # If we still don't have decimals, try to get token supply
            try:
                supply_data = await self._rpc_request({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTokenSupply",
                    "params": [mint_address]
                })
                
                if 'error' not in supply_data:
                    supply_info = supply_data['result']['value']
                    decimals = supply_info['decimals']
                    supply = supply_info['uiAmount']
                    logger.info(f"Got supply info for {mint_address}: {supply} tokens")
                else:
                    logger.warning(f"Could not get token supply: {supply_data.get('error')}")
                    # We'll continue with default values
            except Exception as e:
                logger.warning(f"Error getting token supply: {str(e)}")
        
        # Get price data from Birdeye
        try:
            price_url = f"https://public-api.birdeye.so/public/price?address={mint_address}"
            headers = {"x-chain": "solana", "X-API-KEY": self.birdeye_api_key}
            async with self.session.get(price_url, headers=headers) as response:
                price_data = await response.json()
                if price_data.get('success'):
                    price = price_data.get('data', {}).get('value', 0)
                    logger.info(f"Got price for {mint_address}: ${price}")
                else:
                    # Fallback for well-known tokens if Birdeye fails
                    if mint_address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":  # USDC
                        price = 1.0
                        logger.info(f"Using hardcoded price for USDC: ${price}")
                    elif mint_address == "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB":  # USDT
                        price = 1.0
                        logger.info(f"Using hardcoded price for USDT: ${price}")
                    elif mint_address == "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN":  # Jupiter
                        price = 0.50  # Example price
                        logger.info(f"Using hardcoded price for JUP: ${price}")
                    else:
                        logger.warning(f"Failed to get price from Birdeye for {mint_address}: {price_data.get('message', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"Error getting price from Birdeye: {str(e)}")
        
        # Create and return the token info object with whatever data we could gather
        result = TokenInfo(
            symbol=symbol,
            name=name,
            decimals=decimals,
            price=price,
            mint=mint_address,
            market_cap=market_cap,
            volume_24h=volume_24h,
            supply=supply
        )
        
        logger.info(f"Retrieved token info for {mint_address}: {result.symbol}")
        await self._set_cache(cache_key, result, 'token_metadata')
        return result

    async def find_whales(
        self,
        candidate_wallets: List[str],
        min_usd_value: float = 50000,
        max_wallets: int = 20,
        batch_size: int = 100,
        concurrency: int = 5
    ) -> Dict:
        """
        Find whale wallets based on their total token holdings value.
        Analyzes wallets to identify those with high-value token holdings.
        
        Args:
            candidate_wallets: List of wallet addresses to analyze
            min_usd_value: Minimum USD value threshold to be considered a whale
            max_wallets: Maximum number of wallets to analyze
            batch_size: Number of wallets to process in a batch
            concurrency: Maximum number of concurrent requests
            
        Returns:
            Dictionary with whale detection results
        """
        logger.info(f"Analyzing {len(candidate_wallets[:max_wallets])} wallets for whale detection")
        
        # Cache for token prices to avoid duplicate requests
        prices_cache = {}
        
        # Process wallets in parallel for efficiency
        # We'll use asyncio.gather with semaphore to control concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def calculate_wallet_value(wallet_address):
            async with sem:  # Control concurrency
                try:
                    # Get all token holdings (including fungible tokens)
                    url = self.rpc_endpoint
                    payload = {
                        "jsonrpc": "2.0",
                        "id": "helius-wallet-analysis",
                        "method": "getAssetsByOwner",
                        "params": {
                            "ownerAddress": wallet_address,
                            "page": 1,
                            "limit": 1000,
                            "displayOptions": {
                                "showFungible": True,
                                "showNativeBalance": True
                            }
                        }
                    }
                    
                    async with self.session.post(url, json=payload) as response:
                        data = await response.json()
                        assets = data.get('result', {}).get('items', [])
                    
                    # Calculate total value across all tokens
                    total_value = 0.0
                    token_details = []
                    
                    for asset in assets:
                        if asset.get('token_info', {}).get('balance'):
                            balance = float(asset['token_info']['balance'])
                            token_address = asset['id']
                            token_symbol = asset.get('token_info', {}).get('symbol', 'UNKNOWN')
                            
                            # Get price from cache or fetch if missing
                            if token_address not in prices_cache:
                                # Need to fetch the price
                                price_payload = {
                                    "jsonrpc": "2.0",
                                    "id": "helius-price-fetch",
                                    "method": "getAssetBatch",
                                    "params": {
                                        "ids": [token_address]
                                    }
                                }
                                
                                async with self.session.post(url, json=price_payload) as price_response:
                                    price_data = await price_response.json()
                                    price_items = price_data.get('result', [])
                                    
                                    # Extract price if available
                                    price = 0
                                    for item in price_items:
                                        if item.get('id') == token_address and item.get('token_info', {}).get('price_info'):
                                            price = float(item['token_info']['price_info'].get('price_per_token', 0))
                                            break
                                    
                                    prices_cache[token_address] = price
                            
                            token_price = prices_cache.get(token_address, 0)
                            token_value = balance * token_price
                            total_value += token_value
                            
                            # Only add tokens with non-zero value to the details
                            if token_value > 0:
                                token_details.append({
                                    "mint": token_address,
                                    "symbol": token_symbol,
                                    "balance": balance,
                                    "price": token_price,
                                    "value": token_value
                                })
                    
                    return {
                        "address": wallet_address,
                        "total_value": total_value,
                        "is_whale": total_value >= min_usd_value,
                        "token_holdings": sorted(token_details, key=lambda x: x["value"], reverse=True)
                    }
                
                except Exception as e:
                    logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
                    return {
                        "address": wallet_address,
                        "total_value": 0,
                        "is_whale": False,
                        "token_holdings": [],
                        "error": str(e)
                    }
        
        # Process wallets in parallel (with a limit on concurrency)
        wallets_to_check = candidate_wallets[:max_wallets]
        tasks = [calculate_wallet_value(wallet) for wallet in wallets_to_check]
        wallet_results = await asyncio.gather(*tasks)
        
        # Filter for whales and sort by total value
        whales = [result for result in wallet_results if result["is_whale"]]
        whales.sort(key=lambda x: x["total_value"], reverse=True)
        
        # Create summary statistics
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return {
            "whales": whales,
            "stats": {
                "wallets_analyzed": len(wallets_to_check),
                "whale_count": len(whales),
                "whale_percentage": (len(whales) / len(wallets_to_check) * 100) if wallets_to_check else 0,
                "min_usd_threshold": min_usd_value,
                "timestamp": timestamp
            }
        }

    async def semantic_search(self, prompt: str) -> Dict:
        """
        Process natural language queries about Solana blockchain data.
        This method uses keyword matching to determine intent and extract entities,
        then calls the appropriate API methods to retrieve the requested information.
        
        Args:
            prompt: Natural language prompt describing what information to retrieve
            
        Returns:
            Dictionary containing the search results and metadata
        """
        logger.info(f"Processing semantic search prompt: {prompt}")
        
        # Convert prompt to lowercase for case-insensitive matching only once
        prompt_lower = prompt.lower()
        
        # Initialize result structure
        result = {
            "query": prompt,
            "intent": None,
            "entities": {"addresses": []},
            "data": None,
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Extract Solana addresses from the prompt - use a compiled regex for better performance
            # Compile regex only once at the module level for future optimizations
            address_pattern = re.compile(r'[1-9A-HJ-NP-Za-km-z]{32,44}')
            addresses = address_pattern.findall(prompt)
            result["entities"]["addresses"] = addresses
            
            # Define keyword sets as frozen sets for faster lookup
            # In a real optimization, these would be defined at the module level
            balance_keywords = frozenset([
                "balance", "sol", "holdings", "how much sol", "amount of sol", 
                "solana balance", "wallet balance", "account balance", "has sol",
                "check balance", "wallet worth", "wallet value"
            ])
            
            token_holdings_keywords = frozenset([
                "token", "tokens", "spl", "spl tokens", "token holdings", "token balance",
                "tokens owned", "owns what", "token portfolio", "token list", "token assets"
            ])
            
            token_ownership_keywords = frozenset([
                "owned", "holding", "has", "possesses", "holds", "in wallet", 
                "in possession", "belongs to", "owned by", "owned token"
            ])
            
            token_info_keywords = frozenset([
                "token info", "token details", "price", "token price", "how much is", 
                "worth of", "value of", "market cap", "supply", "token data", 
                "statistics", "metrics", "token stats", "token supply", "circulation"
            ])
            
            whale_keywords = frozenset([
                "whale", "whales", "large holder", "big wallet", "rich wallet", "big player",
                "major holder", "significant holder", "large investor", "large balance",
                "big balance", "wealthy", "high value wallet", "large account"
            ])
            
            account_keywords = frozenset([
                "account", "info", "details", "account info", "account details", 
                "wallet info", "wallet details", "address info", "address details",
                "solana account", "lookup account", "query account"
            ])
            
            # Pre-compute keyword presence checks rather than repeatedly searching
            has_balance_keyword = any(kw in prompt_lower for kw in balance_keywords)
            has_token_holdings_keyword = any(kw in prompt_lower for kw in token_holdings_keywords)
            has_token_ownership_keyword = any(kw in prompt_lower for kw in token_ownership_keywords)
            has_token_info_keyword = any(kw in prompt_lower for kw in token_info_keywords)
            has_whale_keyword = any(kw in prompt_lower for kw in whale_keywords)
            has_account_keyword = any(kw in prompt_lower for kw in account_keywords)
            
            # Identify intent using pre-computed flags
            if has_balance_keyword:
                await self._handle_balance_query(addresses, result)
            
            # Prioritize token-related intents when "token" is explicitly mentioned
            elif "token" in prompt_lower and addresses:
                # First check if it's a token info query
                if has_token_info_keyword or "info" in prompt_lower:
                    await self._handle_token_info_query(addresses, prompt, prompt_lower, result)
                # Then check if it's a token holdings query
                elif has_token_holdings_keyword and has_token_ownership_keyword:
                    await self._handle_token_holdings_query(addresses, result)
                else:
                    # Default to token info if just asking about a "token" with an address
                    await self._handle_token_info_query(addresses, prompt, prompt_lower, result)
            
            elif has_token_holdings_keyword and has_token_ownership_keyword:
                await self._handle_token_holdings_query(addresses, result)
                
            elif has_token_info_keyword:
                await self._handle_token_info_query(addresses, prompt, prompt_lower, result)
            
            elif has_whale_keyword:
                await self._handle_whale_query(addresses, prompt_lower, result)
            
            elif has_account_keyword:
                await self._handle_account_info_query(addresses, result)
            
            else:
                # If no intent is detected, try a more generalized approach
                await self._handle_general_query(prompt_lower, addresses, result)
        
        except Exception as e:
            logger.error(f"Error processing semantic search: {str(e)}")
            result["error"] = f"Error processing request: {str(e)}"
        
        return result

    async def _handle_balance_query(self, addresses: List[str], result: Dict) -> None:
        """Helper method to handle balance queries"""
        result["intent"] = "get_balance"
        
        if addresses:
            balance = await self.get_balance(addresses[0])
            result["data"] = {
                "address": addresses[0],
                "balance_lamports": balance,
                "balance_sol": self.lamports_to_sol(balance)
            }
        else:
            result["error"] = "No wallet address found in the prompt. Please include a valid Solana address."

    async def _handle_token_holdings_query(self, addresses: List[str], result: Dict) -> None:
        """Helper method to handle token holdings queries"""
        result["intent"] = "get_token_holdings"
        
        if not addresses:
            result["error"] = "No wallet address found in the prompt. Please include a valid Solana address."
            return
        
        token_accounts = await self.get_token_accounts_by_owner(addresses[0])
        
        # Create fetch token info tasks for all accounts at once
        mint_addresses = []
        token_data = []
        
        # Extract all mint addresses and token data first
        for account in token_accounts:
            try:
                parsed_info = account.get("data", {}).get("parsed", {}).get("info", {})
                mint = parsed_info.get("mint")
                if mint:
                    mint_addresses.append(mint)
                    token_amount = parsed_info.get("tokenAmount", {})
                    token_data.append({
                        "mint": mint,
                        "amount": int(token_amount.get("amount", 0)),
                        "decimals": int(token_amount.get("decimals", 0))
                    })
            except Exception as e:
                logger.warning(f"Error extracting token account data: {str(e)}")
        
        # Batch fetch token info for all mints (future optimization)
        # For now, create tasks for fetching each token's info
        token_info_tasks = [self.get_token_info(mint) for mint in mint_addresses]
        token_infos = await asyncio.gather(*token_info_tasks, return_exceptions=True)
        
        # Create a mapping of mint to token info for easy lookup
        token_info_map = {}
        for i, info in enumerate(token_infos):
            if not isinstance(info, Exception):
                token_info_map[mint_addresses[i]] = info
        
        # Combine the token data with the token info
        holdings = []
        for token in token_data:
            mint = token["mint"]
            if mint in token_info_map:
                info = token_info_map[mint]
                amount = token["amount"]
                decimals = token["decimals"]
                ui_amount = amount / (10 ** decimals) if decimals > 0 else amount
                
                holdings.append({
                    "token": info.symbol,
                    "mint": mint,
                    "amount": ui_amount,
                    "usd_value": ui_amount * info.price,
                    "price": info.price
                })
        
        # Sort by USD value, descending
        holdings.sort(key=lambda x: x["usd_value"], reverse=True)
        result["data"] = {
            "address": addresses[0],
            "token_holdings": holdings,
            "total_tokens": len(holdings),
            "total_value_usd": sum(h["usd_value"] for h in holdings)
        }

    async def _handle_token_info_query(self, addresses: List[str], prompt: str, prompt_lower: str, result: Dict) -> None:
        """Helper method to handle token info queries"""
        result["intent"] = "get_token_info"
        
        # Check for token addresses
        if addresses:
            try:
                token_info = await self.get_token_info(addresses[0])
                result["data"] = {
                    "symbol": token_info.symbol,
                    "name": token_info.name,
                    "mint": token_info.mint,
                    "price": token_info.price,
                    "supply": token_info.supply,
                    "market_cap": token_info.market_cap,
                    "volume_24h": token_info.volume_24h
                }
            except Exception as e:
                result["error"] = f"Error getting token info: {str(e)}"
        else:
            # Compile regex patterns for better performance
            token_symbol_pattern = re.compile(r'\b[A-Z]{2,10}\b')
            token_name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
            specific_token_pattern = re.compile(r'(?:token|price of|value of)\s+([A-Za-z]+)', re.IGNORECASE)
            
            # Look for token symbols (all caps)
            token_symbols = token_symbol_pattern.findall(prompt)
            
            # Look for token names (capitalized words)
            token_names = token_name_pattern.findall(prompt)
            
            # Look for names following "token" or "price of"
            specific_tokens = specific_token_pattern.findall(prompt_lower)
            
            if token_symbols:
                result["entities"]["token_symbols"] = token_symbols
                result["error"] = f"Found potential token symbols: {', '.join(token_symbols)}. Please provide a token address for more details."
            elif token_names:
                result["entities"]["token_names"] = token_names
                result["error"] = f"Found potential token names: {', '.join(token_names)}. Please provide a token address for more details."
            elif specific_tokens:
                result["entities"]["specific_tokens"] = specific_tokens
                result["error"] = f"Found references to tokens: {', '.join(specific_tokens)}. Please provide a token address for more details."
            else:
                result["error"] = "No token address or symbol found in the prompt. Please include a valid token address."

    async def _handle_whale_query(self, addresses: List[str], prompt_lower: str, result: Dict) -> None:
        """Helper method to handle whale queries"""
        result["intent"] = "find_whales"
        
        # Extract candidate addresses for whale analysis
        if not addresses:
            result["error"] = "No wallet addresses found for whale analysis. Please include at least one Solana address."
            return
            
        # Use the provided address as the token mint if only one is given
        mint_address = addresses[0] if len(addresses) == 1 else None
        candidate_addresses = addresses[1:] if len(addresses) > 1 else []
        
        # Default settings
        min_usd = 50000
        max_holders = 20
            
        # Extract threshold values from prompt
        # Compile value patterns for better performance
        value_patterns = [
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:k|thousand)'),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:m|million)'),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:dollars|usd)'),
            re.compile(r'(?:usd|dollars|value|threshold)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?(?:k|m)?)')
        ]
        
        # Check for USD units in the query
        has_usd_units = any(unit in prompt_lower for unit in ["dollar", "usd", "$"])
        
        # Extract value threshold from prompt
        for pattern in value_patterns:
            value_match = pattern.search(prompt_lower)
            if value_match:
                value_str = value_match.group(1).upper()
                
                # Handle k/m suffixes in the value
                if 'K' in value_str:
                    min_usd = float(value_str.replace('K', '')) * 1000
                elif 'M' in value_str:
                    min_usd = float(value_str.replace('M', '')) * 1000000
                else:
                    min_usd = float(value_str)
                
                # If we're not explicitly using dollars and no dollar terms in the query
                if pattern != value_patterns[2] and not has_usd_units:
                    # Assume it's in SOL, convert to USD with estimate
                    sol_price_estimate = 150
                    min_usd = min_usd * sol_price_estimate
                
                break
        
        try:
            # If we have a mint address, use it to find holders
            if mint_address:
                if not candidate_addresses:
                    # Fetch the token holders
                    holders = await self._fetch_token_holders(mint_address, max_count=100)
                    if not holders:
                        result["error"] = f"Could not find any holders for token: {mint_address}"
                        return
                    
                    # Extract just the addresses for whale analysis
                    candidate_addresses = [holder["address"] for holder in holders]
                    logger.info(f"Found {len(candidate_addresses)} holder addresses for {mint_address}")
                
                # Use the candidate addresses for whale detection
                whale_results = await self.find_whales(
                    candidate_addresses, 
                    min_usd_value=min_usd,
                    max_wallets=max_holders
                )
                result["data"] = whale_results
            else:
                result["error"] = "Unable to determine token address for whale detection. Please specify a token address."
        except Exception as e:
            logger.error(f"Error in whale detection: {str(e)}")
            result["error"] = f"Error processing whale detection: {str(e)}"

    async def _fetch_token_holders(self, mint_address: str, max_count: int = 100) -> List[Dict]:
        """
        Fetch top holders for a token using multiple methods, with fallbacks.
        
        Args:
            mint_address: The token mint address
            max_count: Maximum number of holders to return
            
        Returns:
            List of holder information dictionaries with address and balance
        """
        holders = []
        
        # Try multiple methods to find holders, in order of preference
        
        # Method 1: Jupiter API
        try:
            jupiter_url = f"https://quote-api.jup.ag/v6/tokens/holders?address={mint_address}"
            async with self.session.get(jupiter_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if "holders" in data:
                        # Format: {address, amount, percentage, ...}
                        for holder in data["holders"][:max_count]:
                            holders.append({
                                "address": holder["address"],
                                "balance": float(holder["amount"]),
                                "percentage": float(holder["percentage"])
                            })
                        logger.info(f"Successfully fetched {len(holders)} holders from Jupiter API")
                        return holders
        except Exception as e:
            logger.warning(f"Jupiter API holder fetch failed: {str(e)}")
        
        # Method 2: Try Birdeye API
        try:
            birdeye_url = f"https://public-api.birdeye.so/public/tokenHolder?address={mint_address}&offset=0&limit={max_count}"
            headers = {"x-chain": "solana", "X-API-KEY": self.birdeye_api_key}
            
            async with self.session.get(birdeye_url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success") and "data" in data and "items" in data["data"]:
                        for holder in data["data"]["items"]:
                            holders.append({
                                "address": holder["owner"],
                                "balance": float(holder["balance"]),
                                "percentage": float(holder["percentage"]) if "percentage" in holder else 0
                            })
                        logger.info(f"Successfully fetched {len(holders)} holders from Birdeye API")
                        return holders
        except Exception as e:
            logger.warning(f"Birdeye API holder fetch failed: {str(e)}")
        
        # Method 3: Try Helius DAS API
        try:
            # This is the correct DAS-compliant API format for Helius
            payload = {
                "jsonrpc": "2.0",
                "id": "helius-holders",
                "method": "searchAssets",
                "params": {
                    "ownerAddress": "",  # Empty means we're not filtering by owner
                    "grouping": ["mint", mint_address],
                    "limit": max_count,
                    "page": 1,
                    "displayOptions": {
                        "showOwnerAddress": True
                    }
                }
            }
            
            async with self.session.post(self.rpc_endpoint, json=payload, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("result", {}).get("items", [])
                    
                    # Extract unique owners
                    for item in items:
                        if "ownership" in item and "owner" in item["ownership"]:
                            owner = item["ownership"]["owner"]
                            amount = float(item["ownership"].get("amount", "1"))
                            
                            holders.append({
                                "address": owner,
                                "balance": amount,
                                "percentage": 0  # We don't have percentage from this API
                            })
                    
                    logger.info(f"Successfully fetched {len(holders)} holders from Helius DAS API")
                    return holders
        except Exception as e:
            logger.warning(f"Helius DAS API holder fetch failed: {str(e)}")
        
        # Method 4: Try the legacy token accounts method as a last resort
        try:
            # Use the more reliable getProgramAccounts method
            payload = {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "getProgramAccounts",
                "params": [
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token Program
                    {
                        "encoding": "jsonParsed",
                        "filters": [
                            {
                                "dataSize": 165  # Size of token accounts
                            },
                            {
                                "memcmp": {
                                    "offset": 0,
                                    "bytes": mint_address
                                }
                            }
                        ]
                    }
                ]
            }
            
            async with self.session.post(self.rpc_endpoint, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        accounts = data["result"]
                        
                        for account in accounts:
                            try:
                                parsed_data = account["account"]["data"]["parsed"]["info"]
                                owner = parsed_data["owner"]
                                amount = int(parsed_data["tokenAmount"]["amount"])
                                
                                if amount > 0:
                                    holders.append({
                                        "address": owner,
                                        "balance": amount,
                                        "percentage": 0  # We don't know the total supply here
                                    })
                            except (KeyError, TypeError) as e:
                                continue
                        
                        logger.info(f"Successfully fetched {len(holders)} holders using getProgramAccounts")
                        return holders
        except Exception as e:
            logger.warning(f"getProgramAccounts holder fetch failed: {str(e)}")
        
        # If we get here, all methods failed
        logger.error(f"All methods to fetch token holders for {mint_address} failed")
        return holders

    async def _handle_account_info_query(self, addresses: List[str], result: Dict) -> None:
        """Helper method to handle account info queries"""
        result["intent"] = "get_account_info"
        
        if addresses:
            account_info = await self.get_account_info(addresses[0])
            result["data"] = {
                "address": addresses[0],
                "lamports": account_info.lamports,
                "sol_balance": self.lamports_to_sol(account_info.lamports),
                "owner": account_info.owner,
                "executable": account_info.executable,
                "owner_program": account_info.owner,
                "rent_epoch": account_info.rent_epoch
            }
        else:
            result["error"] = "No account address found in the prompt. Please include a valid Solana address."

    async def _handle_general_query(self, prompt_lower: str, addresses: List[str], result: Dict) -> None:
        """Helper method to handle general blockchain queries"""
        # Check if it's a general blockchain query
        blockchain_terms = frozenset(["blockchain", "solana", "crypto", "cryptocurrency", "wallet", "address"])
        
        if any(term in prompt_lower for term in blockchain_terms):
            if addresses:
                # We have an address but no clear intent, provide general info
                result["intent"] = "general_address_info"
                
                # Create tasks for parallel execution
                balance_task = self.get_balance(addresses[0])
                account_info_task = self.get_account_info(addresses[0])
                token_accounts_task = self.get_token_accounts_by_owner(addresses[0])
                
                # Gather all tasks at once
                balance, account_info, token_accounts = await asyncio.gather(
                    balance_task, account_info_task, token_accounts_task,
                    return_exceptions=True
                )
                
                # Handle potential exceptions
                sol_balance = 0
                token_count = 0
                owner = "Unknown"
                executable = False
                
                if not isinstance(balance, Exception):
                    sol_balance = self.lamports_to_sol(balance)
                
                if not isinstance(account_info, Exception):
                    owner = account_info.owner
                    executable = account_info.executable
                
                if not isinstance(token_accounts, Exception):
                    token_count = len(token_accounts)
                
                result["data"] = {
                    "address": addresses[0],
                    "sol_balance": sol_balance,
                    "token_count": token_count,
                    "owner_program": owner,
                    "executable": executable,
                    "message": f"Address holds {sol_balance} SOL and {token_count} different tokens/token accounts."
                }
            else:
                result["intent"] = "unknown"
                result["error"] = "Could not determine specific intent from prompt. Please provide more details or include a Solana address."
        else:
            result["intent"] = "unknown"
            result["error"] = "Could not determine intent from prompt. Please try rephrasing with keywords like 'balance', 'token', 'transaction', etc."

    async def batch_get_token_info(self, mint_addresses: List[str]) -> Dict[str, TokenInfo]:
        """
        Get token info for multiple mint addresses in a batched request.
        This method reduces API calls by using batch endpoints where available.
        
        Args:
            mint_addresses: List of mint addresses to get info for
            
        Returns:
            Dictionary mapping mint address to TokenInfo object
        """
        if not mint_addresses:
            return {}
            
        # Check cache first
        results = {}
        uncached_mints = []
        
        # Look for cached entries first
        for mint in mint_addresses:
            cache_key = f"token_{mint}"
            cached = await self._get_cached(cache_key, 'token_metadata')
            if cached:
                results[mint] = cached
            else:
                uncached_mints.append(mint)
                
        # If everything was cached, return early
        if not uncached_mints:
            return results
            
        # Split into manageable chunks to avoid oversized requests
        chunk_size = 50  # Adjust based on API limits
        mint_chunks = [uncached_mints[i:i + chunk_size] for i in range(0, len(uncached_mints), chunk_size)]
        
        for chunk in mint_chunks:
            # Get metadata from Helius in batches
            metadata_url = f"https://api.helius.xyz/v0/tokens/metadata?api-key={self.helius_api_key}"
            metadata_payload = {"mintAccounts": chunk}
            
            try:
                async with self.session.post(metadata_url, json=metadata_payload) as response:
                    metadata_resp = await response.json()
                    
                    if not isinstance(metadata_resp, list):
                        logger.warning(f"Unexpected metadata response: {metadata_resp}")
                        continue
                        
                    # Process each token
                    for token_metadata in metadata_resp:
                        mint = token_metadata.get("address")
                        if not mint:
                            continue
                            
                        # Get price data from Birdeye (unfortunately not batchable)
                        price = 0
                        try:
                            price_url = f"https://public-api.birdeye.so/public/price?address={mint}"
                            headers = {"x-chain": "solana", "X-API-KEY": self.birdeye_api_key}
                            
                            async with self.session.get(price_url, headers=headers) as price_response:
                                price_data = await price_response.json()
                                if price_data.get('success'):
                                    price = price_data.get('data', {}).get('value', 0)
                        except Exception as e:
                            logger.warning(f"Error getting price for {mint}: {str(e)}")
                            
                        # Create TokenInfo object
                        token_info = TokenInfo(
                            symbol=token_metadata.get('symbol', 'UNKNOWN'),
                            name=token_metadata.get('name', 'Unknown Token'),
                            decimals=token_metadata.get('decimals', 0),
                            price=price,
                            mint=mint,
                            market_cap=token_metadata.get('marketCap'),
                            volume_24h=token_metadata.get('volume', {}).get('value24h'),
                            supply=token_metadata.get('supply')
                        )
                        
                        # Cache the result
                        cache_key = f"token_{mint}"
                        await self._set_cache(cache_key, token_info, 'token_metadata')
                        
                        # Add to results
                        results[mint] = token_info
            except Exception as e:
                logger.error(f"Error in batch token info request: {str(e)}")
                
        return results

    async def _get_cached(self, key: str, cache_type: str) -> Optional[object]:
        """Retrieve cached data if valid"""
        try:
            if key in self.cache.get(cache_type, {}):
                cached = self.cache[cache_type][key]
                now = datetime.now(timezone.utc)
                if now - cached['timestamp'] < self.cache_expiry.get(cache_type, timedelta(minutes=10)):
                    # Update the timestamp to keep frequently used items fresh
                    self.cache[cache_type][key]['timestamp'] = now
                    return cached['data']
                else:
                    # Remove expired item
                    del self.cache[cache_type][key]
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {str(e)}")
        return None

    async def _set_cache(self, key: str, data: object, cache_type: str):
        """Store data in cache with improved error handling and size management"""
        try:
            if cache_type not in self.cache:
                self.cache[cache_type] = {}
                
            # Check cache size and remove oldest items if needed
            cache_dict = self.cache[cache_type]
            if len(cache_dict) >= self.max_cache_items:
                # Sort by timestamp and remove oldest 10% of items
                items_to_remove = int(self.max_cache_items * 0.1)
                oldest_keys = sorted(
                    cache_dict.keys(), 
                    key=lambda k: cache_dict[k]['timestamp']
                )[:items_to_remove]
                
                for old_key in oldest_keys:
                    del cache_dict[old_key]
                
                logger.debug(f"Cache pruning: removed {len(oldest_keys)} old items from {cache_type} cache")
                
            # Add the new item to cache
            self.cache[cache_type][key] = {
                'data': data,
                'timestamp': datetime.now(timezone.utc)
            }
        except Exception as e:
            logger.warning(f"Error setting cache: {str(e)}")
            
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache data, either all or a specific type"""
        if cache_type:
            if cache_type in self.cache:
                self.cache[cache_type] = {}
                logger.info(f"Cleared {cache_type} cache")
        else:
            for cache_key in self.cache:
                self.cache[cache_key] = {}
            logger.info("Cleared all cache data")

    @staticmethod
    def lamports_to_sol(lamports: int) -> float:
        """Convert lamports to SOL"""
        return lamports / 1_000_000_000

    @staticmethod
    def sol_to_lamports(sol: float) -> int:
        """Convert SOL to lamports"""
        return int(sol * 1_000_000_000)

# Export classes
__all__ = [
    'SolanaAnalyzer',
    'TokenInfo',
    'Whale',
    'AccountInfo',
    'TransactionInfo'
] 