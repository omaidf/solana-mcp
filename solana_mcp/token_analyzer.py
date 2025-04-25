"""Token analysis for Solana tokens with focus on pumpfun tokens."""

import datetime
import logging
from typing import Dict, List, Any, Optional, TypeVar, Callable, Awaitable
from dataclasses import dataclass
import base64
import base58
import asyncio
import functools

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.decorators import validate_solana_key, handle_errors
from solana_mcp.cache import cached, global_cache
from solana_mcp.rpc_utils import (
    get_multiple_accounts, 
    get_filtered_token_accounts,
    get_token_account_owners,
    get_wallet_age,
    retry_with_backoff
)

# Constants
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
DEFAULT_TIMEOUT = 30.0  # Default timeout in seconds

# Set up logging
logger = get_logger(__name__)

# Type variable for generic function
T = TypeVar('T')

async def with_timeout(coro: Awaitable[T], timeout: float = DEFAULT_TIMEOUT, fallback: Optional[T] = None) -> T:
    """Execute a coroutine with a timeout.
    
    Args:
        coro: The coroutine to execute
        timeout: The timeout in seconds
        fallback: The fallback value to return if the timeout is reached
        
    Returns:
        The result of the coroutine or the fallback value if timed out
        
    Raises:
        asyncio.TimeoutError: If no fallback is provided and the timeout is reached
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout} seconds")
        if fallback is not None:
            return fallback
        raise

def with_timeout_decorator(timeout: float = DEFAULT_TIMEOUT, fallback: Optional[Any] = None):
    """Decorator to add timeout handling to an async function.
    
    Args:
        timeout: The timeout in seconds
        fallback: The fallback value to return if the timeout is reached
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_timeout(func(*args, **kwargs), timeout, fallback)
        return wrapper
    return decorator

# Animal/meme related keywords for classification
ANIMAL_KEYWORDS = [
    "dog", "doge", "shib", "cat", "kitty", "monkey", "ape", "bull", "bear", 
    "frog", "pepe", "wolf", "fox", "lion", "tiger", "rabbit", "bunny",
    "bird", "eagle", "fish", "shark", "whale", "dolphin", "octopus", "squid",
    "dragon", "snake", "rat", "mouse", "hamster", "pig", "cow", "goat",
    "sheep", "duck", "swan", "chicken", "rooster", "horse", "unicorn", "pony"
]

FOOD_KEYWORDS = [
    "pizza", "burger", "fries", "nugget", "chicken", "beef", "pork", "meat",
    "vegan", "vegetable", "fruit", "apple", "orange", "banana", "grape",
    "bread", "cake", "pie", "cookie", "donut", "chocolate", "candy", "sweet",
    "soda", "cola", "pop", "juice", "water", "milk", "coffee", "tea", "beer",
    "wine", "vodka", "whiskey", "rum", "tequila", "gin", "brandy"
]

TECH_KEYWORDS = [
    "ai", "robot", "cyber", "tech", "computer", "code", "program", "software",
    "hardware", "chip", "processor", "server", "cloud", "data", "network",
    "internet", "web", "site", "app", "application", "mobile", "phone", "smart",
    "crypto", "token", "coin", "blockchain", "nft", "defi", "finance", "bank"
]

MEME_KEYWORDS = [
    "meme", "lol", "lmao", "rofl", "wtf", "omg", "moon", "rocket", "lambo",
    "rich", "poor", "wojak", "chad", "virgin", "based", "cringe", "stonk",
    "yolo", "fomo", "dump", "pump", "diamond", "hand", "hodl", "hold", "sell",
    "buy", "bro", "bruh", "dude", "guy", "girl", "man", "woman", "wen", "ser"
]

@dataclass
class TokenAnalysis:
    """Data class for token analysis results."""
    token_mint: str
    token_name: str
    token_symbol: str
    decimals: int
    total_supply: str
    circulation_supply: str
    current_price_usd: float
    launch_date: Optional[datetime.datetime]
    age_days: Optional[int]
    owner_can_mint: bool
    owner_can_freeze: bool
    total_holders: int
    largest_holder_percentage: float
    whale_count: int
    whale_percentage: float
    whale_holdings_percentage: float
    whale_holdings_usd_total: float
    fresh_wallet_count: int
    fresh_wallet_percentage: float
    fresh_wallet_holdings_percentage: float
    last_updated: datetime.datetime


class TokenAnalyzer:
    """Analyzer for Solana tokens with comprehensive analysis capabilities."""

    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)
        self.meme_analyzer = MemeTokenAnalyzer(solana_client)

    @validate_solana_key
    @handle_errors
    @with_timeout_decorator(timeout=60.0, fallback=None)
    async def analyze_token(self, mint: str, request_id: Optional[str] = None) -> TokenAnalysis:
        """Perform comprehensive analysis of a token.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Complete token analysis
        """
        log_with_context(
            logger,
            "info",
            f"Comprehensive token analysis requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Initialize default values
        token_name = "Unknown"
        token_symbol = "UNKNOWN"
        decimals = 0
        total_supply = "0"
        total_holders = 0
        largest_holder_percentage = 0.0
        current_price_usd = 0.0
        whale_count = 0
        whale_percentage = 0.0
        whale_holdings_percentage = 0.0
        whale_holdings_usd_total = 0.0
        fresh_wallet_count = 0
        fresh_wallet_percentage = 0.0
        fresh_wallet_holdings_percentage = 0.0
        launch_date = None
        age_days = None
        owner_can_mint = False
        owner_can_freeze = False
        
        try:
            # Get token metadata
            try:
                metadata = await self.get_token_metadata(mint, request_id=request_id)
                token_name = metadata.get("name", "Unknown")
                token_symbol = metadata.get("symbol", "UNKNOWN")
            except Exception as e:
                logger.error(f"Error getting token metadata for {mint}: {str(e)}", exc_info=True)
                
            # Get token supply and decimals
            try:
                supply_info = await self.get_token_supply_and_decimals(mint, request_id=request_id)
                decimals = supply_info.get("value", {}).get("decimals", 0)
                total_supply = supply_info.get("value", {}).get("uiAmountString", "0")
            except Exception as e:
                logger.error(f"Error getting token supply for {mint}: {str(e)}", exc_info=True)
                
            # Get holders data
            try:
                holders_data = await self.get_token_largest_holders(mint, request_id=request_id)
                total_holders = holders_data.get("total_holders", 0)
                largest_holder_percentage = holders_data.get("largest_holder_percentage", 0)
            except Exception as e:
                logger.error(f"Error getting token holders for {mint}: {str(e)}", exc_info=True)
                
            # Get price data before whale and fresh wallet analysis
            # as both depend on accurate price data
            try:
                price_data = await self.get_token_price(mint, request_id=request_id)
                current_price_usd = price_data.get("price_usd", 0.0)
            except Exception as e:
                logger.error(f"Error getting token price for {mint}: {str(e)}", exc_info=True)
                
            # Get whale data
            try:
                whale_data = await self.get_whale_holders(mint, request_id=request_id)
                whale_count = whale_data.get("whale_count", 0)
                whale_percentage = whale_data.get("whale_percentage", 0.0)
                whale_holdings_percentage = whale_data.get("whale_holdings_percentage", 0.0)
                whale_holdings_usd_total = whale_data.get("whale_holdings_usd_total", 0.0)
            except Exception as e:
                logger.error(f"Error getting whale data for {mint}: {str(e)}", exc_info=True)
                
            # Get fresh wallet data
            try:
                fresh_wallet_data = await self.get_fresh_wallets(mint, request_id=request_id)
                fresh_wallet_count = fresh_wallet_data.get("fresh_wallet_count", 0)
                fresh_wallet_percentage = fresh_wallet_data.get("fresh_wallet_percentage", 0.0)
                fresh_wallet_holdings_percentage = fresh_wallet_data.get("fresh_wallet_holdings_percentage", 0.0)
            except Exception as e:
                logger.error(f"Error getting fresh wallet data for {mint}: {str(e)}", exc_info=True)
                
            # Get age data
            try:
                age_data = await self.get_token_age(mint, request_id=request_id)
                launch_date_str = age_data.get("launch_date")
                launch_date = datetime.datetime.fromisoformat(launch_date_str) if launch_date_str else None
                age_days = age_data.get("age_days")
            except Exception as e:
                logger.error(f"Error getting token age data for {mint}: {str(e)}", exc_info=True)
                
            # Get authority data
            try:
                auth_data = await self.get_token_mint_authority(mint, request_id=request_id)
                owner_can_mint = auth_data.get("has_mint_authority", False)
                owner_can_freeze = auth_data.get("has_freeze_authority", False)
            except Exception as e:
                logger.error(f"Error getting token authority for {mint}: {str(e)}", exc_info=True)
                
            # Create analysis result
            analysis = TokenAnalysis(
                token_mint=mint,
                token_name=token_name,
                token_symbol=token_symbol,
                decimals=decimals,
                total_supply=total_supply,
                circulation_supply=total_supply,  # Simplified, could be calculated based on locked tokens
                current_price_usd=current_price_usd,
                launch_date=launch_date,
                age_days=age_days,
                owner_can_mint=owner_can_mint,
                owner_can_freeze=owner_can_freeze,
                total_holders=total_holders,
                largest_holder_percentage=largest_holder_percentage,
                whale_count=whale_count,
                whale_percentage=whale_percentage,
                whale_holdings_percentage=whale_holdings_percentage,
                whale_holdings_usd_total=whale_holdings_usd_total,
                fresh_wallet_count=fresh_wallet_count,
                fresh_wallet_percentage=fresh_wallet_percentage,
                fresh_wallet_holdings_percentage=fresh_wallet_holdings_percentage,
                last_updated=datetime.datetime.now()
            )
            
            log_with_context(
                logger,
                "info",
                f"Comprehensive token analysis completed for: {mint}",
                request_id=request_id,
                mint=mint,
                token_name=token_name,
                token_symbol=token_symbol
            )
            
            return analysis
            
        except Exception as e:
            # Log the error and re-raise to be handled by decorators
            logger.error(f"Error in analyze_token for {mint}: {str(e)}", exc_info=True)
            raise

    @validate_solana_key
    @handle_errors
    async def get_token_metadata(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token metadata.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token metadata
        """
        log_with_context(
            logger,
            "info",
            f"Token metadata requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Use the client to get token metadata
        metadata = await self.client.get_token_metadata(mint)
        
        # Add meme token categorization
        meme_metadata = await self.meme_analyzer.get_token_metadata_with_category(mint, request_id=request_id)
        metadata["is_meme_token"] = meme_metadata.get("is_meme_token", False)
        metadata["category"] = meme_metadata.get("category", "Other")
        
        log_with_context(
            logger,
            "info",
            f"Token metadata completed for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        return metadata

    @validate_solana_key
    @handle_errors
    async def get_token_supply_and_decimals(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token supply and decimals.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Supply information including decimals
        """
        log_with_context(
            logger,
            "info",
            f"Token supply requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        supply_info = await self.client.get_token_supply(mint)
        
        log_with_context(
            logger,
            "info",
            f"Token supply completed for: {mint}",
            request_id=request_id,
            mint=mint,
            supply=supply_info.get("value", {}).get("amount", "0") if "value" in supply_info else "N/A"
        )
        
        return supply_info

    @validate_solana_key
    @handle_errors
    async def get_token_price(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token price.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Price information
        """
        log_with_context(
            logger,
            "info",
            f"Token price requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get market price data from the Solana client
        price_result = await self.client.get_market_price(mint)
        
        # Extract price data or use default values if not available
        price_data = price_result.get("price_data", {})
        
        # Construct a standardized response
        response = {
            "price_usd": price_data.get("price_usd", 0.0),
            "price_sol": price_data.get("price_sol", 0.0),
            "liquidity_usd": price_data.get("liquidity", {}).get("sol_volume", 0.0) if price_data.get("liquidity") else 0.0,
            "market_cap_usd": 0.0,  # Would need additional calculation with supply
            "volume_24h_usd": 0.0,  # Not available in the current implementation
            "change_24h_percent": 0.0,  # Not available in the current implementation
            "dex_info": [],
            "source": price_data.get("source", "unknown"),
            "last_updated": price_data.get("last_updated", datetime.datetime.now().isoformat())
        }
        
        # If there was an error in the price data, include it
        if "error" in price_result:
            response["error"] = price_result["error"]
        
        log_with_context(
            logger,
            "info",
            f"Token price completed for: {mint}",
            request_id=request_id,
            mint=mint,
            price_usd=response["price_usd"],
            price_sol=response["price_sol"]
        )
        
        return response

    @validate_solana_key
    @handle_errors
    async def get_token_largest_holders(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token largest holders.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Holder distribution information
        """
        log_with_context(
            logger,
            "info",
            f"Token holders requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Use the meme analyzer to get holder distribution
        holder_data = await self.meme_analyzer.analyze_holder_distribution(mint, request_id=request_id)
        
        log_with_context(
            logger,
            "info",
            f"Token holders completed for: {mint}",
            request_id=request_id,
            mint=mint,
            total_holders=holder_data.get("total_holders", 0)
        )
        
        return holder_data

    @validate_solana_key
    @handle_errors
    async def get_token_age(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token age information.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Age information
        """
        log_with_context(
            logger,
            "info",
            f"Token age requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Use creation date function from meme analyzer
        creation_data = await self.meme_analyzer.get_creation_date(mint, request_id=request_id)
        
        # Rename fields to match expected API response
        age_data = {
            "token_mint": creation_data.get("token_mint"),
            "token_name": creation_data.get("token_name"),
            "token_symbol": creation_data.get("token_symbol"),
            "launch_date": creation_data.get("creation_date"),
            "creation_signature": creation_data.get("creation_signature"),
            "creation_block": creation_data.get("creation_block"),
            "age_days": creation_data.get("age_days"),
            "last_updated": creation_data.get("last_updated")
        }
        
        log_with_context(
            logger,
            "info",
            f"Token age completed for: {mint}",
            request_id=request_id,
            mint=mint,
            age_days=age_data.get("age_days", "N/A")
        )
        
        return age_data

    @validate_solana_key
    @handle_errors
    async def get_token_mint_authority(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token mint authority information.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Authority information
        """
        log_with_context(
            logger,
            "info",
            f"Token authority requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        result = {
            "token_mint": mint,
            "has_mint_authority": False,
            "mint_authority": None,
            "has_freeze_authority": False,
            "freeze_authority": None,
            "is_mutable": False,
            "decimals": 0,
            "is_initialized": False,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get token account info
            account_info = await self.client.get_account_info(mint)
            
            # Check if account exists
            if account_info and "result" in account_info and account_info["result"]:
                # Process mint account data
                data = account_info["result"].get("data")
                
                # Check data format and decode if it's base64
                if isinstance(data, list) and len(data) >= 2 and data[0] == "base64":
                    try:
                        # Decode base64 data
                        data_bytes = base64.b64decode(data[1])
                        
                        # Make sure we have enough data for SPL Token Mint data structure
                        # Minimum length for token mint data is 82 bytes
                        if len(data_bytes) < 82:
                            logger.error(f"Invalid token mint data length for {mint}: {len(data_bytes)} bytes (expected at least 82)")
                            return result
                        
                        # SPL Token Mint Layout:
                        # Offset 0: Mint authority option (1 byte)
                        # Offset 1: Mint authority pubkey (32 bytes)
                        # Offset 33: Supply (8 bytes)
                        # Offset 41: Decimals (1 byte)
                        # Offset 42: is_initialized (1 byte)
                        # Offset 43: Freeze authority option (1 byte)
                        # Offset 44: Freeze authority pubkey (32 bytes)
                        
                        # Check if mint authority is present
                        mint_authority_option = data_bytes[0]
                        if mint_authority_option not in (0, 1):
                            logger.warning(f"Invalid mint authority option for {mint}: {mint_authority_option}")
                            mint_authority_option = 0  # Default to no authority
                        
                        result["has_mint_authority"] = mint_authority_option == 1
                        
                        if result["has_mint_authority"]:
                            # Extract mint authority pubkey (ensure we have enough data)
                            if len(data_bytes) >= 33:
                                mint_authority = base58.b58encode(data_bytes[1:33]).decode('utf-8')
                                result["mint_authority"] = mint_authority
                        
                        # Get decimals (ensure we have enough data)
                        if len(data_bytes) >= 42:
                            result["decimals"] = data_bytes[41]
                            
                            # Check if initialized
                            result["is_initialized"] = data_bytes[42] == 1
                        
                        # Check if freeze authority is present (ensure we have enough data)
                        if len(data_bytes) >= 44:
                            freeze_authority_option = data_bytes[43]
                            if freeze_authority_option not in (0, 1):
                                logger.warning(f"Invalid freeze authority option for {mint}: {freeze_authority_option}")
                                freeze_authority_option = 0  # Default to no freeze authority
                                
                            result["has_freeze_authority"] = freeze_authority_option == 1
                            
                            if result["has_freeze_authority"] and len(data_bytes) >= 76:
                                # Extract freeze authority pubkey
                                freeze_authority = base58.b58encode(data_bytes[44:76]).decode('utf-8')
                                result["freeze_authority"] = freeze_authority
                    except (ValueError, IndexError, base64.Error) as e:
                        # Handle decoding errors gracefully
                        logger.error(f"Error decoding token mint data for {mint}: {str(e)}", exc_info=True)
                elif isinstance(data, dict) and "parsed" in data and "type" in data and data["type"] == "mint":
                    # Handle pre-parsed data (jsonParsed encoding)
                    try:
                        parsed_data = data.get("parsed", {}).get("info", {})
                        result["decimals"] = parsed_data.get("decimals", 0)
                        result["is_initialized"] = True  # Assuming parsed data means initialized
                        
                        # Extract mint authority
                        mint_authority = parsed_data.get("mintAuthority")
                        result["has_mint_authority"] = mint_authority is not None
                        if mint_authority:
                            result["mint_authority"] = mint_authority
                        
                        # Extract freeze authority
                        freeze_authority = parsed_data.get("freezeAuthority")
                        result["has_freeze_authority"] = freeze_authority is not None
                        if freeze_authority:
                            result["freeze_authority"] = freeze_authority
                    except (KeyError, TypeError) as e:
                        logger.error(f"Error parsing jsonParsed token mint data for {mint}: {str(e)}", exc_info=True)
                else:
                    logger.warning(f"Unsupported data format for token mint {mint}")
        except Exception as e:
            logger.error(f"Error getting token mint authority for {mint}: {str(e)}", exc_info=True)
        
        log_with_context(
            logger,
            "info",
            f"Token authority completed for: {mint}",
            request_id=request_id,
            mint=mint,
            has_mint_authority=result["has_mint_authority"],
            has_freeze_authority=result["has_freeze_authority"]
        )
        
        return result

    @validate_solana_key
    @handle_errors
    async def get_token_holders_count(self, mint: str, request_id: Optional[str] = None) -> int:
        """Get count of token holders.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Number of token holders
        """
        log_with_context(
            logger,
            "info",
            f"Token holders count requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get holder distribution data
        holder_data = await self.get_token_largest_holders(mint, request_id=request_id)
        count = holder_data.get("total_holders", 0)
        
        log_with_context(
            logger,
            "info",
            f"Token holders count completed for: {mint}",
            request_id=request_id,
            mint=mint,
            count=count
        )
        
        return count

    @cached(ttl=300)
    async def get_token_metadata_cached(self, mint, request_id=None):
        """Cached version of get_token_metadata."""
        return await self.client.get_token_metadata(mint)

    @cached(ttl=600)
    async def get_token_supply_cached(self, mint, request_id=None):
        """Cached version of get_token_supply."""
        return await self.client.get_token_supply(mint)

    @cached(ttl=60)
    async def get_market_price_cached(self, mint, request_id=None):
        """Cached version of get_market_price."""
        return await self.client.get_market_price(mint)

    @validate_solana_key
    @handle_errors
    @with_timeout_decorator(timeout=45.0, fallback={"whale_count": 0, "whale_percentage": 0.0, "whale_holdings_percentage": 0.0, "whale_holdings_usd_total": 0.0})
    async def get_whale_holders(self, mint: str, threshold_usd: float = 50000.0, request_id: Optional[str] = None, max_accounts: int = 100) -> Dict[str, Any]:
        """Get token holders with balances over the specified USD threshold.
        
        Args:
            mint: The token mint address
            threshold_usd: The minimum USD value to consider a holder a whale (default: $50K)
            request_id: Optional request ID for tracing
            max_accounts: Maximum number of accounts to analyze
            
        Returns:
            Dictionary with whale holder information
        """
        log_with_context(
            logger,
            "info",
            f"Whale holders analysis requested for: {mint} (threshold: ${threshold_usd})",
            request_id=request_id,
            mint=mint,
            threshold_usd=threshold_usd
        )
        
        result = {
            "success": True,
            "token_mint": mint,
            "threshold_usd": threshold_usd,
            "total_holders_analyzed": 0,
            "total_holders": 0,
            "whale_count": 0,
            "whale_percentage": 0,
            "whale_holders": [],
            "whale_holdings_usd_total": 0,
            "whale_holdings_percentage": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Set default token information in case metadata fetching fails
            result["token_name"] = "Unknown"
            result["token_symbol"] = "UNKNOWN"
            
            # Try to get token metadata, but continue if it fails
            try:
                # Use cached version to reduce RPC calls
                token_metadata = await self.get_token_metadata_cached(mint, request_id)
                result["token_name"] = token_metadata.get("name", "Unknown")
                result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            except Exception as e:
                logger.warning(f"Could not get token metadata: {str(e)}")
                result["warnings"] = result.get("warnings", []) + [f"Could not get token metadata: {str(e)}"]
            
            # Try to get holders count, but continue if it fails
            try:
                holders_data = await self.get_token_largest_holders(mint, request_id=request_id)
                result["total_holders"] = holders_data.get("total_holders", 0)
            except Exception as e:
                logger.warning(f"Could not get total holders count: {str(e)}")
                result["warnings"] = result.get("warnings", []) + [f"Could not get total holders count: {str(e)}"]
                result["total_holders"] = 0
            
            # Get token price with defensive coding
            price_usd = 0.0
            try:
                # Use cached version to reduce RPC calls
                price_data = await self.get_market_price_cached(mint, request_id)
                # Explicitly handle None to avoid NoneType errors
                price_usd = float(price_data.get("price_usd", 0.0) or 0.0)
                result["token_price_usd"] = price_usd
            except Exception as e:
                logger.error(f"Error getting price data: {str(e)}")
                price_usd = 0.0
                result["warnings"] = result.get("warnings", []) + [f"Error getting price data: {str(e)}"]
            
            if price_usd <= 0:
                logger.warning(f"Unable to determine USD price for token {mint}, using fallback price of $0.01")
                price_usd = 0.01  # Use a fallback price instead of 0
                result["token_price_usd"] = price_usd
                result["warnings"] = result.get("warnings", []) + ["Using fallback price of $0.01"]
            
            # Get token supply and decimals, with defensive coding
            total_supply = 1  # Default to avoid division by zero
            try:
                # Use cached version to reduce RPC calls
                supply_info = await self.get_token_supply_cached(mint, request_id)
                decimals = supply_info.get("value", {}).get("decimals", 0)
                total_supply_str = supply_info.get("value", {}).get("uiAmountString", "0")
                result["decimals"] = decimals
                
                try:
                    total_supply = float(total_supply_str or "1")  # Default to 1 if empty string
                    if total_supply <= 0:
                        logger.warning(f"Invalid or zero total supply value: {total_supply_str}")
                        total_supply = 1  # Use a non-zero default to avoid division by zero
                        result["warnings"] = result.get("warnings", []) + ["Invalid total supply value, using default"]
                    result["total_supply"] = total_supply
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse total supply: {total_supply_str}")
                    result["warnings"] = result.get("warnings", []) + [f"Invalid total supply value: {total_supply_str}"]
            except Exception as e:
                logger.warning(f"Error getting token supply: {str(e)}")
                result["warnings"] = result.get("warnings", []) + [f"Error getting token supply: {str(e)}"]
            
            # Get largest token accounts with optimized filtering
            try:
                # Calculate minimum balance threshold to reduce unnecessary processing
                min_balance_threshold = 0
                if price_usd > 0:
                    # Set threshold at 10% of whale threshold to avoid missing edge cases
                    min_balance_threshold = (threshold_usd * 0.1) / price_usd
                
                # Use the optimized filtering function
                accounts_to_process = await get_filtered_token_accounts(
                    self.client,
                    mint,
                    min_balance=min_balance_threshold,
                    max_accounts=max_accounts,
                    order="desc"  # Largest first
                )
                
                result["total_holders_analyzed"] = len(accounts_to_process)
                logger.info(f"Analyzing {len(accounts_to_process)} top accounts for token {mint}")
                
                # If total_holders wasn't successfully fetched earlier, fall back to analyzed count
                if result["total_holders"] == 0:
                    result["total_holders"] = result["total_holders_analyzed"]
                
                if not accounts_to_process:
                    logger.warning(f"No token accounts found for {mint}")
                    result["warnings"] = result.get("warnings", []) + ["No token accounts found or they are all below threshold"]
                    return result
                
                # Get all token account owners in a single batch operation
                token_account_owners = await get_token_account_owners(self.client, accounts_to_process)
                
                # Process accounts that have owners
                whale_holders = []
                whale_holdings_total = 0
                
                # Process accounts in batches to avoid overwhelming the RPC
                BATCH_SIZE = 10
                for i in range(0, len(accounts_to_process), BATCH_SIZE):
                    batch = accounts_to_process[i:i+BATCH_SIZE]
                    
                    # Add a delay between batches
                    if i > 0:
                        await asyncio.sleep(1)
                    
                    # Process each account in the batch
                    for j, account in enumerate(batch):
                        # Small delay between accounts in the same batch
                        if j > 0:
                            await asyncio.sleep(0.2)
                        
                        account_address = account.get("address", "")
                        
                        # Skip missing addresses
                        if not account_address:
                            continue
                        
                        # Get owner from pre-fetched map
                        owner = token_account_owners.get(account_address)
                        
                        # Skip if we couldn't determine the owner
                        if not owner:
                            continue
                        
                        # Get account balance in token units
                        account_balance = float(account.get("uiAmount", 0) or 0)
                        
                        # Calculate USD value
                        balance_usd = account_balance * price_usd
                        
                        # Skip accounts below threshold to reduce unnecessary processing
                        if balance_usd < threshold_usd:
                            continue
                        
                        # Calculate percentage of supply
                        percentage_of_supply = (account_balance / total_supply * 100) if total_supply > 0 else 0
                        
                        # Basic whale data
                        whale_data = {
                            "address": owner,
                            "token_balance": account_balance,
                            "usd_value": balance_usd,
                            "percentage_of_supply": percentage_of_supply
                        }
                        
                        # Add additional information for API results
                        # Get SOL balance for a more complete picture of the wallet
                        try:
                            sol_balance_response = await self.client.get_balance(owner)
                            sol_balance = float(sol_balance_response.get("result", {}).get("value", 0)) / 10**9
                            sol_value = sol_balance * 150.0  # Approximate SOL price
                            
                            # Add SOL info to whale data
                            total_wallet_value = balance_usd + sol_value
                            whale_data["total_wallet_value_usd"] = total_wallet_value
                            whale_data["token_count"] = 1  # Start with this token
                            
                            if sol_balance > 0:
                                whale_data["token_count"] += 1
                                whale_data["top_tokens"] = [{
                                    'token': 'SOL',
                                    'mint': 'So11111111111111111111111111111111111111112',
                                    'amount': sol_balance,
                                    'value_usd': sol_value
                                }]
                            else:
                                whale_data["top_tokens"] = []
                                
                        except Exception as e:
                            logger.error(f"Error getting SOL balance: {str(e)}")
                            whale_data["total_wallet_value_usd"] = balance_usd
                            whale_data["token_count"] = 1
                            whale_data["top_tokens"] = []
                        
                        # Add to whale list
                        whale_holders.append(whale_data)
                        whale_holdings_total += account_balance
                
                # Update result with whale data
                result["whale_count"] = len(whale_holders)
                result["whale_holders"] = whale_holders
                
                if result["total_holders_analyzed"] > 0:
                    result["whale_percentage"] = (result["whale_count"] / result["total_holders_analyzed"]) * 100
                
                if total_supply > 0:
                    result["whale_holdings_percentage"] = (whale_holdings_total / total_supply) * 100
                
                # Calculate total USD value held by whales
                result["whale_holdings_usd_total"] = whale_holdings_total * price_usd
            
            except Exception as e:
                logger.error(f"Error processing largest token accounts: {str(e)}", exc_info=True)
                result["warnings"] = result.get("warnings", []) + [f"Error processing largest token accounts: {str(e)}"]
        
        except Exception as e:
            logger.error(f"Error analyzing whale holders for {mint}: {str(e)}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
        
        log_with_context(
            logger,
            "info",
            f"Whale holders analysis completed for: {mint}",
            request_id=request_id,
            mint=mint,
            whale_count=result.get("whale_count", 0),
            whale_holdings_percentage=f"{result.get('whale_holdings_percentage', 0):.2f}%"
        )
        
        return result

    @validate_solana_key
    @handle_errors
    async def get_fresh_wallets(self, mint: str, request_id: Optional[str] = None, max_accounts: int = 100) -> Dict[str, Any]:
        """Get token holders that are likely new wallets created primarily for this token.
        
        Fresh wallets are wallets that:
        1. Were created recently (less than 30 days old)
        2. Have very few tokens (typically less than 5)
        3. Have a high percentage of this token relative to their total holdings
        
        This is an enhanced version that checks wallet age, transaction ratios, and token diversity.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            max_accounts: Maximum number of accounts to analyze
            
        Returns:
            Dictionary with fresh wallet information
        """
        log_with_context(
            logger,
            "info",
            f"Fresh wallets analysis requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        result = {
            "success": True,
            "token_mint": mint,
            "total_holders_analyzed": 0,
            "fresh_wallet_count": 0,
            "fresh_wallet_percentage": 0,
            "fresh_wallets": [],
            "fresh_wallet_holdings_percentage": 0,
            "fresh_wallet_holdings_token_total": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Set default token information in case metadata fetching fails
            result["token_name"] = "Unknown"
            result["token_symbol"] = "UNKNOWN"
            
            # Try to get token metadata, but continue if it fails
            try:
                token_metadata = await self.get_token_metadata(mint, request_id=request_id)
                result["token_name"] = token_metadata.get("name", "Unknown")
                result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            except Exception as e:
                logger.warning(f"Could not get token metadata: {str(e)}")
                result["warnings"] = result.get("warnings", []) + [f"Could not get token metadata: {str(e)}"]
            
            # Get token price for value calculations, with error handling
            price_usd = 0.01  # Default fallback price
            try:
                price_data = await self.get_token_price(mint, request_id=request_id)
                # Explicitly handle None to avoid NoneType errors
                price_usd = float(price_data.get("price_usd", 0.0) or 0.0)
                if price_usd <= 0:
                    price_usd = 0.01  # Use fallback price
                    result["warnings"] = result.get("warnings", []) + ["Using fallback price of $0.01"]
                
                result["token_price_usd"] = price_usd
            except Exception as e:
                logger.warning(f"Error getting price data, using fallback: {str(e)}")
                result["token_price_usd"] = price_usd
                result["warnings"] = result.get("warnings", []) + [f"Error getting price data, using fallback: {str(e)}"]
            
            # Get token supply and decimals with error handling
            total_supply = 1  # Default to avoid division by zero
            try:
                supply_info = await self.get_token_supply_and_decimals(mint, request_id=request_id)
                decimals = supply_info.get("value", {}).get("decimals", 0)
                total_supply_str = supply_info.get("value", {}).get("uiAmountString", "0")
                result["decimals"] = decimals
                
                try:
                    total_supply = float(total_supply_str or "1")  # Default to 1 if empty string
                    if total_supply <= 0:
                        logger.warning(f"Invalid or zero total supply value: {total_supply_str}")
                        total_supply = 1  # Use a non-zero default to avoid division by zero
                        result["warnings"] = result.get("warnings", []) + ["Invalid total supply value, using default"]
                    result["total_supply"] = total_supply
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse total supply: {total_supply_str}")
                    result["warnings"] = result.get("warnings", []) + [f"Invalid total supply value: {total_supply_str}"]
            except Exception as e:
                logger.warning(f"Error getting token supply: {str(e)}")
                result["warnings"] = result.get("warnings", []) + [f"Error getting token supply: {str(e)}"]
            
            # Get largest token accounts with pagination and filtering
            try:
                # Get the largest token accounts
                largest_accounts_result = await self.client.get_token_largest_accounts(mint)
                
                if "value" in largest_accounts_result and largest_accounts_result["value"]:
                    accounts = largest_accounts_result["value"]
                    
                    # Use the maximum accounts specified (default 100)
                    accounts_to_process = accounts[:max_accounts]
                    
                    result["total_holders_analyzed"] = len(accounts_to_process)
                    logger.info(f"Analyzing {len(accounts_to_process)} top accounts for fresh wallet detection")
                    
                    # Pre-filter accounts to avoid unnecessary RPC calls
                    # For fresh wallets, focus on smaller holders first (they're more likely to be fresh wallets)
                    # Sort by ascending balance to prioritize smaller holders
                    accounts_to_process = sorted(
                        [a for a in accounts_to_process if float(a.get("uiAmount", 0) or 0) > 0],
                        key=lambda x: float(x.get("uiAmount", 0) or 0)
                    )
                    
                    # If accounts to process is still large, limit based on full analysis needs
                    if len(accounts_to_process) > 50:
                        accounts_to_process = accounts_to_process[:50]
                        result["warnings"] = result.get("warnings", []) + [f"Limited analysis to top 50 accounts due to RPC constraints"]
                    
                    logger.info(f"Processing {len(accounts_to_process)} accounts for fresh wallet analysis")
                    
                    fresh_wallets = []
                    fresh_wallet_holdings_total = 0
                    
                    # Define batch size and delay between batches
                    BATCH_SIZE = 10
                    DELAY_SECONDS = 1  # Delay between batches
                    
                    # Process accounts in batches to avoid overwhelming the RPC
                    for i in range(0, len(accounts_to_process), BATCH_SIZE):
                        batch = accounts_to_process[i:i+BATCH_SIZE]
                        
                        # Add a delay between batches
                        if i > 0:
                            await asyncio.sleep(DELAY_SECONDS)
                        
                        # Process each account in the batch
                        for j, account in enumerate(batch):
                            # Small delay between accounts in the same batch
                            if j > 0:
                                await asyncio.sleep(0.2)  # 200ms delay between accounts
                                
                            account_address = account.get("address", "")
                            
                            # Skip missing addresses
                            if not account_address:
                                continue
                            
                            # Get account balance
                            try:
                                account_balance = float(account.get("uiAmount", 0) or 0)
                                
                                # Skip zero balances
                                if account_balance <= 0:
                                    continue
                                    
                                # Get the account owner
                                owner = None
                                
                                try:
                                    account_info = await self.client.get_account_info(account_address)
                                    
                                    # Extract owner from the token account data
                                    if account_info and "result" in account_info and account_info["result"]:
                                        # Check if the account is owned by the Token Program
                                        account_owner = account_info["result"].get("owner")
                                        if account_owner == TOKEN_PROGRAM_ID:
                                            data = account_info["result"].get("data")
                                            if isinstance(data, list) and len(data) >= 2 and data[0] == "base64":
                                                try:
                                                    # Decode base64 data
                                                    data_bytes = base64.b64decode(data[1])
                                                    
                                                    # Check if data length is sufficient
                                                    if len(data_bytes) >= 64:
                                                        # Extract owner pubkey (offset 32 in token account data)
                                                        owner = base58.b58encode(data_bytes[32:64]).decode('utf-8')
                                                except Exception as e:
                                                    logger.error(f"Error decoding account data: {str(e)}")
                                except Exception as e:
                                    logger.error(f"Error getting account info: {str(e)}")
                                    continue
                                
                                if not owner:
                                    continue
                                
                                # Full analysis for wallet freshness
                                # Get wallet age by looking at oldest transaction
                                wallet_age_days = None
                                token_tx_ratio = 0.0
                                
                                try:
                                    # Get a few signatures to analyze wallet age and transaction patterns
                                    signatures = await self.client.get_signatures_for_address(owner, limit=5)
                                    
                                    if signatures and "result" in signatures and signatures["result"]:
                                        # Sort by blockTime to find the oldest
                                        sorted_sigs = sorted(signatures["result"], key=lambda x: x.get("blockTime", 0) if x.get("blockTime") else float("inf"))
                                        
                                        if sorted_sigs:
                                            # Get the oldest transaction's timestamp
                                            if "blockTime" in sorted_sigs[0]:
                                                creation_timestamp = sorted_sigs[0]["blockTime"]
                                                creation_date = datetime.datetime.fromtimestamp(creation_timestamp)
                                                current_date = datetime.datetime.now()
                                                wallet_age_days = (current_date - creation_date).days
                                            
                                            # Calculate token transaction ratio (simplified)
                                            # Check what percentage of transactions involve our target token
                                            token_tx_count = 0
                                            for sig in sorted_sigs:
                                                # For simplicity, we'll just check if the memo contains our token address
                                                if mint in str(sig):
                                                    token_tx_count += 1
                                            
                                            if sorted_sigs:
                                                token_tx_ratio = token_tx_count / len(sorted_sigs)
                                except Exception as e:
                                    logger.error(f"Error checking wallet age for {owner}: {str(e)}")
                                
                                # Check SOL balance for token diversity scoring
                                non_dust_token_count = 0
                                
                                try:
                                    sol_balance_response = await self.client.get_balance(owner)
                                    sol_balance = float(sol_balance_response.get("result", {}).get("value", 0)) / 10**9
                                    
                                    if sol_balance > 0.01:  # Only count if more than 0.01 SOL
                                        non_dust_token_count += 1
                                except Exception as e:
                                    logger.error(f"Error getting SOL balance for {owner}: {str(e)}")
                                
                                # Determine if this is a fresh wallet based on criteria
                                # Calculate freshness score components
                                token_diversity_score = max(0, 1 - (non_dust_token_count / 5))  # 1.0 for 0 tokens, 0.0 for 5+ tokens
                                age_score = 0.0
                                if wallet_age_days is not None:
                                    age_score = max(0, 1 - (wallet_age_days / 30))  # 1.0 for 0 days, 0.0 for 30+ days
                                tx_score = token_tx_ratio  # 1.0 if all transactions involve our token, 0.0 if none
                                
                                # Weight the components
                                freshness_score = (token_diversity_score * 0.4) + (age_score * 0.4) + (tx_score * 0.2)
                                
                                # Mark as fresh if score above threshold
                                is_fresh = freshness_score >= 0.6
                                
                                # If identified as fresh, add to our results
                                if is_fresh:
                                    token_value = account_balance * price_usd
                                    percentage_of_supply = (account_balance / total_supply * 100) if total_supply > 0 else 0
                                    
                                    fresh_wallet = {
                                        "wallet_address": owner,
                                        "token_account": account_address,
                                        "token_balance": account_balance,
                                        "token_value_usd": token_value,
                                        "percentage_of_supply": percentage_of_supply,
                                        "wallet_age_days": wallet_age_days,
                                        "token_tx_ratio": token_tx_ratio,
                                        "freshness_score": freshness_score
                                    }
                                    
                                    fresh_wallets.append(fresh_wallet)
                                    fresh_wallet_holdings_total += account_balance
                            except Exception as e:
                                logger.error(f"Error analyzing account {account_address}: {str(e)}", exc_info=True)
                                continue
                    
                    # Sort fresh wallets by freshness score (highest first)
                    fresh_wallets.sort(key=lambda x: x.get("freshness_score", 0), reverse=True)
                    
                    # Update result with fresh wallet data
                    result["fresh_wallet_count"] = len(fresh_wallets)
                    result["fresh_wallets"] = fresh_wallets
                    
                    if result["total_holders_analyzed"] > 0:
                        result["fresh_wallet_percentage"] = (result["fresh_wallet_count"] / result["total_holders_analyzed"]) * 100
                    
                    if total_supply > 0:
                        result["fresh_wallet_holdings_percentage"] = (fresh_wallet_holdings_total / total_supply) * 100
                    
                    result["fresh_wallet_holdings_token_total"] = fresh_wallet_holdings_total
                else:
                    logger.warning(f"No token accounts found for {mint}")
                    result["warnings"] = result.get("warnings", []) + ["No token accounts found"]
            
            except Exception as e:
                logger.error(f"Error processing largest token accounts: {str(e)}", exc_info=True)
                result["warnings"] = result.get("warnings", []) + [f"Error processing largest token accounts: {str(e)}"]
            
        except Exception as e:
            logger.error(f"Error analyzing fresh wallets for {mint}: {str(e)}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
        
        log_with_context(
            logger,
            "info",
            f"Fresh wallets analysis completed for: {mint}",
            request_id=request_id,
            mint=mint,
            fresh_wallet_count=result.get("fresh_wallet_count", 0),
            fresh_wallet_holdings_percentage=f"{result.get('fresh_wallet_holdings_percentage', 0):.2f}%"
        )
        
        return result


class MemeTokenAnalyzer:
    """Analyzer for Solana tokens with focus on meme token detection and categorization."""

    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)

    @validate_solana_key
    @handle_errors
    async def get_token_metadata_with_category(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token metadata with meme category classification.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token metadata with category
        """
        log_with_context(
            logger,
            "info",
            f"Token metadata with category requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get token metadata
        token_metadata = await self.client.get_token_metadata(mint)
        name = token_metadata.get("name", "Unknown")
        symbol = token_metadata.get("symbol", "UNKNOWN")
        
        # Get token supply
        supply_info = await self.client.get_token_supply(mint)
        
        # Categorize token
        category = self._categorize_token(name, symbol)
        
        # Add category info to metadata
        token_metadata["category"] = category
        token_metadata["is_meme_token"] = category in ["Animal", "Food", "Meme"]
        token_metadata["decimals"] = supply_info.get("value", {}).get("decimals", 0)
        token_metadata["total_supply"] = supply_info.get("value", {}).get("uiAmountString", "0")
        
        log_with_context(
            logger,
            "info",
            f"Token metadata with category completed for: {mint}, category: {category}",
            request_id=request_id,
            mint=mint,
            category=category
        )
        
        return token_metadata

    @validate_solana_key
    @handle_errors
    async def analyze_holder_distribution(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze token holder distribution.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Holder distribution analysis
        """
        log_with_context(
            logger,
            "info",
            f"Holder distribution analysis requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        result = {
            "token_mint": mint,
            "total_holders": 0,
            "top_10_percentage": 0,
            "top_holder_percentage": 0,
            "concentration_index": 0,  # Gini coefficient-like measure of inequality
            "top_holders": []
        }
        
        try:
            # Get token metadata for context
            token_metadata = await self.client.get_token_metadata(mint)
            result["token_name"] = token_metadata.get("name", "Unknown")
            result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            
            # Get largest token holders
            accounts = await self.client.get_token_largest_accounts(mint)
            
            # Process accounts to extract holder data
            if "value" in accounts and len(accounts["value"]) > 0:
                total_supply = 0
                holder_amounts = []
                
                # Extract amounts and calculate total supply
                for account in accounts["value"]:
                    amount = float(account.get("uiAmount", 0))
                    holder_amounts.append(amount)
                    total_supply += amount
                
                # Sort by amount (largest first)
                holder_amounts.sort(reverse=True)
                
                # Calculate top holder percentages
                if total_supply > 0:
                    result["top_holder_percentage"] = (holder_amounts[0] / total_supply) * 100
                    result["largest_holder_percentage"] = result["top_holder_percentage"]  # Add for compatibility
                    
                    # Calculate top 10 percentage
                    top_10_sum = sum(holder_amounts[:min(10, len(holder_amounts))])
                    result["top_10_percentage"] = (top_10_sum / total_supply) * 100
                
                # Calculate concentration index (simplified Gini coefficient)
                if len(holder_amounts) > 1:
                    n = len(holder_amounts)
                    holder_amounts_sum = sum(holder_amounts)
                    
                    if holder_amounts_sum > 0:
                        # Sort holder amounts (ascending for the Gini formula)
                        holder_amounts.sort()
                        
                        # Calculate Gini coefficient
                        indices = range(1, n + 1)
                        gini = sum((2 * i - n - 1) * amt for i, amt in zip(indices, holder_amounts))
                        gini = gini / (n * holder_amounts_sum)
                        
                        result["concentration_index"] = gini
                
                # Extract top holders data
                result["total_holders"] = len(accounts["value"])
                for i, account in enumerate(accounts["value"][:10]):  # Get top 10
                    holder_info = {
                        "address": account.get("address", "unknown"),
                        "amount": account.get("uiAmount", 0),
                        "percentage": (account.get("uiAmount", 0) / total_supply) * 100 if total_supply > 0 else 0
                    }
                    result["top_holders"].append(holder_info)
                
        except Exception as e:
            logger.error(f"Error analyzing holder distribution for {mint}: {str(e)}")
            result["error"] = str(e)
            
        result["last_updated"] = datetime.datetime.now().isoformat()
            
        log_with_context(
            logger,
            "info",
            f"Holder distribution analysis completed for: {mint}",
            request_id=request_id,
            mint=mint,
            total_holders=result.get("total_holders", 0)
        )
        
        return result

    @validate_solana_key
    @handle_errors
    async def get_creation_date(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the creation date for a token.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token creation information
        """
        log_with_context(
            logger,
            "info",
            f"Token creation date requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        result = {
            "token_mint": mint,
            "creation_date": None,
            "creation_signature": None,
            "creation_block": None,
            "age_days": None,
        }
        
        try:
            # Get token metadata for context
            token_metadata = await self.client.get_token_metadata(mint)
            result["token_name"] = token_metadata.get("name", "Unknown")
            result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            
            # Get the earliest signature for this token mint
            signatures = await self.client.get_signatures_for_address(mint, limit=1, before=None)
            
            if signatures and len(signatures) > 0:
                creation_signature = signatures[-1]["signature"]  # Get the earliest signature
                result["creation_signature"] = creation_signature
                
                # Get full transaction details
                creation_tx = await self.client.get_transaction(creation_signature)
                
                # Extract creation timestamp
                if "blockTime" in creation_tx:
                    creation_timestamp = creation_tx["blockTime"]
                    creation_date = datetime.datetime.fromtimestamp(creation_timestamp)
                    
                    # Calculate token age
                    now = datetime.datetime.now()
                    age_days = (now - creation_date).days
                    
                    result["creation_date"] = creation_date.isoformat()
                    result["age_days"] = age_days
                
                # Extract creation block
                if "slot" in creation_tx:
                    result["creation_block"] = creation_tx["slot"]
                    
        except Exception as e:
            logger.error(f"Error getting creation date for {mint}: {str(e)}")
            result["error"] = str(e)
            
        result["last_updated"] = datetime.datetime.now().isoformat()
            
        log_with_context(
            logger,
            "info",
            f"Token creation date completed for: {mint}",
            request_id=request_id,
            mint=mint,
            creation_date=result.get("creation_date")
        )
        
        return result

    def _categorize_token(self, name: str, symbol: str) -> str:
        """Categorize token based on its name and symbol.
        
        Args:
            name: Token name
            symbol: Token symbol
            
        Returns:
            Token category (Animal, Food, Tech, Meme, Other)
        """
        combined_text = (name + " " + symbol).lower()
        
        # Check if token contains animal references
        if any(keyword in combined_text for keyword in ANIMAL_KEYWORDS):
            return "Animal"
        
        # Check if token contains food references
        if any(keyword in combined_text for keyword in FOOD_KEYWORDS):
            return "Food"
        
        # Check if token contains tech references
        if any(keyword in combined_text for keyword in TECH_KEYWORDS):
            return "Tech"
        
        # Check if token contains meme references
        if any(keyword in combined_text for keyword in MEME_KEYWORDS):
            return "Meme"
        
        # Default category
        return "Other" 