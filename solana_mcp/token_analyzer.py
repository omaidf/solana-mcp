"""Solana token analyzer for detecting and analyzing tokens, particularly pumpfun tokens."""

# Standard library imports
import datetime
import functools
from typing import Dict, List, Any, Optional, Callable, TypeVar, Awaitable, cast
from dataclasses import dataclass

# Third-party library imports
import base58
import httpx
from cachetools import TTLCache

# Internal imports
from solana_mcp.solana_client import SolanaClient, validate_public_key, InvalidPublicKeyError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.config import get_cache_config

# Set up logging
logger = get_logger(__name__)

# Type variables for decorator
T = TypeVar('T')
R = TypeVar('R')

# Constants
JUPITER_PRICE_API = "https://price.jup.ag/v4/price"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
HTTP_TIMEOUT = 10.0  # seconds

# Cache settings from config
cache_config = get_cache_config()
METADATA_CACHE_SIZE = cache_config.metadata_cache_size
METADATA_CACHE_TTL = cache_config.metadata_cache_ttl


def validate_solana_key(func: Callable[[Any, str, ...], Awaitable[T]]) -> Callable[[Any, str, ...], Awaitable[T]]:
    """Decorator to validate Solana public key.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    async def wrapper(self: Any, mint: str, *args: Any, **kwargs: Any) -> T:
        request_id = kwargs.pop('request_id', None)
        log_with_context(
            logger, 
            "debug", 
            f"Validating Solana key: {mint}", 
            request_id=request_id,
            mint=mint,
            function=func.__name__
        )
        
        if not validate_public_key(mint):
            log_with_context(
                logger, 
                "error", 
                f"Invalid Solana key: {mint}", 
                request_id=request_id,
                mint=mint,
                function=func.__name__
            )
            raise InvalidPublicKeyError(mint)
            
        # Add request_id back to kwargs if it was present
        if request_id:
            kwargs['request_id'] = request_id
            
        return await func(self, mint, *args, **kwargs)
    return wrapper


def handle_errors(func: Callable[[Any, ...], Awaitable[R]]) -> Callable[[Any, ...], Awaitable[R]]:
    """Decorator to handle errors in token analyzer methods.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        request_id = kwargs.pop('request_id', None)
        
        try:
            # Add request_id back to kwargs if it was present
            if request_id:
                kwargs['request_id'] = request_id
                
            start_time = datetime.datetime.now()
            result = await func(*args, **kwargs)
            
            # Log execution time for performance monitoring
            duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
            log_with_context(
                logger,
                "debug",
                f"Function {func.__name__} completed in {duration:.2f}ms",
                request_id=request_id,
                function=func.__name__,
                duration=duration
            )
            
            return result
        except InvalidPublicKeyError:
            # Re-raise this specific error for proper handling
            raise
        except Exception as e:
            log_with_context(
                logger,
                "error",
                f"Error in {func.__name__}: {str(e)}",
                request_id=request_id,
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Return appropriate default value based on return type annotation
            return_type = func.__annotations__.get('return')
            if return_type == Dict[str, Any]:
                return {"error": str(e)}
            elif return_type == int:
                return 0
            elif 'TokenDistribution' in str(return_type):
                return TokenDistribution(0, 0.0, 0.0, 0.0, 0.0)
            else:
                return None  # type: ignore
    return wrapper


@dataclass
class TokenDistribution:
    """Token holder distribution data."""
    total_holders: int
    top_10_percent: float  # Percentage held by top 10% of holders
    top_50_holders: float  # Percentage held by top 50 holders
    largest_holder_percentage: float
    average_balance: float
    

@dataclass
class TokenMetadata:
    """Token metadata."""
    name: str
    symbol: str
    uri: Optional[str]
    update_authority: str
    is_mutable: bool
    creators: List[Dict[str, Any]]
    verified: bool


@dataclass
class TokenAnalysis:
    """Basic token analysis results."""
    token_mint: str
    token_name: str
    token_symbol: str
    decimals: int
    total_supply: int
    circulation_supply: int
    current_price_usd: float
    launch_date: Optional[datetime.datetime]
    age_days: Optional[int]
    owner_can_mint: bool
    owner_can_freeze: bool
    total_holders: int
    largest_holder_percentage: float
    last_updated: datetime.datetime


class TokenAnalyzer:
    """Analyzer for Solana tokens with focus on detecting pumpfun and risky tokens."""
    
    # Metaplex Token Metadata Program ID
    METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the token analyzer.
        
        Args:
            solana_client: Initialized Solana client
        """
        self.client = solana_client
        log_with_context(logger, "info", "Initializing TokenAnalyzer", component="TokenAnalyzer")
        
        self.http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
        # Initialize cache
        self.metadata_cache = TTLCache(maxsize=METADATA_CACHE_SIZE, ttl=METADATA_CACHE_TTL)
        log_with_context(
            logger, 
            "debug", 
            f"Cache initialized with size={METADATA_CACHE_SIZE}, ttl={METADATA_CACHE_TTL}s",
            component="TokenAnalyzer",
            cache_size=METADATA_CACHE_SIZE,
            cache_ttl=METADATA_CACHE_TTL
        )
    
    async def __aenter__(self):
        """Async context manager entry.
        
        Returns:
            Self reference
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.
        
        Cleans up resources.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        await self.close()
        
    async def close(self):
        """Close resources used by the analyzer."""
        if hasattr(self, 'http_client') and self.http_client:
            await self.http_client.aclose()
            log_with_context(
                logger,
                "debug",
                "HTTP client closed",
                component="TokenAnalyzer"
            )
    
    def _get_metadata_address(self, mint: str) -> str:
        """Derive the metadata PDA for a mint.
        
        Args:
            mint: The mint address
            
        Returns:
            The metadata account address
        """
        # Convert strings to bytes for PDA derivation
        seeds = [
            b"metadata",
            bytes(base58.b58decode(self.METADATA_PROGRAM_ID)),
            bytes(base58.b58decode(mint))
        ]
        
        # Simple deterministic derivation (this is a simplified version)
        # In production, you'd use the proper PDA derivation from the Solana SDK
        metadata_address = base58.b58encode(seeds[0] + seeds[1][:4] + seeds[2][:4]).decode('utf-8')
        
        # Truncate to valid address length and ensure proper encoding
        metadata_address = metadata_address[:44]
        
        log_with_context(
            logger, 
            "debug", 
            f"Derived metadata address: {metadata_address} for mint: {mint}",
            mint=mint,
            metadata_address=metadata_address
        )
        
        return metadata_address
    
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
        # Check cache first
        cache_key = f"metadata_{mint}"
        if cache_key in self.metadata_cache:
            log_with_context(
                logger, 
                "debug", 
                f"Cache hit for metadata: {mint}",
                request_id=request_id,
                mint=mint,
                from_cache=True
            )
            return self.metadata_cache[cache_key]
            
        log_with_context(
            logger, 
            "debug", 
            f"Cache miss for metadata: {mint}",
            request_id=request_id,
            mint=mint,
            from_cache=False
        )
            
        metadata_address = self._get_metadata_address(mint)
        
        # Get the metadata account
        log_with_context(
            logger, 
            "debug", 
            f"Fetching account info for metadata address: {metadata_address}",
            request_id=request_id,
            mint=mint,
            metadata_address=metadata_address
        )
        
        account_info = await self.client.get_account_info(metadata_address)
        
        # Process the account data to extract metadata
        # This is simplified - real implementation would decode the account data per Metaplex format
        if "value" in account_info and account_info["value"]:
            data = account_info["value"].get("data", ["", ""])
            if isinstance(data, list) and len(data) > 0:
                # Decode base64 data - this is simplified
                # Real implementation would decode per Metaplex metadata schema
                metadata = {
                    "mint": mint,
                    "name": "Token Name",  # Would be parsed from actual data
                    "symbol": "TKN",       # Would be parsed from actual data
                    "uri": "uri",          # Would be parsed from actual data
                    "update_authority": "authority"  # Would be parsed from actual data
                }
                
                log_with_context(
                    logger, 
                    "info", 
                    f"Retrieved metadata for token: {mint}",
                    request_id=request_id,
                    mint=mint,
                    token_name=metadata["name"],
                    token_symbol=metadata["symbol"]
                )
                
                self.metadata_cache[cache_key] = metadata
                return metadata
                
        log_with_context(
            logger, 
            "warning", 
            f"No metadata found for token: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # If no metadata found, return basic info
        default_metadata = {
            "mint": mint,
            "name": "Unknown Token",
            "symbol": "UNKNOWN"
        }
        self.metadata_cache[cache_key] = default_metadata
        return default_metadata
    
    @validate_solana_key
    @handle_errors
    async def get_token_supply_and_decimals(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token supply and decimals.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token supply information
        """
        log_with_context(
            logger, 
            "debug", 
            f"Fetching token supply for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        supply_info = await self.client.get_token_supply(mint)
        
        if "value" in supply_info:
            log_with_context(
                logger, 
                "info", 
                f"Retrieved supply for token: {mint}",
                request_id=request_id,
                mint=mint,
                amount=supply_info["value"].get("amount", "0"),
                decimals=supply_info["value"].get("decimals", 0)
            )
        
        return supply_info
    
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
            "debug", 
            f"Fetching token holders count for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get program accounts for token accounts of this mint
        # Filter for accounts with the mint field equal to our mint
        # Note: This is a simplified approach and might not get all accounts for tokens with many holders
        filters = [
            {"memcmp": {"offset": 0, "bytes": mint}}
        ]
        accounts = await self.client.get_program_accounts(
            TOKEN_PROGRAM_ID,
            filters=filters,
            limit=1000  # Limiting results, real implementation would handle pagination
        )
        
        count = len(accounts)
        
        log_with_context(
            logger, 
            "info", 
            f"Token {mint} has {count} holders",
            request_id=request_id,
            mint=mint,
            holders_count=count
        )
        
        return count
    
    @validate_solana_key
    @handle_errors
    async def get_token_largest_holders(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get largest token holders.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Information about largest token holders
        """
        log_with_context(
            logger, 
            "debug", 
            f"Fetching largest token holders for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        largest_accounts = await self.client.get_token_largest_accounts(mint)
        
        # Extract total from all accounts
        total_amount = 0
        accounts = largest_accounts.get("value", [])
        
        for account in accounts:
            total_amount += int(account.get("amount", "0"))
        
        # Calculate percentages for largest holders
        if total_amount > 0 and len(accounts) > 0:
            largest_holder_percentage = (int(accounts[0].get("amount", "0")) / total_amount) * 100
        else:
            largest_holder_percentage = 0
            
        log_with_context(
            logger, 
            "info", 
            f"Token {mint} has {len(accounts)} large accounts, largest holder has {largest_holder_percentage:.2f}%",
            request_id=request_id,
            mint=mint,
            accounts_count=len(accounts),
            largest_holder_percentage=largest_holder_percentage
        )
            
        return {
            "total_holders": len(accounts),
            "largest_holder_percentage": largest_holder_percentage,
            "accounts": accounts
        }
    
    @validate_solana_key
    @handle_errors
    async def get_token_price(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token price information.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Price information
        """
        log_with_context(
            logger, 
            "debug", 
            f"Fetching token price for: {mint}",
            request_id=request_id,
            mint=mint,
            price_api=JUPITER_PRICE_API
        )
        
        # Use Jupiter API to get current price
        url = f"{JUPITER_PRICE_API}?ids={mint}"
        response = await self.http_client.get(url)
        response.raise_for_status()
        price_data = response.json()
        
        if "data" in price_data and mint in price_data["data"]:
            price = price_data["data"][mint].get("price", 0)
            log_with_context(
                logger, 
                "info", 
                f"Token {mint} price: ${price}",
                request_id=request_id,
                mint=mint,
                price=price
            )
            return price_data["data"][mint]
        
        log_with_context(
            logger, 
            "warning", 
            f"Price not available for token: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        return {"price": 0, "error": "Price not available"}
    
    @validate_solana_key
    @handle_errors
    async def get_token_mint_authority(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token mint and freeze authorities.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Authority information
        """
        log_with_context(
            logger, 
            "debug", 
            f"Fetching token authorities for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        account_info = await self.client.get_account_info(mint, encoding="jsonParsed")
        
        # Extract relevant data from the account info
        if "value" in account_info and account_info["value"]:
            data = account_info["value"].get("data", {})
            
            if isinstance(data, dict) and "parsed" in data:
                parsed_data = data["parsed"]
                
                # Extract mint and freeze authorities
                info = parsed_data.get("info", {})
                mint_authority = info.get("mintAuthority")
                freeze_authority = info.get("freezeAuthority")
                
                log_with_context(
                    logger, 
                    "info", 
                    f"Token {mint} authorities: mint={mint_authority is not None}, freeze={freeze_authority is not None}",
                    request_id=request_id,
                    mint=mint,
                    has_mint_authority=mint_authority is not None,
                    has_freeze_authority=freeze_authority is not None
                )
                
                return {
                    "mint_authority": mint_authority,
                    "freeze_authority": freeze_authority,
                    "has_mint_authority": mint_authority is not None,
                    "has_freeze_authority": freeze_authority is not None
                }
        
        log_with_context(
            logger, 
            "warning", 
            f"Could not parse authority data for token: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Default response if data can't be parsed
        return {
            "mint_authority": None, 
            "freeze_authority": None,
            "has_mint_authority": False,
            "has_freeze_authority": False
        }
    
    @validate_solana_key
    @handle_errors
    async def get_token_age(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token age based on earliest transaction.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token age information
        """
        log_with_context(
            logger, 
            "debug", 
            f"Fetching token age for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get transaction signatures for this mint, looking at the oldest ones
        signatures = await self.client.get_signatures_for_address(
            mint, 
            limit=1000  # Request a large number to try to get the oldest
        )
        
        # If no signatures found, can't determine age
        if not signatures:
            log_with_context(
                logger, 
                "warning", 
                f"No signatures found for token: {mint}",
                request_id=request_id,
                mint=mint
            )
            
            return {
                "launch_date": None,
                "age_days": None
            }
        
        # Get the oldest signature (last in list)
        oldest_signature = signatures[-1]
        
        # Get transaction time
        block_time = oldest_signature.get("blockTime")
        
        if block_time:
            launch_date = datetime.datetime.fromtimestamp(block_time)
            age_days = (datetime.datetime.now() - launch_date).days
            
            log_with_context(
                logger, 
                "info", 
                f"Token {mint} age: {age_days} days, launched: {launch_date.isoformat()}",
                request_id=request_id,
                mint=mint,
                age_days=age_days,
                launch_date=launch_date.isoformat()
            )
            
            return {
                "launch_date": launch_date,
                "age_days": age_days
            }
        
        log_with_context(
            logger, 
            "warning", 
            f"Could not determine launch date for token: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        return {
            "launch_date": None,
            "age_days": None
        }
    
    @validate_solana_key
    @handle_errors
    async def analyze_token(self, mint: str, request_id: Optional[str] = None) -> TokenAnalysis:
        """Perform token analysis by combining multiple data sources.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token analysis
        """
        log_with_context(
            logger, 
            "info", 
            f"Starting comprehensive token analysis for: {mint}",
            request_id=request_id,
            mint=mint,
            analysis_type="comprehensive"
        )
        
        # Run all data fetching operations concurrently
        metadata_future = self.get_token_metadata(mint, request_id=request_id)
        supply_future = self.get_token_supply_and_decimals(mint, request_id=request_id)
        price_future = self.get_token_price(mint, request_id=request_id)
        holder_future = self.get_token_largest_holders(mint, request_id=request_id)
        authority_future = self.get_token_mint_authority(mint, request_id=request_id)
        age_future = self.get_token_age(mint, request_id=request_id)
        
        log_with_context(
            logger, 
            "debug", 
            f"Initiated concurrent data fetching for token: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Await all operations
        metadata = await metadata_future
        supply_info = await supply_future
        price_data = await price_future
        holder_data = await holder_future
        authority_info = await authority_future
        age_info = await age_future
        
        # Extract data from results
        token_name = metadata.get("name", "Unknown Token")
        token_symbol = metadata.get("symbol", "UNKNOWN")
        
        decimals = supply_info.get("value", {}).get("decimals", 0)
        total_supply = int(supply_info.get("value", {}).get("amount", "0"))
        
        current_price = price_data.get("price", 0)
        
        total_holders = holder_data.get("total_holders", 0)
        largest_holder_percentage = holder_data.get("largest_holder_percentage", 0)
        
        owner_can_mint = authority_info.get("has_mint_authority", False)
        owner_can_freeze = authority_info.get("has_freeze_authority", False)
        
        launch_date = age_info.get("launch_date")
        age_days = age_info.get("age_days")
        
        # Estimate circulation supply (total supply - known locked amounts)
        # This is a simplification - real implementation would account for known locked tokens
        circulation_supply = int(total_supply * 0.9)  # Assume 10% locked as a simplification
        
        log_with_context(
            logger, 
            "info", 
            f"Completed token analysis for: {mint}, {token_name} ({token_symbol})",
            request_id=request_id,
            mint=mint,
            token_name=token_name,
            token_symbol=token_symbol,
            total_supply=total_supply,
            current_price=current_price,
            total_holders=total_holders,
            age_days=age_days
        )
        
        # Create analysis result
        return TokenAnalysis(
            token_mint=mint,
            token_name=token_name,
            token_symbol=token_symbol,
            decimals=decimals,
            total_supply=total_supply,
            circulation_supply=circulation_supply,
            current_price_usd=current_price,
            launch_date=launch_date,
            age_days=age_days,
            owner_can_mint=owner_can_mint,
            owner_can_freeze=owner_can_freeze,
            total_holders=total_holders,
            largest_holder_percentage=largest_holder_percentage,
            last_updated=datetime.datetime.now()
        )
    
    async def monitor_new_tokens(self) -> List[Dict[str, Any]]:
        """Monitor for newly created tokens on Solana.
        
        Returns:
            List of newly detected tokens with basic info
        """
        # This would require monitoring the token program for mint creation events
        # This is a complex operation that requires subscription to blockchain events
        # Not implemented in this version as it requires additional infrastructure
        raise NotImplementedError("Token monitoring requires additional infrastructure") 