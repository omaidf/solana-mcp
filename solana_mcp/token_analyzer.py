"""Token analysis for Solana tokens with focus on pumpfun tokens."""

import datetime
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.decorators import validate_solana_key, handle_errors

# Set up logging
logger = get_logger(__name__)

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
        
        try:
            # Get token metadata
            metadata = await self.get_token_metadata(mint, request_id=request_id)
            token_name = metadata.get("name", "Unknown")
            token_symbol = metadata.get("symbol", "UNKNOWN")
            
            # Get token supply and decimals
            supply_info = await self.get_token_supply_and_decimals(mint, request_id=request_id)
            decimals = supply_info.get("value", {}).get("decimals", 0)
            total_supply = supply_info.get("value", {}).get("uiAmountString", "0")
            
            # Get holders data
            holders_data = await self.get_token_largest_holders(mint, request_id=request_id)
            total_holders = holders_data.get("total_holders", 0)
            largest_holder_percentage = holders_data.get("largest_holder_percentage", 0)
            
            # Get price data before whale and fresh wallet analysis
            # as both depend on accurate price data
            price_data = await self.get_token_price(mint, request_id=request_id)
            current_price_usd = price_data.get("price_usd", 0.0)
            
            # Get whale data
            try:
                whale_data = await self.get_whale_holders(mint, request_id=request_id)
                whale_count = whale_data.get("whale_count", 0)
                whale_percentage = whale_data.get("whale_percentage", 0.0)
                whale_holdings_percentage = whale_data.get("whale_holdings_percentage", 0.0)
                whale_holdings_usd_total = whale_data.get("whale_holdings_usd_total", 0.0)
            except Exception as e:
                logger.error(f"Error getting whale data: {str(e)}", exc_info=True)
                whale_count = 0
                whale_percentage = 0.0
                whale_holdings_percentage = 0.0
                whale_holdings_usd_total = 0.0
            
            # Get fresh wallet data
            try:
                fresh_wallet_data = await self.get_fresh_wallets(mint, request_id=request_id)
                fresh_wallet_count = fresh_wallet_data.get("fresh_wallet_count", 0)
                fresh_wallet_percentage = fresh_wallet_data.get("fresh_wallet_percentage", 0.0)
                fresh_wallet_holdings_percentage = fresh_wallet_data.get("fresh_wallet_holdings_percentage", 0.0)
            except Exception as e:
                logger.error(f"Error getting fresh wallet data: {str(e)}", exc_info=True)
                fresh_wallet_count = 0
                fresh_wallet_percentage = 0.0
                fresh_wallet_holdings_percentage = 0.0
            
            # Get age data
            age_data = await self.get_token_age(mint, request_id=request_id)
            launch_date_str = age_data.get("launch_date")
            launch_date = datetime.datetime.fromisoformat(launch_date_str) if launch_date_str else None
            age_days = age_data.get("age_days")
            
            # Get authority data
            auth_data = await self.get_token_mint_authority(mint, request_id=request_id)
            owner_can_mint = auth_data.get("has_mint_authority", False)
            owner_can_freeze = auth_data.get("has_freeze_authority", False)
            
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
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get token account info
            account_info = await self.client.get_account_info(mint)
            
            # Check if account exists
            if account_info and "result" in account_info and account_info["result"]:
                # Process mint account data
                data = account_info["result"].get("data")
                
                if isinstance(data, list) and data[0] == "base64":
                    # Decode base64 data
                    import base64
                    data_bytes = base64.b64decode(data[1])
                    
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
                    result["has_mint_authority"] = mint_authority_option == 1
                    
                    if result["has_mint_authority"]:
                        # Extract mint authority pubkey
                        import base58
                        mint_authority = base58.b58encode(data_bytes[1:33]).decode('utf-8')
                        result["mint_authority"] = mint_authority
                    
                    # Get decimals for additional info
                    result["decimals"] = data_bytes[41]
                    
                    # Check if initialized
                    result["is_initialized"] = data_bytes[42] == 1
                    
                    # Check if freeze authority is present
                    freeze_authority_option = data_bytes[43]
                    result["has_freeze_authority"] = freeze_authority_option == 1
                    
                    if result["has_freeze_authority"]:
                        # Extract freeze authority pubkey
                        freeze_authority = base58.b58encode(data_bytes[44:76]).decode('utf-8')
                        result["freeze_authority"] = freeze_authority
                    
                    # A token is mutable if it has a mint authority
                    result["is_mutable"] = result["has_mint_authority"]
                
                else:
                    result["error"] = "Invalid data format"
                
        except Exception as e:
            logger.error(f"Error getting token authority for {mint}: {str(e)}")
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"Token authority completed for: {mint}",
            request_id=request_id,
            mint=mint,
            has_mint_authority=result.get("has_mint_authority", False),
            has_freeze_authority=result.get("has_freeze_authority", False)
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

    @validate_solana_key
    @handle_errors
    async def get_whale_holders(self, mint: str, threshold_usd: float = 50000.0, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token holders with balances over the specified USD threshold.
        
        Args:
            mint: The token mint address
            threshold_usd: The minimum USD value to consider a holder a whale (default: $50K)
            request_id: Optional request ID for tracing
            
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
            "token_mint": mint,
            "threshold_usd": threshold_usd,
            "total_holders_analyzed": 0,
            "total_holders": 0,  # Initialize total_holders explicitly
            "whale_count": 0,
            "whale_percentage": 0,
            "whale_holders": [],
            "whale_holdings_usd_total": 0,
            "whale_holdings_percentage": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get token metadata
            token_metadata = await self.get_token_metadata(mint, request_id=request_id)
            result["token_name"] = token_metadata.get("name", "Unknown")
            result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            
            # Always get total holders count first to ensure this data is included
            try:
                total_holders = await self.get_token_holders_count(mint, request_id=request_id)
                result["total_holders"] = total_holders
            except Exception as e:
                logger.error(f"Error getting total holders count: {str(e)}", exc_info=True)
                # Will fall back to analyzed count later if needed
            
            # Get current token price
            price_data = await self.get_token_price(mint, request_id=request_id)
            price_usd = price_data.get("price_usd", 0.0)
            
            # Handle price data reliability
            if price_usd <= 0:
                # Try other price sources or estimation methods if available
                price_sol = price_data.get("price_sol", 0.0)
                if price_sol > 0:
                    # Estimate USD price using SOL price if SOL/USD rate is available
                    # This is a simplistic fallback - in a real implementation, you'd use a reliable oracle
                    estimated_sol_usd = 100.0  # Fallback estimated SOL/USD price
                    price_usd = price_sol * estimated_sol_usd
                    logger.warning(f"Using estimated USD price from SOL price: ${price_usd}")
                    result["price_source"] = "estimated_from_sol"
            
            if price_usd <= 0:
                result["error"] = "Unable to determine token price in USD"
                result["price_usd"] = 0.0
                logger.error(f"Unable to determine USD price for token {mint}")
                # Continue anyway to gather holder data, but whale detection will be limited
            else:
                result["token_price_usd"] = price_usd
            
            # Get token supply and decimals
            supply_info = await self.get_token_supply_and_decimals(mint, request_id=request_id)
            decimals = supply_info.get("value", {}).get("decimals", 0)
            total_supply_str = supply_info.get("value", {}).get("uiAmountString", "0")
            
            try:
                total_supply = float(total_supply_str)
                if total_supply <= 0:
                    logger.warning(f"Invalid or zero total supply value: {total_supply_str}")
                    total_supply = 1  # Use a non-zero default to avoid division by zero
                result["total_supply"] = total_supply
            except (ValueError, TypeError):
                logger.warning(f"Could not parse total supply: {total_supply_str}")
                total_supply = 1  # Use a non-zero default to avoid division by zero
                result["error"] = f"Invalid total supply value: {total_supply_str}"
            
            # Get largest token accounts (limit 100)
            largest_accounts = await self.client.get_token_largest_accounts(mint)
            accounts = largest_accounts.get("value", [])
            
            result["total_holders_analyzed"] = len(accounts)
            
            # If total_holders wasn't successfully fetched earlier, fall back to analyzed count
            if result["total_holders"] == 0:
                result["total_holders"] = result["total_holders_analyzed"]
            
            whale_holders = []
            whale_holdings_total = 0
            
            for account in accounts:
                try:
                    # Get account balance in token units with safer parsing
                    account_balance = 0.0
                    ui_amount = account.get("uiAmount")
                    
                    # Handle different types of uiAmount - could be string or number
                    if ui_amount is not None:
                        if isinstance(ui_amount, (int, float)):
                            account_balance = float(ui_amount)
                        elif isinstance(ui_amount, str):
                            try:
                                account_balance = float(ui_amount)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not parse account balance: {ui_amount}")
                                continue
                    
                    # Calculate USD value (will be 0 if price_usd is 0)
                    balance_usd = account_balance * price_usd
                    
                    # Check if this is a whale, but only if we have price data
                    if price_usd > 0 and balance_usd >= threshold_usd:
                        percentage_of_supply = (account_balance / total_supply * 100) if total_supply > 0 else 0
                        
                        whale_holder = {
                            "address": account.get("address", "unknown"),
                            "token_balance": account_balance,
                            "usd_value": balance_usd,
                            "percentage_of_supply": percentage_of_supply
                        }
                        
                        whale_holders.append(whale_holder)
                        whale_holdings_total += account_balance
                except Exception as e:
                    logger.error(f"Error processing account balance: {str(e)}", exc_info=True)
                    continue
            
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
            logger.error(f"Error analyzing whale holders for {mint}: {str(e)}", exc_info=True)
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
    async def get_fresh_wallets(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token holders that only hold this specific token (fresh wallets).
        
        Fresh wallets are wallets that have only bought this specific token and no others,
        which could indicate artificial activity or targeted pump activity.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
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
            # Get token metadata
            token_metadata = await self.get_token_metadata(mint, request_id=request_id)
            result["token_name"] = token_metadata.get("name", "Unknown")
            result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            
            # Get token supply and decimals
            supply_info = await self.get_token_supply_and_decimals(mint, request_id=request_id)
            decimals = supply_info.get("value", {}).get("decimals", 0)
            total_supply_str = supply_info.get("value", {}).get("uiAmountString", "0")
            
            try:
                total_supply = float(total_supply_str)
                if total_supply <= 0:
                    logger.warning(f"Invalid or zero total supply value: {total_supply_str}")
                    total_supply = 1  # Use a non-zero default to avoid division by zero
                result["total_supply"] = total_supply
            except (ValueError, TypeError):
                logger.warning(f"Could not parse total supply: {total_supply_str}")
                total_supply = 1  # Use a non-zero default to avoid division by zero
                result["error"] = f"Invalid total supply value: {total_supply_str}"
            
            # Get largest token accounts (limit 100)
            largest_accounts = await self.client.get_token_largest_accounts(mint)
            accounts = largest_accounts.get("value", [])
            
            result["total_holders_analyzed"] = len(accounts)
            
            fresh_wallets = []
            fresh_wallet_holdings_total = 0
            
            # Set of addresses already analyzed to avoid duplicates
            analyzed_owners = set()
            
            # For each token account, check if the owner only holds this token
            for account in accounts:
                account_address = account.get("address", "")
                account_balance = float(account.get("uiAmount", 0))
                
                # Get the owner of this token account
                try:
                    token_account_info = await self.client.get_account_info(account_address)
                    
                    # Extract owner from the token account data
                    owner = None
                    
                    # Verify that account exists and is a token account
                    if token_account_info and "result" in token_account_info and token_account_info["result"]:
                        # Check if the account is owned by the Token Program
                        account_owner = token_account_info["result"].get("owner")
                        if account_owner == TOKEN_PROGRAM_ID:
                            data = token_account_info["result"].get("data")
                            if isinstance(data, list) and len(data) >= 2 and data[0] == "base64":
                                try:
                                    # Decode base64 data
                                    import base64
                                    data_bytes = base64.b64decode(data[1])
                                    
                                    # Check if data length is sufficient
                                    if len(data_bytes) >= 64:
                                        # Extract owner pubkey (offset 32 in token account data)
                                        import base58
                                        owner = base58.b58encode(data_bytes[32:64]).decode('utf-8')
                                except Exception as e:
                                    logger.error(f"Error decoding account data: {str(e)}")
                                    continue
                    
                    if owner and owner not in analyzed_owners:
                        analyzed_owners.add(owner)
                        
                        # Get all token accounts owned by this wallet
                        token_accounts = await self.client.get_token_accounts_by_owner(owner)
                        
                        # Count how many different tokens this wallet holds
                        unique_tokens = set()
                        token_balances = {}
                        
                        if "value" in token_accounts:
                            for token_account in token_accounts["value"]:
                                account_data = token_account.get("account", {}).get("data", {})
                                if isinstance(account_data, dict) and "parsed" in account_data:
                                    parsed_data = account_data["parsed"]
                                    if "info" in parsed_data and "mint" in parsed_data["info"]:
                                        token_mint_addr = parsed_data["info"]["mint"]
                                        token_amount = parsed_data["info"].get("tokenAmount", {})
                                        token_balance = token_amount.get("uiAmount", 0)
                                        
                                        # Only count if balance > 0
                                        if token_balance > 0:
                                            unique_tokens.add(token_mint_addr)
                                            token_balances[token_mint_addr] = token_balance
                        
                        # Check if wallet only holds our target token
                        # A wallet is considered "fresh" if it only holds this specific token
                        # (checking for only one token in unique_tokens is sufficient since we're only 
                        # counting tokens with balance > 0, and native SOL isn't counted in token accounts)
                        is_fresh_wallet = len(unique_tokens) == 1 and mint in unique_tokens
                        
                        if is_fresh_wallet:
                            # Get the specific balance for our target token
                            target_token_balance = token_balances.get(mint, 0)
                            percentage_of_supply = (target_token_balance / total_supply * 100) if total_supply > 0 else 0
                            
                            fresh_wallet = {
                                "wallet_address": owner,
                                "token_account": account_address,
                                "token_balance": target_token_balance,
                                "percentage_of_supply": percentage_of_supply
                            }
                            
                            fresh_wallets.append(fresh_wallet)
                            fresh_wallet_holdings_total += target_token_balance
                
                except Exception as e:
                    logger.error(f"Error processing account {account_address}: {str(e)}", exc_info=True)
                    continue
            
            # Update result with fresh wallet data
            result["fresh_wallet_count"] = len(fresh_wallets)
            result["fresh_wallets"] = fresh_wallets
            
            if result["total_holders_analyzed"] > 0:
                result["fresh_wallet_percentage"] = (result["fresh_wallet_count"] / result["total_holders_analyzed"]) * 100
            
            if total_supply > 0:
                result["fresh_wallet_holdings_percentage"] = (fresh_wallet_holdings_total / total_supply) * 100
            
            result["fresh_wallet_holdings_token_total"] = fresh_wallet_holdings_total
            
        except Exception as e:
            logger.error(f"Error analyzing fresh wallets for {mint}: {str(e)}", exc_info=True)
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