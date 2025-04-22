"""Token analysis for Solana tokens with focus on pumpfun tokens."""

import datetime
from typing import Dict, List, Any, Optional
import logging

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