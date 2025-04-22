"""Token categorization module for Solana tokens."""

import re
from typing import Dict, List, Any, Optional, Set
import logging
from dataclasses import dataclass

from solana_mcp.solana_client import SolanaClient
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.decorators import validate_solana_key, handle_errors

# Set up logging
logger = get_logger(__name__)

# Token category definitions with keywords
TOKEN_CATEGORIES = {
    "meme": [
        "meme", "doge", "pepe", "shib", "cat", "inu", "dog", "elon", "wojak", "chad",
        "moon", "rocket", "gme", "based", "bonk", "pepsi", "cola", "cheese", "monkey",
        "ape", "fun", "dogwifhat", "doggo", "pup", "floki", "corgi"
    ],
    "defi": [
        "swap", "yield", "farm", "stake", "dao", "governance", "curve", "lend", "borrow",
        "amm", "pool", "index", "oracle", "synthetic", "options", "margin", "leverage",
        "collateral", "liquid", "vault", "aave", "compound", "sushi", "pancake", "uniswap"
    ],
    "stablecoin": [
        "usd", "stable", "dollar", "usdt", "usdc", "dai", "peg", "tether", "busd", "tusd",
        "eurc", "eur", "euro", "jpy", "yen", "gold", "pax", "reserve", "fiat"
    ],
    "gaming": [
        "game", "play", "nft", "meta", "verse", "vr", "virtual", "reality", "token", "reward",
        "item", "loot", "quest", "guild", "raid", "battle", "war", "skill", "craft", "adventure",
        "dungeon", "arena", "player", "score", "level", "boss", "weapon"
    ],
    "infrastructure": [
        "chain", "protocol", "bridge", "layer", "cross", "oracle", "network", "node", "validator",
        "consensus", "block", "transaction", "data", "storage", "compute", "cloud", "api", "rpc",
        "web3", "eth", "sol", "gas", "fee", "security", "private", "rollup", "zero", "proof"
    ]
}


@dataclass
class TokenClassification:
    """Classification result for a token."""
    token_mint: str
    token_name: str
    token_symbol: str
    primary_category: str
    secondary_categories: List[str]
    category_scores: Dict[str, float]
    keywords_matched: Dict[str, List[str]]
    description: Optional[str] = None


class TokenCategorizer:
    """Categorizes Solana tokens based on keywords and metadata."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)
        self.categories = TOKEN_CATEGORIES
        
    @validate_solana_key
    @handle_errors
    async def categorize_token(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Categorize a token based on its name, symbol, and metadata.
        
        Args:
            mint: Token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token classification information
        """
        log_with_context(
            logger,
            "info",
            f"Token categorization requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get token metadata
        token_metadata = await self.client.get_token_metadata(mint)
        
        # Extract basic token info
        token_name = token_metadata.get("name", "Unknown")
        token_symbol = token_metadata.get("symbol", "UNKNOWN")
        token_description = token_metadata.get("description", "")
        
        # Combined text for searching
        combined_text = f"{token_name.lower()} {token_symbol.lower()} {token_description.lower()}"
        
        # Match keywords from each category
        category_matches = {}
        keywords_matched = {}
        
        for category, keywords in self.categories.items():
            category_matches[category] = 0
            keywords_matched[category] = []
            
            for keyword in keywords:
                # Check if keyword is present in the combined text
                matches = re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text)
                match_count = len(matches)
                
                if match_count > 0:
                    category_matches[category] += match_count
                    keywords_matched[category].append(keyword)
        
        # Determine total matches for percentage calculation
        total_matches = sum(category_matches.values())
        
        # Calculate scores as percentages
        category_scores = {}
        for category, count in category_matches.items():
            if total_matches > 0:
                category_scores[category] = round((count / total_matches) * 100, 2)
            else:
                category_scores[category] = 0.0
        
        # Get primary and secondary categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        primary_category = sorted_categories[0][0] if sorted_categories else "unknown"
        secondary_categories = [cat for cat, score in sorted_categories[1:3] if score > 0]
        
        # Create classification object
        classification = TokenClassification(
            token_mint=mint,
            token_name=token_name,
            token_symbol=token_symbol,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            category_scores=category_scores,
            keywords_matched=keywords_matched,
            description=token_description
        )
        
        # Prepare result dictionary
        result = {
            "token_mint": mint,
            "token_name": token_name,
            "token_symbol": token_symbol,
            "primary_category": primary_category,
            "secondary_categories": secondary_categories,
            "category_scores": category_scores,
            "keywords_matched": keywords_matched,
            "description": token_description
        }
        
        log_with_context(
            logger,
            "info",
            f"Token categorized as {primary_category} with {len(secondary_categories)} secondary categories",
            request_id=request_id,
            mint=mint,
            primary_category=primary_category
        )
        
        return result
    
    @handle_errors
    async def batch_categorize_tokens(self, mints: List[str], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Categorize multiple tokens at once.
        
        Args:
            mints: List of token mint addresses
            request_id: Optional request ID for tracing
            
        Returns:
            Dictionary of token categorizations
        """
        log_with_context(
            logger,
            "info",
            f"Batch token categorization requested for {len(mints)} tokens",
            request_id=request_id,
            token_count=len(mints)
        )
        
        results = {}
        
        for mint in mints:
            try:
                validation_error = self._validate_solana_key_format(mint)
                if validation_error:
                    results[mint] = {"error": validation_error}
                    continue
                    
                result = await self.categorize_token(mint, request_id)
                results[mint] = result
            except Exception as e:
                logger.error(f"Error categorizing token {mint}: {str(e)}")
                results[mint] = {"error": str(e)}
        
        log_with_context(
            logger,
            "info",
            f"Batch categorization completed for {len(mints)} tokens",
            request_id=request_id,
            token_count=len(mints)
        )
        
        return {
            "tokens": results,
            "count": len(mints),
            "categories_found": self._summarize_categories(results)
        }
    
    def _validate_solana_key_format(self, key: str) -> Optional[str]:
        """Validate a Solana public key format.
        
        Args:
            key: The key to validate
            
        Returns:
            Error message if invalid, None otherwise
        """
        if not key:
            return "Key cannot be empty"
            
        if not isinstance(key, str):
            return "Key must be a string"
            
        if len(key) != 44 and len(key) != 43:
            return f"Invalid key length: {len(key)}, expected 43 or 44"
            
        # Check if key contains only base58 characters
        if not re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', key):
            return "Key contains invalid characters"
            
        return None
    
    def _summarize_categories(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Summarize the categories found in the results.
        
        Args:
            results: Token categorization results
            
        Returns:
            Summary of categories
        """
        category_counts = {}
        
        for token_result in results.values():
            if "error" in token_result:
                continue
                
            primary_category = token_result.get("primary_category")
            if primary_category:
                category_counts[primary_category] = category_counts.get(primary_category, 0) + 1
        
        return category_counts
    
    @handle_errors
    async def get_category_tokens(self, category: str, limit: int = 50, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get tokens belonging to a specific category.
        
        Note: In a production system, this would query a database of pre-categorized tokens.
        In this simplified version, we'll query some active tokens and categorize them on the fly.
        
        Args:
            category: The category to filter by
            limit: Maximum number of tokens to return
            request_id: Optional request ID for tracing
            
        Returns:
            Tokens in the specified category
        """
        log_with_context(
            logger,
            "info",
            f"Category tokens requested for: {category}",
            request_id=request_id,
            category=category
        )
        
        # Check if the category is valid
        if category not in self.categories and category != "all":
            return {
                "error": f"Invalid category: {category}",
                "valid_categories": list(self.categories.keys())
            }
            
        # Query some active tokens
        # In production, this would query a database
        # For simplicity, we'll use a small sample of known tokens
        sample_tokens = [
            # Some well-known SPL tokens
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
            "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",  # mSOL
            "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # stSOL
            "So11111111111111111111111111111111111111112",  # Wrapped SOL
            "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",  # RAY
            "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",  # ORCA
            "7i5KKsX2weiTkry7jA4ZwSuXGhs5eJBEjY8vVxR4pfRx",  # GMT
            "AFbX8oGjGpmVFywbVouvhQSRmiW2aR1mohfahi4Y2AdB",  # GST
            "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac",  # MNGO
            "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            "kinXdEcpDQeHPEuQnqmUgtYykqKGVFq6CeVX5iAHJq6",  # KIN
            "ATLASXmbPQxBUYbxPsV97usA3fPQYEqzQBUHgiFCUsXx",  # ATLAS
            "PoLisWXnNRwC6oBu1vHiuKQzFjGL4XDSu4g9qjz9qVk",  # POLIS
            "SHDWyBxihqiCj6YekG2GUr7wqKLeLAMK1gHZck9pL6y",  # SHDW
        ]
        
        # Categorize each token
        results = []
        count = 0
        
        for mint in sample_tokens:
            try:
                token_result = await self.categorize_token(mint, request_id)
                
                # Filter by category if specified
                if category != "all":
                    if token_result.get("primary_category") == category or category in token_result.get("secondary_categories", []):
                        results.append(token_result)
                        count += 1
                else:
                    results.append(token_result)
                    count += 1
                    
                # Stop if we've reached the limit
                if count >= limit:
                    break
            except Exception as e:
                logger.error(f"Error categorizing token {mint}: {str(e)}")
                continue
        
        # Sort by primary category score
        sorted_results = sorted(
            results,
            key=lambda x: x.get("category_scores", {}).get(x.get("primary_category", ""), 0),
            reverse=True
        )
        
        log_with_context(
            logger,
            "info",
            f"Found {len(sorted_results)} tokens in category: {category}",
            request_id=request_id,
            category=category,
            token_count=len(sorted_results)
        )
        
        return {
            "category": category,
            "tokens": sorted_results,
            "count": len(sorted_results)
        }
    
    @handle_errors
    async def extend_categories(self, new_categories: Dict[str, List[str]], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Extend the category definitions with new categories or keywords.
        
        Args:
            new_categories: Dictionary of category names to lists of keywords
            request_id: Optional request ID for tracing
            
        Returns:
            Updated categories
        """
        log_with_context(
            logger,
            "info",
            f"Extending categories with {len(new_categories)} new or updated categories",
            request_id=request_id,
            category_count=len(new_categories)
        )
        
        for category, keywords in new_categories.items():
            # Filter out empty or invalid keywords
            valid_keywords = [k.lower() for k in keywords if k and isinstance(k, str) and len(k.strip()) > 0]
            
            if category in self.categories:
                # Add new keywords to existing category
                existing_keywords = set(self.categories[category])
                new_keywords = set(valid_keywords) - existing_keywords
                
                self.categories[category].extend(list(new_keywords))
                
                log_with_context(
                    logger,
                    "info",
                    f"Added {len(new_keywords)} new keywords to category: {category}",
                    request_id=request_id,
                    category=category,
                    keyword_count=len(new_keywords)
                )
            else:
                # Create new category
                self.categories[category] = valid_keywords
                
                log_with_context(
                    logger,
                    "info",
                    f"Created new category: {category} with {len(valid_keywords)} keywords",
                    request_id=request_id,
                    category=category,
                    keyword_count=len(valid_keywords)
                )
        
        return {
            "categories": self.categories,
            "category_count": len(self.categories)
        } 