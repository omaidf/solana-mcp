"""Market service for Solana MCP.

This module provides services for working with Solana token market data.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from solana_mcp.services.base_service import BaseService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.decorators import validate_solana_key

class MarketService(BaseService):
    """Service for working with Solana token market data."""
    
    def __init__(self, solana_client: SolanaClient, cache_service: Optional[CacheService] = None):
        """Initialize the market service.
        
        Args:
            solana_client: The Solana client to use
            cache_service: Optional cache service
        """
        super().__init__()
        self.client = solana_client
        self.cache = cache_service
    
    @validate_solana_key
    async def get_token_price(self, mint: str) -> Dict[str, Any]:
        """Get token price information.
        
        Args:
            mint: The token mint address
            
        Returns:
            Token price information
        """
        self.log_with_context(
            "info", 
            f"Getting price for token {mint}"
        )
        
        # Use cache if available
        if self.cache:
            cached_price = self.cache.get(f"token_price:{mint}")
            if cached_price:
                return cached_price
        
        # Get price data - implementation depends on price data source
        price_data = await self.execute_with_fallback(
            self.client.get_market_price(mint),
            fallback_value=None,
            error_message=f"Error fetching price for token {mint}"
        )
        
        # Format the response
        if price_data:
            result = {
                "mint": mint,
                "price_usd": price_data.get("price_usd"),
                "price_sol": price_data.get("price_sol"),
                "liquidity_usd": price_data.get("liquidity_usd"),
                "market_cap_usd": price_data.get("market_cap"),
                "volume_24h_usd": price_data.get("volume_24h"),
                "change_24h_percent": price_data.get("price_change_24h"),
                "source": price_data.get("source", "Unknown"),
                "last_updated": datetime.now().isoformat()
            }
            
            # Cache the result
            if self.cache:
                # Price data should have a shorter TTL since it changes frequently
                self.cache.set(f"token_price:{mint}", result, ttl=60)  # 1 minute TTL
                
            return result
        else:
            return {
                "mint": mint,
                "price_usd": None,
                "price_sol": None,
                "error": "Price data not available"
            }
    
    @validate_solana_key
    async def get_historical_prices(
        self, 
        mint: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Get historical price data for a token.
        
        Args:
            mint: The token mint address
            days: Number of days of historical data to retrieve
            
        Returns:
            Historical price data
        """
        self.log_with_context(
            "info", 
            f"Getting historical prices for token {mint}",
            days=days
        )
        
        # Use cache if available
        cache_key = f"token_history:{mint}:{days}"
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        # Get historical price data - implementation depends on data source
        historical_data = await self.execute_with_fallback(
            self.client.get_historical_prices(mint, days),
            fallback_value=[],
            error_message=f"Error fetching historical prices for token {mint}"
        )
        
        # Format the response
        result = {
            "mint": mint,
            "days": days,
            "prices": historical_data,
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache the result
        if self.cache:
            # Historical data can be cached longer
            self.cache.set(cache_key, result, ttl=3600)  # 1 hour TTL
        
        return result
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get an overview of the Solana token market.
        
        Returns:
            Market overview data
        """
        self.log_with_context("info", "Getting market overview")
        
        # Use cache if available
        if self.cache:
            cached_overview = self.cache.get("market_overview")
            if cached_overview:
                return cached_overview
        
        # Get market data - implementation depends on data source
        market_data = await self.execute_with_fallback(
            self.client.get_market_overview(),
            fallback_value={},
            error_message="Error fetching market overview"
        )
        
        # Format the response
        result = {
            "sol_price_usd": market_data.get("sol_price_usd"),
            "sol_market_cap_usd": market_data.get("sol_market_cap"),
            "sol_volume_24h_usd": market_data.get("sol_volume_24h"),
            "sol_change_24h_percent": market_data.get("sol_price_change_24h"),
            "total_value_locked_usd": market_data.get("tvl"),
            "top_tokens": market_data.get("top_tokens", []),
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache the result
        if self.cache:
            self.cache.set("market_overview", result, ttl=300)  # 5 minute TTL
        
        return result
    
    @validate_solana_key
    async def get_token_liquidity(self, mint: str) -> Dict[str, Any]:
        """Get liquidity information for a token.
        
        Args:
            mint: The token mint address
            
        Returns:
            Token liquidity information
        """
        self.log_with_context(
            "info", 
            f"Getting liquidity for token {mint}"
        )
        
        # Use cache if available
        if self.cache:
            cached_liquidity = self.cache.get(f"token_liquidity:{mint}")
            if cached_liquidity:
                return cached_liquidity
        
        # Get liquidity data - implementation depends on data source
        liquidity_data = await self.execute_with_fallback(
            self.client.get_token_liquidity(mint),
            fallback_value={},
            error_message=f"Error fetching liquidity for token {mint}"
        )
        
        # Format the response
        result = {
            "mint": mint,
            "liquidity_usd": liquidity_data.get("liquidity_usd"),
            "liquidity_sol": liquidity_data.get("liquidity_sol"),
            "pools": liquidity_data.get("pools", []),
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache the result
        if self.cache:
            self.cache.set(f"token_liquidity:{mint}", result, ttl=300)  # 5 minute TTL
        
        return result
    
    async def get_trending_tokens(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending tokens.
        
        Args:
            limit: Maximum number of tokens to return
            
        Returns:
            List of trending tokens
        """
        self.log_with_context(
            "info", 
            f"Getting {limit} trending tokens"
        )
        
        # Use cache if available
        cache_key = f"trending_tokens:{limit}"
        if self.cache:
            cached_tokens = self.cache.get(cache_key)
            if cached_tokens:
                return cached_tokens
        
        # Get trending tokens - implementation depends on data source
        trending_data = await self.execute_with_fallback(
            self.client.get_trending_tokens(limit),
            fallback_value=[],
            error_message="Error fetching trending tokens"
        )
        
        # Cache the result
        if self.cache:
            self.cache.set(cache_key, trending_data, ttl=600)  # 10 minute TTL
        
        return trending_data 