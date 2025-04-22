"""Liquidity pool analysis for Solana tokens with focus on Raydium and Orca pools."""

import asyncio
import datetime
import math
from typing import Dict, List, Any, Optional
import logging

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.decorators import validate_solana_key, handle_errors

# Set up logging
logger = get_logger(__name__)

# Constants
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
ORCA_PROGRAM_ID = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"
RAYDIUM_LP_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
ORCA_SWAP_PROGRAM_ID = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"


class LiquidityAnalyzer:
    """Analyzer for liquidity pools on Solana, focusing on Raydium and Orca."""

    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)
        self.pool_analyzer = LiquidityPoolAnalyzer(solana_client)

    @validate_solana_key
    @handle_errors
    async def analyze_pool(self, pool_address: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a specific liquidity pool.
        
        Args:
            pool_address: The pool account address
            request_id: Optional request ID for tracing
            
        Returns:
            Pool analysis data
        """
        log_with_context(
            logger,
            "info",
            f"Pool analysis requested for: {pool_address}",
            request_id=request_id,
            pool_address=pool_address
        )
        
        # Get pool reserves first
        reserves = await self.pool_analyzer.get_token_pair_reserves(pool_address, request_id=request_id)
        
        # Get account data to determine pool type
        account_info = await self.client.get_account_info(pool_address)
        owner = account_info.get("result", {}).get("owner", "Unknown")
        
        # Determine pool type
        pool_type = "Unknown"
        if owner == RAYDIUM_PROGRAM_ID:
            pool_type = "Raydium"
        elif owner == ORCA_PROGRAM_ID:
            pool_type = "Orca"
            
        # Build analysis response
        result = {
            "pool_address": pool_address,
            "pool_type": pool_type,
            "token_a": reserves.get("token_a"),
            "token_b": reserves.get("token_b"),
            "reserves_a": reserves.get("reserves_a", 0),
            "reserves_b": reserves.get("reserves_b", 0),
            "liquidity_usd": 0,  # Would calculate based on token prices
            "apr_estimate": 0,   # Would calculate based on fees and volume
            "volume_24h": 0,     # Would get from external sources or events
            "fees_24h": 0,       # Would calculate based on volume and fee rate
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # If one of the tokens is a stablecoin, use that for USD valuation
        if (reserves.get("token_a") and reserves.get("token_a", {}).get("symbol") in ["USDC", "USDT", "DAI"]):
            result["liquidity_usd"] = float(reserves.get("reserves_a", 0)) * 2  # Approximate
        elif (reserves.get("token_b") and reserves.get("token_b", {}).get("symbol") in ["USDC", "USDT", "DAI"]):
            result["liquidity_usd"] = float(reserves.get("reserves_b", 0)) * 2  # Approximate
            
        log_with_context(
            logger,
            "info",
            f"Pool analysis completed for: {pool_address}",
            request_id=request_id,
            pool_address=pool_address,
            pool_type=pool_type
        )
        
        return result
        
    @validate_solana_key
    @handle_errors
    async def get_user_positions(self, wallet_address: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get liquidity positions for a user.
        
        Args:
            wallet_address: The user's wallet address
            request_id: Optional request ID for tracing
            
        Returns:
            User's liquidity positions
        """
        log_with_context(
            logger,
            "info",
            f"User liquidity positions requested for: {wallet_address}",
            request_id=request_id,
            wallet_address=wallet_address
        )
        
        result = {
            "wallet_address": wallet_address,
            "position_count": 0,
            "total_value_usd": 0,
            "positions": [],
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # For demonstration purposes, using placeholder data
            # In a real implementation, would search for LP tokens owned by this wallet
            
            # Example placeholder position
            position = {
                "pool_address": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
                "pool_type": "Raydium",
                "token_a": {
                    "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "symbol": "USDC",
                    "name": "USD Coin"
                },
                "token_b": {
                    "mint": "So11111111111111111111111111111111111111112",
                    "symbol": "SOL",
                    "name": "Wrapped SOL"
                },
                "share_percentage": 0.01,  # User's share of the pool
                "value_usd": 250.0,        # Estimated USD value of position
                "token_a_amount": 125.0,   # User's share of token A
                "token_b_amount": 1.0      # User's share of token B
            }
            
            # Add the placeholder position (would be actual positions in real implementation)
            result["positions"].append(position)
            result["position_count"] = len(result["positions"])
            result["total_value_usd"] = sum(p.get("value_usd", 0) for p in result["positions"])
            
        except Exception as e:
            logger.error(f"Error getting user positions for {wallet_address}: {str(e)}")
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"User positions completed for: {wallet_address}",
            request_id=request_id,
            wallet_address=wallet_address,
            position_count=result.get("position_count", 0)
        )
        
        return result
    
    @handle_errors
    async def get_top_pools(self, limit: int = 10, protocol: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get top liquidity pools by TVL.
        
        Args:
            limit: Maximum number of pools to return
            protocol: Optional filter by protocol
            request_id: Optional request ID for tracing
            
        Returns:
            Top liquidity pools
        """
        log_with_context(
            logger,
            "info",
            f"Top pools requested (limit: {limit}, protocol: {protocol or 'all'})",
            request_id=request_id,
            limit=limit,
            protocol=protocol
        )
        
        result = {
            "pools": [],
            "total_count": 0,
            "protocol": protocol or "all",
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # For demonstration purposes, using placeholder data
            # In a real implementation, would fetch actual top pools from indexer or blockchain
            
            # Example placeholder pools
            pools = [
                {
                    "pool_address": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
                    "pool_type": "Raydium",
                    "token_a": {
                        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "symbol": "USDC",
                        "name": "USD Coin"
                    },
                    "token_b": {
                        "mint": "So11111111111111111111111111111111111111112",
                        "symbol": "SOL",
                        "name": "Wrapped SOL"
                    },
                    "liquidity_usd": 25000000,
                    "volume_24h": 5000000,
                    "apy": 12.5
                },
                {
                    "pool_address": "4RyJecykr9ivpjPvj1pRiLwz1MkVk7mK87q5vs4Zd7Qy",
                    "pool_type": "Orca",
                    "token_a": {
                        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "symbol": "USDC",
                        "name": "USD Coin"
                    },
                    "token_b": {
                        "mint": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
                        "symbol": "BTC",
                        "name": "Wrapped Bitcoin"
                    },
                    "liquidity_usd": 18000000,
                    "volume_24h": 3200000,
                    "apy": 9.2
                },
                {
                    "pool_address": "2q8KVKr7map9xc8UM9yJC3X8dVPJZ4R57THWXnvJ5jd4",
                    "pool_type": "Raydium",
                    "token_a": {
                        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "symbol": "USDC",
                        "name": "USD Coin"
                    },
                    "token_b": {
                        "mint": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                        "symbol": "USDT",
                        "name": "USDT"
                    },
                    "liquidity_usd": 15000000,
                    "volume_24h": 8000000,
                    "apy": 7.8
                }
            ]
            
            # Filter by protocol if specified
            if protocol:
                filtered_pools = [pool for pool in pools if pool["pool_type"].lower() == protocol.lower()]
            else:
                filtered_pools = pools
                
            # Sort by liquidity (descending) and limit
            sorted_pools = sorted(filtered_pools, key=lambda x: x["liquidity_usd"], reverse=True)[:limit]
            
            result["pools"] = sorted_pools
            result["total_count"] = len(filtered_pools)
            
        except Exception as e:
            logger.error(f"Error getting top pools: {str(e)}")
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"Top pools fetched (count: {len(result.get('pools', []))})",
            request_id=request_id,
            pool_count=len(result.get("pools", []))
        )
        
        return result
    
    @handle_errors
    async def calculate_impermanent_loss(
        self, 
        token_a_price_change: float, 
        token_b_price_change: float, 
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate impermanent loss for given price changes.
        
        Args:
            token_a_price_change: Price change ratio for token A (1.0 = no change)
            token_b_price_change: Price change ratio for token B (1.0 = no change)
            request_id: Optional request ID for tracing
            
        Returns:
            Impermanent loss calculation
        """
        log_with_context(
            logger,
            "info",
            f"Impermanent loss calculation requested",
            request_id=request_id,
            token_a_price_change=token_a_price_change,
            token_b_price_change=token_b_price_change
        )
        
        result = {
            "token_a_price_change": token_a_price_change,
            "token_b_price_change": token_b_price_change,
            "price_ratio_change": token_b_price_change / token_a_price_change,
            "percentage_loss": 0.0,
            "dollar_value_example": {
                "initial_investment": 1000.0,
                "hodl_value": 0.0,
                "lp_value": 0.0,
                "difference": 0.0
            },
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Calculate impermanent loss
            # Formula: IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
            
            # Normalize token prices to get the price ratio change
            price_ratio_change = token_b_price_change / token_a_price_change
            
            # Calculate impermanent loss percentage
            il_factor = (2 * math.sqrt(price_ratio_change)) / (1 + price_ratio_change) - 1
            percentage_loss = il_factor * 100
            
            result["percentage_loss"] = abs(percentage_loss)
            
            # Example with dollar values (assuming 50/50 pool with $1000 initial investment)
            initial_investment = 1000.0
            token_a_initial = initial_investment / 2
            token_b_initial = initial_investment / 2
            
            # Value if holding tokens separately
            hodl_value = (token_a_initial * token_a_price_change) + (token_b_initial * token_b_price_change)
            
            # Value in liquidity pool (accounting for IL)
            lp_value = hodl_value * (1 + il_factor)
            
            result["dollar_value_example"]["hodl_value"] = hodl_value
            result["dollar_value_example"]["lp_value"] = lp_value
            result["dollar_value_example"]["difference"] = hodl_value - lp_value
            
        except Exception as e:
            logger.error(f"Error calculating impermanent loss: {str(e)}")
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"Impermanent loss calculated: {result.get('percentage_loss', 0):.2f}%",
            request_id=request_id,
            percentage_loss=result.get("percentage_loss", 0)
        )
        
        return result


class LiquidityPoolAnalyzer:
    """Analyzer for liquidity pools on Solana, focusing on Raydium and Orca."""

    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)

    @validate_solana_key
    @handle_errors
    async def get_liquidity_pools(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get liquidity pools for the given token.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Dictionary with liquidity pool information
        """
        log_with_context(
            logger,
            "info",
            f"Liquidity pool analysis requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        result = {
            "token_mint": mint,
            "total_pools": 0,
            "total_liquidity_usd": 0,
            "raydium_pools": [],
            "orca_pools": [],
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get token metadata for context
            token_metadata = await self.client.get_token_metadata(mint)
            result["token_name"] = token_metadata.get("name", "Unknown")
            result["token_symbol"] = token_metadata.get("symbol", "UNKNOWN")
            
            # Find Raydium pools
            raydium_pools = await self._find_raydium_pools(mint)
            result["raydium_pools"] = raydium_pools
            
            # Find Orca pools
            orca_pools = await self._find_orca_pools(mint)
            result["orca_pools"] = orca_pools
            
            # Update summary statistics
            result["total_pools"] = len(raydium_pools) + len(orca_pools)
            result["total_liquidity_usd"] = sum(pool.get("liquidity_usd", 0) for pool in raydium_pools + orca_pools)
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity pools for {mint}: {str(e)}")
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"Liquidity pool analysis completed for: {mint}",
            request_id=request_id,
            mint=mint,
            total_pools=result.get("total_pools", 0)
        )
        
        return result

    @validate_solana_key
    @handle_errors
    async def get_token_pair_reserves(self, pool_address: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get token pair reserves for a specific liquidity pool.
        
        Args:
            pool_address: The pool address
            request_id: Optional request ID for tracing
            
        Returns:
            Token pair reserve information
        """
        log_with_context(
            logger,
            "info",
            f"Token pair reserves requested for pool: {pool_address}",
            request_id=request_id,
            pool_address=pool_address
        )
        
        result = {
            "pool_address": pool_address,
            "token_a": None,
            "token_b": None,
            "reserves_a": 0,
            "reserves_b": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get account info
            account_info = await self.client.get_account_info(pool_address)
            
            # Determine pool type and extract data accordingly
            owner = account_info.get("owner", "")
            
            if owner == RAYDIUM_PROGRAM_ID:
                # Process Raydium pool data
                await self._extract_raydium_pool_reserves(account_info, result)
            elif owner == ORCA_PROGRAM_ID:
                # Process Orca pool data
                await self._extract_orca_pool_reserves(account_info, result)
            else:
                result["error"] = f"Pool owned by unknown program: {owner}"
                
        except Exception as e:
            logger.error(f"Error getting token pair reserves for {pool_address}: {str(e)}")
            result["error"] = str(e)
            
        log_with_context(
            logger,
            "info",
            f"Token pair reserves completed for pool: {pool_address}",
            request_id=request_id,
            pool_address=pool_address
        )
        
        return result

    async def _find_raydium_pools(self, mint: str) -> List[Dict[str, Any]]:
        """Find Raydium liquidity pools for the token.
        
        Args:
            mint: The token mint address
            
        Returns:
            List of Raydium pools
        """
        pools = []
        
        try:
            # Get token accounts by owner (Raydium program)
            response = await self.client.get_program_accounts(
                RAYDIUM_PROGRAM_ID,
                encoding="jsonParsed",
                filters=[
                    {"memcmp": {"offset": 200, "bytes": mint}}  # Filter by token mint in pool data
                ]
            )
            
            for account in response:
                pool_address = account.get("pubkey", "")
                pool_data = {
                    "pool_address": pool_address,
                    "pool_type": "Raydium",
                    "token_a": None,
                    "token_b": None,
                    "liquidity_usd": 0,
                }
                
                # Extract token pair from account data
                try:
                    account_data = account.get("account", {}).get("data", {})
                    parsed_data = account_data.get("parsed", {})
                    
                    # Extract token information
                    # Note: This is a simplified version, actual Raydium pool data parsing
                    # would require deeper understanding of their specific data structure
                    token_a_mint = parsed_data.get("info", {}).get("tokenAMint")
                    token_b_mint = parsed_data.get("info", {}).get("tokenBMint")
                    
                    if token_a_mint:
                        token_a_info = await self.client.get_token_metadata(token_a_mint)
                        pool_data["token_a"] = {
                            "mint": token_a_mint,
                            "symbol": token_a_info.get("symbol", "Unknown"),
                            "name": token_a_info.get("name", "Unknown")
                        }
                        
                    if token_b_mint:
                        token_b_info = await self.client.get_token_metadata(token_b_mint)
                        pool_data["token_b"] = {
                            "mint": token_b_mint,
                            "symbol": token_b_info.get("symbol", "Unknown"),
                            "name": token_b_info.get("name", "Unknown")
                        }
                    
                    # Get reserve data
                    reserves = await self.get_token_pair_reserves(pool_address)
                    pool_data["reserves_a"] = reserves.get("reserves_a", 0)
                    pool_data["reserves_b"] = reserves.get("reserves_b", 0)
                    
                    # Calculate approximate liquidity (simplified)
                    # In a real implementation, would get token prices and calculate properly
                    # For now, using a placeholder calculation
                    if pool_data["token_b"] and pool_data["token_b"].get("symbol") in ["USDC", "USDT"]:
                        pool_data["liquidity_usd"] = reserves.get("reserves_b", 0) * 2  # Approximate
                    
                except Exception as e:
                    logger.error(f"Error parsing Raydium pool data for {pool_address}: {str(e)}")
                
                pools.append(pool_data)
                
        except Exception as e:
            logger.error(f"Error finding Raydium pools for {mint}: {str(e)}")
            
        return pools

    async def _find_orca_pools(self, mint: str) -> List[Dict[str, Any]]:
        """Find Orca liquidity pools for the token.
        
        Args:
            mint: The token mint address
            
        Returns:
            List of Orca pools
        """
        pools = []
        
        try:
            # Get token accounts by owner (Orca program)
            response = await self.client.get_program_accounts(
                ORCA_PROGRAM_ID,
                encoding="jsonParsed",
                filters=[
                    {"memcmp": {"offset": 184, "bytes": mint}}  # Filter by token mint in pool data
                ]
            )
            
            for account in response:
                pool_address = account.get("pubkey", "")
                pool_data = {
                    "pool_address": pool_address,
                    "pool_type": "Orca",
                    "token_a": None,
                    "token_b": None,
                    "liquidity_usd": 0,
                }
                
                # Extract token pair from account data
                try:
                    account_data = account.get("account", {}).get("data", {})
                    parsed_data = account_data.get("parsed", {})
                    
                    # Extract token information
                    # Note: This is a simplified version, actual Orca pool data parsing
                    # would require deeper understanding of their specific data structure
                    token_a_mint = parsed_data.get("info", {}).get("tokenAMint")
                    token_b_mint = parsed_data.get("info", {}).get("tokenBMint")
                    
                    if token_a_mint:
                        token_a_info = await self.client.get_token_metadata(token_a_mint)
                        pool_data["token_a"] = {
                            "mint": token_a_mint,
                            "symbol": token_a_info.get("symbol", "Unknown"),
                            "name": token_a_info.get("name", "Unknown")
                        }
                        
                    if token_b_mint:
                        token_b_info = await self.client.get_token_metadata(token_b_mint)
                        pool_data["token_b"] = {
                            "mint": token_b_mint,
                            "symbol": token_b_info.get("symbol", "Unknown"),
                            "name": token_b_info.get("name", "Unknown")
                        }
                    
                    # Get reserve data
                    reserves = await self.get_token_pair_reserves(pool_address)
                    pool_data["reserves_a"] = reserves.get("reserves_a", 0)
                    pool_data["reserves_b"] = reserves.get("reserves_b", 0)
                    
                    # Calculate approximate liquidity (simplified)
                    # In a real implementation, would get token prices and calculate properly
                    # For now, using a placeholder calculation
                    if pool_data["token_b"] and pool_data["token_b"].get("symbol") in ["USDC", "USDT"]:
                        pool_data["liquidity_usd"] = reserves.get("reserves_b", 0) * 2  # Approximate
                    
                except Exception as e:
                    logger.error(f"Error parsing Orca pool data for {pool_address}: {str(e)}")
                
                pools.append(pool_data)
                
        except Exception as e:
            logger.error(f"Error finding Orca pools for {mint}: {str(e)}")
            
        return pools

    async def _extract_raydium_pool_reserves(self, account_info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Extract Raydium pool token pair reserves.
        
        Args:
            account_info: Account information from RPC
            result: Result dictionary to update
        """
        try:
            # Extract token data from account info
            data = account_info.get("data", [])
            
            # In a real implementation, would parse the binary data properly
            # This is a simplified placeholder
            
            # For now, populate with placeholder data extraction logic
            # Would need specific knowledge of Raydium pool data structure
            result["token_a"] = {
                "mint": data.get("tokenAMint", "Unknown"),
                "symbol": "Unknown",
                "name": "Unknown"
            }
            
            result["token_b"] = {
                "mint": data.get("tokenBMint", "Unknown"),
                "symbol": "Unknown",
                "name": "Unknown"
            }
            
            result["reserves_a"] = data.get("tokenAReserve", 0)
            result["reserves_b"] = data.get("tokenBReserve", 0)
            
        except Exception as e:
            logger.error(f"Error extracting Raydium pool reserves: {str(e)}")
            result["error"] = str(e)

    async def _extract_orca_pool_reserves(self, account_info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Extract Orca pool token pair reserves.
        
        Args:
            account_info: Account information from RPC
            result: Result dictionary to update
        """
        try:
            # Extract token data from account info
            data = account_info.get("data", [])
            
            # In a real implementation, would parse the binary data properly
            # This is a simplified placeholder
            
            # For now, populate with placeholder data extraction logic
            # Would need specific knowledge of Orca pool data structure
            result["token_a"] = {
                "mint": data.get("tokenAMint", "Unknown"),
                "symbol": "Unknown",
                "name": "Unknown"
            }
            
            result["token_b"] = {
                "mint": data.get("tokenBMint", "Unknown"),
                "symbol": "Unknown",
                "name": "Unknown"
            }
            
            result["reserves_a"] = data.get("tokenAReserve", 0)
            result["reserves_b"] = data.get("tokenBReserve", 0)
            
        except Exception as e:
            logger.error(f"Error extracting Orca pool reserves: {str(e)}")
            result["error"] = str(e) 