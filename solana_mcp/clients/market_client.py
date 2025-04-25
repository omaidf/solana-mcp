"""Market-related Solana RPC client operations.

This module provides specialized client functionality for Solana market and price data.
"""

import base64
import datetime
from typing import Dict, List, Any, Optional, Union

import base58

from solana_mcp.clients.base_client import BaseSolanaClient, InvalidPublicKeyError, validate_public_key
from solana_mcp.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

# Jupiter Aggregator Program ID for DEX data
JUPITER_PROGRAM_ID = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"

# Raydium Program ID
RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Orca Program ID
ORCA_PROGRAM_ID = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"

# Common token mints for price references
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
SOL_MINT = "So11111111111111111111111111111111111111112"  # Native SOL wrapped mint

class MarketClient(BaseSolanaClient):
    """Client for Solana market and price operations."""
    
    async def get_market_price(self, token_mint: str) -> Dict[str, Any]:
        """Get market price data for a token.
        
        This method fetches price information for a token from available DEX liquidity.
        It uses various sources to determine the token price in SOL and USD.
        
        Args:
            token_mint: The mint address of the token
            
        Returns:
            Price information including:
            - price_sol: Price in SOL
            - price_usd: Estimated price in USD (if SOL price is available)
            - liquidity: Estimated liquidity information
            - source: Data source used for the price
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        result = {
            "mint": token_mint,
            "price_data": {
                "price_sol": None,
                "price_usd": None,
                "liquidity": None,
                "source": None,
                "last_updated": datetime.datetime.now().isoformat()
            }
        }
        
        try:
            # Fetch SOL/USD price first to convert SOL prices to USD
            sol_usd_price = await self._get_sol_usd_price()
            
            # Now find pools containing our token
            token_pools = await self._find_token_pools(token_mint)
            
            # Calculate prices and liquidity from pools
            token_pools = await self._calculate_pool_prices(token_pools, token_mint, sol_usd_price)
            
            # Select the best pool based on liquidity
            if token_pools:
                # Sort pools by liquidity (highest first)
                sorted_pools = sorted(token_pools, key=lambda x: x.get("liquidity_usd", 0), reverse=True)
                best_pool = sorted_pools[0]
                
                result["price_data"] = {
                    "price_sol": best_pool.get("price_sol", 0),
                    "price_usd": best_pool.get("price_usd", 0),
                    "liquidity": {
                        "total_usd": best_pool.get("liquidity_usd", 0),
                        "best_pool_protocol": best_pool.get("protocol", "unknown"),
                        "best_pool_pair": best_pool.get("pair", "unknown")
                    },
                    "source": f"{best_pool.get('protocol', 'dex')}",
                    "last_updated": datetime.datetime.now().isoformat()
                }
            else:
                # If no pools found, return empty result
                logger.warning(f"No liquidity pools found for token {token_mint}")
                
        except Exception as e:
            # Log and return error information
            logger.error(f"Error fetching price data for {token_mint}: {str(e)}", exc_info=True)
            result["error"] = str(e)
            
        return result
    
    async def _get_sol_usd_price(self) -> float:
        """Get the SOL/USD price from available DEX pools.
        
        Returns:
            SOL/USD price or default value (100) if not found
        """
        # Try Raydium pools for SOL/USDC price
        sol_usdc_pools = await self._make_request(
            "getProgramAccounts",
            [
                RAYDIUM_PROGRAM_ID,
                {
                    "encoding": "base64",
                    "filters": [
                        {"memcmp": {"offset": 200, "bytes": SOL_MINT}},
                        {"memcmp": {"offset": 232, "bytes": USDC_MINT}}
                    ]
                }
            ]
        )
        
        if sol_usdc_pools and len(sol_usdc_pools) > 0:
            # Get the largest pool by reserves
            largest_pool = None
            max_reserves = 0
            
            for pool in sol_usdc_pools:
                if "account" in pool and "data" in pool["account"]:
                    data = pool["account"]["data"]
                    if isinstance(data, list) and data[1] == "base64":
                        decoded_data = base64.b64decode(data[0])
                        
                        # Extract reserves data (simplified, actual layout may vary)
                        sol_reserves = int.from_bytes(decoded_data[264:272], byteorder="little")
                        usdc_reserves = int.from_bytes(decoded_data[296:304], byteorder="little")
                        
                        if sol_reserves > max_reserves:
                            max_reserves = sol_reserves
                            largest_pool = {
                                "sol_reserves": sol_reserves,
                                "usdc_reserves": usdc_reserves
                            }
            
            if largest_pool:
                # Calculate SOL/USD price from pool data
                # USDC has 6 decimals, SOL has 9 decimals
                sol_usd_price = (largest_pool["usdc_reserves"] / 10**6) / (largest_pool["sol_reserves"] / 10**9)
                return sol_usd_price
        
        # If we couldn't get SOL/USD price, use a default value
        # In a production system, you'd use a proper price oracle
        return 100.0  # Default fallback price
    
    async def _find_token_pools(self, token_mint: str) -> List[Dict[str, Any]]:
        """Find liquidity pools containing the specified token.
        
        Args:
            token_mint: The token mint address
            
        Returns:
            List of liquidity pool information
        """
        token_pools = []
        
        # Check Raydium pools
        raydium_pools = await self._make_request(
            "getProgramAccounts",
            [
                RAYDIUM_PROGRAM_ID,
                {
                    "encoding": "base64",
                    "filters": [
                        {"memcmp": {"offset": 200, "bytes": token_mint}}
                    ]
                }
            ]
        )
        
        if raydium_pools:
            for pool in raydium_pools:
                if "account" in pool and "data" in pool["account"]:
                    data = pool["account"]["data"]
                    if isinstance(data, list) and data[1] == "base64":
                        try:
                            decoded_data = base64.b64decode(data[0])
                            
                            # Extract pool data
                            token_a_mint_offset = 200
                            token_b_mint_offset = 232
                            token_a_reserves_offset = 264
                            token_b_reserves_offset = 296
                            
                            # Validate that the data is long enough
                            if len(decoded_data) < token_b_reserves_offset + 8:
                                continue
                            
                            # Extract mint addresses (32 bytes each)
                            token_a_mint_bytes = decoded_data[token_a_mint_offset:token_a_mint_offset+32]
                            token_b_mint_bytes = decoded_data[token_b_mint_offset:token_b_mint_offset+32]
                            
                            # Convert to base58
                            token_a_mint = base58.encode(bytes(token_a_mint_bytes))
                            token_b_mint = base58.encode(bytes(token_b_mint_bytes))
                            
                            # Extract reserve data (8 bytes each)
                            token_a_reserves = int.from_bytes(
                                decoded_data[token_a_reserves_offset:token_a_reserves_offset+8], 
                                byteorder="little"
                            )
                            token_b_reserves = int.from_bytes(
                                decoded_data[token_b_reserves_offset:token_b_reserves_offset+8], 
                                byteorder="little"
                            )
                            
                            token_pools.append({
                                "protocol": "raydium",
                                "pair": f"{token_a_mint}-{token_b_mint}",
                                "token_a_mint": token_a_mint,
                                "token_b_mint": token_b_mint,
                                "token_a_reserves": token_a_reserves,
                                "token_b_reserves": token_b_reserves,
                                "pool_address": pool.get("pubkey", "unknown")
                            })
                        except Exception as e:
                            logger.warning(f"Error decoding Raydium pool data: {str(e)}")
        
        # Similar approach for Orca pools - to be implemented
        
        return token_pools
    
    async def _calculate_pool_prices(
        self, 
        pools: List[Dict[str, Any]], 
        token_mint: str, 
        sol_usd_price: float
    ) -> List[Dict[str, Any]]:
        """Calculate price and liquidity data for each pool.
        
        Args:
            pools: List of pool information
            token_mint: The token mint address
            sol_usd_price: Current SOL/USD price
            
        Returns:
            List of pools with added price data
        """
        result_pools = []
        
        for pool in pools:
            try:
                # Determine which token is our target token
                is_token_a = pool["token_a_mint"] == token_mint
                target_token = "a" if is_token_a else "b"
                paired_token = "b" if is_token_a else "a"
                
                paired_token_mint = pool[f"token_{paired_token}_mint"]
                
                # Get decimal information for both tokens
                # In a real implementation, this would use caching to avoid repeated calls
                token_decimal_info = await self._get_token_decimals(token_mint)
                token_decimals = token_decimal_info.get("decimals", 9)
                
                paired_decimal_info = await self._get_token_decimals(paired_token_mint)
                paired_decimals = paired_decimal_info.get("decimals", 9)
                
                # Calculate price based on the paired token
                token_reserves = pool[f"token_{target_token}_reserves"] / (10 ** token_decimals)
                paired_reserves = pool[f"token_{paired_token}_reserves"] / (10 ** paired_decimals)
                
                if token_reserves == 0:
                    continue  # Skip pools with no reserves
                
                # Price is how much of paired token you get for 1 of the target token
                price_in_paired = paired_reserves / token_reserves
                
                # Calculate SOL and USD prices
                price_sol = 0
                price_usd = 0
                
                if paired_token_mint == SOL_MINT:
                    price_sol = price_in_paired
                    price_usd = price_sol * sol_usd_price
                elif paired_token_mint == USDC_MINT or paired_token_mint == USDT_MINT:
                    price_usd = price_in_paired
                    price_sol = price_usd / sol_usd_price if sol_usd_price > 0 else 0
                else:
                    # If paired with another token, we'd need to find that token's price first
                    # For simplicity, we'll skip these pools for now
                    continue
                
                # Calculate liquidity
                liquidity_usd = 0
                if price_usd > 0:
                    token_liquidity = token_reserves * price_usd
                    paired_liquidity = 0
                    
                    if paired_token_mint == USDC_MINT or paired_token_mint == USDT_MINT:
                        paired_liquidity = paired_reserves
                    elif paired_token_mint == SOL_MINT and sol_usd_price > 0:
                        paired_liquidity = paired_reserves * sol_usd_price
                    
                    liquidity_usd = token_liquidity + paired_liquidity
                
                # Add price data to pool info
                result_pools.append({
                    **pool,
                    "price_sol": price_sol,
                    "price_usd": price_usd,
                    "liquidity_usd": liquidity_usd
                })
                
            except Exception as e:
                logger.warning(f"Error calculating pool prices: {str(e)}")
        
        return result_pools
    
    async def _get_token_decimals(self, token_mint: str) -> Dict[str, Any]:
        """Get token decimal information.
        
        Args:
            token_mint: The token mint address
            
        Returns:
            Dict with token decimals
        """
        # Common token decimals
        known_decimals = {
            SOL_MINT: 9,
            USDC_MINT: 6,
            USDT_MINT: 6
        }
        
        if token_mint in known_decimals:
            return {"decimals": known_decimals[token_mint]}
        
        try:
            # Get token supply to find decimals
            response = await self._make_request("getTokenSupply", [token_mint])
            
            if response and isinstance(response, dict) and "value" in response:
                decimals = int(response["value"].get("decimals", 0))
                return {"decimals": decimals}
            
        except Exception as e:
            logger.warning(f"Error getting token decimals: {str(e)}")
        
        # Default to 9 decimals if we can't determine
        return {"decimals": 9}
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get an overview of current market conditions.
        
        Returns:
            Market overview including top tokens
        """
        result = {
            "sol_price_usd": 0,
            "top_tokens": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Get SOL price
            sol_usd_price = await self._get_sol_usd_price()
            result["sol_price_usd"] = sol_usd_price
            
            # In a real implementation, you would get data for top tokens
            # For simplicity, we'll include a few hardcoded major tokens
            top_token_mints = [
                USDC_MINT,
                USDT_MINT,
                "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",  # mSOL
                "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # ETH (Wormhole)
                "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj"   # stSOL
            ]
            
            # Get data for each token
            for mint in top_token_mints:
                try:
                    price_data = await self.get_market_price(mint)
                    if "price_data" in price_data and price_data["price_data"]["price_usd"]:
                        result["top_tokens"].append({
                            "mint": mint,
                            "price_usd": price_data["price_data"]["price_usd"],
                            "price_sol": price_data["price_data"]["price_sol"],
                            "liquidity_usd": price_data["price_data"]["liquidity"].get("total_usd", 0) if price_data["price_data"]["liquidity"] else 0,
                        })
                except Exception as e:
                    logger.warning(f"Error getting price for {mint}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            result["error"] = str(e)
        
        return result 