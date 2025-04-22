"""Liquidity pool analysis for Solana tokens with focus on Raydium and Orca pools."""

import asyncio
import datetime
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