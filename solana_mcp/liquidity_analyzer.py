"""Liquidity pool analysis for Solana tokens with focus on Raydium and Orca pools."""

import asyncio
import datetime
import math
from typing import Dict, List, Any, Optional
import logging
import base64
import base58

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
            # Get all token accounts owned by this wallet
            token_accounts = await self.client.get_token_accounts_by_owner(wallet_address)
            
            # No token accounts found
            if not token_accounts or "value" not in token_accounts:
                return result
                
            # List of LP token accounts
            lp_positions = []
            
            # Process each token account to identify LP tokens
            for account in token_accounts.get("value", []):
                # Skip accounts with no data
                if "data" not in account["account"] or "parsed" not in account["account"]["data"]:
                    continue
                    
                account_data = account["account"]["data"]["parsed"]["info"]
                
                # Skip accounts with zero balance
                if float(account_data.get("tokenAmount", {}).get("uiAmount", 0)) <= 0:
                    continue
                    
                # Get the mint address
                mint = account_data.get("mint")
                if not mint:
                    continue
                
                # Check if this is an LP token by examining related pools
                # This is a simplified approach - in a real implementation,
                # you would have a database or lookup mechanism to map LP token mints to pools
                
                # Try to get metadata to check if it's an LP token
                metadata = await self.client.get_token_metadata(mint)
                token_name = metadata.get("metadata", {}).get("name", "").lower()
                token_symbol = metadata.get("metadata", {}).get("symbol", "").lower()
                
                # Check if name or symbol suggests it's an LP token
                is_lp_token = any(keyword in token_name or keyword in token_symbol 
                                 for keyword in ["lp", "pool", "liquidity", "swap"])
                
                if not is_lp_token:
                    continue
                
                # Find the associated pool
                # For this implementation, we'll search for Raydium and Orca pools directly
                
                # First check Raydium pools with this LP token
                raydium_pool = await self._find_pool_by_lp_token(mint, "raydium")
                
                # Then check Orca pools
                orca_pool = await self._find_pool_by_lp_token(mint, "orca") if not raydium_pool else None
                
                pool_address = None
                pool_type = None
                
                if raydium_pool:
                    pool_address = raydium_pool.get("pool_address")
                    pool_type = "Raydium"
                elif orca_pool:
                    pool_address = orca_pool.get("pool_address")
                    pool_type = "Orca"
                
                # If we found a pool, analyze it
                if pool_address:
                    # Get pool data
                    pool_data = await self.analyze_pool(pool_address)
                    
                    # User's token amount
                    lp_token_amount = float(account_data.get("tokenAmount", {}).get("uiAmount", 0))
                    
                    # For a real implementation, you would get the total LP token supply
                    # and calculate the user's percentage of the pool
                    # This is a simplified calculation
                    total_lp_supply = await self._get_lp_token_supply(mint)
                    share_percentage = (lp_token_amount / total_lp_supply) * 100 if total_lp_supply > 0 else 0
                    
                    # Calculate value of position using pool reserves and token prices
                    value_usd = 0
                    
                    # If one of the tokens is a stablecoin, use it to estimate position value
                    if pool_data.get("token_a", {}).get("symbol") in ["USDC", "USDT", "DAI"]:
                        reserves_a = float(pool_data.get("reserves_a", 0))
                        value_usd = (reserves_a * 2 * share_percentage / 100) if reserves_a > 0 else 0
                    elif pool_data.get("token_b", {}).get("symbol") in ["USDC", "USDT", "DAI"]:
                        reserves_b = float(pool_data.get("reserves_b", 0))
                        value_usd = (reserves_b * 2 * share_percentage / 100) if reserves_b > 0 else 0
                    else:
                        # If no stablecoin, try to get token prices
                        token_a_mint = pool_data.get("token_a", {}).get("mint")
                        token_a_price = 0
                        
                        if token_a_mint:
                            price_data = await self.client.get_market_price(token_a_mint)
                            token_a_price = price_data.get("price_data", {}).get("price_usd", 0)
                            
                        if token_a_price > 0:
                            reserves_a = float(pool_data.get("reserves_a", 0))
                            value_usd = (reserves_a * token_a_price * 2 * share_percentage / 100)
                    
                    # Create position data
                    position = {
                        "pool_address": pool_address,
                        "pool_type": pool_type,
                        "token_a": pool_data.get("token_a", {}),
                        "token_b": pool_data.get("token_b", {}),
                        "lp_token": {
                            "mint": mint,
                            "amount": lp_token_amount,
                            "name": token_name,
                            "symbol": token_symbol
                        },
                        "share_percentage": share_percentage,
                        "value_usd": value_usd,
                        "token_a_amount": (float(pool_data.get("reserves_a", 0)) * share_percentage / 100) 
                                          if share_percentage > 0 else 0,
                        "token_b_amount": (float(pool_data.get("reserves_b", 0)) * share_percentage / 100)
                                          if share_percentage > 0 else 0
                    }
                    
                    lp_positions.append(position)
            
            # Update the result with positions
            result["positions"] = lp_positions
            result["position_count"] = len(lp_positions)
            result["total_value_usd"] = sum(position.get("value_usd", 0) for position in lp_positions)
            
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
            # Get pools from Raydium and/or Orca
            raydium_pools = []
            orca_pools = []
            
            # Get Raydium pools if no protocol filter or protocol is "raydium"
            if not protocol or protocol.lower() == "raydium":
                raydium_pools = await self._get_raydium_pools(limit=limit)
                
            # Get Orca pools if no protocol filter or protocol is "orca"
            if not protocol or protocol.lower() == "orca":
                orca_pools = await self._get_orca_pools(limit=limit)
                
            # Combine and sort pools by TVL
            all_pools = raydium_pools + orca_pools
            sorted_pools = sorted(all_pools, key=lambda p: p.get("liquidity_usd", 0), reverse=True)
            
            # Limit results
            result["pools"] = sorted_pools[:limit]
            result["total_count"] = len(all_pools)
            result["total_tvl"] = sum(pool.get("liquidity_usd", 0) for pool in result["pools"])
            
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

    async def _find_pool_by_lp_token(self, lp_token_mint: str, protocol: str = None) -> Optional[Dict[str, Any]]:
        """Find a liquidity pool that uses the given LP token.
        
        Args:
            lp_token_mint: The LP token mint address
            protocol: Protocol to search (raydium, orca, or None for both)
            
        Returns:
            Pool data if found, None otherwise
        """
        if protocol and protocol.lower() not in ["raydium", "orca"]:
            return None
            
        # Determine which program ID to query based on protocol
        program_ids = []
        if not protocol or protocol.lower() == "raydium":
            program_ids.append(RAYDIUM_PROGRAM_ID)
        if not protocol or protocol.lower() == "orca":
            program_ids.append(ORCA_PROGRAM_ID)
            
        for program_id in program_ids:
            try:
                # Try to find pool accounts that reference this LP token
                # There are several approaches to find this association:
                
                # 1. Some pools store their LP token mint directly in their account data
                # Filter by searching for the LP token mint address in the account data
                response = await self.client.get_program_accounts(
                    program_id,
                    filters=[
                        {"memcmp": {"offset": 0, "bytes": lp_token_mint}}
                    ],
                    limit=5  # Limit to a few results
                )
                
                # Check if we found any matching pools
                if response and len(response) > 0:
                    pool_address = response[0].get("pubkey")
                    current_protocol = "raydium" if program_id == RAYDIUM_PROGRAM_ID else "orca"
                    return {
                        "pool_address": pool_address,
                        "lp_token_mint": lp_token_mint,
                        "protocol": current_protocol
                    }
                
                # 2. Alternative approach: Look at the LP token's mint authority
                # Many pools are the mint authority for their LP tokens
                token_info = await self.client.get_account_info(lp_token_mint)
                if token_info and "result" in token_info and token_info["result"]:
                    data = token_info["result"].get("data")
                    if isinstance(data, list) and data[0] == "base64":
                        # Decode base64 data
                        import base64
                        data_bytes = base64.b64decode(data[1])
                        
                        # Check if there's a mint authority
                        mint_authority_option = data_bytes[0]
                        if mint_authority_option == 1:
                            # Extract mint authority pubkey
                            import base58
                            mint_authority = base58.b58encode(data_bytes[1:33]).decode('utf-8')
                            
                            # Check if this mint authority is a program account
                            authority_info = await self.client.get_account_info(mint_authority)
                            if authority_info and "result" in authority_info:
                                owner = authority_info["result"].get("owner")
                                if owner == program_id:
                                    # This pool is likely the mint authority for the LP token
                                    current_protocol = "raydium" if program_id == RAYDIUM_PROGRAM_ID else "orca"
                                    return {
                                        "pool_address": mint_authority,
                                        "lp_token_mint": lp_token_mint,
                                        "protocol": current_protocol
                                    }
                
            except Exception as e:
                logger.error(f"Error searching for pool with LP token {lp_token_mint} in {program_id}: {str(e)}")
                
        # If we get here, we couldn't find a matching pool
        return None
            
    async def _get_lp_token_supply(self, lp_token_mint: str) -> float:
        """Get the total supply of an LP token.
        
        Args:
            lp_token_mint: The LP token mint address
            
        Returns:
            Total supply as a float
        """
        try:
            supply_info = await self.client.get_token_supply(lp_token_mint)
            if "value" in supply_info:
                return float(supply_info["value"]["uiAmountString"])
            return 0
        except Exception as e:
            logger.error(f"Error getting LP token supply for {lp_token_mint}: {str(e)}")
            return 0

    async def _get_raydium_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get Raydium liquidity pools.
        
        Args:
            limit: Maximum number of pools to return
            
        Returns:
            List of Raydium pools
        """
        pools = []
        
        try:
            # Query Raydium program accounts to find pools
            program_accounts = await self.client.get_program_accounts(
                RAYDIUM_PROGRAM_ID,
                limit=limit * 2  # Get more than needed to account for filtering
            )
            
            # Process up to 'limit' accounts
            for i, account in enumerate(program_accounts):
                if i >= limit * 2:
                    break
                    
                try:
                    # Get account address
                    pool_address = account.get("pubkey", "Unknown")
                    
                    # Skip if no address
                    if pool_address == "Unknown":
                        continue
                        
                    # Analyze the pool
                    pool_data = await self.analyze_pool(pool_address)
                    
                    # Skip pools with errors or without tokens
                    if "error" in pool_data or not pool_data.get("token_a") or not pool_data.get("token_b"):
                        continue
                        
                    # Format pool data for response
                    formatted_pool = {
                        "pool_address": pool_address,
                        "pool_type": "Raydium",
                        "token_a": pool_data.get("token_a", {}),
                        "token_b": pool_data.get("token_b", {}),
                        "liquidity_usd": pool_data.get("liquidity_usd", 0),
                        "volume_24h": pool_data.get("volume_24h", 0),
                        "apy": pool_data.get("apr_estimate", 0)
                    }
                    
                    pools.append(formatted_pool)
                    
                    # Stop if we have enough pools
                    if len(pools) >= limit:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing Raydium pool {account.get('pubkey', 'Unknown')}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching Raydium pools: {str(e)}")
            
        return pools
        
    async def _get_orca_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get Orca liquidity pools.
        
        Args:
            limit: Maximum number of pools to return
            
        Returns:
            List of Orca pools
        """
        pools = []
        
        try:
            # Query Orca program accounts to find pools
            program_accounts = await self.client.get_program_accounts(
                ORCA_PROGRAM_ID,
                limit=limit * 2  # Get more than needed to account for filtering
            )
            
            # Process up to 'limit' accounts
            for i, account in enumerate(program_accounts):
                if i >= limit * 2:
                    break
                    
                try:
                    # Get account address
                    pool_address = account.get("pubkey", "Unknown")
                    
                    # Skip if no address
                    if pool_address == "Unknown":
                        continue
                        
                    # Analyze the pool
                    pool_data = await self.analyze_pool(pool_address)
                    
                    # Skip pools with errors or without tokens
                    if "error" in pool_data or not pool_data.get("token_a") or not pool_data.get("token_b"):
                        continue
                        
                    # Format pool data for response
                    formatted_pool = {
                        "pool_address": pool_address,
                        "pool_type": "Orca",
                        "token_a": pool_data.get("token_a", {}),
                        "token_b": pool_data.get("token_b", {}),
                        "liquidity_usd": pool_data.get("liquidity_usd", 0),
                        "volume_24h": pool_data.get("volume_24h", 0),
                        "apy": pool_data.get("apr_estimate", 0)
                    }
                    
                    pools.append(formatted_pool)
                    
                    # Stop if we have enough pools
                    if len(pools) >= limit:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing Orca pool {account.get('pubkey', 'Unknown')}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching Orca pools: {str(e)}")
            
        return pools


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
                
                # Extract reserves and token info
                reserves = await self.get_token_pair_reserves(pool_address)
                
                # Get token metadata
                token_a_mint = reserves.get("token_a", {}).get("mint")
                token_b_mint = reserves.get("token_b", {}).get("mint")
                
                if token_a_mint:
                    pool_data["token_a"] = reserves.get("token_a")
                    
                if token_b_mint:
                    pool_data["token_b"] = reserves.get("token_b")
                
                pool_data["reserves_a"] = reserves.get("reserves_a", 0)
                pool_data["reserves_b"] = reserves.get("reserves_b", 0)
                
                # Calculate liquidity using actual prices
                if token_a_mint and token_b_mint and pool_data["reserves_a"] and pool_data["reserves_b"]:
                    # Reference stablecoin mints
                    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                    USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    # Get token decimals
                    token_a_supply = await self.client.get_token_supply(token_a_mint)
                    token_a_decimals = token_a_supply.get("value", {}).get("decimals", 9)
                    
                    token_b_supply = await self.client.get_token_supply(token_b_mint)
                    token_b_decimals = token_b_supply.get("value", {}).get("decimals", 9)
                    
                    # Calculate USD value based on token types
                    if token_a_mint == USDC_MINT or token_a_mint == USDT_MINT:
                        # Token A is a stablecoin
                        token_a_usd_value = pool_data["reserves_a"] / (10 ** token_a_decimals)
                        token_b_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                        
                        if token_b_amount > 0:
                            token_b_price = token_a_usd_value / token_b_amount
                            token_b_usd_value = token_b_amount * token_b_price
                            pool_data["liquidity_usd"] = token_a_usd_value + token_b_usd_value
                            
                    elif token_b_mint == USDC_MINT or token_b_mint == USDT_MINT:
                        # Token B is a stablecoin
                        token_b_usd_value = pool_data["reserves_b"] / (10 ** token_b_decimals)
                        token_a_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                        
                        if token_a_amount > 0:
                            token_a_price = token_b_usd_value / token_a_amount
                            token_a_usd_value = token_a_amount * token_a_price
                            pool_data["liquidity_usd"] = token_a_usd_value + token_b_usd_value
                            
                    elif token_a_mint == SOL_MINT or token_b_mint == SOL_MINT:
                        # One token is SOL, get SOL price from a SOL/USDC pool
                        sol_price = await self._get_sol_price()
                        
                        if sol_price > 0:
                            if token_a_mint == SOL_MINT:
                                # Token A is SOL
                                sol_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                                sol_value = sol_amount * sol_price
                                
                                token_b_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                                token_b_price = sol_value / token_b_amount if token_b_amount > 0 else 0
                                token_b_value = token_b_amount * token_b_price
                                
                                pool_data["liquidity_usd"] = sol_value + token_b_value
                            else:
                                # Token B is SOL
                                sol_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                                sol_value = sol_amount * sol_price
                                
                                token_a_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                                token_a_price = sol_value / token_a_amount if token_a_amount > 0 else 0
                                token_a_value = token_a_amount * token_a_price
                                
                                pool_data["liquidity_usd"] = sol_value + token_a_value
                    else:
                        # Neither token is a stablecoin or SOL
                        # Try to get price data for both tokens
                        token_a_price_data = await self.client.get_market_price(token_a_mint)
                        token_a_price = token_a_price_data.get("price_data", {}).get("price_usd", 0)
                        
                        token_b_price_data = await self.client.get_market_price(token_b_mint)
                        token_b_price = token_b_price_data.get("price_data", {}).get("price_usd", 0)
                        
                        token_a_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                        token_b_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                        
                        token_a_value = token_a_amount * token_a_price
                        token_b_value = token_b_amount * token_b_price
                        
                        pool_data["liquidity_usd"] = token_a_value + token_b_value
                    
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
                
                # Extract reserves and token info
                reserves = await self.get_token_pair_reserves(pool_address)
                
                # Get token metadata
                token_a_mint = reserves.get("token_a", {}).get("mint")
                token_b_mint = reserves.get("token_b", {}).get("mint")
                
                if token_a_mint:
                    pool_data["token_a"] = reserves.get("token_a")
                    
                if token_b_mint:
                    pool_data["token_b"] = reserves.get("token_b")
                
                pool_data["reserves_a"] = reserves.get("reserves_a", 0)
                pool_data["reserves_b"] = reserves.get("reserves_b", 0)
                
                # Calculate liquidity using actual prices
                if token_a_mint and token_b_mint and pool_data["reserves_a"] and pool_data["reserves_b"]:
                    # Reference stablecoin mints
                    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                    USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    # Get token decimals
                    token_a_supply = await self.client.get_token_supply(token_a_mint)
                    token_a_decimals = token_a_supply.get("value", {}).get("decimals", 9)
                    
                    token_b_supply = await self.client.get_token_supply(token_b_mint)
                    token_b_decimals = token_b_supply.get("value", {}).get("decimals", 9)
                    
                    # Calculate USD value based on token types
                    if token_a_mint == USDC_MINT or token_a_mint == USDT_MINT:
                        # Token A is a stablecoin
                        token_a_usd_value = pool_data["reserves_a"] / (10 ** token_a_decimals)
                        token_b_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                        
                        if token_b_amount > 0:
                            token_b_price = token_a_usd_value / token_b_amount
                            token_b_usd_value = token_b_amount * token_b_price
                            pool_data["liquidity_usd"] = token_a_usd_value + token_b_usd_value
                            
                    elif token_b_mint == USDC_MINT or token_b_mint == USDT_MINT:
                        # Token B is a stablecoin
                        token_b_usd_value = pool_data["reserves_b"] / (10 ** token_b_decimals)
                        token_a_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                        
                        if token_a_amount > 0:
                            token_a_price = token_b_usd_value / token_a_amount
                            token_a_usd_value = token_a_amount * token_a_price
                            pool_data["liquidity_usd"] = token_a_usd_value + token_b_usd_value
                            
                    elif token_a_mint == SOL_MINT or token_b_mint == SOL_MINT:
                        # One token is SOL, get SOL price from a SOL/USDC pool
                        sol_price = await self._get_sol_price()
                        
                        if sol_price > 0:
                            if token_a_mint == SOL_MINT:
                                # Token A is SOL
                                sol_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                                sol_value = sol_amount * sol_price
                                
                                token_b_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                                token_b_price = sol_value / token_b_amount if token_b_amount > 0 else 0
                                token_b_value = token_b_amount * token_b_price
                                
                                pool_data["liquidity_usd"] = sol_value + token_b_value
                            else:
                                # Token B is SOL
                                sol_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                                sol_value = sol_amount * sol_price
                                
                                token_a_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                                token_a_price = sol_value / token_a_amount if token_a_amount > 0 else 0
                                token_a_value = token_a_amount * token_a_price
                                
                                pool_data["liquidity_usd"] = sol_value + token_a_value
                    else:
                        # Neither token is a stablecoin or SOL
                        # Try to get price data for both tokens
                        token_a_price_data = await self.client.get_market_price(token_a_mint)
                        token_a_price = token_a_price_data.get("price_data", {}).get("price_usd", 0)
                        
                        token_b_price_data = await self.client.get_market_price(token_b_mint)
                        token_b_price = token_b_price_data.get("price_data", {}).get("price_usd", 0)
                        
                        token_a_amount = pool_data["reserves_a"] / (10 ** token_a_decimals)
                        token_b_amount = pool_data["reserves_b"] / (10 ** token_b_decimals)
                        
                        token_a_value = token_a_amount * token_a_price
                        token_b_value = token_b_amount * token_b_price
                        
                        pool_data["liquidity_usd"] = token_a_value + token_b_value
                
                pools.append(pool_data)
                
        except Exception as e:
            logger.error(f"Error finding Orca pools for {mint}: {str(e)}")
            
        return pools

    async def _get_sol_price(self) -> float:
        """Get the current SOL price in USD from the largest SOL/USDC pool.
        
        Returns:
            SOL price in USD
        """
        try:
            # Reference token mints
            SOL_MINT = "So11111111111111111111111111111111111111112"
            USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            
            # Look for SOL/USDC pools on Raydium
            raydium_pools = await self.client.get_program_accounts(
                RAYDIUM_PROGRAM_ID,
                filters=[
                    {"memcmp": {"offset": 200, "bytes": SOL_MINT}},
                    {"memcmp": {"offset": 232, "bytes": USDC_MINT}}
                ],
                limit=5
            )
            
            largest_pool = None
            max_sol_reserves = 0
            
            # Find the largest pool by SOL reserves
            for pool_data in raydium_pools:
                pool_address = pool_data.get("pubkey", "")
                try:
                    reserves = await self.get_token_pair_reserves(pool_address)
                    
                    # Check which reserve is SOL
                    token_a_mint = reserves.get("token_a", {}).get("mint")
                    token_b_mint = reserves.get("token_b", {}).get("mint")
                    
                    sol_reserves = 0
                    usdc_reserves = 0
                    
                    if token_a_mint == SOL_MINT:
                        sol_reserves = reserves.get("reserves_a", 0)
                        usdc_reserves = reserves.get("reserves_b", 0)
                    elif token_b_mint == SOL_MINT:
                        sol_reserves = reserves.get("reserves_b", 0)
                        usdc_reserves = reserves.get("reserves_a", 0)
                    
                    if sol_reserves > max_sol_reserves:
                        max_sol_reserves = sol_reserves
                        largest_pool = {
                            "sol_reserves": sol_reserves,
                            "usdc_reserves": usdc_reserves
                        }
                except Exception as e:
                    logger.warning(f"Error getting reserves for pool {pool_address}: {str(e)}")
            
            # If we found a pool, calculate the SOL price
            if largest_pool:
                # USDC has 6 decimals, SOL has 9 decimals
                sol_amount = largest_pool["sol_reserves"] / 10**9
                usdc_amount = largest_pool["usdc_reserves"] / 10**6
                
                if sol_amount > 0:
                    return usdc_amount / sol_amount
            
            # If we can't find a Raydium pool, try Orca
            orca_pools = await self.client.get_program_accounts(
                ORCA_PROGRAM_ID,
                filters=[
                    {"memcmp": {"offset": 184, "bytes": SOL_MINT}},
                    {"memcmp": {"offset": 216, "bytes": USDC_MINT}}
                ],
                limit=5
            )
            
            largest_pool = None
            max_sol_reserves = 0
            
            # Find the largest pool by SOL reserves
            for pool_data in orca_pools:
                pool_address = pool_data.get("pubkey", "")
                try:
                    reserves = await self.get_token_pair_reserves(pool_address)
                    
                    # Check which reserve is SOL
                    token_a_mint = reserves.get("token_a", {}).get("mint")
                    token_b_mint = reserves.get("token_b", {}).get("mint")
                    
                    sol_reserves = 0
                    usdc_reserves = 0
                    
                    if token_a_mint == SOL_MINT:
                        sol_reserves = reserves.get("reserves_a", 0)
                        usdc_reserves = reserves.get("reserves_b", 0)
                    elif token_b_mint == SOL_MINT:
                        sol_reserves = reserves.get("reserves_b", 0)
                        usdc_reserves = reserves.get("reserves_a", 0)
                    
                    if sol_reserves > max_sol_reserves:
                        max_sol_reserves = sol_reserves
                        largest_pool = {
                            "sol_reserves": sol_reserves,
                            "usdc_reserves": usdc_reserves
                        }
                except Exception as e:
                    logger.warning(f"Error getting reserves for pool {pool_address}: {str(e)}")
            
            # If we found a pool, calculate the SOL price
            if largest_pool:
                # USDC has 6 decimals, SOL has 9 decimals
                sol_amount = largest_pool["sol_reserves"] / 10**9
                usdc_amount = largest_pool["usdc_reserves"] / 10**6
                
                if sol_amount > 0:
                    return usdc_amount / sol_amount
            
            # If we still can't get a price, use a fallback value
            return 100  # Fallback SOL price in USD
            
        except Exception as e:
            logger.error(f"Error getting SOL price: {str(e)}")
            return 100  # Default fallback SOL price in USD

    async def _extract_raydium_pool_reserves(self, account_info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Extract token pair reserves from Raydium pool account data.
        
        Args:
            account_info: The account info from RPC
            result: The result dictionary to update with extracted data
        """
        try:
            # Check if account data is available
            if not account_info or "result" not in account_info or not account_info["result"]:
                result["error"] = "Account data not available"
                return

            # Get the data bytes
            data = account_info["result"]["data"]
            if isinstance(data, list) and data[0] == "base64":
                data_bytes = base64.b64decode(data[1])
            else:
                result["error"] = "Invalid data format"
                return
                
            # Get the owner to verify it's a Raydium pool
            owner = account_info["result"]["owner"]
            if owner != RAYDIUM_PROGRAM_ID:
                result["error"] = f"Not a Raydium pool (owner: {owner})"
                return
                
            # Raydium pools have different layouts based on version
            # This implementation uses the most common layout for v4 pools
            
            # Extract token A mint (32 bytes)
            # In v4 pools, token mints are at the beginning of the data
            token_a_offset = 8  # After 8-byte discriminator
            token_a_mint = base58.b58encode(data_bytes[token_a_offset:token_a_offset+32]).decode('utf-8')
            
            # Extract token B mint (32 bytes after token A)
            token_b_offset = token_a_offset + 32
            token_b_mint = base58.b58encode(data_bytes[token_b_offset:token_b_offset+32]).decode('utf-8')
            
            # Extract token accounts (where reserves are stored)
            # Token accounts come after the mints
            token_a_account_offset = token_b_offset + 32
            token_a_account = base58.b58encode(data_bytes[token_a_account_offset:token_a_account_offset+32]).decode('utf-8')
            
            token_b_account_offset = token_a_account_offset + 32
            token_b_account = base58.b58encode(data_bytes[token_b_account_offset:token_b_account_offset+32]).decode('utf-8')
            
            # Extract reserves directly from token accounts
            reserves_a = 0
            reserves_b = 0
            
            try:
                # Get token A account info to get reserves
                token_a_account_info = await self.client.get_account_info(token_a_account)
                if token_a_account_info and "result" in token_a_account_info and token_a_account_info["result"]:
                    token_a_data = token_a_account_info["result"]["data"]
                    if isinstance(token_a_data, list) and token_a_data[0] == "base64":
                        token_a_bytes = base64.b64decode(token_a_data[1])
                        # Token amount is at offset 64 (8 bytes, little-endian)
                        reserves_a = int.from_bytes(token_a_bytes[64:72], byteorder='little')
                        
                # Get token B account info to get reserves
                token_b_account_info = await self.client.get_account_info(token_b_account)
                if token_b_account_info and "result" in token_b_account_info and token_b_account_info["result"]:
                    token_b_data = token_b_account_info["result"]["data"]
                    if isinstance(token_b_data, list) and token_b_data[0] == "base64":
                        token_b_bytes = base64.b64decode(token_b_data[1])
                        # Token amount is at offset 64 (8 bytes, little-endian)
                        reserves_b = int.from_bytes(token_b_bytes[64:72], byteorder='little')
            except Exception as e:
                logger.error(f"Error fetching token account reserves: {str(e)}")
                # Fallback to older method - try to extract reserves directly from pool account
                # Note: This is less reliable but can work for some pool versions
                try:
                    # For some pool versions, reserves are stored directly in pool account
                    # Try typical offsets for reserves (these vary by version)
                    reserves_offset = 200  # Common offset in some versions
                    reserves_a = int.from_bytes(data_bytes[reserves_offset:reserves_offset+8], byteorder='little')
                    reserves_b = int.from_bytes(data_bytes[reserves_offset+8:reserves_offset+16], byteorder='little')
                except Exception as inner_e:
                    logger.error(f"Error extracting reserves from pool account: {str(inner_e)}")
            
            # Get token metadata for better context
            token_a_metadata = await self.client.get_token_metadata(token_a_mint)
            token_b_metadata = await self.client.get_token_metadata(token_b_mint)
            
            # Update the result with extracted data
            result["token_a"] = {
                "mint": token_a_mint,
                "symbol": token_a_metadata.get("metadata", {}).get("symbol", "UNKNOWN"),
                "name": token_a_metadata.get("metadata", {}).get("name", "Unknown"),
                "account": token_a_account
            }
            
            result["token_b"] = {
                "mint": token_b_mint,
                "symbol": token_b_metadata.get("metadata", {}).get("symbol", "UNKNOWN"),
                "name": token_b_metadata.get("metadata", {}).get("name", "Unknown"),
                "account": token_b_account
            }
            
            result["reserves_a"] = reserves_a
            result["reserves_b"] = reserves_b
            
        except Exception as e:
            logger.error(f"Error extracting Raydium pool reserves: {str(e)}")
            result["error"] = f"Failed to extract pool reserves: {str(e)}"

    async def _extract_orca_pool_reserves(self, account_info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Extract token pair reserves from Orca pool account data.
        
        Args:
            account_info: The account info from RPC
            result: The result dictionary to update with extracted data
        """
        try:
            # Check if account data is available
            if not account_info or "result" not in account_info or not account_info["result"]:
                result["error"] = "Account data not available"
                return

            # Get the data bytes
            data = account_info["result"]["data"]
            if isinstance(data, list) and data[0] == "base64":
                data_bytes = base64.b64decode(data[1])
            else:
                result["error"] = "Invalid data format"
                return
                
            # Get the owner to verify it's an Orca pool
            owner = account_info["result"]["owner"]
            if owner != ORCA_PROGRAM_ID:
                result["error"] = f"Not an Orca pool (owner: {owner})"
                return
                
            # Orca pool layout is different from Raydium
            # The layout can vary by pool version, but we'll implement a common one
            
            # After the 8-byte discriminator, many Orca pools have the following structure:
            # - Token program ID: 32 bytes
            # - Token A mint: 32 bytes
            # - Token B mint: 32 bytes
            # - LP mint: 32 bytes
            # - ... (various other pool parameters)
            # - Token accounts come much later
            
            # Extract token A mint (32 bytes)
            token_a_offset = 40  # After discriminator (8) and token program ID (32)
            token_a_mint = base58.b58encode(data_bytes[token_a_offset:token_a_offset+32]).decode('utf-8')
            
            # Extract token B mint (32 bytes after token A)
            token_b_offset = token_a_offset + 32
            token_b_mint = base58.b58encode(data_bytes[token_b_offset:token_b_offset+32]).decode('utf-8')
            
            # Extract LP mint (optional, for context)
            lp_mint_offset = token_b_offset + 32
            lp_mint = base58.b58encode(data_bytes[lp_mint_offset:lp_mint_offset+32]).decode('utf-8')
            
            # Extract token A and B accounts
            # In most Orca pools, token accounts come after several other fields
            token_a_account_offset = lp_mint_offset + 96  # Skip LP mint (32) and other fields (~64)
            token_a_account = base58.b58encode(data_bytes[token_a_account_offset:token_a_account_offset+32]).decode('utf-8')
            
            token_b_account_offset = token_a_account_offset + 32
            token_b_account = base58.b58encode(data_bytes[token_b_account_offset:token_b_account_offset+32]).decode('utf-8')
            
            # Extract reserves directly from token accounts
            reserves_a = 0
            reserves_b = 0
            
            try:
                # Get token A account info to get reserves
                token_a_account_info = await self.client.get_account_info(token_a_account)
                if token_a_account_info and "result" in token_a_account_info and token_a_account_info["result"]:
                    token_a_data = token_a_account_info["result"]["data"]
                    if isinstance(token_a_data, list) and token_a_data[0] == "base64":
                        token_a_bytes = base64.b64decode(token_a_data[1])
                        # Token amount is at offset 64 (8 bytes, little-endian)
                        reserves_a = int.from_bytes(token_a_bytes[64:72], byteorder='little')
                        
                # Get token B account info to get reserves
                token_b_account_info = await self.client.get_account_info(token_b_account)
                if token_b_account_info and "result" in token_b_account_info and token_b_account_info["result"]:
                    token_b_data = token_b_account_info["result"]["data"]
                    if isinstance(token_b_data, list) and token_b_data[0] == "base64":
                        token_b_bytes = base64.b64decode(token_b_data[1])
                        # Token amount is at offset 64 (8 bytes, little-endian)
                        reserves_b = int.from_bytes(token_b_bytes[64:72], byteorder='little')
            except Exception as e:
                logger.error(f"Error fetching token account reserves: {str(e)}")
                # Fallback to older method - try to extract reserves directly from pool data
                try:
                    # Some Orca pools have reserves directly in them, around this offset
                    reserves_offset = 240  # Approximate offset for some pools
                    reserves_a = int.from_bytes(data_bytes[reserves_offset:reserves_offset+8], byteorder='little')
                    reserves_b = int.from_bytes(data_bytes[reserves_offset+8:reserves_offset+16], byteorder='little')
                except Exception as inner_e:
                    logger.error(f"Error extracting reserves from pool data: {str(inner_e)}")
            
            # Get token metadata for better context
            token_a_metadata = await self.client.get_token_metadata(token_a_mint)
            token_b_metadata = await self.client.get_token_metadata(token_b_mint)
            
            # Update the result with extracted data
            result["token_a"] = {
                "mint": token_a_mint,
                "symbol": token_a_metadata.get("metadata", {}).get("symbol", "UNKNOWN"),
                "name": token_a_metadata.get("metadata", {}).get("name", "Unknown"),
                "account": token_a_account
            }
            
            result["token_b"] = {
                "mint": token_b_mint,
                "symbol": token_b_metadata.get("metadata", {}).get("symbol", "UNKNOWN"),
                "name": token_b_metadata.get("metadata", {}).get("name", "Unknown"),
                "account": token_b_account
            }
            
            result["lp_mint"] = lp_mint
            result["reserves_a"] = reserves_a
            result["reserves_b"] = reserves_b
            
        except Exception as e:
            logger.error(f"Error extracting Orca pool reserves: {str(e)}")
            result["error"] = f"Failed to extract pool reserves: {str(e)}" 