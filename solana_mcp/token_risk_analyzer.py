"""Token risk analysis for Solana tokens with focus on pumpfun tokens."""

import datetime
import math
import re
import base64
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
import logging

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError, SolanaRpcError
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.decorators import validate_solana_key, handle_errors

# Set up logging
logger = get_logger(__name__)

# Constants
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
KNOWN_SAFE_TOKEN_CREATORS = [
    "BPFLoaderUpgradeab1e11111111111111111111111",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
]

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
class TokenRiskProfile:
    """Risk profile for a token."""
    token_mint: str
    name: str
    symbol: str
    supply_risk_score: float  # 0-100 (higher = riskier)
    authority_risk_score: float  # 0-100 (higher = riskier)
    liquidity_risk_score: float  # 0-100 (higher = riskier)
    ownership_risk_score: float  # 0-100 (higher = riskier)
    overall_risk_score: float  # 0-100 (higher = riskier)
    risk_level: str  # Low, Medium, High, Extreme
    flags: List[str]  # List of specific risk flags
    token_category: str  # Animal, Food, Tech, Meme, Other
    created_at: Optional[datetime.datetime] = None
    last_updated: datetime.datetime = datetime.datetime.now()


class TokenRiskAnalyzer:
    """Analyzer for Solana tokens with focus on detecting risky or pumpfun tokens."""

    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)

    @validate_solana_key
    @handle_errors
    async def analyze_token_risks(self, mint: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive risk analysis for a token.
        
        Args:
            mint: The token mint address
            request_id: Optional request ID for tracing
            
        Returns:
            Token risk analysis data
        """
        log_with_context(
            logger,
            "info",
            f"Token risk analysis requested for: {mint}",
            request_id=request_id,
            mint=mint
        )
        
        # Get token metadata
        token_metadata = await self.client.get_token_metadata(mint)
        name = token_metadata.get("name", "Unknown")
        symbol = token_metadata.get("symbol", "UNKNOWN")
        
        # Get token mint info
        mint_info = await self.client.get_account_info(mint)
        
        # Get token supply
        supply_info = await self.client.get_token_supply(mint)
        
        # Get token holders
        holder_data = await self._analyze_holder_distribution(mint)
        
        # Analyze token authorities
        authority_data = await self._analyze_authority_risks(mint, mint_info)
        
        # Analyze liquidity
        liquidity_data = await self._analyze_liquidity_risks(mint)
        
        # Calculate risk scores
        supply_risk = await self._calculate_supply_risk(mint, supply_info)
        authority_risk = authority_data.get("risk_score", 50)
        liquidity_risk = liquidity_data.get("risk_score", 50)
        ownership_risk = await self._calculate_ownership_risk(mint, holder_data)
        
        # Calculate overall risk score (weighted average)
        overall_risk = (
            supply_risk * 0.15 +
            authority_risk * 0.35 +
            liquidity_risk * 0.25 +
            ownership_risk * 0.25
        )
        
        # Determine risk level
        risk_level = "Low"
        if overall_risk >= 75:
            risk_level = "Extreme"
        elif overall_risk >= 50:
            risk_level = "High"
        elif overall_risk >= 25:
            risk_level = "Medium"
        
        # Categorize token
        token_category = self._categorize_token(name, symbol)
        
        # Collect all risk flags
        flags = []
        flags.extend(authority_data.get("flags", []))
        flags.extend(liquidity_data.get("flags", []))
        
        if supply_risk >= 75:
            flags.append("High supply concentration risk")
        if ownership_risk >= 75:
            flags.append("High ownership concentration risk")
        
        # Create token risk profile
        risk_profile = {
            "token_mint": mint,
            "name": name,
            "symbol": symbol,
            "supply_risk_score": round(supply_risk, 2),
            "authority_risk_score": round(authority_risk, 2),
            "liquidity_risk_score": round(liquidity_risk, 2),
            "ownership_risk_score": round(ownership_risk, 2),
            "overall_risk_score": round(overall_risk, 2),
            "risk_level": risk_level,
            "flags": flags,
            "token_category": token_category,
            "creation_analysis": authority_data.get("creation_info", {}),
            "liquidity_analysis": liquidity_data,
            "holder_analysis": holder_data,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        log_with_context(
            logger,
            "info",
            f"Token risk analysis completed for: {mint}, risk level: {risk_level}",
            request_id=request_id,
            mint=mint,
            risk_level=risk_level,
            overall_risk_score=round(overall_risk, 2)
        )
        
        return risk_profile

    async def _analyze_authority_risks(self, mint: str, mint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risks related to token authorities.
        
        Args:
            mint: The token mint address
            mint_info: The token mint account info
            
        Returns:
            Authority risk analysis
        """
        result = {
            "has_mint_authority": False,
            "has_freeze_authority": False,
            "mint_authority": None,
            "freeze_authority": None,
            "creator_address": None,
            "is_verified_creator": False,
            "creation_info": {},
            "risk_score": 50,
            "flags": []
        }
        
        try:
            # Extract mint and freeze authorities from the mint account
            if "data" in mint_info and mint_info["data"]:
                data = mint_info["data"]
                if isinstance(data, list) and len(data) >= 2 and data[1] == "base64":
                    decoded_data = base64.b64decode(data[0])
                    
                    # Parse the SPL Token mint data
                    # Real implementation would use proper layout parsing
                    # For a simplified version, we're checking specific bytes
                    
                    # Check if mint authority is present (not null)
                    # In SPL Token, mint authority is at offset 0 (after header)
                    has_mint_authority = not all(b == 0 for b in decoded_data[4:36])
                    
                    # Freeze authority is after mint authority
                    has_freeze_authority = not all(b == 0 for b in decoded_data[36:68])
                    
                    result["has_mint_authority"] = has_mint_authority
                    result["has_freeze_authority"] = has_freeze_authority
                    
                    # If authorities exist, try to get their addresses
                    if has_mint_authority:
                        # In real implementation, extract the actual authority address
                        # For now, we'll use a placeholder
                        result["mint_authority"] = "MINT_AUTHORITY_ADDRESS"
                        
                    if has_freeze_authority:
                        # In real implementation, extract the actual freeze authority
                        # For now, we'll use a placeholder
                        result["freeze_authority"] = "FREEZE_AUTHORITY_ADDRESS"
                        
            # Try to get the token creator
            try:
                # Get token creation transaction
                signatures = await self.client.get_signatures_for_address(mint, limit=1, before=None)
                
                if signatures and len(signatures) > 0:
                    creation_signature = signatures[-1]["signature"]  # Get the earliest signature
                    creation_tx = await self.client.get_transaction(creation_signature)
                    
                    # Extract creator from transaction
                    if creation_tx and "transaction" in creation_tx:
                        creator = creation_tx["transaction"]["message"]["accountKeys"][0]
                        result["creator_address"] = creator
                        
                        # Check if creator is a known safe address
                        if creator in KNOWN_SAFE_TOKEN_CREATORS:
                            result["is_verified_creator"] = True
                        
                        # Get creation timestamp
                        if "blockTime" in creation_tx:
                            creation_timestamp = creation_tx["blockTime"]
                            creation_date = datetime.datetime.fromtimestamp(creation_timestamp)
                            
                            # Calculate token age
                            now = datetime.datetime.now()
                            age_days = (now - creation_date).days
                            
                            result["creation_info"] = {
                                "creation_date": creation_date.isoformat(),
                                "age_days": age_days
                            }
                            
                            # Age risk factor - new tokens are riskier
                            if age_days < 7:
                                result["flags"].append("Token less than 1 week old")
                            
                            # Add creation block
                            if "slot" in creation_tx:
                                result["creation_info"]["creation_block"] = creation_tx["slot"]
            except Exception as e:
                logger.warning(f"Error getting creation info for {mint}: {str(e)}")
            
            # Calculate authority risk score
            authority_risk = 0
            
            # Having mint authority is a significant risk
            if result["has_mint_authority"]:
                authority_risk += 50
                result["flags"].append("Mint authority present (supply can be increased)")
                
            # Having freeze authority is a moderate risk
            if result["has_freeze_authority"]:
                authority_risk += 30
                result["flags"].append("Freeze authority present (accounts can be frozen)")
                
            # Unverified creator is a risk
            if not result["is_verified_creator"]:
                authority_risk += 20
                
            # New token is a risk factor
            if "creation_info" in result and "age_days" in result["creation_info"]:
                age_days = result["creation_info"]["age_days"]
                if age_days < 1:
                    authority_risk += 30
                    result["flags"].append("Token created less than 24 hours ago")
                elif age_days < 7:
                    authority_risk += 20
                elif age_days < 30:
                    authority_risk += 10
                    
            # Cap the risk score at 100
            result["risk_score"] = min(100, authority_risk)
                
        except Exception as e:
            logger.error(f"Error analyzing authority risks for {mint}: {str(e)}")
            result["flags"].append("Error analyzing authority risks")
            
        return result

    async def _analyze_liquidity_risks(self, mint: str) -> Dict[str, Any]:
        """Analyze liquidity risks for a token.
        
        Args:
            mint: The token mint address
            
        Returns:
            Liquidity risk analysis
        """
        result = {
            "total_liquidity_usd": 0,
            "largest_pool": None,
            "liquidity_to_mcap_ratio": 0,
            "has_locked_liquidity": False,
            "lock_details": [],
            "risk_score": 50,
            "flags": []
        }
        
        try:
            # Get token market data
            price_data = await self.client.get_market_price(mint)
            price = price_data.get("price", 0)
            
            # Get token supply
            supply_info = await self.client.get_token_supply(mint)
            total_supply = float(supply_info.get("value", {}).get("uiAmountString", "0"))
            
            # Calculate market cap
            market_cap = price * total_supply
            
            # Get liquidity pools for the token
            # In a real implementation, we would scan major DEXes (Raydium, Orca, etc.)
            # For simplicity, we'll use a placeholder with reasonable values
            
            # Search for the token in Raydium and Orca pools
            from solana_mcp.liquidity_analyzer import LiquidityAnalyzer, RAYDIUM_LP_PROGRAM_ID, ORCA_SWAP_PROGRAM_ID
            
            liquidity_analyzer = LiquidityAnalyzer(self.client)
            
            # Get total liquidity in USD
            total_liquidity = 0
            largest_pool = None
            max_pool_liquidity = 0
            
            # Check Raydium pools
            try:
                raydium_pools = await self.client.get_program_accounts(
                    RAYDIUM_LP_PROGRAM_ID,
                    filters=[{"dataSize": 380}],  # Typical size for Raydium pools
                    limit=20  # Keep the limit reasonable
                )
                
                for pool_data in raydium_pools:
                    pool_address = pool_data.get("pubkey")
                    
                    # Analyze the pool
                    try:
                        pool_analysis = await liquidity_analyzer.analyze_pool(pool_address)
                        
                        # Check if this pool contains our token
                        pool_tokens = [
                            pool_analysis.get("pool_data", {}).get("tokenA", {}).get("mint"),
                            pool_analysis.get("pool_data", {}).get("tokenB", {}).get("mint")
                        ]
                        
                        if mint in pool_tokens:
                            # This pool contains our token, add its liquidity
                            pool_liquidity = pool_analysis.get("metrics", {}).get("liquidity_usd", 0)
                            total_liquidity += pool_liquidity
                            
                            # Check if this is the largest pool
                            if pool_liquidity > max_pool_liquidity:
                                max_pool_liquidity = pool_liquidity
                                largest_pool = {
                                    "address": pool_address,
                                    "protocol": "raydium",
                                    "liquidity_usd": pool_liquidity,
                                    "pair": f"{pool_analysis.get('pool_data', {}).get('tokenA', {}).get('symbol', 'UNKNOWN')}-{pool_analysis.get('pool_data', {}).get('tokenB', {}).get('symbol', 'UNKNOWN')}"
                                }
                    except Exception as e:
                        logger.warning(f"Error analyzing Raydium pool {pool_address}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error fetching Raydium pools: {str(e)}")
            
            # Check Orca pools
            try:
                orca_pools = await self.client.get_program_accounts(
                    ORCA_SWAP_PROGRAM_ID,
                    filters=[{"dataSize": 324}],  # Typical size for Orca pools
                    limit=20  # Keep the limit reasonable
                )
                
                for pool_data in orca_pools:
                    pool_address = pool_data.get("pubkey")
                    
                    # Analyze the pool
                    try:
                        pool_analysis = await liquidity_analyzer.analyze_pool(pool_address)
                        
                        # Check if this pool contains our token
                        pool_tokens = [
                            pool_analysis.get("pool_data", {}).get("tokenA", {}).get("mint"),
                            pool_analysis.get("pool_data", {}).get("tokenB", {}).get("mint")
                        ]
                        
                        if mint in pool_tokens:
                            # This pool contains our token, add its liquidity
                            pool_liquidity = pool_analysis.get("metrics", {}).get("liquidity_usd", 0)
                            total_liquidity += pool_liquidity
                            
                            # Check if this is the largest pool
                            if pool_liquidity > max_pool_liquidity:
                                max_pool_liquidity = pool_liquidity
                                largest_pool = {
                                    "address": pool_address,
                                    "protocol": "orca",
                                    "liquidity_usd": pool_liquidity,
                                    "pair": f"{pool_analysis.get('pool_data', {}).get('tokenA', {}).get('symbol', 'UNKNOWN')}-{pool_analysis.get('pool_data', {}).get('tokenB', {}).get('symbol', 'UNKNOWN')}"
                                }
                    except Exception as e:
                        logger.warning(f"Error analyzing Orca pool {pool_address}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error fetching Orca pools: {str(e)}")
            
            result["total_liquidity_usd"] = total_liquidity
            result["largest_pool"] = largest_pool
            
            # Calculate liquidity to market cap ratio
            # A healthy token should have a significant portion of its market cap in liquidity
            if market_cap > 0:
                result["liquidity_to_mcap_ratio"] = total_liquidity / market_cap
            
            # Check for liquidity locks
            # In a real implementation, we would look for known liquidity locker contracts
            # For now, we'll use a placeholder method to simulate lock detection
            
            # Simulate liquidity lock detection
            # In practice, this would involve checking token vesting programs
            # or timelock contracts that hold LP tokens
            has_locked_liquidity = total_liquidity > 50000  # Just a placeholder condition
            lock_percentage = 0.75 if has_locked_liquidity else 0
            
            result["has_locked_liquidity"] = has_locked_liquidity
            if has_locked_liquidity:
                result["lock_details"] = [
                    {
                        "lock_contract": "SIMULATED_LOCK_CONTRACT",
                        "locked_amount_usd": total_liquidity * lock_percentage,
                        "lock_end_date": (datetime.datetime.now() + datetime.timedelta(days=180)).isoformat(),
                        "lock_percentage": lock_percentage * 100
                    }
                ]
            
            # Calculate liquidity risk score
            liquidity_risk = 0
            
            # Low liquidity is a risk
            if total_liquidity < 10000:
                liquidity_risk += 70
                result["flags"].append("Extremely low liquidity (< $10,000)")
            elif total_liquidity < 50000:
                liquidity_risk += 50
                result["flags"].append("Very low liquidity (< $50,000)")
            elif total_liquidity < 100000:
                liquidity_risk += 30
                result["flags"].append("Low liquidity (< $100,000)")
                
            # Low liquidity to market cap ratio is a risk
            if result["liquidity_to_mcap_ratio"] < 0.03:
                liquidity_risk += 50
                result["flags"].append("Extremely low liquidity to market cap ratio (< 3%)")
            elif result["liquidity_to_mcap_ratio"] < 0.1:
                liquidity_risk += 30
                result["flags"].append("Low liquidity to market cap ratio (< 10%)")
                
            # No locked liquidity is a risk
            if not has_locked_liquidity:
                liquidity_risk += 30
                result["flags"].append("No detected liquidity locks")
                
            # Cap the risk score at 100
            result["risk_score"] = min(100, liquidity_risk)
                
        except Exception as e:
            logger.error(f"Error analyzing liquidity risks for {mint}: {str(e)}")
            result["flags"].append("Error analyzing liquidity risks")
            result["risk_score"] = 75  # Default to high risk on error
            
        return result

    async def _analyze_holder_distribution(self, mint: str) -> Dict[str, Any]:
        """Analyze token holder distribution.
        
        Args:
            mint: The token mint address
            
        Returns:
            Holder distribution analysis
        """
        result = {
            "total_holders": 0,
            "top_10_percentage": 0,
            "top_holder_percentage": 0,
            "concentration_index": 0,  # Gini coefficient-like measure of inequality
            "top_holders": []
        }
        
        try:
            # Get largest token holders
            # For a real implementation, you would need to scan all token accounts
            # This is a simplified approach for demonstration
            
            # Get token accounts for this mint
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
                        # Sort holder amounts (ascending)
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
            
        return result

    async def _calculate_supply_risk(self, mint: str, supply_info: Dict[str, Any]) -> float:
        """Calculate supply-related risk score.
        
        Args:
            mint: The token mint address
            supply_info: Token supply information
            
        Returns:
            Supply risk score (0-100, higher = riskier)
        """
        supply_risk = 50  # Default medium risk
        
        try:
            # Extract total supply
            total_supply = float(supply_info.get("value", {}).get("uiAmountString", "0"))
            
            # Extremely large or small supplies can be risky
            if total_supply > 1e15:
                supply_risk += 25  # Extremely large supply
            elif total_supply < 1000:
                supply_risk += 20  # Very small supply
            
            # Adjust based on decimals (very low decimals can indicate potential manipulation)
            decimals = supply_info.get("value", {}).get("decimals", 9)
            if decimals < 6:
                supply_risk += 15
            
            # Cap the risk score at 100
            supply_risk = min(100, supply_risk)
            
        except Exception as e:
            logger.error(f"Error calculating supply risk for {mint}: {str(e)}")
            
        return supply_risk

    async def _calculate_ownership_risk(self, mint: str, holder_data: Dict[str, Any]) -> float:
        """Calculate ownership concentration risk score.
        
        Args:
            mint: The token mint address
            holder_data: Token holder distribution data
            
        Returns:
            Ownership risk score (0-100, higher = riskier)
        """
        ownership_risk = 0  # Start with no risk
        
        try:
            # High concentration in top holder is very risky
            top_holder_percentage = holder_data.get("top_holder_percentage", 0)
            if top_holder_percentage > 80:
                ownership_risk += 80
            elif top_holder_percentage > 50:
                ownership_risk += 60
            elif top_holder_percentage > 30:
                ownership_risk += 40
            elif top_holder_percentage > 20:
                ownership_risk += 20
            
            # High concentration in top 10 holders is risky
            top_10_percentage = holder_data.get("top_10_percentage", 0)
            if top_10_percentage > 95:
                ownership_risk += 50
            elif top_10_percentage > 90:
                ownership_risk += 40
            elif top_10_percentage > 80:
                ownership_risk += 30
            elif top_10_percentage > 70:
                ownership_risk += 20
            
            # High concentration index is risky
            concentration_index = holder_data.get("concentration_index", 0)
            if concentration_index > 0.9:
                ownership_risk += 30
            elif concentration_index > 0.8:
                ownership_risk += 20
            elif concentration_index > 0.7:
                ownership_risk += 10
            
            # Small number of holders is risky
            total_holders = holder_data.get("total_holders", 0)
            if total_holders < 10:
                ownership_risk += 40
            elif total_holders < 50:
                ownership_risk += 30
            elif total_holders < 100:
                ownership_risk += 20
            elif total_holders < 500:
                ownership_risk += 10
            
            # Cap the risk score at 100
            ownership_risk = min(100, ownership_risk)
            
        except Exception as e:
            logger.error(f"Error calculating ownership risk for {mint}: {str(e)}")
            
        return ownership_risk

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