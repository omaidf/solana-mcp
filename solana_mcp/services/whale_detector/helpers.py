"""Helper functions for whale detection."""

import logging
import requests
from typing import Dict, List, Any, Optional
from decimal import Decimal

from solana_mcp.services.whale_detector.models import TokenInfo, TokenValue, WalletValue

logger = logging.getLogger(__name__)

# Constants
SIGNIFICANT_VALUE_USD = 50000  # Minimum USD value to consider significant


def get_token_price(token_address: str) -> Decimal:
    """Get token price from Jupiter API.
    
    Args:
        token_address: Token mint address
        
    Returns:
        Token price in USD
    """
    try:
        # First try Jupiter API
        jupiter_url = f"https://price.jup.ag/v4/price?ids={token_address}"
        response = requests.get(jupiter_url)
        data = response.json()
        
        if data.get("data") and token_address in data["data"]:
            price = data["data"][token_address]["price"]
            return Decimal(str(price))
            
        # If Jupiter doesn't have it, try Birdeye API
        birdeye_url = f"https://public-api.birdeye.so/public/price?address={token_address}"
        headers = {"X-API-KEY": "5c88c2dddead4a1094fbcc5f2d86d6fe"}
        response = requests.get(birdeye_url, headers=headers)
        data = response.json()
        
        if data.get("success") and data.get("data", {}).get("value"):
            price = data["data"]["value"]
            return Decimal(str(price))
            
        logger.info(f"Could not find price for token {token_address}, using default price of $0.01")
        return Decimal("0.01")  # Default fallback price
    except Exception as e:
        logger.warning(f"Error fetching token price: {str(e)}")
        logger.info("Using default price of $0.01")
        return Decimal("0.01")  # Default fallback price


def get_token_prices(token_mints: List[str]) -> Dict[str, Decimal]:
    """Get prices for multiple tokens.
    
    Args:
        token_mints: List of token mint addresses
        
    Returns:
        Dictionary of token prices
    """
    prices = {}
    
    # Add default prices for common tokens
    default_prices = {
        "So11111111111111111111111111111111111111112": Decimal("150"),  # SOL - hardcoded estimate
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": Decimal("1"),   # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": Decimal("1"),   # USDT
        "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": Decimal("1"),   # stUSDC
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": Decimal("30"),  # BONK
    }
    
    # Use the default prices
    prices.update(default_prices)
    
    # For token mints not in defaults, try to get prices
    for mint in token_mints:
        if mint not in prices:
            try:
                price = get_token_price(mint)
                prices[mint] = price
            except:
                # If we can't get price, use a very low default
                prices[mint] = Decimal("0.0001")
    
    return prices


def calculate_wallet_value(token_balances: List[Dict[str, Any]], 
                          prices: Dict[str, Decimal], 
                          target_token: str = None) -> WalletValue:
    """Calculate the total value of all tokens in a wallet.
    
    Args:
        token_balances: List of token balances
        prices: Dictionary of token prices
        target_token: Optional target token to track separately
        
    Returns:
        Wallet value information
    """
    total_value = Decimal(0)
    target_token_value = Decimal(0)
    token_values = []
    
    for token in token_balances:
        mint = token['mint']
        amount = token['amount']
        price = prices.get(mint, Decimal("0.0001"))  # Default very low price if unknown
        
        value = amount * price
        total_value += value
        
        # Track value of the target token separately
        if target_token and mint == target_token:
            target_token_value = value
            
        token_values.append(TokenValue(
            mint=mint,
            token=token.get('token', mint[:6]),
            amount=float(amount),
            value_usd=float(value)
        ))
    
    # Sort tokens by value (highest first)
    token_values.sort(key=lambda x: x.value_usd, reverse=True)
    
    return WalletValue(
        total_value_usd=float(total_value),
        target_token_value_usd=float(target_token_value),
        tokens=token_values[:10]  # Include top 10 tokens by value
    )


def determine_whale_threshold(wallets: List[Dict[str, Any]]) -> float:
    """Determine a reasonable threshold for whale classification.
    
    Args:
        wallets: List of wallet information
        
    Returns:
        Threshold for whale classification
    """
    if not wallets:
        return SIGNIFICANT_VALUE_USD
    
    # Sort wallets by total value
    sorted_wallets = sorted(wallets, key=lambda x: x['total_value_usd'], reverse=True)
    
    # Get values for analysis
    values = [w['total_value_usd'] for w in sorted_wallets]
    
    # Calculate some statistical thresholds
    try:
        # 1. Top 20% percentile of holders are considered whales
        percentile_threshold = values[int(len(values) * 0.2)] if len(values) >= 5 else values[-1]
        
        # 2. Anyone holding more than 1% of supply is a whale
        target_token_threshold = next((w['total_value_usd'] for w in sorted_wallets 
                                if w['target_token_supply_percentage'] >= 1), 0)
        
        # 3. Base minimum threshold for significance
        base_threshold = SIGNIFICANT_VALUE_USD
        
        # Take the minimum of the three thresholds (excluding 0)
        candidates = [t for t in [percentile_threshold, target_token_threshold, base_threshold] if t > 0]
        threshold = min(candidates) if candidates else base_threshold
        
        # Make sure it's at least the base threshold
        return max(threshold, base_threshold)
    except:
        # Fallback to base threshold if statistical calculation fails
        return SIGNIFICANT_VALUE_USD 