"""Helper functions for fresh wallet detection."""

import logging
import datetime
import time
from typing import Dict, List, Any, Optional
from decimal import Decimal

from solana_mcp.solana_client import SolanaClient

logger = logging.getLogger(__name__)

# Constants
FRESH_WALLET_AGE_DAYS = 7  # Maximum age for a fresh wallet


async def get_wallet_age_days(wallet_address: str, solana_client: SolanaClient) -> Optional[int]:
    """Get wallet age in days.
    
    Args:
        wallet_address: Wallet address
        solana_client: Solana client
        
    Returns:
        Age in days or None if can't be determined
    """
    try:
        # Get signatures with the highest possible limit for better age estimate
        signatures = await solana_client.get_signatures_for_address(wallet_address, limit=100)
        
        if not signatures:
            return None
            
        # Get the oldest transaction
        oldest_tx = signatures[-1]
        
        if "blockTime" not in oldest_tx:
            return None
            
        # Calculate age from the oldest transaction
        block_time = oldest_tx["blockTime"]
        tx_date = datetime.datetime.fromtimestamp(block_time)
        current_date = datetime.datetime.now()
        
        return (current_date - tx_date).days
    except Exception as e:
        logger.warning(f"Error determining wallet age for {wallet_address}: {str(e)}")
        return None


async def get_transaction_count(wallet_address: str, solana_client: SolanaClient, days: int = None) -> int:
    """Get transaction count for a wallet.
    
    Args:
        wallet_address: Wallet address
        solana_client: Solana client
        days: Optional number of days to limit to
        
    Returns:
        Transaction count
    """
    try:
        # Get signatures for the address with a reasonable limit
        signatures = await solana_client.get_signatures_for_address(wallet_address, limit=100)
        
        if not days:
            return len(signatures)
            
        # Filter by timestamp if days is specified
        current_time = int(time.time())
        cutoff_time = current_time - (days * 86400)  # Convert days to seconds
        
        filtered_count = 0
        for sig in signatures:
            if sig.get("blockTime", 0) >= cutoff_time:
                filtered_count += 1
                
        return filtered_count
    except Exception as e:
        logger.warning(f"Error getting transaction count for {wallet_address}: {str(e)}")
        return 0


def calculate_freshness_score(
    token_count: int,
    non_dust_token_count: int,
    is_new_wallet: bool
) -> tuple:
    """Calculate a freshness score for a wallet.
    
    Args:
        token_count: Total token count
        non_dust_token_count: Number of non-dust tokens
        is_new_wallet: Whether the wallet is new
        
    Returns:
        Tuple of (is_fresh, freshness_score)
    """
    # Fast early return - if too many tokens, not fresh
    if non_dust_token_count > 8:
        return False, 0.1
    
    # Determine freshness
    is_fresh = (
        (is_new_wallet and non_dust_token_count < 5) or  # New wallet with few tokens
        (non_dust_token_count < 4)  # Very few tokens
    )
    
    # Calculate freshness score
    freshness_score = 0.5
    if is_fresh:
        freshness_score = (1 - (non_dust_token_count / 10)) * 0.5 + (0.8 if is_new_wallet else 0.3)
        freshness_score = min(0.95, freshness_score)  # Cap at 0.95
    
    return is_fresh, freshness_score 