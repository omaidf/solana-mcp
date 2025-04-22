"""Wallet classifier module for identifying high-risk wallet behavior."""

import time
import re
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from datetime import datetime, timedelta
import statistics
from dataclasses import dataclass, field

from solana_mcp.solana_client import SolanaClient
from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.decorators import validate_solana_key, handle_errors
# Import these at the module level to avoid circular imports
from solana_mcp.token_categorizer import TokenCategorizer
from solana_mcp.token_risk_analyzer import TokenRiskAnalyzer

# Set up logging
logger = get_logger(__name__)

# Token Program ID constant
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

# Transaction activity thresholds for classification
ACTIVITY_THRESHOLDS = {
    "high_velocity": {
        "tx_count_24h": 50,  # More than 50 transactions in 24 hours
        "unique_token_transfers": 20,  # More than 20 unique tokens transferred
    },
    "whale": {
        "total_value_usd": 100000,  # More than $100k in assets
        "large_transfer_usd": 50000,  # Transfers over $50k
    },
    "new_wallet": {
        "age_days": 7,  # Less than 7 days old
        "tx_count_total": 10,  # Less than 10 total transactions
    },
    "temporary_holder": {
        "avg_hold_time_hours": 6,  # Average hold time less than 6 hours
        "token_in_out_ratio": 0.9,  # 90% of tokens that come in also go out
    }
}

# Risk levels and their thresholds
RISK_LEVELS = {
    "low": 30,      # 0-29
    "medium": 60,   # 30-59
    "high": 85,     # 60-84
    "very_high": 100  # 85-100
}

@dataclass
class WalletStats:
    """Statistics for a wallet's activity."""
    tx_count_total: int = 0
    tx_count_24h: int = 0
    unique_tokens_transferred: Set[str] = field(default_factory=set)
    token_in_out_map: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # (in_count, out_count)
    total_value_usd: float = 0.0
    largest_transfer_usd: float = 0.0
    first_activity_timestamp: Optional[int] = None
    last_activity_timestamp: Optional[int] = None
    avg_hold_time_hours: Optional[float] = None
    
    @property
    def age_days(self) -> Optional[float]:
        """Get the age of the wallet in days."""
        if not self.first_activity_timestamp:
            return None
        
        try:
            first_activity = datetime.fromtimestamp(self.first_activity_timestamp)
            now = datetime.now()
            
            return (now - first_activity).total_seconds() / 86400  # Convert seconds to days
        except Exception:
            # Handle potential timestamp conversion errors
            return None
    
    @property
    def token_in_out_ratio(self) -> Optional[float]:
        """Calculate the ratio of tokens that come in and go out."""
        total_in = 0
        total_out = 0
        
        for token, (in_count, out_count) in self.token_in_out_map.items():
            total_in += in_count
            total_out += out_count
        
        if total_in == 0:
            return None
        
        return total_out / total_in


class WalletClassifier:
    """Identifies wallet behavior patterns and classifies them."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize with a Solana client.
        
        Args:
            solana_client: The Solana client
        """
        self.client = solana_client
        self.logger = get_logger(__name__)
        self.thresholds = ACTIVITY_THRESHOLDS
    
    @validate_solana_key
    @handle_errors
    async def classify_wallet(self, wallet_address: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Classify a wallet based on its transaction history and behavior.
        
        Args:
            wallet_address: The wallet address to classify
            request_id: Optional request ID for tracing
            
        Returns:
            Classification results with behavior patterns
        """
        log_with_context(
            logger,
            "info",
            f"Wallet classification requested for: {wallet_address}",
            request_id=request_id,
            wallet=wallet_address
        )
        
        # Get wallet stats from transaction history
        wallet_stats = await self._compute_wallet_stats(wallet_address, request_id)
        
        # Classify wallet based on activity patterns
        classifications = self._apply_classification_rules(wallet_stats)
        
        # Compute overall risk score (0-100) based on classifications
        risk_score = self._compute_risk_score(classifications, wallet_stats)
        
        # Prepare result with all classification information
        result = {
            "wallet_address": wallet_address,
            "classifications": classifications,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "stats": {
                "tx_count_total": wallet_stats.tx_count_total,
                "tx_count_24h": wallet_stats.tx_count_24h,
                "unique_tokens_count": len(wallet_stats.unique_tokens_transferred),
                "total_value_usd": wallet_stats.total_value_usd,
                "largest_transfer_usd": wallet_stats.largest_transfer_usd,
                "age_days": wallet_stats.age_days,
                "avg_hold_time_hours": wallet_stats.avg_hold_time_hours,
                "token_in_out_ratio": wallet_stats.token_in_out_ratio
            }
        }
        
        log_with_context(
            logger,
            "info",
            f"Wallet classified with risk score: {risk_score}, level: {result['risk_level']}",
            request_id=request_id,
            wallet=wallet_address,
            risk_score=risk_score,
            risk_level=result["risk_level"]
        )
        
        return result
    
    @handle_errors
    async def _compute_wallet_stats(self, wallet_address: str, request_id: Optional[str] = None) -> WalletStats:
        """Compute statistics for a wallet's activity.
        
        Args:
            wallet_address: The wallet address
            request_id: Optional request ID for tracing
            
        Returns:
            WalletStats object with computed statistics
        """
        # Initialize wallet stats
        stats = WalletStats()
        
        try:
            # Get wallet transaction signatures with pagination
            # Start with a reasonable limit for performance
            signatures = []
            before = None
            page_size = 50
            max_pages = 10  # Limit to 10 pages (500 transactions) for performance
            
            for i in range(max_pages):
                page = await self.client.get_signatures_for_address(
                    wallet_address, 
                    limit=page_size, 
                    before=before
                )
                
                if not page or len(page) == 0:
                    break
                    
                signatures.extend(page)
                if len(page) < page_size:
                    break
                    
                # Set 'before' parameter for next page
                before = page[-1].get("signature")
                
            stats.tx_count_total = len(signatures)
            
            # Early exit if no transactions
            if stats.tx_count_total == 0:
                return stats
            
            # Get current time for 24h window
            current_time = int(time.time())
            day_ago = current_time - 86400  # 24 hours in seconds
            
            # Process transaction timestamps
            if signatures:
                # Set first and last activity timestamps
                last_sig = signatures[0]
                first_sig = signatures[-1]
                
                stats.first_activity_timestamp = first_sig.get("blockTime", current_time) if "blockTime" in first_sig else current_time
                stats.last_activity_timestamp = last_sig.get("blockTime", current_time) if "blockTime" in last_sig else current_time
                
                # Count recent transactions (24h)
                for signature in signatures:
                    if "blockTime" not in signature:
                        continue
                        
                    block_time = signature.get("blockTime", 0)
                    if block_time >= day_ago:
                        stats.tx_count_24h += 1
            
            # Get token balances and calculate total value
            try:
                token_balances = await self.client.get_token_accounts_by_owner(wallet_address)
                
                for balance in token_balances:
                    try:
                        mint = balance.get("mint", "")
                        if not mint:
                            continue
                            
                        amount = float(balance.get("amount", 0))
                        
                        # Get token price and calculate USD value
                        try:
                            token_price = await self.client.get_token_price(mint)
                            usd_value = amount * token_price
                            stats.total_value_usd += usd_value
                        except Exception as e:
                            self.logger.debug(f"Error getting token price for {mint}: {str(e)}")
                            continue
                    except Exception as e:
                        self.logger.warning(f"Error calculating token value for a balance: {str(e)}")
                        continue
            except Exception as e:
                self.logger.warning(f"Error getting token accounts for {wallet_address}: {str(e)}")
            
            # Calculate average hold time for tokens
            if signatures:
                hold_times = []
                token_timestamps = {}  # { token_mint: { (tx_id, timestamp) } }
                
                for signature in signatures:
                    token_mint = signature.get("mint", "")
                    if not token_mint:
                        continue
                        
                    block_time = signature.get("blockTime", 0)
                    if block_time >= day_ago:
                        # Track token in/out counts
                        direction = "in" if signature.get("source", "") == wallet_address else "out"
                        current_counts = stats.token_in_out_map.get(token_mint, (0, 0))
                        
                        if direction == "in":
                            stats.token_in_out_map[token_mint] = (current_counts[0] + 1, current_counts[1])
                        elif direction == "out":
                            stats.token_in_out_map[token_mint] = (current_counts[0], current_counts[1] + 1)
                        
                        # Track token timestamp
                        if token_mint not in token_timestamps:
                            token_timestamps[token_mint] = []
                        token_timestamps[token_mint].append((signature.get("signature"), block_time))
                
                # Calculate hold times from in/out pairs
                for token_mint, transfers in token_timestamps.items():
                    in_timestamps = [ts for (tx, ts), direction in transfers.items() if direction == "in"]
                    out_timestamps = [ts for (tx, ts), direction in transfers.items() if direction == "out"]
                    
                    if not in_timestamps or not out_timestamps:
                        continue
                        
                    in_timestamps.sort()
                    out_timestamps.sort()
                    
                    # Match each outgoing transfer with the closest preceding incoming transfer
                    for out_ts in out_timestamps:
                        # Find the closest preceding in timestamp
                        preceding_ins = [ts for ts in in_timestamps if ts < out_ts]
                        
                        if preceding_ins:
                            closest_in = max(preceding_ins)
                            hold_time = (out_ts - closest_in) / 3600  # Convert seconds to hours
                            hold_times.append(hold_time)
                
                if hold_times:
                    stats.avg_hold_time_hours = statistics.mean(hold_times)
                    
        except Exception as e:
            self.logger.error(f"Error computing wallet stats for {wallet_address}: {str(e)}")
            
        return stats
    
    def _apply_classification_rules(self, stats: WalletStats) -> Dict[str, bool]:
        """Apply classification rules to wallet stats.
        
        Args:
            stats: Wallet statistics
            
        Returns:
            Dictionary of classification results
        """
        classifications = {
            "high_velocity": False,
            "whale": False,
            "new_wallet": False,
            "temporary_holder": False
        }
        
        # Check high velocity - require valid stats data
        tx_count_24h = stats.tx_count_24h if stats.tx_count_24h is not None else 0
        unique_tokens = len(stats.unique_tokens_transferred) if stats.unique_tokens_transferred is not None else 0
        
        if (tx_count_24h >= self.thresholds["high_velocity"]["tx_count_24h"] or
            unique_tokens >= self.thresholds["high_velocity"]["unique_token_transfers"]):
            classifications["high_velocity"] = True
        
        # Check whale status - require valid stats data
        total_value = stats.total_value_usd if stats.total_value_usd is not None else 0
        largest_transfer = stats.largest_transfer_usd if stats.largest_transfer_usd is not None else 0
        
        if (total_value >= self.thresholds["whale"]["total_value_usd"] or
            largest_transfer >= self.thresholds["whale"]["large_transfer_usd"]):
            classifications["whale"] = True
        
        # Check new wallet - require valid age data
        age_days = stats.age_days
        tx_count_total = stats.tx_count_total if stats.tx_count_total is not None else 0
        
        if (age_days is not None and age_days <= self.thresholds["new_wallet"]["age_days"] and
            tx_count_total <= self.thresholds["new_wallet"]["tx_count_total"]):
            classifications["new_wallet"] = True
        
        # Check temporary holder - require valid holding data
        avg_hold_time = stats.avg_hold_time_hours
        token_ratio = stats.token_in_out_ratio
        
        if (avg_hold_time is not None and token_ratio is not None and
            avg_hold_time <= self.thresholds["temporary_holder"]["avg_hold_time_hours"] and
            token_ratio >= self.thresholds["temporary_holder"]["token_in_out_ratio"]):
            classifications["temporary_holder"] = True
        
        return classifications
    
    def _compute_risk_score(self, classifications: Dict[str, bool], stats: WalletStats) -> int:
        """Compute a risk score based on classifications and other factors.
        
        Args:
            classifications: Classification results
            stats: Wallet statistics
            
        Returns:
            Risk score (0-100)
        """
        # Base score
        score = 0
        
        # Add points for each classification with proper weighting
        if classifications["high_velocity"]:
            score += 25
        
        if classifications["whale"]:
            score += 10
        
        if classifications["new_wallet"]:
            score += 20
        
        if classifications["temporary_holder"]:
            score += 30
        
        # Additional factors - token diversity
        token_count = len(stats.unique_tokens_transferred) if stats.unique_tokens_transferred else 0
        if token_count > 100:
            score += 15
        elif token_count > 50:
            score += 10
        elif token_count > 20:
            score += 5
        
        # Transaction velocity factor
        tx_count_24h = stats.tx_count_24h if stats.tx_count_24h is not None else 0
        if tx_count_24h > 200:
            score += 15
        elif tx_count_24h > 100:
            score += 10
        elif tx_count_24h > 75:
            score += 5
        
        # Cap score at 100
        return min(score, 100)
    
    def _get_risk_level(self, score: int) -> str:
        """Convert a risk score to a risk level.
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            Risk level (low, medium, high, very_high)
        """
        if score < RISK_LEVELS["low"]:
            return "low"
        elif score < RISK_LEVELS["medium"]:
            return "medium"
        elif score < RISK_LEVELS["high"]:
            return "high"
        else:
            return "very_high"
    
    @handle_errors
    async def batch_classify_wallets(self, wallet_addresses: List[str], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Classify multiple wallets at once.
        
        Args:
            wallet_addresses: List of wallet addresses
            request_id: Optional request ID for tracing
            
        Returns:
            Dictionary of wallet classifications
        """
        log_with_context(
            logger,
            "info",
            f"Batch wallet classification requested for {len(wallet_addresses)} wallets",
            request_id=request_id,
            wallet_count=len(wallet_addresses)
        )
        
        results = {}
        
        for wallet in wallet_addresses:
            try:
                validation_error = self._validate_solana_key_format(wallet)
                if validation_error:
                    results[wallet] = {"error": validation_error}
                    continue
                    
                result = await self.classify_wallet(wallet, request_id)
                results[wallet] = result
            except Exception as e:
                logger.error(f"Error classifying wallet {wallet}: {str(e)}")
                results[wallet] = {"error": str(e)}
        
        log_with_context(
            logger,
            "info",
            f"Batch classification completed for {len(wallet_addresses)} wallets",
            request_id=request_id,
            wallet_count=len(wallet_addresses)
        )
        
        # Summarize risk levels
        risk_levels = {"low": 0, "medium": 0, "high": 0, "very_high": 0, "error": 0}
        
        for wallet, result in results.items():
            if "error" in result:
                risk_levels["error"] += 1
            else:
                risk_level = result.get("risk_level", "low")
                if risk_level in risk_levels:
                    risk_levels[risk_level] += 1
                else:
                    # Handle unexpected risk level
                    risk_levels["error"] += 1
        
        return {
            "wallets": results,
            "count": len(wallet_addresses),
            "risk_summary": risk_levels
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