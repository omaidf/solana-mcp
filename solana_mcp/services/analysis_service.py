"""Analysis service for Solana MCP.

This module provides services for analyzing Solana blockchain data.
"""

import asyncio
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

from solana_mcp.services.base_service import BaseService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.token_service import TokenService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.decorators import validate_solana_key
from solana_mcp.constants import SYSTEM_PROGRAM_ID, TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID, METADATA_PROGRAM_ID

class AnalysisService(BaseService):
    """Service for analyzing Solana blockchain data."""
    
    def __init__(
        self, 
        solana_client: SolanaClient, 
        transaction_service: TransactionService,
        token_service: Optional[TokenService] = None,
        cache_service: Optional[CacheService] = None
    ):
        """Initialize the analysis service.
        
        Args:
            solana_client: The Solana client to use
            transaction_service: Transaction service for fetching transaction data
            token_service: Optional token service for token-related analysis
            cache_service: Optional cache service
        """
        super().__init__()
        self.client = solana_client
        self.transaction_service = transaction_service
        self.token_service = token_service
        self.cache = cache_service
    
    @validate_solana_key
    async def analyze_token_flow(
        self, 
        address: str, 
        limit: int = 100, 
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze token flow in and out of an account.
        
        Args:
            address: The account address
            limit: Maximum number of transactions to analyze
            days: Optional number of days to analyze (from now)
            
        Returns:
            Token flow analysis
        """
        self.log_with_context(
            "info", 
            f"Analyzing token flow for account {address}",
            limit=limit,
            days=days
        )
        
        # Get transaction history
        history = await self.transaction_service.get_transactions_for_address(
            address,
            limit=limit
        )
        
        if not history or not history.get("transactions"):
            return {
                "address": address,
                "inflow": [],
                "outflow": [],
                "total_inflow": 0,
                "total_outflow": 0,
                "net_flow": 0
            }
        
        # Analyze token flow from transactions
        inflow = []
        outflow = []
        total_inflow = 0
        total_outflow = 0
        
        for tx_info in history.get("transactions", []):
            # Extract token transfers from transaction if details are available
            if "details" in tx_info:
                tx = tx_info["details"]
                transfers = self._extract_token_transfers(tx, address)
                
                for transfer in transfers:
                    if transfer["direction"] == "in":
                        inflow.append(transfer)
                        total_inflow += transfer["amount"]
                    else:
                        outflow.append(transfer)
                        total_outflow += transfer["amount"]
        
        return {
            "address": address,
            "inflow": inflow,
            "outflow": outflow,
            "total_inflow": total_inflow,
            "total_outflow": total_outflow,
            "net_flow": total_inflow - total_outflow
        }
    
    @validate_solana_key
    async def analyze_activity_pattern(
        self, 
        address: str, 
        limit: int = 200
    ) -> Dict[str, Any]:
        """Analyze activity patterns for an account.
        
        Args:
            address: The account address
            limit: Maximum number of transactions to analyze
            
        Returns:
            Activity pattern analysis
        """
        self.log_with_context(
            "info", 
            f"Analyzing activity pattern for account {address}",
            limit=limit
        )
        
        # Get transaction history
        history = await self.transaction_service.get_transactions_for_address(
            address,
            limit=limit
        )
        
        if not history or not history.get("transactions"):
            return {
                "address": address,
                "total_transactions": 0,
                "programs_interaction": {},
                "temporal_pattern": {},
                "transaction_types": {}
            }
        
        transactions = history.get("transactions", [])
        
        # Extract transaction timestamps
        timestamps = []
        programs = Counter()
        transaction_types = Counter()
        
        for tx_info in transactions:
            # Add timestamp
            if "block_time" in tx_info:
                timestamps.append(tx_info["block_time"])
            
            # Analyze program interactions if details are available
            if "details" in tx_info:
                tx = tx_info["details"]
                
                # Count program interactions
                self._count_program_interactions(tx, programs)
                
                # Determine transaction type
                tx_type = self._detect_transaction_type(tx)
                if tx_type:
                    transaction_types[tx_type] += 1
        
        # Generate temporal pattern
        temporal_pattern = self._analyze_temporal_pattern(timestamps)
        
        return {
            "address": address,
            "total_transactions": len(transactions),
            "programs_interaction": dict(programs.most_common(10)),
            "temporal_pattern": temporal_pattern,
            "transaction_types": dict(transaction_types.most_common())
        }
    
    @validate_solana_key
    async def wallet_profile(self, address: str) -> Dict[str, Any]:
        """Generate a profile for a wallet based on its activity.
        
        Args:
            address: The account address
            
        Returns:
            Wallet profile
        """
        self.log_with_context(
            "info", 
            f"Generating wallet profile for account {address}"
        )
        
        # Check cache
        if self.cache:
            cached_profile = self.cache.get(f"wallet_profile:{address}")
            if cached_profile:
                return cached_profile
        
        # Get account balance
        balance = await self.execute_with_fallback(
            self.client.get_balance(address),
            fallback_value=0,
            error_message=f"Error fetching balance for {address}"
        )
        
        # Get token balances if token service is available
        token_balances = []
        if self.token_service:
            # Implementation depends on token service capabilities
            pass
        
        # Run activity pattern analysis
        activity = await self.analyze_activity_pattern(address, limit=100)
        
        # Generate profile based on collected data
        profile = {
            "address": address,
            "sol_balance": balance / 1e9,  # Convert lamports to SOL
            "token_balances": token_balances,
            "activity_level": self._determine_activity_level(activity["total_transactions"]),
            "transaction_types": activity["transaction_types"],
            "programs_interaction": activity["programs_interaction"],
            "temporal_pattern": activity["temporal_pattern"],
        }
        
        # Add user type classification
        profile["user_type"] = self._classify_user_type(profile)
        
        # Cache the profile
        if self.cache:
            self.cache.set(f"wallet_profile:{address}", profile, ttl=3600)
        
        return profile
    
    def _extract_token_transfers(self, transaction: Dict[str, Any], address: str) -> List[Dict[str, Any]]:
        """Extract token transfers from a transaction.
        
        Args:
            transaction: Transaction data
            address: The address to analyze transfers for
            
        Returns:
            List of token transfers
        """
        # Import TransactionClient to use its implementation
        from solana_mcp.clients.transaction_client import TransactionClient
        
        try:
            # Create a TransactionClient with appropriate configuration
            transaction_client = TransactionClient(
                # Pass the same configuration the client would have
                rpc_url=self.client.config.rpc_url if hasattr(self.client, 'config') else None,
                timeout=self.timeout
            )
            
            # Use TransactionClient's parse_transaction to get token transfers
            parsed_tx = transaction_client.parse_transaction(transaction)
            
            # Extract token transfers from the parsed transaction
            token_transfers = parsed_tx.get("token_transfers", [])
            
            # Convert to our expected format
            transfers = []
            for transfer in token_transfers:
                # Determine if this transfer involves our address
                token_account = transfer.get("token_account", "")
                owner = transfer.get("owner", "")
                
                if owner == address or token_account == address:
                    # Determine direction
                    direction = "in" if transfer.get("change", 0) > 0 else "out"
                    
                    transfers.append({
                        "direction": direction,
                        "counterparty": token_account if direction == "in" else owner,
                        "amount": abs(transfer.get("change", 0)),
                        "token": transfer.get("mint", "Unknown"),
                        "time": transaction.get("block_time")
                    })
                    
            return transfers
        except Exception as e:
            self.logger.error(f"Error extracting token transfers: {str(e)}", exc_info=True)
            return []
        
    def _count_program_interactions(self, transaction: Dict[str, Any], counter: Counter) -> None:
        """Count program interactions in a transaction.
        
        Args:
            transaction: Transaction data
            counter: Counter to update
        """
        # Import TransactionClient to use its program name mapping
        from solana_mcp.clients.transaction_client import TransactionClient
        transaction_client = TransactionClient()
        
        if "transaction" in transaction and "message" in transaction["transaction"]:
            message = transaction["transaction"]["message"]
            
            for instr in message.get("instructions", []):
                if "programId" in instr:
                    program_id = instr["programId"]
                    # Use TransactionClient's program name mapping
                    program_name = transaction_client._get_program_name(program_id)
                    counter[program_name] += 1
    
    def _detect_transaction_type(self, transaction: Dict[str, Any]) -> Optional[str]:
        """Detect the type of transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Transaction type or None if unknown
        """
        # This is a simplified implementation
        # A real implementation would need more sophisticated logic
        
        if "instructions" in transaction:
            for instr in transaction.get("instructions", []):
                program_id = instr.get("program_id")
                
                if program_id == SYSTEM_PROGRAM_ID:
                    return "SOL Transfer"
                elif program_id == TOKEN_PROGRAM_ID:
                    return "Token Transfer"
                elif program_id == ASSOCIATED_TOKEN_PROGRAM_ID:
                    return "Token Account Creation"
                elif program_id == METADATA_PROGRAM_ID:
                    return "NFT Operation"
        
        return "Unknown"
    
    def _analyze_temporal_pattern(self, timestamps: List[int]) -> Dict[str, Any]:
        """Analyze temporal patterns from transaction timestamps.
        
        Args:
            timestamps: List of Unix timestamps
            
        Returns:
            Temporal pattern analysis
        """
        if not timestamps:
            return {}
        
        # Convert timestamps to datetime for analysis
        from datetime import datetime
        
        # Simplified implementation - a real one would be more sophisticated
        hour_activity = defaultdict(int)
        day_activity = defaultdict(int)
        
        for ts in timestamps:
            dt = datetime.fromtimestamp(ts)
            hour_activity[dt.hour] += 1
            day_activity[dt.weekday()] += 1
        
        # Find most active periods
        most_active_hour = max(hour_activity.items(), key=lambda x: x[1])[0] if hour_activity else None
        most_active_day = max(day_activity.items(), key=lambda x: x[1])[0] if day_activity else None
        
        # Convert day number to name
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        most_active_day_name = day_names[most_active_day] if most_active_day is not None else None
        
        return {
            "most_active_hour": most_active_hour,
            "most_active_day": most_active_day_name,
            "hourly_activity": dict(hour_activity),
            "daily_activity": {day_names[day]: count for day, count in day_activity.items()}
        }
    
    def _determine_activity_level(self, transaction_count: int) -> str:
        """Determine activity level based on transaction count.
        
        Args:
            transaction_count: Number of transactions
            
        Returns:
            Activity level description
        """
        if transaction_count == 0:
            return "Inactive"
        elif transaction_count < 10:
            return "Low"
        elif transaction_count < 50:
            return "Medium"
        elif transaction_count < 200:
            return "High"
        else:
            return "Very High"
    
    def _classify_user_type(self, profile: Dict[str, Any]) -> str:
        """Classify user type based on profile data.
        
        Args:
            profile: Wallet profile data
            
        Returns:
            User type classification
        """
        # This is a simplified implementation
        # A real implementation would use more sophisticated logic or ML
        
        txs = profile.get("transaction_types", {})
        programs = profile.get("programs_interaction", {})
        activity_level = profile.get("activity_level", "Low")
        
        # Simple classification logic
        if "NFT Operation" in txs and txs.get("NFT Operation", 0) > 5:
            return "NFT Collector"
        elif "Token Transfer" in txs and txs.get("Token Transfer", 0) > 10:
            return "Token Trader"
        elif activity_level in ["High", "Very High"]:
            return "Active Trader"
        elif profile.get("sol_balance", 0) > 10:
            return "SOL Holder"
        else:
            return "Casual User" 