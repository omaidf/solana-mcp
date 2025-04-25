"""Transaction client module for Solana MCP.

This module provides a client for transaction-related operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from solana.publickey import PublicKey
from solana.transaction import Transaction

from solana_mcp.models.enhanced_transaction import EnhancedTransaction
from solana_mcp.models.token_transfer import TokenTransfer
from solana_mcp.services.rpc_service import RPCService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.constants import ZERO_PUBKEY
from solana_mcp.utils.dependency_injection import inject
from solana_mcp.utils.error_handling import TransactionError, NotFoundError, handle_errors
from solana_mcp.utils.transaction_parser import (
    extract_sol_transfers,
    extract_token_transfers,
    parse_instruction_logs,
)

from solana_mcp.clients.base_client import BaseSolanaClient, InvalidPublicKeyError, validate_public_key
from solana_mcp.logging_config import get_logger
from solana_mcp.utils.validation import validate_transaction_signature
from solana_mcp.constants import SYSTEM_PROGRAM_ID, TOKEN_PROGRAM_ID, PROGRAM_NAMES

# Get logger
logger = get_logger(__name__)

class TransactionClient:
    """Client for transaction-related operations."""

    @inject
    def __init__(self, solana_client: Optional[SolanaClient] = None, rpc_service: Optional[RPCService] = None):
        """Initialize transaction client.

        Args:
            solana_client: Solana client instance (injected)
            rpc_service: RPC service instance (injected)
        """
        self.solana_client = solana_client
        self.rpc_service = rpc_service

    @handle_errors(error_map={
        Exception: TransactionError,
        ValueError: NotFoundError
    })
    async def get_transaction(
        self, signature: str, encoding: Optional[str] = None
    ) -> EnhancedTransaction:
        """Get transaction details.

        Args:
            signature: Transaction signature
            encoding: Encoding type

        Returns:
            Enhanced transaction details
        
        Raises:
            NotFoundError: If transaction not found
            TransactionError: For any other transaction-related error
        """
        # Get transaction from RPC
        tx_response = await self.rpc_service.get_transaction(signature, encoding)
        if not tx_response or not tx_response.get("transaction"):
            raise NotFoundError(f"Transaction not found: {signature}")

        # Parse transaction
        return await self.parse_transaction(tx_response)

    @handle_errors(error_map={
        Exception: TransactionError
    })
    async def get_block(
        self, slot: int, encoding: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get block details.

        Args:
            slot: Block slot
            encoding: Encoding type

        Returns:
            Block details
        
        Raises:
            TransactionError: For any transaction-related error
        """
        block = await self.rpc_service.get_block(slot, encoding)
        return block

    @handle_errors(error_map={
        Exception: TransactionError
    })
    async def get_blocks(
        self, start_slot: int, end_slot: Optional[int] = None
    ) -> List[int]:
        """Get blocks in slot range.

        Args:
            start_slot: Start slot
            end_slot: End slot (optional)

        Returns:
            List of block slots
        
        Raises:
            TransactionError: For any transaction-related error
        """
        blocks = await self.rpc_service.get_blocks(start_slot, end_slot)
        return blocks

    @handle_errors(error_map={
        Exception: TransactionError
    })
    async def parse_transaction(self, tx_data: Dict[str, Any]) -> EnhancedTransaction:
        """Parse transaction data into enhanced transaction model.

        Args:
            tx_data: Transaction data from RPC

        Returns:
            Enhanced transaction object
        
        Raises:
            TransactionError: For any transaction-related error
        """
        # Extract transaction data
        meta = tx_data.get("meta", {})
        transaction = tx_data.get("transaction", {})

        # Extract transaction data
        signature = transaction.get("signatures", [""])[0]
        slot = tx_data.get("slot")
        block_time = tx_data.get("blockTime")
        timestamp = datetime.fromtimestamp(block_time) if block_time else None
        success = not meta.get("err")

        # Extract logs if available
        logs = meta.get("logMessages", [])
        parsed_logs = parse_instruction_logs(logs) if logs else []

        # Extract fee info
        fee = meta.get("fee", 0)

        # Extract account keys
        account_keys = [
            PublicKey(key) for key in transaction.get("message", {}).get("accountKeys", [])
        ]

        # Extract SOL transfers
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        sol_transfers = extract_sol_transfers(account_keys, pre_balances, post_balances)

        # Extract token transfers
        token_transfers = self._extract_token_transfers(meta, account_keys)

        # Create enhanced transaction object
        enhanced_tx = EnhancedTransaction(
            signature=signature,
            slot=slot,
            timestamp=timestamp,
            success=success,
            fee=fee,
            logs=logs,
            parsed_logs=parsed_logs,
            sol_transfers=sol_transfers,
            token_transfers=token_transfers,
            raw_data=tx_data,
        )
        
        return enhanced_tx

    def _extract_token_transfers(
        self, meta: Dict[str, Any], account_keys: List[PublicKey]
    ) -> List[TokenTransfer]:
        """Extract token transfers from transaction metadata.

        Args:
            meta: Transaction metadata
            account_keys: List of account public keys

        Returns:
            List of token transfers
        """
        token_transfers = []
        if "postTokenBalances" in meta and "preTokenBalances" in meta:
            token_transfers = extract_token_transfers(
                meta.get("preTokenBalances", []),
                meta.get("postTokenBalances", []),
                account_keys,
            )
        return token_transfers

    def _extract_program_ids(self, transaction: Transaction) -> Set[str]:
        """Extract program IDs from transaction.

        Args:
            transaction: Transaction object

        Returns:
            Set of program IDs
        """
        program_ids = set()
        for instruction in transaction.instructions:
            if instruction.program_id:
                program_ids.add(str(instruction.program_id))
        return program_ids

    def _get_program_name(self, program_id: str) -> str:
        """Get a human-readable name for a program ID.
        
        Args:
            program_id: The program ID
            
        Returns:
            Program name or 'Unknown Program'
        """
        return PROGRAM_NAMES.get(program_id, "Unknown Program")
    
    def _identify_instruction_type(self, program_id: str, data: str, accounts: List[str]) -> str:
        """Identify the type of instruction based on program and data.
        
        Args:
            program_id: The program ID
            data: The instruction data
            accounts: The accounts involved
            
        Returns:
            Instruction type or 'Unknown'
        """
        # This is a simplified version - in a real implementation, you would
        # decode the data based on the program's instruction format
        
        # System program instructions
        if program_id == SYSTEM_PROGRAM_ID:
            # First byte of data indicates instruction type
            if data.startswith("3Bxs"):  # Example: Transfer instruction starts with 3Bxs
                return "Transfer"
            if data.startswith("2"):  # Example: Create Account starts with 2
                return "CreateAccount"
            return "SystemInstruction"
            
        # Token program instructions
        elif program_id == TOKEN_PROGRAM_ID:
            # This would need proper decoding based on actual data format
            # Using simplified examples
            if data.startswith("3"):  # Example: Transfer instruction
                return "TokenTransfer"
            if data.startswith("7"):  # Example: Mint instruction
                return "TokenMint"
            if data.startswith("8"):  # Example: Burn instruction
                return "TokenBurn"
            return "TokenInstruction"
            
        # For other programs, return generic type
        return "Instruction"
    
    def _extract_token_transfers(
        self,
        pre_balances: List[Dict[str, Any]],
        post_balances: List[Dict[str, Any]],
        account_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract token transfers from pre and post token balances.
        
        Args:
            pre_balances: Pre-execution token balances
            post_balances: Post-execution token balances
            account_keys: Account keys in the transaction
            
        Returns:
            List of token transfers
        """
        transfers = []
        
        # Create maps of pre and post balances by account index and mint
        pre_map = {}
        for balance in pre_balances:
            idx = balance.get("accountIndex")
            mint = balance.get("mint")
            key = f"{idx}:{mint}"
            pre_map[key] = balance
        
        # Compare with post balances to find transfers
        for post in post_balances:
            idx = post.get("accountIndex")
            mint = post.get("mint")
            key = f"{idx}:{mint}"
            
            if key in pre_map:
                pre = pre_map[key]
                pre_amount = int(pre.get("uiTokenAmount", {}).get("amount", "0"))
                post_amount = int(post.get("uiTokenAmount", {}).get("amount", "0"))
                
                # Check if balance changed
                if post_amount != pre_amount:
                    owner = post.get("owner", "unknown")
                    account = account_keys[idx] if idx < len(account_keys) else "unknown"
                    
                    transfers.append({
                        "token_account": account,
                        "owner": owner,
                        "mint": mint,
                        "pre_amount": pre_amount,
                        "post_amount": post_amount,
                        "change": post_amount - pre_amount,
                        "decimals": post.get("uiTokenAmount", {}).get("decimals", 0)
                    })
            else:
                # New token account might have been created
                owner = post.get("owner", "unknown")
                account = account_keys[idx] if idx < len(account_keys) else "unknown"
                amount = int(post.get("uiTokenAmount", {}).get("amount", "0"))
                
                if amount > 0:
                    transfers.append({
                        "token_account": account,
                        "owner": owner,
                        "mint": mint,
                        "pre_amount": 0,
                        "post_amount": amount,
                        "change": amount,
                        "decimals": post.get("uiTokenAmount", {}).get("decimals", 0)
                    })
        
        return transfers
    
    def _extract_sol_transfers(
        self,
        pre_balances: List[int],
        post_balances: List[int],
        account_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract SOL transfers from pre and post balances.
        
        Args:
            pre_balances: Pre-execution SOL balances in lamports
            post_balances: Post-execution SOL balances in lamports
            account_keys: Account keys in the transaction
            
        Returns:
            List of SOL transfers
        """
        transfers = []
        
        min_length = min(len(pre_balances), len(post_balances), len(account_keys))
        
        for i in range(min_length):
            pre = pre_balances[i]
            post = post_balances[i]
            account = account_keys[i]
            
            if post != pre:
                change = post - pre
                # Ignore very small changes that might be due to rent changes
                if abs(change) >= 10000:  # 0.00001 SOL threshold
                    transfers.append({
                        "account": account,
                        "pre_balance": pre,
                        "post_balance": post,
                        "change": change,
                        "change_sol": change / 1_000_000_000  # Convert lamports to SOL
                    })
        
        return transfers 