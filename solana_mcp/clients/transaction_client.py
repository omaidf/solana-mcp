"""Transaction client module for Solana MCP.

This module provides a client for transaction-related operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from solana.rpc.api import Pubkey
from solana.transaction import Transaction

from solana_mcp.models.enhanced_transaction import EnhancedTransaction
from solana_mcp.models.token_transfer import TokenTransfer
from solana_mcp.services.rpc_service import RPCService
from solana_mcp.solana_client import SolanaClient
from solana_mcp.utils.constants import ZERO_PUBKEY
from solana_mcp.utils.dependency_injection import inject_by_type
from solana_mcp.utils.error_handling import (
    SolanaMCPError,
    TransactionError,
    ValidationError,
    DataError,
    ErrorCode,
    handle_async_exceptions,
    handle_async_data_exceptions,
    map_rpc_errors,
    validate_async_parameters
)
from solana_mcp.utils.transaction_parser import (
    extract_sol_transfers,
    extract_token_transfers,
    parse_instruction_logs,
)

from solana_mcp.clients.base_client import BaseSolanaClient
from solana_mcp.logging_config import get_logger
from solana_mcp.constants import SYSTEM_PROGRAM_ID, TOKEN_PROGRAM_ID, PROGRAM_NAMES

# Get logger
logger = get_logger(__name__)

class TransactionClient:
    """Client for transaction-related operations."""

    @inject_by_type
    def __init__(self, solana_client: Optional[SolanaClient] = None, rpc_service: Optional[RPCService] = None):
        """Initialize transaction client.

        Args:
            solana_client: Solana client instance (injected)
            rpc_service: RPC service instance (injected)
        """
        self.solana_client = solana_client
        self.rpc_service = rpc_service
        self.logger = logger

    @validate_async_parameters
    @map_rpc_errors
    @handle_async_exceptions(
        (ValueError, TransactionError),
        log_level=logging.WARNING,
        reraise=False,
        default_error=TransactionError
    )
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
            TransactionError: If transaction not found or for any other transaction-related error
        """
        # Validate transaction signature
        if not signature or not isinstance(signature, str) or len(signature) < 32:
            raise ValidationError(
                f"Invalid transaction signature: {signature}",
                details={"signature": signature}
            )
            
        # Get transaction from RPC
        tx_response = await self.rpc_service.get_transaction(signature)
        if not tx_response or not tx_response.get("transaction"):
            raise TransactionError(
                f"Transaction not found: {signature}",
                ErrorCode.NOT_IMPLEMENTED_ERROR,
                details={"signature": signature}
            )

        # Parse transaction
        return await self.parse_transaction(tx_response)

    @validate_async_parameters
    @map_rpc_errors
    @handle_async_exceptions(
        default_error=TransactionError,
        log_level=logging.WARNING
    )
    async def get_block(self, slot: int) -> Dict[str, Any]:
        """Get block details.

        Args:
            slot: Block slot

        Returns:
            Block details
        
        Raises:
            TransactionError: For any transaction-related error
        """
        if not isinstance(slot, int) or slot < 0:
            raise ValidationError(f"Invalid slot number: {slot}", details={"slot": slot})
            
        block = await self.rpc_service.get_block(slot)
        return block

    @validate_async_parameters
    @map_rpc_errors
    @handle_async_exceptions(
        default_error=TransactionError,
        log_level=logging.WARNING
    )
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
        if not isinstance(start_slot, int) or start_slot < 0:
            raise ValidationError(f"Invalid start slot: {start_slot}", details={"start_slot": start_slot})
            
        if end_slot is not None and (not isinstance(end_slot, int) or end_slot < start_slot):
            raise ValidationError(
                f"Invalid end slot: {end_slot}",
                details={"start_slot": start_slot, "end_slot": end_slot}
            )
            
        blocks = await self.rpc_service.get_blocks(start_slot, end_slot)
        return blocks

    @handle_async_data_exceptions
    async def parse_transaction(self, tx_data: Dict[str, Any]) -> EnhancedTransaction:
        """Parse transaction data into enhanced transaction model.

        Args:
            tx_data: Transaction data from RPC

        Returns:
            Enhanced transaction object
        
        Raises:
            DataError: If there is an error parsing the transaction data
            TransactionError: For any other transaction-related error
        """
        if not tx_data or not isinstance(tx_data, dict):
            raise DataError(
                "Invalid transaction data structure",
                ErrorCode.PARSING_ERROR,
                details={"data_type": type(tx_data).__name__}
            )
            
        # Extract transaction data
        meta = tx_data.get("meta", {})
        transaction = tx_data.get("transaction", {})

        if not transaction:
            raise DataError(
                "Missing transaction information in transaction data",
                ErrorCode.PARSING_ERROR,
                details={"available_keys": list(tx_data.keys())}
            )

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
        message = transaction.get("message", {})
        account_keys_raw = message.get("accountKeys", [])
        
        try:
            account_keys = [Pubkey.from_string(key) for key in account_keys_raw]
        except ValueError as e:
            raise DataError(
                f"Invalid public key in transaction: {str(e)}",
                ErrorCode.PARSING_ERROR,
                details={"account_keys": account_keys_raw}
            )

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
        self, meta: Dict[str, Any], account_keys: List[Pubkey]
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