"""Transaction-related Solana RPC client operations.

This module provides specialized client functionality for Solana transaction operations.
"""

import re
from typing import Dict, List, Any, Optional

from solana_mcp.clients.base_client import BaseSolanaClient, InvalidPublicKeyError, validate_public_key
from solana_mcp.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

class TransactionClient(BaseSolanaClient):
    """Client for Solana transaction operations."""
    
    async def get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get transaction details.
        
        Args:
            signature: The transaction signature
            
        Returns:
            Transaction details
            
        Raises:
            ValueError: If the signature format is invalid
        """
        # Validate transaction signature format - base58 encoded signatures
        # should be alphanumeric and typical length is 88 characters, but allow some flexibility
        if not signature or not isinstance(signature, str):
            raise ValueError(f"Transaction signature must be a non-empty string")
        
        # Use a more general validation for base58 encoded data
        if not re.match(r"^[1-9A-HJ-NP-Za-km-z]{43,128}$", signature):
            raise ValueError(f"Invalid transaction signature format: {signature}")
        
        return await self._make_request(
            "getTransaction", 
            [signature, {"encoding": "json"}]
        )
    
    async def get_block(self, slot: int) -> Dict[str, Any]:
        """Get information about a block.
        
        Args:
            slot: The slot number
            
        Returns:
            Block information
        """
        return await self._make_request(
            "getBlock",
            [slot, {"encoding": "json", "transactionDetails": "full", "rewards": True}]
        )
    
    async def get_blocks(
        self, 
        start_slot: int, 
        end_slot: Optional[int] = None,
        commitment: Optional[str] = None
    ) -> List[int]:
        """Get a list of confirmed blocks.
        
        Args:
            start_slot: Start slot (inclusive)
            end_slot: End slot (inclusive, optional)
            commitment: Commitment level
            
        Returns:
            List of block slots
        """
        params = [start_slot]
        if end_slot is not None:
            params.append(end_slot)
        if commitment:
            params.append({"commitment": commitment})
        
        return await self._make_request("getBlocks", params)
    
    async def parse_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a transaction into a more human-readable format.
        
        Args:
            transaction_data: The transaction data from get_transaction
            
        Returns:
            Parsed transaction data
        """
        if not transaction_data:
            return {"error": "No transaction data provided"}
        
        try:
            # Extract basic transaction info
            result = {}
            
            # Check if we have a proper transaction response
            if "meta" not in transaction_data and "transaction" not in transaction_data:
                return {"error": "Invalid transaction data format"}
            
            # Get basic transaction data
            result["signature"] = transaction_data.get("transaction", {}).get("signatures", ["unknown"])[0]
            result["slot"] = transaction_data.get("slot", 0)
            result["block_time"] = transaction_data.get("blockTime", None)
            
            # Check transaction status
            meta = transaction_data.get("meta", {})
            if meta.get("err") is not None:
                result["status"] = "failed"
                result["error"] = meta.get("err")
            else:
                result["status"] = "success"
            
            # Get transaction fee
            result["fee"] = meta.get("fee", 0)
            
            # Extract instructions
            instructions = []
            tx = transaction_data.get("transaction", {})
            message = tx.get("message", {})
            
            # Extract accounts referenced in the transaction
            account_keys = []
            for key in message.get("accountKeys", []):
                if isinstance(key, str):
                    account_keys.append(key)
                elif isinstance(key, dict) and "pubkey" in key:
                    account_keys.append(key["pubkey"])
            
            # Process instructions
            for idx, instruction in enumerate(message.get("instructions", [])):
                program_id_idx = instruction.get("programIdIndex")
                program_id = account_keys[program_id_idx] if 0 <= program_id_idx < len(account_keys) else "unknown"
                
                # Get accounts used by this instruction
                accounts = []
                for acc_idx in instruction.get("accounts", []):
                    if 0 <= acc_idx < len(account_keys):
                        accounts.append(account_keys[acc_idx])
                
                # Add instruction data
                instr_data = {
                    "program_id": program_id,
                    "accounts": accounts,
                    "data": instruction.get("data", "")
                }
                
                # Add program name if we recognize it
                instr_data["program_name"] = self._get_program_name(program_id)
                
                # Try to identify instruction type based on program and data
                instr_data["type"] = self._identify_instruction_type(program_id, instruction.get("data", ""), accounts)
                
                instructions.append(instr_data)
            
            result["instructions"] = instructions
            
            # Extract token balances if available
            if "preTokenBalances" in meta and "postTokenBalances" in meta:
                result["token_transfers"] = self._extract_token_transfers(
                    meta.get("preTokenBalances", []),
                    meta.get("postTokenBalances", []),
                    account_keys
                )
            
            # Extract SOL transfers from account balance changes
            if "preBalances" in meta and "postBalances" in meta:
                result["sol_transfers"] = self._extract_sol_transfers(
                    meta.get("preBalances", []),
                    meta.get("postBalances", []),
                    account_keys
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing transaction: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to parse transaction: {str(e)}",
                "raw_data": transaction_data
            }
    
    def _get_program_name(self, program_id: str) -> str:
        """Get a human-readable name for a program ID.
        
        Args:
            program_id: The program ID
            
        Returns:
            Program name or 'Unknown Program'
        """
        program_names = {
            "11111111111111111111111111111111": "System Program",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "Token Program",
            "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL": "Token Associated Program",
            "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s": "Metaplex Metadata",
            "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "Jupiter Aggregator",
            "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca Program",
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium Program",
            "So11111111111111111111111111111111111111112": "Wrapped SOL",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC Mint",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT Mint",
        }
        return program_names.get(program_id, "Unknown Program")
    
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
        if program_id == "11111111111111111111111111111111":
            # First byte of data indicates instruction type
            if data.startswith("3Bxs"):  # Example: Transfer instruction starts with 3Bxs
                return "Transfer"
            if data.startswith("2"):  # Example: Create Account starts with 2
                return "CreateAccount"
            return "SystemInstruction"
            
        # Token program instructions
        elif program_id == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
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