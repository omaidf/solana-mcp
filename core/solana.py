"""
Solana blockchain client for the MCP server
"""
import os
import json
import base64
import asyncio
import logging
from typing import Dict, Any, Optional, List

from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

# Setup logger
logger = logging.getLogger(__name__)

class SolanaClient:
    """Client for interacting with the Solana blockchain"""
    
    NETWORKS = {
        "mainnet": "https://api.mainnet-beta.solana.com",
        "testnet": "https://api.testnet.solana.com",
        "devnet": "https://api.devnet.solana.com",
    }
    
    def __init__(self, network: str = "mainnet", custom_url: Optional[str] = None):
        """Initialize Solana client with network configuration"""
        # Use custom URL if provided, otherwise use predefined networks
        if custom_url:
            self.endpoint = custom_url
        else:
            self.endpoint = self.NETWORKS.get(network, self.NETWORKS["mainnet"])
            
        # Override with environment variable if set
        if os.getenv("SOLANA_RPC_URL"):
            self.endpoint = os.getenv("SOLANA_RPC_URL")
            
        # Log the endpoint (without API key)
        sanitized_endpoint = self.endpoint.split("?")[0]
        logger.info(f"Initializing Solana client with endpoint: {sanitized_endpoint}")
            
        self.client = AsyncClient(self.endpoint)
        self.network = network
        
    async def get_account_info(self, address: str) -> Dict[str, Any]:
        """Get account information for a Solana address"""
        try:
            pubkey = Pubkey.from_string(address)
            response = await self.client.get_account_info(pubkey, encoding="jsonParsed")
            
            if response.value is None:
                return {"error": "Account not found"}
            
            # Get data in appropriate format
            if hasattr(response.value, 'data') and isinstance(response.value.data, list) and len(response.value.data) >= 2:
                data_encoding = response.value.data[1]
                if data_encoding == "base64":
                    # Decode base64 data
                    data = base64.b64decode(response.value.data[0])
                    data_str = str(data)
                else:
                    # Use the data as is for jsonParsed
                    data_str = response.value.data[0]
            else:
                data_str = "Unknown data format"
                
            account_info = {
                "address": str(pubkey),
                "lamports": response.value.lamports,
                "owner": str(response.value.owner),
                "executable": response.value.executable,
                "rent_epoch": response.value.rent_epoch,
                "data_size": len(data_str) if isinstance(data_str, bytes) else "N/A",
                "data": data_str
            }
            
            return account_info
        except Exception as e:
            logger.exception(f"Error fetching account info for {address}")
            raise Exception(f"Error fetching account info: {str(e)}")
    
    async def get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get transaction details by signature"""
        try:
            sig = Signature.from_string(signature)
            response = await self.client.get_transaction(sig)
            
            if response.value is None:
                return {"error": "Transaction not found"}
                
            # Process and clean transaction data for the response
            tx_data = {
                "signature": signature,
                "slot": response.value.slot,
                "block_time": response.value.block_time,
                "success": not response.value.meta.err,
            }
            
            if response.value.meta:
                tx_data["fee"] = response.value.meta.fee
                tx_data["pre_balances"] = response.value.meta.pre_balances
                tx_data["post_balances"] = response.value.meta.post_balances
            
            if response.value.transaction:
                tx_data["message"] = self._process_transaction_message(response.value.transaction)
                
            return tx_data
        except Exception as e:
            logger.exception(f"Error fetching transaction {signature}")
            raise Exception(f"Error fetching transaction: {str(e)}")
    
    async def get_balance(self, address: str) -> Dict[str, Any]:
        """Get SOL balance for an address"""
        try:
            pubkey = Pubkey.from_string(address)
            response = await self.client.get_balance(pubkey)
            
            # Convert lamports to SOL with proper precision
            balance_sol = self.lamports_to_sol(response.value)
            
            return {
                "address": address,
                "balance": response.value,
                "balance_sol": balance_sol
            }
        except Exception as e:
            logger.exception(f"Error fetching balance for {address}")
            raise Exception(f"Error fetching balance: {str(e)}")
    
    async def get_token_accounts(self, address: str) -> List[Dict[str, Any]]:
        """Get token accounts for an address with proper decimal handling"""
        try:
            pubkey = Pubkey.from_string(address)
            
            # Use jsonParsed encoding for proper token data
            response = await self.client.get_token_accounts_by_owner(
                pubkey,
                {"programId": Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")},
                encoding="jsonParsed"
            )
            
            token_accounts = []
            for item in response.value:
                try:
                    # Extract data using jsonParsed format which has decimals info
                    if item.account.data and hasattr(item.account.data, '__getitem__') and item.account.data[1] == "jsonParsed":
                        parsed_data = item.account.data[0]
                        
                        if "parsed" in parsed_data and "info" in parsed_data["parsed"]:
                            info = parsed_data["parsed"]["info"]
                            
                            # Get token amount with proper decimal handling
                            raw_amount = 0
                            ui_amount = 0
                            decimals = 0
                            
                            if "tokenAmount" in info:
                                token_amount = info["tokenAmount"]
                                raw_amount = int(token_amount.get("amount", "0"))
                                ui_amount = float(token_amount.get("uiAmount", 0))
                                decimals = int(token_amount.get("decimals", 0))
                            
                            # Build token account data
                            token_account = {
                                "address": str(item.pubkey),
                                "mint": info.get("mint", ""),
                                "owner": info.get("owner", ""),
                                "amount": raw_amount,
                                "ui_amount": ui_amount,
                                "decimals": decimals,
                                "state": info.get("state", ""),
                            }
                            
                            token_accounts.append(token_account)
                    else:
                        # Fallback to raw decoding if jsonParsed is not available
                        account_data = base64.b64decode(item.account.data[0]) if isinstance(item.account.data, list) else b""
                        
                        # This is a simplified approach that won't work for all token programs
                        # A complete implementation should handle different token program layouts
                        if len(account_data) >= 72:  # Minimum size for a token account
                            mint_bytes = account_data[0:32]
                            owner_bytes = account_data[32:64]
                            amount_bytes = account_data[64:72]
                            
                            mint = str(Pubkey(bytes(mint_bytes)))
                            owner = str(Pubkey(bytes(owner_bytes)))
                            amount = int.from_bytes(amount_bytes, byteorder="little")
                            
                            token_accounts.append({
                                "address": str(item.pubkey),
                                "mint": mint,
                                "owner": owner,
                                "amount": amount,
                                "ui_amount": amount,  # No decimals info available
                                "decimals": 0,
                                "state": "Unknown",
                            })
                except Exception as e:
                    logger.warning(f"Error processing token account {item.pubkey}: {str(e)}")
                    # Skip this account and continue with others
                    
            return token_accounts
        except Exception as e:
            logger.exception(f"Error fetching token accounts for {address}")
            raise Exception(f"Error fetching token accounts: {str(e)}")
    
    def _process_transaction_message(self, transaction: VersionedTransaction) -> Dict[str, Any]:
        """Process transaction message data"""
        # This is a simplified implementation
        # In a production environment, you would want to fully decode the transaction
        if hasattr(transaction, 'message'):
            return {
                "accounts": [str(pk) for pk in transaction.message.account_keys],
                "instructions": len(transaction.message.instructions),
                # Additional message processing would go here
            }
        return {}
        
    @staticmethod
    def lamports_to_sol(lamports: int) -> float:
        """Convert lamports to SOL with proper precision"""
        return lamports / 1_000_000_000
        
    @staticmethod
    def sol_to_lamports(sol: float) -> int:
        """Convert SOL to lamports with proper precision"""
        return int(sol * 1_000_000_000) 