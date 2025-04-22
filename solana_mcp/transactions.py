"""Transaction building and signing for Solana MCP server."""

import base64
import json
from typing import Any, Dict, List, Optional, Union, Tuple

from solana_mcp.solana_client import SolanaClient, validate_public_key


class TransactionBuilder:
    """Builder for Solana transactions."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the transaction builder.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
    
    async def build_transfer_transaction(
        self,
        from_address: str,
        to_address: str,
        lamports: int,
        recent_blockhash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a SOL transfer transaction.
        
        Args:
            from_address: Sender address
            to_address: Recipient address
            lamports: Amount in lamports
            recent_blockhash: Recent blockhash to use. If None, fetches one.
            
        Returns:
            Transaction data
        """
        # Validate addresses
        if not validate_public_key(from_address):
            raise ValueError(f"Invalid from_address: {from_address}")
        if not validate_public_key(to_address):
            raise ValueError(f"Invalid to_address: {to_address}")
        
        # Get recent blockhash if not provided
        if not recent_blockhash:
            blockhash_resp = await self.solana_client.get_recent_blockhash()
            recent_blockhash = blockhash_resp["blockhash"]
        
        # This is a simplified implementation that returns the transaction data
        # In a real implementation, you'd use the Solana SDK to build the transaction
        # For now, we're returning a structure that could be used with solana-web3.js
        
        # Simple transfer transaction structure
        transaction = {
            "recentBlockhash": recent_blockhash,
            "feePayer": from_address,
            "instructions": [
                {
                    "programId": "11111111111111111111111111111111",  # System program
                    "accounts": [
                        {"pubkey": from_address, "isSigner": True, "isWritable": True},
                        {"pubkey": to_address, "isSigner": False, "isWritable": True}
                    ],
                    "data": f"transfer:{lamports}"  # In real implementation, this would be properly encoded
                }
            ]
        }
        
        return transaction
    
    async def build_token_transfer_transaction(
        self,
        from_address: str,
        to_address: str,
        mint: str,
        amount: int,
        decimals: int = 9,
        create_associated_token_account: bool = True,
        recent_blockhash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a token transfer transaction.
        
        Args:
            from_address: Sender address
            to_address: Recipient address
            mint: Token mint address
            amount: Amount of tokens (in smallest units)
            decimals: Token decimals
            create_associated_token_account: Whether to create ATA if needed
            recent_blockhash: Recent blockhash to use. If None, fetches one.
            
        Returns:
            Transaction data
        """
        # Validate addresses
        if not validate_public_key(from_address):
            raise ValueError(f"Invalid from_address: {from_address}")
        if not validate_public_key(to_address):
            raise ValueError(f"Invalid to_address: {to_address}")
        if not validate_public_key(mint):
            raise ValueError(f"Invalid mint: {mint}")
        
        # Get recent blockhash if not provided
        if not recent_blockhash:
            blockhash_resp = await self.solana_client.get_recent_blockhash()
            recent_blockhash = blockhash_resp["blockhash"]
            
        # In a real implementation, you'd:
        # 1. Check if source token account exists
        # 2. Check if destination token account exists
        # 3. If not, add instruction to create associated token account
        # 4. Add token transfer instruction
        
        # This is a simplified version
        transaction = {
            "recentBlockhash": recent_blockhash,
            "feePayer": from_address,
            "instructions": [
                {
                    "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
                    "accounts": [
                        {"pubkey": f"source_token_account_{from_address}_{mint}", "isSigner": False, "isWritable": True},
                        {"pubkey": f"destination_token_account_{to_address}_{mint}", "isSigner": False, "isWritable": True},
                        {"pubkey": from_address, "isSigner": True, "isWritable": False}
                    ],
                    "data": f"transfer:{amount}"  # In real implementation, this would be properly encoded
                }
            ]
        }
        
        # If creating associated token account is needed
        if create_associated_token_account:
            transaction["instructions"].insert(0, {
                "programId": "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # Associated token program
                "accounts": [
                    {"pubkey": from_address, "isSigner": True, "isWritable": True},
                    {"pubkey": f"destination_token_account_{to_address}_{mint}", "isSigner": False, "isWritable": True},
                    {"pubkey": to_address, "isSigner": False, "isWritable": False},
                    {"pubkey": mint, "isSigner": False, "isWritable": False},
                    {"pubkey": "11111111111111111111111111111111", "isSigner": False, "isWritable": False},
                    {"pubkey": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", "isSigner": False, "isWritable": False},
                    {"pubkey": "SysvarRent111111111111111111111111111111111", "isSigner": False, "isWritable": False}
                ],
                "data": "create"  # In real implementation, this would be properly encoded
            })
        
        return transaction
    
    async def build_nft_mint_transaction(
        self,
        creator_address: str,
        recipient_address: str,
        metadata_uri: str,
        name: str,
        symbol: str,
        recent_blockhash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build an NFT minting transaction (using Metaplex standards).
        
        Args:
            creator_address: Creator/minter address
            recipient_address: Recipient address
            metadata_uri: URI for NFT metadata
            name: NFT name
            symbol: NFT symbol
            recent_blockhash: Recent blockhash to use. If None, fetches one.
            
        Returns:
            Transaction data
        """
        # Validate addresses
        if not validate_public_key(creator_address):
            raise ValueError(f"Invalid creator_address: {creator_address}")
        if not validate_public_key(recipient_address):
            raise ValueError(f"Invalid recipient_address: {recipient_address}")
        
        # Get recent blockhash if not provided
        if not recent_blockhash:
            blockhash_resp = await self.solana_client.get_recent_blockhash()
            recent_blockhash = blockhash_resp["blockhash"]
            
        # This is a simplified placeholder
        # In a real implementation, you'd use the Metaplex SDK to build the transaction
        transaction = {
            "recentBlockhash": recent_blockhash,
            "feePayer": creator_address,
            "instructions": [
                # Create mint account
                {
                    "programId": "11111111111111111111111111111111",  # System program
                    "accounts": [],
                    "data": "createMint"
                },
                # Initialize mint
                {
                    "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
                    "accounts": [],
                    "data": "initializeMint"
                },
                # Create metadata
                {
                    "programId": "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s",  # Metadata program
                    "accounts": [],
                    "data": {
                        "name": name,
                        "symbol": symbol,
                        "uri": metadata_uri,
                        "sellerFeeBasisPoints": 0,
                        "creators": [
                            {
                                "address": creator_address,
                                "verified": True,
                                "share": 100
                            }
                        ]
                    }
                },
                # Create token account
                {
                    "programId": "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # Associated token program
                    "accounts": [],
                    "data": "createATA"
                },
                # Mint to
                {
                    "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
                    "accounts": [],
                    "data": "mintTo"
                }
            ]
        }
        
        return transaction
    
    async def serialize_transaction(self, transaction: Dict[str, Any]) -> str:
        """Serialize a transaction to base64 string.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Base64 encoded transaction
        """
        # This is a placeholder - in a real implementation you'd use Solana SDK
        # to properly serialize the transaction
        return base64.b64encode(json.dumps(transaction).encode()).decode()
    
    async def deserialize_transaction(self, serialized_transaction: str) -> Dict[str, Any]:
        """Deserialize a transaction from base64 string.
        
        Args:
            serialized_transaction: Base64 encoded transaction
            
        Returns:
            Transaction data
        """
        # This is a placeholder - in a real implementation you'd use Solana SDK
        # to properly deserialize the transaction
        return json.loads(base64.b64decode(serialized_transaction).decode())
    
    async def sign_transaction(
        self, 
        transaction: Dict[str, Any], 
        keypair_bytes: bytes
    ) -> Dict[str, Any]:
        """Sign a transaction (placeholder implementation).
        
        Args:
            transaction: Transaction data
            keypair_bytes: Ed25519 keypair bytes
            
        Returns:
            Signed transaction data
        """
        # This is a placeholder - in a real implementation you'd use Solana SDK
        # to properly sign the transaction
        transaction["signatures"] = {
            "dummy_signature": "signed_transaction_placeholder"
        }
        return transaction
    
    async def send_transaction(
        self, 
        serialized_transaction: str, 
        skip_preflight: bool = False
    ) -> str:
        """Send a transaction to the Solana network.
        
        Args:
            serialized_transaction: Base64 encoded transaction
            skip_preflight: Whether to skip preflight checks
            
        Returns:
            Transaction signature
        """
        # Make the RPC call to send the transaction
        result = await self.solana_client._make_request(
            "sendTransaction",
            [
                serialized_transaction,
                {
                    "skipPreflight": skip_preflight,
                    "encoding": "base64"
                }
            ]
        )
        
        return result 