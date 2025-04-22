"""Wallet portfolio functionality for Solana MCP server."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from solana_mcp.cache import cache
from solana_mcp.defi import JupiterClient
from solana_mcp.solana_client import SolanaClient, validate_public_key


class PortfolioManager:
    """Manager for wallet portfolio functionality."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the portfolio manager.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
        self.jupiter_client = JupiterClient()
    
    async def close(self):
        """Close all client connections."""
        await self.jupiter_client.close()
    
    @cache(category="portfolio", ttl=60)
    async def get_wallet_balances(
        self,
        address: str,
        include_nfts: bool = True,
        include_positions: bool = True,
        include_prices: bool = True
    ) -> Dict[str, Any]:
        """Get all token balances for a wallet.
        
        Args:
            address: The wallet address
            include_nfts: Whether to include NFTs
            include_positions: Whether to include DeFi positions
            include_prices: Whether to include price data
            
        Returns:
            Wallet balance information
        """
        if not validate_public_key(address):
            raise ValueError(f"Invalid wallet address: {address}")
            
        result = {
            "address": address,
            "timestamp": datetime.now().isoformat(),
            "sol_balance": None,
            "tokens": [],
            "total_value_usd": 0.0,
            "nfts": [] if include_nfts else None,
            "positions": [] if include_positions else None
        }
        
        # Get native SOL balance
        try:
            sol_balance_lamports = await self.solana_client.get_balance(address)
            sol_balance = sol_balance_lamports / 1_000_000_000  # Convert to SOL
            result["sol_balance"] = {
                "amount": sol_balance,
                "raw_amount": sol_balance_lamports,
                "symbol": "SOL",
                "name": "Solana"
            }
            
            # Get SOL price if requested
            if include_prices:
                try:
                    # Solana mint address (native SOL)
                    sol_mint = "So11111111111111111111111111111111111111112"
                    sol_price_data = await self.jupiter_client.get_price(sol_mint)
                    if "price" in sol_price_data and sol_price_data["price"]:
                        sol_price = float(sol_price_data["price"])
                        sol_value = sol_balance * sol_price
                        result["sol_balance"]["price_usd"] = sol_price
                        result["sol_balance"]["value_usd"] = sol_value
                        result["total_value_usd"] += sol_value
                except Exception as e:
                    # Price fetch failure shouldn't fail the whole request
                    result["sol_balance"]["price_error"] = str(e)
        except Exception as e:
            result["sol_balance_error"] = str(e)
            
        # Get token accounts
        try:
            token_accounts = await self.solana_client.get_token_accounts_by_owner(address)
            
            # Process each token account
            for account in token_accounts.get("value", []):
                try:
                    account_data = account.get("account", {}).get("data", {})
                    parsed_data = account_data.get("parsed", {}).get("info", {})
                    
                    token_info = {
                        "mint": parsed_data.get("mint"),
                        "account": account.get("pubkey"),
                        "amount": parsed_data.get("tokenAmount", {}).get("amount"),
                        "ui_amount": parsed_data.get("tokenAmount", {}).get("uiAmount"),
                        "decimals": parsed_data.get("tokenAmount", {}).get("decimals"),
                        "symbol": "Unknown",  # Will be filled in if available
                        "name": "Unknown Token"  # Will be filled in if available
                    }
                    
                    # Get token metadata if available
                    if token_info["mint"] and include_prices:
                        try:
                            price_data = await self.jupiter_client.get_price(token_info["mint"])
                            if price_data and "price" in price_data and price_data["price"]:
                                token_info["price_usd"] = float(price_data["price"])
                                token_info["value_usd"] = token_info["ui_amount"] * token_info["price_usd"]
                                result["total_value_usd"] += token_info["value_usd"]
                                
                                # If we have price data, we might also have name/symbol
                                token_info["symbol"] = price_data.get("symbol", token_info["symbol"])
                                token_info["name"] = price_data.get("name", token_info["name"])
                        except Exception as e:
                            # Price fetch failure shouldn't fail the whole token
                            token_info["price_error"] = str(e)
                    
                    # Skip tokens with zero balance
                    if token_info.get("ui_amount", 0) > 0:
                        # Determine if it's likely an NFT
                        is_nft = (
                            token_info.get("decimals", 0) == 0 and
                            token_info.get("ui_amount", 0) == 1
                        )
                        
                        if is_nft and include_nfts:
                            # Add to NFTs list
                            result["nfts"].append(token_info)
                        else:
                            # Add to tokens list
                            result["tokens"].append(token_info)
                except Exception as e:
                    # Individual token processing error shouldn't fail the whole request
                    if "token_errors" not in result:
                        result["token_errors"] = []
                    result["token_errors"].append({
                        "account": account.get("pubkey"),
                        "error": str(e)
                    })
                    
        except Exception as e:
            result["token_accounts_error"] = str(e)
            
        # Get DeFi positions if requested
        if include_positions:
            # This is a placeholder - in a real implementation, you would
            # query various DeFi protocols for user positions
            result["positions"] = []
            
        return result
    
    @cache(category="portfolio", ttl=300)
    async def get_transaction_history(
        self,
        address: str,
        limit: int = 20,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Get transaction history for a wallet.
        
        Args:
            address: The wallet address
            limit: Maximum number of transactions to return
            include_details: Whether to include detailed transaction information
            
        Returns:
            Transaction history for the wallet
        """
        if not validate_public_key(address):
            raise ValueError(f"Invalid wallet address: {address}")
            
        # Get signatures for the address
        try:
            signatures = await self.solana_client.get_signatures_for_address(
                address,
                limit=limit
            )
            
            result = {
                "address": address,
                "transactions": signatures,
                "count": len(signatures)
            }
            
            # Get detailed transaction information if requested
            if include_details and signatures:
                details = []
                
                # Process up to 10 transactions to avoid overloading the RPC
                for signature_info in signatures[:min(10, len(signatures))]:
                    try:
                        signature = signature_info.get("signature")
                        if signature:
                            tx_details = await self.solana_client.get_transaction(signature)
                            details.append(tx_details)
                    except Exception as e:
                        # Individual transaction processing error shouldn't fail the whole request
                        if "tx_errors" not in result:
                            result["tx_errors"] = []
                        result["tx_errors"].append({
                            "signature": signature,
                            "error": str(e)
                        })
                        
                result["transaction_details"] = details
                
            return result
        except Exception as e:
            return {
                "address": address,
                "error": str(e),
                "transactions": []
            }
    
    @cache(category="portfolio", ttl=60)
    async def analyze_wallet(self, address: str) -> Dict[str, Any]:
        """Perform a comprehensive analysis of a wallet.
        
        Args:
            address: The wallet address
            
        Returns:
            Wallet analysis
        """
        if not validate_public_key(address):
            raise ValueError(f"Invalid wallet address: {address}")
            
        # Get wallet balances and transaction history concurrently
        balances_task = asyncio.create_task(
            self.get_wallet_balances(address, include_nfts=True, include_positions=True)
        )
        
        history_task = asyncio.create_task(
            self.get_transaction_history(address, limit=50, include_details=False)
        )
        
        # Wait for both tasks to complete
        balances, history = await asyncio.gather(balances_task, history_task)
        
        # Combine the results
        analysis = {
            "address": address,
            "timestamp": datetime.now().isoformat(),
            "balances": balances,
            "transaction_count": history.get("count", 0),
            "recent_transactions": history.get("transactions", [])[:5],
            "summary": {}
        }
        
        # Create a summary
        summary = {}
        
        # Portfolio value
        if "total_value_usd" in balances:
            summary["total_value_usd"] = balances["total_value_usd"]
        
        # Token counts
        token_count = len(balances.get("tokens", []))
        nft_count = len(balances.get("nfts", []))
        summary["token_count"] = token_count
        summary["nft_count"] = nft_count
        
        # Activity summary
        if "transactions" in history:
            # Simple activity classification based on recency
            recent_tx_count = sum(
                1 for tx in history["transactions"] 
                if datetime.fromisoformat(tx.get("blockTime", "2000-01-01")) 
                > datetime.now() - timedelta(days=7)
            )
            
            if recent_tx_count > 10:
                summary["activity_level"] = "High"
            elif recent_tx_count > 3:
                summary["activity_level"] = "Medium"
            else:
                summary["activity_level"] = "Low"
                
            summary["recent_tx_count"] = recent_tx_count
        
        # Add the summary to the analysis
        analysis["summary"] = summary
        
        return analysis


class NFTManager:
    """Manager for NFT-specific functionality."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the NFT manager.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
    
    @cache(category="nfts", ttl=300)
    async def get_nft_metadata(self, mint: str) -> Dict[str, Any]:
        """Get metadata for an NFT.
        
        Args:
            mint: The NFT mint address
            
        Returns:
            NFT metadata
        """
        if not validate_public_key(mint):
            raise ValueError(f"Invalid mint address: {mint}")
            
        # This is a simplified implementation. In practice, you would:
        # 1. Get the metadata account for the mint
        # 2. Parse the metadata to extract URI and other information
        # 3. Fetch the metadata JSON from the URI
        
        # For now, return a placeholder response
        try:
            # Get token metadata account
            metadata_program_id = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"
            
            # Get accounts owned by the metadata program that reference this mint
            # This is a simplified implementation
            metadata_accounts = await self.solana_client.get_program_accounts(
                metadata_program_id,
                filters=[
                    {
                        "memcmp": {
                            "offset": 33,  # Offset for mint in metadata accounts
                            "bytes": mint
                        }
                    }
                ]
            )
            
            if not metadata_accounts:
                return {
                    "mint": mint,
                    "error": "No metadata account found"
                }
                
            # Parse the metadata account data
            # This is a simplified implementation
            metadata_account = metadata_accounts[0]
            account_data = metadata_account.get("account", {}).get("data", [])
            
            # In practice, you'd properly parse the metadata account data
            # For now, return a placeholder
            return {
                "mint": mint,
                "metadata_account": metadata_account.get("pubkey"),
                "uri": f"https://example.com/metadata/{mint}.json",
                "name": f"NFT {mint[:6]}",
                "symbol": "NFT",
                "collection": "Unknown Collection",
                "attributes": []
            }
        except Exception as e:
            return {
                "mint": mint,
                "error": str(e)
            }
    
    @cache(category="nfts", ttl=300)
    async def get_nft_collection(self, collection_mint: str) -> Dict[str, Any]:
        """Get NFTs in a collection.
        
        Args:
            collection_mint: The collection mint address
            
        Returns:
            Collection information and NFTs
        """
        if not validate_public_key(collection_mint):
            raise ValueError(f"Invalid collection mint address: {collection_mint}")
            
        # This is a simplified implementation. In practice, you would:
        # 1. Get metadata accounts that are part of this collection
        # 2. Parse the metadata to extract information about each NFT
        
        # For now, return a placeholder response
        return {
            "collection_mint": collection_mint,
            "name": f"Collection {collection_mint[:6]}",
            "description": "A collection of NFTs",
            "total_items": 0,
            "items": []
        } 