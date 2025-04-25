"""
Token service for the Solana MCP.

This module provides a service for retrieving and managing token data
from the Solana blockchain, including token info, metadata, and price data.
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from functools import lru_cache

import httpx
from fastapi import Depends
from pydantic import BaseModel

from solana_mcp.models.token import TokenInfo, TokenMetadata, TokenPrice
from solana_mcp.services.base_service import BaseService, handle_errors
from solana_mcp.services.cache_service import CacheService, get_cache_service
from solana_mcp.utils.errors import (DataParsingError, RpcConnectionError,
                                     RpcError, ResourceNotFoundError, DataNotFoundError, ExternalServiceError)
from solana_mcp.utils.batch_processor import batch_process_requests, BatchProcessor
from solana_mcp.services.rpc_service import RPCService
from solana_mcp.utils.api_response import ValidationError

# Constants for Solana tokens
SPL_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

logger = logging.getLogger(__name__)


class TokenService(BaseService):
    """Service for retrieving and managing token data."""
    
    def __init__(
        self,
        rpc_service: RPCService,
        timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 100
    ):
        """
        Initialize the token service.
        
        Args:
            rpc_service: The RPC service for Solana API calls
            timeout: Default timeout for operations in seconds
            logger: Optional logger instance
            batch_size: Maximum batch size for RPC requests
        """
        super().__init__(timeout=timeout, logger=logger)
        self.rpc_service = rpc_service
        self.batch_size = batch_size
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.logger.info("TokenService initialized")
    
    async def start(self) -> None:
        """Start the token service."""
        self.logger.info(f"Token service started")
    
    async def stop(self) -> None:
        """Stop the token service."""
        self.logger.info("Token service stopped")
    
    async def _make_rpc_request(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Make an RPC request to the Solana API.
        
        Args:
            method: RPC method name
            params: RPC method parameters
            
        Returns:
            RPC response data
            
        Raises:
            RpcConnectionError: If there is a connection error
            RpcError: If the RPC request fails
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            async with self.log_timing(f"rpc_request.{method}"):
                response = await self.client.post(
                    self.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as e:
            raise RpcError(
                message=f"RPC request failed with status {e.response.status_code}",
                details={"method": method, "status_code": e.response.status_code}
            )
        except httpx.RequestError as e:
            raise RpcConnectionError(
                message=f"Connection error during RPC request: {str(e)}",
                details={"method": method, "error": str(e)}
            )
        except Exception as e:
            raise RpcError(
                message=f"Unexpected error during RPC request: {str(e)}",
                details={"method": method, "error": str(e)}
            )
        
        if "error" in result:
            raise RpcError(
                message=f"RPC error: {result['error'].get('message', 'Unknown error')}",
                details={"method": method, "error": result["error"]}
            )
        
        return result.get("result", {})
    
    async def list_tokens(self, offset: int = 0, limit: int = 10) -> List[TokenInfo]:
        """
        Get a paginated list of tokens.

        Args:
            offset: The pagination offset (starting index)
            limit: The maximum number of items to return

        Returns:
            A list of token information objects
        """
        cache_key = f"tokens:list:{offset}:{limit}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return [TokenInfo.model_validate(token) for token in cached_result]

        try:
            # Get tokens from Solana RPC
            response = await self._make_rpc_request("getTokenAccountsByOwner", {
                "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program ID
                "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                "encoding": "jsonParsed",
                "commitment": "finalized"
            })

            result = response.get("result", {})
            token_accounts = result.get("value", [])

            # Process token accounts to get token info
            tokens = []
            for account in token_accounts[offset:offset+limit]:
                parsed_info = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                mint = parsed_info.get("mint")
                
                if mint:
                    # Get additional token info
                    token_info = await self.get_token_info(mint)
                    tokens.append(token_info)

            # Cache the results
            await self.cache_service.set(
                cache_key, 
                [token.model_dump() for token in tokens], 
                ttl=300  # 5 minutes
            )
            
            return tokens
        except Exception as e:
            logger.error(f"Error fetching token list: {str(e)}")
            self._handle_error(e, "Failed to fetch token list")
            return []
    
    async def get_token_info(self, address: str) -> TokenInfo:
        """
        Get detailed information about a specific token.

        Args:
            address: The token's mint address

        Returns:
            Detailed token information

        Raises:
            DataNotFoundError: If the token is not found
        """
        cache_key = f"token:info:{address}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return TokenInfo.model_validate(cached_result)

        try:
            # Get token info from Solana RPC
            response = await self._make_rpc_request("getTokenSupply", {
                "mint": address,
                "commitment": "finalized"
            })

            if not response.get("result"):
                raise DataNotFoundError(f"Token with address {address} not found")

            # Get metadata for the token
            metadata = await self.get_token_metadata(address)
            
            # Get price data if available
            price = None
            try:
                price = await self.get_token_price(address)
            except Exception as e:
                logger.warning(f"Error fetching price for token {address}: {str(e)}")
                # Continue without price data

            # Create the token info object
            supply_data = response.get("result", {}).get("value", {})
            decimals = supply_data.get("decimals", 0)
            amount = supply_data.get("amount", "0")
            
            token_info = TokenInfo(
                address=address,
                name=metadata.name if metadata else None,
                symbol=metadata.symbol if metadata else None,
                decimals=decimals,
                total_supply=amount,
                metadata=metadata,
                price=price,
                is_nft=metadata.is_nft if metadata else False
            )

            # Cache the result
            await self.cache_service.set(
                cache_key, 
                token_info.model_dump(), 
                ttl=300  # 5 minutes
            )
            
            return token_info
        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error fetching token info for {address}: {str(e)}")
            self._handle_error(e, f"Failed to fetch token info for {address}")
    
    @handle_errors()
    async def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """
        Get metadata for a specific token.
        
        Args:
            token_address: The token mint address
            
        Returns:
            Token metadata including name, symbol, decimals, etc.
            
        Raises:
            ValidationError: If the token address is invalid
            DataNotFoundError: If the token metadata cannot be found
        """
        if not token_address or len(token_address) < 32:
            raise ValidationError(f"Invalid token address: {token_address}")
            
        try:
            return await self.with_timeout(
                self.rpc_service.get_token_metadata(token_address)
            )
        except Exception as e:
            self.logger.error(f"Failed to get token metadata for {token_address}: {str(e)}")
            raise DataNotFoundError(f"Token metadata not found for {token_address}")
    
    async def get_token_price(self, address: str) -> TokenPrice:
        """
        Get price information for a specific token.

        Args:
            address: The token's mint address

        Returns:
            Token price information

        Raises:
            DataNotFoundError: If the token price is not found
        """
        cache_key = f"token:price:{address}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return TokenPrice.model_validate(cached_result)

        try:
            # This would typically use a price API like CoinGecko or similar
            # For this example, we'll simulate a price response
            
            # In a real implementation, you would make an external API call here
            # Example:
            # response = await self._make_external_request(
            #     "GET", 
            #     f"https://api.coingecko.com/api/v3/coins/solana/contract/{address}"
            # )
            
            # Simulated price data
            import time
            import random
            
            # Generate some realistic-looking but random price data
            # In a real implementation, this would come from an external price API
            base_price = random.uniform(0.01, 100.0)
            if base_price > 10:
                base_price = round(base_price, 2)
            elif base_price > 1:
                base_price = round(base_price, 4)
            else:
                base_price = round(base_price, 6)
                
            change_pct = random.uniform(-15.0, 15.0)
            
            price_data = {
                "price_usd": base_price,
                "price_sol": base_price / 25.0,  # Assuming SOL is around $25
                "price_change_24h": change_pct,
                "volume_24h": random.uniform(10000, 10000000),
                "market_cap": base_price * random.uniform(1000000, 100000000),
                "last_updated": int(time.time())
            }
            
            token_price = TokenPrice(**price_data)

            # Cache the result with a shorter TTL since prices change frequently
            await self.cache_service.set(
                cache_key, 
                token_price.model_dump(), 
                ttl=60  # 1 minute
            )
            
            return token_price
        except Exception as e:
            logger.error(f"Error fetching token price for {address}: {str(e)}")
            self._handle_error(e, f"Failed to fetch token price for {address}")
    
    async def get_tokens_by_owner(self, owner_address: str) -> List[TokenInfo]:
        """
        Get all tokens owned by a specific wallet address.

        Args:
            owner_address: The owner's wallet address

        Returns:
            A list of token information for tokens owned by the address
        """
        cache_key = f"tokens:owner:{owner_address}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return [TokenInfo.model_validate(token) for token in cached_result]

        try:
            # Get token accounts owned by the address
            response = await self._make_rpc_request("getTokenAccountsByOwner", {
                "owner": owner_address,
                "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                "encoding": "jsonParsed",
                "commitment": "finalized"
            })

            result = response.get("result", {})
            token_accounts = result.get("value", [])

            # Extract mint addresses
            mint_addresses = []
            for account in token_accounts:
                parsed_info = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                mint = parsed_info.get("mint")
                if mint:
                    mint_addresses.append(mint)

            # Batch process token info requests
            async def fetch_token_info(mint_address):
                try:
                    return await self.get_token_info(mint_address)
                except Exception:
                    logger.warning(f"Failed to fetch info for token {mint_address}")
                    return None

            tokens = await batch_process_requests(
                fetch_token_info,
                mint_addresses,
                batch_size=5,
                concurrency=3
            )

            # Filter out None values (failed requests)
            tokens = [token for token in tokens if token is not None]

            # Cache the results
            await self.cache_service.set(
                cache_key, 
                [token.model_dump() for token in tokens], 
                ttl=300  # 5 minutes
            )
            
            return tokens
        except Exception as e:
            logger.error(f"Error fetching tokens for owner {owner_address}: {str(e)}")
            self._handle_error(e, f"Failed to fetch tokens for owner {owner_address}")
            return []
    
    async def search_tokens(self, query: str, limit: int = 10) -> List[TokenInfo]:
        """
        Search for tokens by name, symbol, or address.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            A list of matching token information
        """
        query = query.lower()
        
        try:
            # First, try to get exact match by address
            if len(query) >= 32:
                try:
                    token = await self.get_token_info(query)
                    return [token]
                except DataNotFoundError:
                    pass  # Continue with search
            
            # Get a list of tokens to search through
            all_tokens = await self.list_tokens(limit=100)
            
            # Filter tokens by query
            matching_tokens = []
            for token in all_tokens:
                if (token.name and query in token.name.lower()) or \
                   (token.symbol and query in token.symbol.lower()) or \
                   (token.address and query in token.address.lower()):
                    matching_tokens.append(token)
                    
                if len(matching_tokens) >= limit:
                    break
            
            return matching_tokens
        except Exception as e:
            logger.error(f"Error searching tokens with query '{query}': {str(e)}")
            self._handle_error(e, f"Failed to search tokens with query '{query}'")
            return []
    
    async def _find_metadata_pda(self, mint_address: str) -> str:
        """
        Find the metadata PDA for a given mint address.
        
        This is a simplified implementation. In a real system, this would
        compute the actual PDA using the Metaplex metadata program.
        
        Args:
            mint_address: The token's mint address
            
        Returns:
            The metadata account address
        """
        # In a real implementation, you would compute this using:
        # PublicKey.findProgramAddress(
        #     [Buffer.from('metadata'), METADATA_PROGRAM_ID.toBytes(), mint.toBytes()],
        #     METADATA_PROGRAM_ID
        # )
        
        # This is a placeholder that just appends 'metadata' to the mint address
        # In a real implementation, use the proper PDA derivation
        return f"{mint_address}_metadata"

    @handle_errors()
    async def get_token_accounts(self, owner_address: str) -> List[Dict[str, Any]]:
        """
        Get all token accounts for a wallet address.
        
        Args:
            owner_address: The wallet address to get token accounts for
            
        Returns:
            List of token accounts with balances
            
        Raises:
            ValidationError: If the owner address is invalid
        """
        if not owner_address or len(owner_address) < 32:
            raise ValidationError(f"Invalid owner address: {owner_address}")
            
        return await self.with_timeout(
            self.rpc_service.get_token_accounts_by_owner(owner_address)
        )

    @handle_errors()
    async def get_token_supply(self, token_address: str) -> Dict[str, Any]:
        """
        Get total supply information for a token.
        
        Args:
            token_address: The token mint address
            
        Returns:
            Supply information including amount and decimals
            
        Raises:
            ValidationError: If the token address is invalid
            DataNotFoundError: If the token supply cannot be found
        """
        if not token_address or len(token_address) < 32:
            raise ValidationError(f"Invalid token address: {token_address}")
            
        try:
            return await self.with_timeout(
                self.rpc_service.get_token_supply(token_address)
            )
        except Exception as e:
            self.logger.error(f"Failed to get token supply for {token_address}: {str(e)}")
            raise DataNotFoundError(f"Token supply not found for {token_address}")

    @handle_errors()
    async def get_token_largest_accounts(self, token_address: str) -> List[Dict[str, Any]]:
        """
        Get the largest accounts holding a specific token.
        
        Args:
            token_address: The token mint address
            
        Returns:
            List of largest token accounts
            
        Raises:
            ValidationError: If the token address is invalid
        """
        if not token_address or len(token_address) < 32:
            raise ValidationError(f"Invalid token address: {token_address}")
            
        return await self.with_timeout(
            self.rpc_service.get_token_largest_accounts(token_address)
        )

    @handle_errors()
    async def get_multiple_token_metadata(self, token_addresses: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for multiple tokens in an efficient batch.
        
        Args:
            token_addresses: List of token mint addresses
            
        Returns:
            Dictionary mapping token addresses to their metadata
            
        Raises:
            ValidationError: If any token address is invalid
        """
        # Validate addresses
        invalid_addresses = [addr for addr in token_addresses if not addr or len(addr) < 32]
        if invalid_addresses:
            raise ValidationError(f"Invalid token addresses: {invalid_addresses}")
        
        # Process in batches
        result: Dict[str, Dict[str, Any]] = {}
        
        async def fetch_metadata(address: str) -> tuple[str, Dict[str, Any]]:
            try:
                metadata = await self.rpc_service.get_token_metadata(address)
                return address, metadata
            except Exception as e:
                self.logger.warning(f"Failed to get metadata for token {address}: {str(e)}")
                return address, {"error": str(e)}
        
        # Process tokens in batches
        batches = [token_addresses[i:i+self.batch_size] for i in range(0, len(token_addresses), self.batch_size)]
        
        for batch in batches:
            tasks = [fetch_metadata(addr) for addr in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Update the result dictionary
            for addr, metadata in batch_results:
                result[addr] = metadata
        
        return result


@lru_cache()
def get_token_service() -> TokenService:
    """
    Get or create a TokenService instance.
    
    Returns:
        A singleton instance of TokenService
    """
    from solana_mcp.services.rpc_service import get_rpc_service
    from solana_mcp.services.cache_service import get_cache_service
    from solana_mcp.utils.config import get_settings
    
    settings = get_settings()
    return TokenService(
        rpc_service=get_rpc_service(),
        timeout=settings.RPC_TIMEOUT,
        cache_service=get_cache_service(),
        batch_size=getattr(settings, "BATCH_SIZE", 100)
    ) 