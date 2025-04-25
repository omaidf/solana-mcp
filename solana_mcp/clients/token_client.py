"""Token-related Solana RPC client operations.

This module provides specialized client functionality for Solana token operations.
"""

import base64
from typing import Dict, List, Any, Optional
import datetime
import httpx

from solana_mcp.clients.base_client import BaseSolanaClient, InvalidPublicKeyError, validate_public_key
from solana_mcp.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

# Default SPL Token Program ID
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

# Metaplex Token Metadata Program ID
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

class TokenClient(BaseSolanaClient):
    """Client for Solana token operations."""
    
    async def get_token_accounts_by_owner(
        self, 
        owner: str, 
        mint: Optional[str] = None,
        program_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get token accounts by owner.
        
        Args:
            owner: The owner public key
            mint: Filter by mint. Defaults to None.
            program_id: Filter by token program ID. Defaults to None.
            
        Returns:
            List of token accounts
            
        Raises:
            InvalidPublicKeyError: If the owner, mint, or program_id is not a valid Solana public key
        """
        if not validate_public_key(owner):
            raise InvalidPublicKeyError(owner)
            
        if mint and not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        if program_id and not validate_public_key(program_id):
            raise InvalidPublicKeyError(program_id)
            
        if mint:
            params = [owner, {"mint": mint}]
        elif program_id:
            params = [owner, {"programId": program_id}]
        else:
            # Default to SPL Token program
            params = [
                owner, 
                {"programId": TOKEN_PROGRAM_ID}
            ]
        
        params.append({"encoding": "jsonParsed"})
        return await self._make_request("getTokenAccountsByOwner", params)
    
    async def get_token_accounts_by_delegate(
        self,
        delegate: str,
        mint: Optional[str] = None,
        program_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get token accounts by delegate.
        
        Args:
            delegate: The delegate public key
            mint: Filter by mint. Defaults to None.
            program_id: Filter by token program ID. Defaults to None.
            
        Returns:
            List of token accounts
            
        Raises:
            InvalidPublicKeyError: If the delegate, mint, or program_id is not a valid Solana public key
        """
        if not validate_public_key(delegate):
            raise InvalidPublicKeyError(delegate)
            
        if mint and not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        if program_id and not validate_public_key(program_id):
            raise InvalidPublicKeyError(program_id)
            
        if mint:
            params = [delegate, {"mint": mint}]
        elif program_id:
            params = [delegate, {"programId": program_id}]
        else:
            params = [
                delegate, 
                {"programId": TOKEN_PROGRAM_ID}
            ]
        
        params.append({"encoding": "jsonParsed"})
        return await self._make_request("getTokenAccountsByDelegate", params)
    
    async def get_token_largest_accounts(self, mint: str) -> List[Dict[str, Any]]:
        """Get the largest accounts for a token.
        
        Args:
            mint: The mint address
            
        Returns:
            List of token accounts sorted by balance
            
        Raises:
            InvalidPublicKeyError: If the mint is not a valid Solana public key
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
            
        return await self._make_request("getTokenLargestAccounts", [mint])
    
    async def get_token_supply(self, mint: str) -> Dict[str, Any]:
        """Get token supply information for a specific mint.
        
        Args:
            mint: The mint address of the token
            
        Returns:
            Dict containing supply info and decimals.
            If token supply info cannot be retrieved, returns a default supply object.
            
        Raises:
            InvalidPublicKeyError: If the mint address is invalid
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
        
        try:
            # Set default supply info
            default_supply_info = {
                "mint": mint,
                "supply": {
                    "amount": "0",
                    "decimals": 0,
                    "ui_amount": 0.0,
                    "ui_amount_string": "0"
                }
            }
            
            # Get token supply using more robust error handling
            logger.debug(f"Fetching token supply for mint: {mint}")
            
            response = await self._make_request("getTokenSupply", [mint])
            
            if response and "result" in response:
                result = response["result"]
                
                # Check for error in result
                if "error" in result:
                    error_message = result.get("error", {}).get("message", "Unknown error")
                    logger.warning(f"Error getting token supply for {mint}: {error_message}")
                    return {**default_supply_info, "error": error_message}
                
                # Extract the supply value
                if "value" in result:
                    supply_value = result["value"]
                    
                    # Validate supply value structure
                    if "amount" in supply_value and "decimals" in supply_value:
                        decimals = int(supply_value.get("decimals", 0))
                        amount = supply_value.get("amount", "0")
                        
                        # Calculate UI amount (human-readable)
                        try:
                            ui_amount = float(amount) / (10 ** decimals) if decimals > 0 else float(amount)
                            ui_amount_string = f"{ui_amount:,.{decimals}f}"
                        except (ValueError, ZeroDivisionError):
                            ui_amount = 0.0
                            ui_amount_string = "0"
                        
                        return {
                            "mint": mint,
                            "supply": {
                                "amount": amount,
                                "decimals": decimals,
                                "ui_amount": ui_amount,
                                "ui_amount_string": ui_amount_string
                            }
                        }
            
            # If we get here, return the default supply info
            logger.warning(f"Could not parse token supply for {mint}, using default values")
            return default_supply_info
            
        except Exception as e:
            # Log and return error information
            logger.error(f"Error fetching token supply for {mint}: {str(e)}")
            return {
                "mint": mint,
                "supply": {
                    "amount": "0",
                    "decimals": 0,
                    "ui_amount": 0.0,
                    "ui_amount_string": "0"
                },
                "error": str(e)
            }
    
    async def get_token_metadata(self, mint: str) -> Dict[str, Any]:
        """Get token metadata from Metaplex program.
        
        Args:
            mint: The mint address
            
        Returns:
            Dict with token metadata (name, symbol, uri)
            
        Raises:
            InvalidPublicKeyError: If the mint address is invalid
        """
        if not validate_public_key(mint):
            raise InvalidPublicKeyError(mint)
        
        # Default metadata if metadata doesn't exist
        default_metadata = {
            "name": "Unknown Token",
            "symbol": "UNKNOWN",
            "uri": "",
        }
        
        try:
            logger.debug(f"Fetching token metadata for mint: {mint}")
            
            # Get metadata account using memcmp filter
            metadata_response = await self._make_request(
                "getProgramAccounts",
                [
                    METADATA_PROGRAM_ID,
                    {
                        "encoding": "base64",
                        "filters": [
                            {"memcmp": {"offset": 33, "bytes": mint}}
                        ]
                    }
                ]
            )
            
            if metadata_response and "result" in metadata_response and len(metadata_response["result"]) > 0:
                # Extract metadata from the first account
                account_info = metadata_response["result"][0]
                if "account" in account_info and "data" in account_info["account"]:
                    data = account_info["account"]["data"]
                    if isinstance(data, list) and data[1] == "base64":
                        decoded_data = base64.b64decode(data[0])
                        
                        # Ensure we have enough data
                        if len(decoded_data) > 50:
                            # Name length is at offset 40 (1 byte)
                            name_length_offset = 40
                            if len(decoded_data) <= name_length_offset:
                                return default_metadata
                                
                            name_length = decoded_data[name_length_offset]
                            
                            # Name starts at offset 41
                            name_start = name_length_offset + 1
                            name_end = name_start + name_length
                            if name_end > len(decoded_data):
                                name_end = len(decoded_data)
                                
                            name = decoded_data[name_start:name_end].decode('utf-8').rstrip('\x00')
                            
                            # Symbol length comes after name
                            symbol_length_offset = name_end
                            if len(decoded_data) <= symbol_length_offset:
                                return {**default_metadata, "name": name or default_metadata["name"]}
                                
                            symbol_length = decoded_data[symbol_length_offset]
                            
                            # Symbol starts after symbol length
                            symbol_start = symbol_length_offset + 1
                            symbol_end = symbol_start + symbol_length
                            if symbol_end > len(decoded_data):
                                symbol_end = len(decoded_data)
                                
                            symbol = decoded_data[symbol_start:symbol_end].decode('utf-8').rstrip('\x00')
                            
                            # URI length comes after symbol
                            uri_length_offset = symbol_end
                            if len(decoded_data) <= uri_length_offset:
                                return {
                                    "name": name or default_metadata["name"],
                                    "symbol": symbol or default_metadata["symbol"],
                                    "uri": default_metadata["uri"]
                                }
                                
                            uri_length = decoded_data[uri_length_offset]
                            
                            # URI starts after URI length
                            uri_start = uri_length_offset + 1
                            uri_end = uri_start + uri_length
                            if uri_end > len(decoded_data):
                                uri_end = len(decoded_data)
                                
                            uri = decoded_data[uri_start:uri_end].decode('utf-8').rstrip('\x00')
                            
                            metadata = {
                                "name": name if name else default_metadata["name"],
                                "symbol": symbol if symbol else default_metadata["symbol"],
                                "uri": uri if uri else default_metadata["uri"]
                            }
                            
                            logger.debug(f"Successfully parsed metadata for {mint}: {metadata}")
                            return metadata
                            
            # If we get here, try an alternative approach using Jupiter token list API
            try:
                # Try to get from Jupiter token list API
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"https://token.jup.ag/all")
                    if response.status_code == 200:
                        tokens = response.json()
                        for token in tokens:
                            if token.get("address") == mint:
                                return {
                                    "name": token.get("name", default_metadata["name"]),
                                    "symbol": token.get("symbol", default_metadata["symbol"]),
                                    "uri": token.get("logoURI", default_metadata["uri"]),
                                    "source": "jupiter"
                                }
            except Exception as e:
                logger.warning(f"Error fetching from Jupiter API: {str(e)}")
                
            # If we get here, return the default metadata
            logger.warning(f"Could not parse token metadata for {mint}, using default values")
            return default_metadata
            
        except Exception as e:
            logger.error(f"Error fetching token metadata for {mint}: {str(e)}")
            return default_metadata
    
    async def get_token_price(self, token_mint: str) -> Dict[str, Any]:
        """Get token price from Birdeye API.
        
        Args:
            token_mint: The mint address of the token
            
        Returns:
            Dict containing price information including:
            - price: current price in USD
            - price_change_24h: 24-hour price change percentage
            - last_updated: timestamp of last update
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Default response with fallback price
        default_response = {
            "price": 0.0,
            "price_change_24h": 0.0,
            "last_updated": datetime.datetime.now().isoformat(),
            "source": "fallback"
        }
        
        try:
            # Birdeye API endpoint
            BIRDEYE_API = "https://public-api.birdeye.so/public/price"
            headers = {"x-chain": "solana"}
            
            # Use httpx instead of requests to maintain async
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{BIRDEYE_API}?address={token_mint}",
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('success'):
                    price_data = data['data']
                    return {
                        "price": price_data['value'],
                        "price_change_24h": price_data['change24h'],
                        "last_updated": datetime.datetime.fromtimestamp(price_data['updateUnixTime']).isoformat(),
                        "source": "birdeye"
                    }
                
                logger.warning(f"Birdeye API returned unsuccessful response for {token_mint}")
                return default_response
                
        except Exception as e:
            logger.error(f"Error fetching price from Birdeye for {token_mint}: {str(e)}")
            return default_response
            
    async def get_token_holders(self, token_mint: str, limit: int = 10) -> Dict[str, Any]:
        """Get top token holders from Solana FM API.
        
        Args:
            token_mint: The mint address of the token
            limit: Maximum number of holders to return. Defaults to 10.
            
        Returns:
            Dict containing list of holders with their balances
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Default response with empty holders
        default_response = {
            "holders": [],
            "total_holders": 0,
            "source": "fallback"
        }
        
        try:
            # Solana FM API endpoint
            SOLANA_FM_API = f"https://api.solana.fm/v0/tokens/{token_mint}/holders?limit={limit}"
            
            # Use httpx instead of requests to maintain async
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(SOLANA_FM_API)
                response.raise_for_status()
                data = response.json()
                
                # Process and return the holder data
                if "data" in data and "items" in data["data"]:
                    holders = []
                    for holder in data["data"]["items"]:
                        try:
                            holders.append({
                                "address": holder.get("address", ""),
                                "amount": holder.get("amount", "0"),
                                "ui_amount": holder.get("uiAmount", 0),
                                "percentage": holder.get("percentage", 0)
                            })
                        except (KeyError, TypeError) as e:
                            logger.warning(f"Error parsing holder data: {str(e)}")
                    
                    return {
                        "holders": holders,
                        "total_holders": data["data"].get("totalItems", len(holders)),
                        "source": "solana_fm"
                    }
                
                logger.warning(f"Unexpected response format from Solana FM API for {token_mint}")
                return default_response
                
        except Exception as e:
            logger.error(f"Error fetching holders from Solana FM for {token_mint}: {str(e)}")
            return default_response
            
    async def get_all_token_data(self, token_mint: str, holder_limit: int = 10) -> Dict[str, Any]:
        """Get comprehensive token data combining metadata, supply, price, and holder information.
        
        Args:
            token_mint: The mint address of the token
            holder_limit: Maximum number of holders to return. Defaults to 10.
            
        Returns:
            Consolidated token data including metadata, supply, price, and holders
            
        Raises:
            InvalidPublicKeyError: If the token_mint is not a valid Solana public key
        """
        if not validate_public_key(token_mint):
            raise InvalidPublicKeyError(token_mint)
            
        # Get token metadata (name, symbol, etc.)
        metadata = await self.get_token_metadata(token_mint)
        
        # Get token supply information
        supply_data = await self.get_token_supply(token_mint)
        
        # Get token price information
        price_data = await self.get_token_price(token_mint)
        
        # Get token holders information
        holders_data = await self.get_token_holders(token_mint, limit=holder_limit)
        
        # Combine all data into a comprehensive response
        result = {
            "token_mint": token_mint,
            "name": metadata.get("name", "Unknown"),
            "symbol": metadata.get("symbol", "UNKNOWN"),
            "decimals": supply_data.get("supply", {}).get("decimals", 0),
            "supply": {
                "amount": supply_data.get("supply", {}).get("amount", "0"),
                "ui_amount": supply_data.get("supply", {}).get("ui_amount", 0.0),
                "ui_amount_string": supply_data.get("supply", {}).get("ui_amount_string", "0")
            },
            "price": {
                "current_price_usd": price_data.get("price", 0.0),
                "price_change_24h": price_data.get("price_change_24h", 0.0),
                "last_updated": price_data.get("last_updated", datetime.datetime.now().isoformat()),
                "source": price_data.get("source", "unknown")
            },
            "holders": {
                "total_holders": holders_data.get("total_holders", 0),
                "top_holders": holders_data.get("holders", [])
            },
            "metadata": {
                "uri": metadata.get("uri", ""),
                "source": metadata.get("source", "metaplex")
            },
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Include error information if any of the API calls failed
        errors = []
        if "error" in metadata:
            errors.append({"metadata_error": metadata["error"]})
        if "error" in supply_data:
            errors.append({"supply_error": supply_data["error"]})
        if "error" in price_data:
            errors.append({"price_error": price_data["error"]})
        if "error" in holders_data:
            errors.append({"holders_error": holders_data["error"]})
            
        if errors:
            result["errors"] = errors
            
        return result 