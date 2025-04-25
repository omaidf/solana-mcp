"""Token-related Solana RPC client operations.

This module provides specialized client functionality for Solana token operations.
"""

import base64
from typing import Dict, List, Any, Optional

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
            
            # Try to find the metadata account using two approaches
            metadata_response = None
            try:
                # First approach: Use more specific filters
                metadata_response = await self._make_request(
                    "getProgramAccounts",
                    [
                        METADATA_PROGRAM_ID,
                        {
                            "encoding": "base64",
                            "filters": [
                                {"dataSize": 679},  # Expected size of metadata accounts
                                {
                                    "memcmp": {
                                        "offset": 33,
                                        "bytes": mint
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                if not metadata_response or len(metadata_response.get("result", [])) == 0:
                    # Second approach: Try with just the mint filter
                    metadata_response = await self._make_request(
                        "getProgramAccounts",
                        [
                            METADATA_PROGRAM_ID,
                            {
                                "encoding": "base64",
                                "filters": [
                                    {
                                        "memcmp": {
                                            "offset": 33,
                                            "bytes": mint
                                        }
                                    }
                                ]
                            }
                        ]
                    )
            except Exception as e:
                logger.error(f"Error fetching metadata accounts: {str(e)}")
                return default_metadata
            
            if metadata_response and "result" in metadata_response:
                result = metadata_response["result"]
                
                if len(result) > 0:
                    # Extract metadata from the first account
                    account_info = result[0]
                    if "account" in account_info and "data" in account_info["account"]:
                        data = account_info["account"]["data"]
                        if isinstance(data, list) and data[1] == "base64":
                            decoded_data = base64.b64decode(data[0])
                            
                            # Parsing metadata - using safer methods with bounds checking
                            if len(decoded_data) < 70:  # Minimum length for valid metadata
                                logger.warning(f"Metadata account data too short for mint: {mint}")
                                return default_metadata
                            
                            # Read name length (1-byte at offset 0x28/40)
                            name_length_offset = 40
                            if len(decoded_data) <= name_length_offset:
                                return default_metadata
                                
                            name_length = decoded_data[name_length_offset]
                            
                            # Read name bytes with bounds checking
                            name_start = name_length_offset + 1
                            name_end = name_start + name_length
                            if name_end > len(decoded_data):
                                name_end = len(decoded_data)
                                
                            name = decoded_data[name_start:name_end].decode('utf-8').rstrip('\x00')
                            
                            # Read symbol length (1-byte after name)
                            symbol_length_offset = name_end
                            if len(decoded_data) <= symbol_length_offset:
                                return {**default_metadata, "name": name or default_metadata["name"]}
                                
                            symbol_length = decoded_data[symbol_length_offset]
                            
                            # Read symbol bytes with bounds checking
                            symbol_start = symbol_length_offset + 1
                            symbol_end = symbol_start + symbol_length
                            if symbol_end > len(decoded_data):
                                symbol_end = len(decoded_data)
                                
                            symbol = decoded_data[symbol_start:symbol_end].decode('utf-8').rstrip('\x00')
                            
                            # Read URI length (1-byte after symbol)
                            uri_length_offset = symbol_end
                            if len(decoded_data) <= uri_length_offset:
                                return {
                                    "name": name or default_metadata["name"],
                                    "symbol": symbol or default_metadata["symbol"],
                                    "uri": default_metadata["uri"]
                                }
                                
                            uri_length = decoded_data[uri_length_offset]
                            
                            # Read URI bytes with bounds checking
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
                
            # If we get here, return the default supply info
            logger.warning(f"Could not parse token metadata for {mint}, using default values")
            return default_metadata
            
        except Exception as e:
            logger.error(f"Error fetching token metadata for {mint}: {str(e)}")
            return default_metadata 