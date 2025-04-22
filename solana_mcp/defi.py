"""DeFi protocol integration for Solana MCP server."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from solana_mcp.cache import cache
from solana_mcp.solana_client import SolanaClient, validate_public_key


class JupiterClient:
    """Client for Jupiter Aggregator API."""
    
    # Jupiter API base URL
    BASE_URL = "https://quote-api.jup.ag/v6"
    
    def __init__(self):
        """Initialize the Jupiter client."""
        self.session = None
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create an HTTP session.
        
        Returns:
            An async HTTP client
        """
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SolanaMCPServer/1.0.0"
                }
            )
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    @cache(category="defi", ttl=300)
    async def get_tokens(self) -> Dict[str, Any]:
        """Get all tokens supported by Jupiter.
        
        Returns:
            A dictionary of token information
        """
        session = await self._get_session()
        response = await session.get(f"{self.BASE_URL}/tokens")
        response.raise_for_status()
        return response.json()
    
    @cache(category="defi", ttl=10)
    async def get_price(self, mint: str) -> Dict[str, Any]:
        """Get token price information.
        
        Args:
            mint: The token mint address
            
        Returns:
            Price information
        """
        if not validate_public_key(mint):
            raise ValueError(f"Invalid mint address: {mint}")
            
        session = await self._get_session()
        response = await session.get(f"https://price.jup.ag/v4/price?ids={mint}")
        response.raise_for_status()
        data = response.json()
        
        if mint not in data.get("data", {}):
            return {"mint": mint, "price": None, "error": "Price not available"}
            
        return data["data"][mint]
    
    @cache(category="defi", ttl=5)
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: Union[int, float],
        slippage_bps: int = 50,
        only_direct_routes: bool = False
    ) -> Dict[str, Any]:
        """Get a swap quote.
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in input token (integer for raw units, float for decimal)
            slippage_bps: Slippage tolerance in basis points (1 bp = 0.01%)
            only_direct_routes: Only include direct swap routes
            
        Returns:
            Quote information
        """
        if not validate_public_key(input_mint):
            raise ValueError(f"Invalid input mint address: {input_mint}")
        if not validate_public_key(output_mint):
            raise ValueError(f"Invalid output mint address: {output_mint}")
            
        # Construct quote parameters
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": slippage_bps,
            "onlyDirectRoutes": only_direct_routes
        }
        
        session = await self._get_session()
        response = await session.get(
            f"{self.BASE_URL}/quote", 
            params=params
        )
        
        if response.status_code == 404:
            return {
                "success": False,
                "error": "No routes found",
                "inputMint": input_mint,
                "outputMint": output_mint
            }
            
        response.raise_for_status()
        return response.json()
    
    async def get_swap_instruction(
        self,
        quote: Dict[str, Any],
        user_public_key: str
    ) -> Dict[str, Any]:
        """Get swap instructions based on a quote.
        
        Args:
            quote: The quote from get_quote
            user_public_key: The user's wallet public key
            
        Returns:
            Swap transaction instructions
        """
        if not validate_public_key(user_public_key):
            raise ValueError(f"Invalid user public key: {user_public_key}")
            
        # Extract route from quote
        if "routePlan" not in quote:
            raise ValueError("Invalid quote: missing routePlan")
            
        # Prepare request body
        body = {
            "quoteResponse": quote,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True
        }
        
        session = await self._get_session()
        response = await session.post(
            f"{self.BASE_URL}/swap-instructions",
            json=body
        )
        response.raise_for_status()
        return response.json()


class OrcaClient:
    """Client for Orca DEX API."""
    
    # Orca API base URL (placeholder - Orca doesn't have a public API)
    BASE_URL = "https://api.orca.so"
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the Orca client.
        
        Args:
            solana_client: Solana RPC client
        """
        self.solana_client = solana_client
        self.session = None
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create an HTTP session.
        
        Returns:
            An async HTTP client
        """
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SolanaMCPServer/1.0.0"
                }
            )
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    @cache(category="defi", ttl=300)
    async def get_whirlpools(self) -> Dict[str, Any]:
        """Get all Orca whirlpools.
        
        Note: This is a placeholder implementation. In a real world scenario,
        you would query the Orca program for whirlpool accounts.
        
        Returns:
            A dictionary of whirlpool information
        """
        # In a real implementation, you would query the Orca program for whirlpools
        # This would involve using getProgramAccounts with the appropriate filters
        
        try:
            # Orca Whirlpool Program ID
            program_id = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
            
            # Get all whirlpool accounts
            # In practice, you'd need more specific filters to reduce the result set
            accounts = await self.solana_client.get_program_accounts(
                program_id,
                encoding="jsonParsed",
                filters=[
                    {"dataSize": 1400}  # Approximate size of a whirlpool account
                ]
            )
            
            return {
                "whirlpools": accounts,
                "count": len(accounts)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @cache(category="defi", ttl=60)
    async def get_pool_stats(self, pool_address: str) -> Dict[str, Any]:
        """Get statistics for a specific Orca pool.
        
        Note: This is a placeholder implementation. In a real implementation,
        you would get the pool account data and parse it.
        
        Args:
            pool_address: The pool address
            
        Returns:
            Pool statistics
        """
        if not validate_public_key(pool_address):
            raise ValueError(f"Invalid pool address: {pool_address}")
            
        # In a real implementation, you would get the pool account data and parse it
        try:
            account_info = await self.solana_client.get_account_info(
                pool_address,
                encoding="jsonParsed"
            )
            
            # Create a placeholder response
            return {
                "address": pool_address,
                "liquidity": "1000000",
                "fee_rate": "0.003",
                "account_data": account_info,
                "note": "This is a placeholder implementation"
            }
        except Exception as e:
            return {"error": str(e)}


class SolendClient:
    """Client for Solend lending protocol API."""
    
    # Solend API base URL
    BASE_URL = "https://api.solend.fi"
    
    def __init__(self):
        """Initialize the Solend client."""
        self.session = None
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create an HTTP session.
        
        Returns:
            An async HTTP client
        """
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SolanaMCPServer/1.0.0"
                }
            )
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    @cache(category="defi", ttl=300)
    async def get_lending_markets(self) -> Dict[str, Any]:
        """Get all Solend lending markets.
        
        Returns:
            A dictionary of lending market information
        """
        session = await self._get_session()
        response = await session.get(f"{self.BASE_URL}/v1/markets/configs")
        response.raise_for_status()
        return response.json()
    
    @cache(category="defi", ttl=60)
    async def get_reserve_stats(self, reserve_id: str) -> Dict[str, Any]:
        """Get statistics for a specific Solend reserve.
        
        Args:
            reserve_id: The reserve ID
            
        Returns:
            Reserve statistics
        """
        if not validate_public_key(reserve_id):
            raise ValueError(f"Invalid reserve ID: {reserve_id}")
            
        session = await self._get_session()
        response = await session.get(f"{self.BASE_URL}/v1/reserves/{reserve_id}/stats")
        
        if response.status_code == 404:
            return {"error": "Reserve not found", "reserve_id": reserve_id}
            
        response.raise_for_status()
        return response.json()
    
    @cache(category="defi", ttl=60)
    async def get_user_stats(self, wallet_address: str) -> Dict[str, Any]:
        """Get Solend user statistics.
        
        Args:
            wallet_address: The user's wallet address
            
        Returns:
            User statistics
        """
        if not validate_public_key(wallet_address):
            raise ValueError(f"Invalid wallet address: {wallet_address}")
            
        session = await self._get_session()
        response = await session.get(f"{self.BASE_URL}/v1/users/{wallet_address}/stats")
        
        if response.status_code == 404:
            return {"error": "User not found", "wallet_address": wallet_address}
            
        response.raise_for_status()
        return response.json()


class DeFiManager:
    """Manager for DeFi protocol integrations."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the DeFi manager.
        
        Args:
            solana_client: Solana RPC client
        """
        self.solana_client = solana_client
        self.jupiter = JupiterClient()
        self.orca = OrcaClient(solana_client)
        self.solend = SolendClient()
    
    async def close(self):
        """Close all DeFi clients."""
        await self.jupiter.close()
        await self.orca.close()
        await self.solend.close()
    
    @cache(category="defi", ttl=300)
    async def get_supported_tokens(self) -> Dict[str, Any]:
        """Get all tokens supported across DeFi protocols.
        
        Returns:
            Token information
        """
        # Get Jupiter tokens
        jupiter_tokens = await self.jupiter.get_tokens()
        
        # In a real implementation, you would also query other protocols
        # and combine the results
        
        return jupiter_tokens
    
    @cache(category="defi", ttl=10)
    async def get_swap_quotes(
        self,
        input_mint: str,
        output_mint: str,
        amount: Union[int, float]
    ) -> Dict[str, Any]:
        """Get swap quotes from multiple protocols.
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in input token
            
        Returns:
            Quotes from different DEXes
        """
        if not validate_public_key(input_mint):
            raise ValueError(f"Invalid input mint address: {input_mint}")
        if not validate_public_key(output_mint):
            raise ValueError(f"Invalid output mint address: {output_mint}")
            
        # Get Jupiter quote
        jupiter_quote = await self.jupiter.get_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount
        )
        
        # In a real implementation, you would also query other DEXes
        
        # Compare quotes and return the best one
        return {
            "input_mint": input_mint,
            "output_mint": output_mint,
            "amount": amount,
            "quotes": {
                "jupiter": jupiter_quote
                # Add other DEX quotes here
            }
        } 