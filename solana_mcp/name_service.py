"""Name service integration for Solana MCP server."""

import base58
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from solana_mcp.cache import cache
from solana_mcp.solana_client import SolanaClient, validate_public_key


class NameServiceClient:
    """Client for Solana Name Service (SNS)."""
    
    # SNS Program ID
    SNS_PROGRAM_ID = "namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX"
    
    # TLD accounts
    SOL_TLD_ACCOUNT = "58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx"
    
    # Hash prefix for domain name
    HASH_PREFIX = "SPL Name Service"
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the name service client.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
    
    def _get_name_account_key(self, name: str, parent_name_account: str = None) -> str:
        """Derive the name account key for a given name.
        
        Args:
            name: The name to derive the key for
            parent_name_account: The parent name account key
            
        Returns:
            The derived name account key
        """
        # Hash the name with the prefix
        hasher = hashlib.sha256()
        hasher.update(self.HASH_PREFIX.encode())
        hasher.update(name.encode())
        
        # If parent account is provided, include it in the hash
        if parent_name_account:
            hasher.update(base58.b58decode(parent_name_account))
            
        hashed_name = hasher.digest()
        
        # Find a valid program address
        # Note: This is a simplified implementation. In practice, 
        # you would use the Solana SDK's findProgramAddress function
        seeds = [hashed_name, bytes([0])]  # Using bump seed 0 for simplicity
        
        # This is a placeholder - in practice you'd derive this correctly
        name_account_key = f"name_{base58.b58encode(hashed_name[:16]).decode()}"
        
        return name_account_key
    
    @cache(category="name_service", ttl=300)
    async def resolve_domain(
        self, 
        domain_name: str, 
        root_domain: str = "sol"
    ) -> Dict[str, Any]:
        """Resolve a domain name to Solana address.
        
        Args:
            domain_name: The domain name to resolve
            root_domain: The root domain (default: "sol")
            
        Returns:
            Domain resolution information
        """
        if not domain_name:
            raise ValueError("Domain name cannot be empty")
            
        # Get TLD account based on root domain
        tld_account = self.SOL_TLD_ACCOUNT if root_domain == "sol" else None
        if not tld_account:
            return {"error": f"Unsupported root domain: {root_domain}"}
        
        # Get the name account key
        name_account_key = self._get_name_account_key(domain_name, tld_account)
        
        try:
            # Get the name account data
            account_info = await self.solana_client.get_account_info(
                name_account_key,
                encoding="jsonParsed"
            )
            
            # Parse the account data to extract the owner address
            # This is a simplified implementation. In practice, you would 
            # properly parse the SNS account data structure
            if "data" in account_info and account_info.get("owner") == self.SNS_PROGRAM_ID:
                # For now, return a mock response
                return {
                    "domain": f"{domain_name}.{root_domain}",
                    "address": f"placeholder_address_for_{domain_name}",
                    "account": name_account_key,
                    "exists": True,
                    "data": account_info.get("data")
                }
            else:
                return {
                    "domain": f"{domain_name}.{root_domain}",
                    "exists": False,
                    "error": "Domain not found or not registered"
                }
        except Exception as e:
            return {
                "domain": f"{domain_name}.{root_domain}",
                "exists": False,
                "error": str(e)
            }
    
    @cache(category="name_service", ttl=300)
    async def reverse_lookup(self, address: str) -> Dict[str, Any]:
        """Perform a reverse lookup to find domain names for an address.
        
        Args:
            address: The Solana address to look up
            
        Returns:
            Domain information for the address
        """
        if not validate_public_key(address):
            raise ValueError(f"Invalid address: {address}")
            
        try:
            # Query for "reverse" accounts owned by the SNS program
            # This is a simplified implementation. In practice, you would
            # need specific filters to find reverse lookup accounts
            program_accounts = await self.solana_client.get_program_accounts(
                self.SNS_PROGRAM_ID,
                filters=[
                    {
                        "memcmp": {
                            "offset": 32,  # Assuming owner address is at offset 32
                            "bytes": address
                        }
                    }
                ]
            )
            
            # Parse the accounts to find domain names
            # This is a simplified implementation
            domains = []
            for account in program_accounts:
                # In practice, you would properly parse the account data
                # to extract the domain name
                domains.append({
                    "name": f"placeholder_name_for_{account['pubkey'][:8]}",
                    "account": account["pubkey"]
                })
                
            return {
                "address": address,
                "domains": domains,
                "count": len(domains)
            }
        except Exception as e:
            return {
                "address": address,
                "error": str(e),
                "domains": []
            }
    
    @cache(category="name_service", ttl=60)
    async def get_domain_registry_info(self) -> Dict[str, Any]:
        """Get information about the domain registry.
        
        Returns:
            Registry information
        """
        try:
            # Get the .sol TLD account
            sol_tld_info = await self.solana_client.get_account_info(
                self.SOL_TLD_ACCOUNT,
                encoding="jsonParsed"
            )
            
            # In practice, you would properly parse the account data
            return {
                "sol_tld": {
                    "address": self.SOL_TLD_ACCOUNT,
                    "owner": sol_tld_info.get("owner"),
                    "data_size": sol_tld_info.get("data", {}).get("size", 0)
                },
                "program_id": self.SNS_PROGRAM_ID
            }
        except Exception as e:
            return {"error": str(e)}


class BonfidaNameService:
    """Client for Bonfida Name Service API."""
    
    # Bonfida API base URL
    BASE_URL = "https://sns-api.bonfida.com/v2"
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the Bonfida name service client.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
    
    @cache(category="name_service", ttl=300)
    async def resolve_domain(
        self, 
        domain_name: str, 
        root_domain: str = "sol"
    ) -> Dict[str, Any]:
        """Resolve a domain name to Solana address using Bonfida's API.
        
        Args:
            domain_name: The domain name to resolve
            root_domain: The root domain (default: "sol")
            
        Returns:
            Domain resolution information
        """
        if not domain_name:
            raise ValueError("Domain name cannot be empty")
            
        # Only .sol domains are supported
        if root_domain != "sol":
            return {"error": f"Unsupported root domain: {root_domain}"}
            
        import httpx
        
        full_domain = f"{domain_name}.{root_domain}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/domains/{full_domain}"
                )
                
                if response.status_code == 404:
                    return {
                        "domain": full_domain,
                        "exists": False,
                        "error": "Domain not found"
                    }
                    
                response.raise_for_status()
                data = response.json()
                
                return {
                    "domain": full_domain,
                    "exists": True,
                    "owner": data.get("owner"),
                    "registry": "Bonfida SNS"
                }
        except httpx.HTTPError as e:
            return {
                "domain": full_domain,
                "exists": False,
                "error": f"HTTP error: {str(e)}"
            }
        except Exception as e:
            return {
                "domain": full_domain,
                "exists": False,
                "error": str(e)
            }
    
    @cache(category="name_service", ttl=300)
    async def reverse_lookup(self, address: str) -> Dict[str, Any]:
        """Perform a reverse lookup to find domain names for an address.
        
        Args:
            address: The Solana address to look up
            
        Returns:
            Domain information for the address
        """
        if not validate_public_key(address):
            raise ValueError(f"Invalid address: {address}")
            
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/addresses/{address}/domains"
                )
                
                if response.status_code == 404:
                    return {
                        "address": address,
                        "domains": [],
                        "count": 0
                    }
                    
                response.raise_for_status()
                data = response.json()
                
                # Extract domains from the response
                domains = [
                    {"name": domain["domain"], "type": "sns"}
                    for domain in data.get("domains", [])
                ]
                
                return {
                    "address": address,
                    "domains": domains,
                    "count": len(domains),
                    "registry": "Bonfida SNS"
                }
        except httpx.HTTPError as e:
            return {
                "address": address,
                "error": f"HTTP error: {str(e)}",
                "domains": []
            }
        except Exception as e:
            return {
                "address": address,
                "error": str(e),
                "domains": []
            }


class NameServiceManager:
    """Manager for name service integrations."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the name service manager.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
        self.sns = NameServiceClient(solana_client)
        self.bonfida = BonfidaNameService(solana_client)
    
    @cache(category="name_service", ttl=300)
    async def resolve_domain(
        self, 
        domain: str,
        use_bonfida_api: bool = True
    ) -> Dict[str, Any]:
        """Resolve a domain name to Solana address.
        
        Args:
            domain: The domain name to resolve (e.g., "example.sol")
            use_bonfida_api: Whether to use Bonfida's API for resolution
            
        Returns:
            Domain resolution information
        """
        # Parse the domain
        parts = domain.split('.')
        if len(parts) < 2:
            return {"error": "Invalid domain format. Expected format: name.tld"}
            
        name = parts[0]
        tld = parts[-1]
        
        if use_bonfida_api and tld == "sol":
            # Use Bonfida's API for .sol domains
            return await self.bonfida.resolve_domain(name, tld)
        else:
            # Use local name service client
            return await self.sns.resolve_domain(name, tld)
    
    @cache(category="name_service", ttl=300)
    async def reverse_lookup(
        self, 
        address: str,
        use_bonfida_api: bool = True
    ) -> Dict[str, Any]:
        """Perform a reverse lookup to find domain names for an address.
        
        Args:
            address: The Solana address to look up
            use_bonfida_api: Whether to use Bonfida's API for lookup
            
        Returns:
            Domain information for the address
        """
        if not validate_public_key(address):
            raise ValueError(f"Invalid address: {address}")
            
        if use_bonfida_api:
            # Use Bonfida's API
            bonfida_result = await self.bonfida.reverse_lookup(address)
            
            # If error or no domains found, fall back to local lookup
            if "error" in bonfida_result or not bonfida_result.get("domains"):
                sns_result = await self.sns.reverse_lookup(address)
                domains = bonfida_result.get("domains", []) + sns_result.get("domains", [])
                
                return {
                    "address": address,
                    "domains": domains,
                    "count": len(domains),
                    "sources": ["bonfida", "sns"]
                }
            
            return bonfida_result
        else:
            # Use local lookup only
            return await self.sns.reverse_lookup(address) 