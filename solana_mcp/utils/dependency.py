"""
Dependency provider utility for FastAPI.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use `solana_mcp.dependencies` instead.
"""

import warnings
from functools import lru_cache
from typing import Callable, Dict, Optional, Type, TypeVar

from solana_mcp.dependencies import (
    get_service, get_account_service, get_token_service,
    get_transaction_service, get_analysis_service, get_nlp_service,
    get_cache_service, initialize_providers
)

# Show deprecation warning
warnings.warn(
    "The `solana_mcp.utils.dependency` module is deprecated. "
    "Use `solana_mcp.dependencies` instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = [
    'get_service',
    'get_account_service',
    'get_token_service',
    'get_transaction_service',
    'get_nlp_service',
    'get_cache_service',
    'initialize_providers'
]

from fastapi import Depends

from solana_mcp.services.account_service import AccountService
from solana_mcp.services.base_service import BaseService
from solana_mcp.services.cache_service import CacheService, get_cache_service
from solana_mcp.services.nlp_service import NLPService
from solana_mcp.services.rpc_client import RPCClient
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.transaction_service import TransactionService

# Type variable for service classes
T = TypeVar('T', bound=BaseService)

# Registry to store service instances
_service_registry: Dict[Type[BaseService], BaseService] = {}

# Get or create a service instance
def get_service(service_class: Type[T]) -> T:
    """
    Get or create a service instance.
    
    This function returns a cached instance of the specified service class.
    If an instance doesn't exist, it creates one and caches it.
    
    Args:
        service_class: The service class to instantiate
        
    Returns:
        An instance of the specified service class
    """
    if service_class not in _service_registry:
        # Initialize with common dependencies
        cache_service = get_cache_service()
        rpc_client = RPCClient()
        
        # Create the service instance
        _service_registry[service_class] = service_class(
            cache_service=cache_service,
            rpc_client=rpc_client
        )
    
    return _service_registry[service_class]

# Dependency providers for specific services
@lru_cache()
def get_token_service() -> TokenService:
    """Dependency provider for TokenService."""
    return get_service(TokenService)

@lru_cache()
def get_account_service() -> AccountService:
    """Dependency provider for AccountService."""
    return get_service(AccountService)

@lru_cache()
def get_transaction_service() -> TransactionService:
    """Dependency provider for TransactionService."""
    return get_service(TransactionService)

@lru_cache()
def get_nlp_service() -> NLPService:
    """Dependency provider for NLPService."""
    return get_service(NLPService) 