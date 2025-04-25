"""
Dependency injection and service provider management for the Solana MCP API.

This module contains FastAPI dependency providers and service initialization functions
to ensure consistent access to services across the application.
"""

import logging
from functools import lru_cache
from typing import Callable, Dict, Type, TypeVar, Any

from fastapi import Depends

from solana_mcp.services.base_service import BaseService
from solana_mcp.services.account_service import AccountService
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.analysis_service import AnalysisService
from solana_mcp.services.nlp_service import NLPService
from solana_mcp.services.cache_service import CacheService
from solana_mcp.utils.config import settings
from solana_mcp.utils.solana_client import SolanaClient


# Type variable for service classes
T = TypeVar('T', bound=BaseService)

# Global registry of service instances
_SERVICE_REGISTRY: Dict[Type[BaseService], BaseService] = {}


def initialize_providers():
    """
    Initialize all service providers when the application starts.

    This ensures that any required setup for services is completed during
    application startup and services are properly registered for dependency injection.
    """
    logging.info("Initializing service providers")
    
    # Initialize shared resources
    solana_client = SolanaClient(
        rpc_url=settings.SOLANA_RPC_URL,
        timeout=settings.RPC_TIMEOUT
    )
    
    # Initialize cache service
    cache_service = CacheService(
        max_size=settings.MAX_CACHE_SIZE,
        ttl=settings.CACHE_TTL
    )
    
    # Register services
    register_service(
        CacheService,
        cache_service
    )
    
    register_service(
        AccountService,
        AccountService(
            solana_client=solana_client,
            cache_service=cache_service
        )
    )
    
    register_service(
        TokenService,
        TokenService(
            solana_client=solana_client,
            cache_service=cache_service
        )
    )
    
    register_service(
        TransactionService,
        TransactionService(
            solana_client=solana_client,
            cache_service=cache_service
        )
    )
    
    register_service(
        AnalysisService,
        AnalysisService(
            solana_client=solana_client,
            cache_service=cache_service
        )
    )
    
    register_service(
        NLPService,
        NLPService(
            solana_client=solana_client,
            cache_service=cache_service
        )
    )
    
    logging.info("Service providers initialized successfully")


def register_service(service_type: Type[T], service_instance: T):
    """
    Register a service instance with the dependency injection system.
    
    Args:
        service_type: The service class type
        service_instance: The instance of the service to register
    """
    _SERVICE_REGISTRY[service_type] = service_instance
    logging.debug(f"Registered service: {service_type.__name__}")


def get_service(service_type: Type[T]) -> T:
    """
    Get a service instance by its type.
    
    Args:
        service_type: The service class type to retrieve
        
    Returns:
        The registered service instance
        
    Raises:
        KeyError: If the service is not registered
    """
    if service_type not in _SERVICE_REGISTRY:
        raise KeyError(f"Service not registered: {service_type.__name__}")
    return _SERVICE_REGISTRY[service_type]


# FastAPI dependency providers
@lru_cache
def get_account_service() -> AccountService:
    """Dependency provider for AccountService."""
    return get_service(AccountService)


@lru_cache
def get_token_service() -> TokenService:
    """Dependency provider for TokenService."""
    return get_service(TokenService)


@lru_cache
def get_transaction_service() -> TransactionService:
    """Dependency provider for TransactionService."""
    return get_service(TransactionService)


@lru_cache
def get_analysis_service() -> AnalysisService:
    """Dependency provider for AnalysisService."""
    return get_service(AnalysisService)


@lru_cache
def get_nlp_service() -> NLPService:
    """Dependency provider for NLPService."""
    return get_service(NLPService)


@lru_cache
def get_cache_service() -> CacheService:
    """Dependency provider for CacheService."""
    return get_service(CacheService) 