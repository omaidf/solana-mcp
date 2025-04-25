"""Dependency management utilities for Solana MCP.

This module provides a centralized dependency provider to simplify dependency injection
in FastAPI routes and services.
"""

from typing import Dict, Any, Optional, Type, TypeVar, Callable, cast, AsyncGenerator
import asyncio
from contextlib import asynccontextmanager
import inspect
from functools import wraps

from fastapi import Depends, Request

from solana_mcp.clients import SolanaClient, get_solana_client
from solana_mcp.services.cache_service import CacheService
from solana_mcp.utils.errors import SolanaMCPError, ServiceUnavailableError
from solana_mcp.logging_config import get_logger
from solana_mcp.services.rpc_service import RPCService
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.nlp_service import NLPService
from solana_mcp.utils.config import get_settings

# Get logger
logger = get_logger(__name__)

# Type variable for dependency types
T = TypeVar('T')

class DependencyProvider:
    """Centralized provider for application dependencies.
    
    This class manages the creation, caching, and retrieval of application
    dependencies such as clients, services, and utilities.
    """
    
    def __init__(self):
        """Initialize the dependency provider."""
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._async_factories: Dict[str, Callable[[], AsyncGenerator[Any, None]]] = {}
    
    def register(
        self, 
        dependency_type: Type[T], 
        factory: Optional[Callable[[], T]] = None,
        instance: Optional[T] = None,
        singleton: bool = True
    ) -> None:
        """Register a dependency with the provider.
        
        Args:
            dependency_type: Type of the dependency
            factory: Optional factory function to create the dependency
            instance: Optional pre-created instance
            singleton: Whether to cache the instance for reuse
            
        Raises:
            ValueError: If neither factory nor instance is provided
        """
        if factory is None and instance is None:
            raise ValueError("Either factory or instance must be provided")
            
        key = self._get_dependency_key(dependency_type)
        
        if instance is not None:
            if singleton:
                self._instances[key] = instance
            else:
                # For non-singletons, create a factory that returns a copy
                self._factories[key] = lambda: instance
        else:
            if singleton and factory is not None:
                # For singletons, we'll create the instance on first use
                self._factories[key] = factory
            elif factory is not None:
                # For non-singletons, just store the factory
                self._factories[key] = factory
    
    def register_async(
        self, 
        dependency_type: Type[T], 
        factory: Callable[[], AsyncGenerator[T, None]],
        singleton: bool = True
    ) -> None:
        """Register an async dependency with the provider.
        
        Args:
            dependency_type: Type of the dependency
            factory: Async factory function to create the dependency
            singleton: Whether to cache the instance for reuse
        """
        key = self._get_dependency_key(dependency_type)
        
        if singleton:
            # For singletons, we'll create and cache the instance on first use
            self._async_factories[key] = factory
        else:
            # For non-singletons, just store the factory
            self._async_factories[key] = factory
    
    def get(self, dependency_type: Type[T]) -> T:
        """Get a dependency instance.
        
        Args:
            dependency_type: Type of the dependency to get
            
        Returns:
            Dependency instance
            
        Raises:
            ServiceUnavailableError: If the dependency is not registered or cannot be created
        """
        key = self._get_dependency_key(dependency_type)
        
        # Check if we already have an instance
        if key in self._instances:
            return cast(T, self._instances[key])
        
        # Check if we have a factory
        if key in self._factories:
            try:
                instance = self._factories[key]()
                # Cache singleton instances
                self._instances[key] = instance
                return cast(T, instance)
            except Exception as e:
                logger.error(f"Error creating dependency {key}: {str(e)}", exc_info=True)
                raise ServiceUnavailableError(
                    message=f"Failed to create dependency {dependency_type.__name__}",
                    service=dependency_type.__name__,
                    details={"error": str(e)}
                ) from e
        
        # If we reach here, the dependency is not registered
        raise ServiceUnavailableError(
            message=f"Dependency {dependency_type.__name__} not found",
            service=dependency_type.__name__
        )
    
    async def get_async(self, dependency_type: Type[T]) -> T:
        """Get an async dependency instance.
        
        Args:
            dependency_type: Type of the dependency to get
            
        Returns:
            Dependency instance
            
        Raises:
            ServiceUnavailableError: If the dependency is not registered or cannot be created
        """
        key = self._get_dependency_key(dependency_type)
        
        # Check if we already have an instance
        if key in self._instances:
            return cast(T, self._instances[key])
        
        # Check if we have an async factory
        if key in self._async_factories:
            try:
                # Get the factory and create a context manager
                factory = self._async_factories[key]
                # Use the context manager to get the instance
                async for instance in factory():
                    # Cache singleton instances
                    self._instances[key] = instance
                    return cast(T, instance)
            except Exception as e:
                logger.error(f"Error creating async dependency {key}: {str(e)}", exc_info=True)
                raise ServiceUnavailableError(
                    message=f"Failed to create dependency {dependency_type.__name__}",
                    service=dependency_type.__name__,
                    details={"error": str(e)}
                ) from e
        
        # If we reach here, the dependency is not registered
        raise ServiceUnavailableError(
            message=f"Dependency {dependency_type.__name__} not found",
            service=dependency_type.__name__
        )
    
    def _get_dependency_key(self, dependency_type: Type[T]) -> str:
        """Get a unique key for a dependency type.
        
        Args:
            dependency_type: Type of the dependency
            
        Returns:
            Unique key for the dependency
        """
        return dependency_type.__name__

# Singleton instance
_provider = DependencyProvider()

def get_provider() -> DependencyProvider:
    """Get the dependency provider instance.
    
    Returns:
        DependencyProvider instance
    """
    return _provider

# Standard dependencies for use with FastAPI Depends
async def get_solana_client_dependency() -> SolanaClient:
    """Get a Solana client for dependency injection.
    
    Returns:
        SolanaClient instance
    """
    client = await get_provider().get_async(SolanaClient)
    return client

async def get_cache_service_dependency() -> CacheService:
    """Get a cache service for dependency injection.
    
    Returns:
        CacheService instance
    """
    service = await get_provider().get_async(CacheService)
    return service

def setup_dependencies(app: Any) -> None:
    """Setup standard dependencies for the application.
    
    Args:
        app: FastAPI application instance
    """
    # Register async dependencies
    get_provider().register_async(
        SolanaClient,
        lambda: get_solana_client()
    )
    
    # Register cache service with a factory
    get_provider().register_async(
        CacheService,
        lambda: _create_cache_service()
    )
    
    # Register in the app state for access
    app.state.dependency_provider = get_provider()

@asynccontextmanager
async def _create_cache_service() -> AsyncGenerator[CacheService, None]:
    """Create and manage a cache service context.
    
    Yields:
        CacheService instance
    """
    service = CacheService(max_size=1000, ttl=300)
    try:
        yield service
    finally:
        await service.close()

async def get_rpc_service() -> RPCService:
    """
    Provides an instance of the RPCService.
    
    Used as a FastAPI dependency to inject RPC service into route handlers.
    """
    settings = get_settings()
    return RPCService(
        rpc_url=settings.SOLANA_RPC_URL,
        timeout=settings.RPC_TIMEOUT
    )

async def get_cache_service() -> CacheService:
    """
    Provides an instance of the CacheService.
    
    Used as a FastAPI dependency to inject cache service into route handlers.
    """
    settings = get_settings()
    return CacheService(
        max_size=settings.MAX_CACHE_SIZE,
        ttl=settings.CACHE_TTL
    )

async def get_token_service(
    rpc_service: RPCService = Depends(get_rpc_service),
    cache_service: CacheService = Depends(get_cache_service)
) -> TokenService:
    """
    Provides an instance of the TokenService.
    
    Used as a FastAPI dependency to inject token service into route handlers.
    """
    settings = get_settings()
    return TokenService(
        rpc_service=rpc_service,
        timeout=settings.RPC_TIMEOUT,
        cache_service=cache_service,
        batch_size=settings.BATCH_SIZE
    )

async def get_nlp_service(
    cache_service: CacheService = Depends(get_cache_service)
) -> NLPService:
    """
    Provides an instance of the NLPService.
    
    Used as a FastAPI dependency to inject NLP service into route handlers.
    """
    settings = get_settings()
    return NLPService(
        model_name=settings.NLP_MODEL_NAME,
        cache_service=cache_service
    ) 