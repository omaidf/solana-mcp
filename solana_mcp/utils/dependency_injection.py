"""
Dependency injection system for Solana MCP.

This module provides a simple dependency injection system to manage
and reuse service and client instances throughout the application.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast, get_type_hints

from solana_mcp.utils.error_handling import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceProvider:
    """Service provider for dependency injection.
    
    This class manages service instances and provides them when needed.
    It supports singleton and transient service lifetimes.
    """
    
    def __init__(self):
        """Initialize the service provider."""
        self._services: Dict[Type, Dict[str, Any]] = {}
        self._factories: Dict[Type, Callable[..., Any]] = {}
        self._instances: Dict[str, Any] = {}
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register a singleton service instance.
        
        Args:
            service_type: The type of the service
            instance: The instance of the service
        """
        if service_type not in self._services:
            self._services[service_type] = {}
        
        # Use the default key for the singleton
        self._services[service_type]['default'] = instance
    
    def register_singleton_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for a singleton service.
        
        Args:
            service_type: The type of the service
            factory: A function that creates the service
        """
        if service_type not in self._factories:
            self._factories[service_type] = {}
        
        # Use the default key for the singleton factory
        self._factories[service_type]['default'] = factory
    
    def register_named_singleton(self, service_type: Type[T], name: str, instance: T) -> None:
        """Register a named singleton service instance.
        
        Args:
            service_type: The type of the service
            name: The name of the service
            instance: The instance of the service
        """
        if service_type not in self._services:
            self._services[service_type] = {}
        
        self._services[service_type][name] = instance
    
    def register_named_singleton_factory(self, service_type: Type[T], name: str, factory: Callable[[], T]) -> None:
        """Register a factory function for a named singleton service.
        
        Args:
            service_type: The type of the service
            name: The name of the service
            factory: A function that creates the service
        """
        if service_type not in self._factories:
            self._factories[service_type] = {}
        
        self._factories[service_type][name] = factory
    
    def get_service(self, service_type: Type[T], name: str = 'default') -> T:
        """Get a service instance.
        
        Args:
            service_type: The type of the service
            name: The name of the service (default: 'default')
            
        Returns:
            The service instance
            
        Raises:
            ConfigurationError: If the service is not registered
        """
        # Check if we already have an instance
        if service_type in self._services and name in self._services[service_type]:
            return cast(T, self._services[service_type][name])
        
        # Check if we have a factory for this service
        if service_type in self._factories and name in self._factories[service_type]:
            # Create the instance and cache it for future use
            instance = self._factories[service_type][name]()
            
            if service_type not in self._services:
                self._services[service_type] = {}
            
            self._services[service_type][name] = instance
            return cast(T, instance)
        
        raise ConfigurationError(f"Service of type {service_type.__name__} with name '{name}' is not registered")
    
    def has_service(self, service_type: Type[T], name: str = 'default') -> bool:
        """Check if a service is registered.
        
        Args:
            service_type: The type of the service
            name: The name of the service (default: 'default')
            
        Returns:
            True if the service is registered, False otherwise
        """
        if service_type in self._services and name in self._services[service_type]:
            return True
        
        if service_type in self._factories and name in self._factories[service_type]:
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered services and factories."""
        self._services.clear()
        self._factories.clear()
        self._instances.clear()


# Global service provider instance
_service_provider: Optional[ServiceProvider] = None


def get_service_provider() -> ServiceProvider:
    """Get the global service provider instance.
    
    Returns:
        The service provider
    """
    global _service_provider
    if _service_provider is None:
        _service_provider = ServiceProvider()
    return _service_provider


def get_service(service_type: Type[T], name: str = 'default') -> T:
    """Get a service instance from the global service provider.
    
    Args:
        service_type: The type of the service
        name: The name of the service (default: 'default')
        
    Returns:
        The service instance
    """
    provider = get_service_provider()
    return provider.get_service(service_type, name)


def inject(
    **dependencies: Union[Type, tuple]
) -> Callable:
    """Decorator to inject dependencies into a function.
    
    This decorator injects dependencies from the service provider
    into the decorated function.
    
    Args:
        **dependencies: Mapping of parameter names to service types,
            optionally with a tuple of (service_type, name)
    
    Returns:
        The decorated function
    
    Example:
        @inject(client=SolanaClient, rpc_service=(RPCService, 'mainnet'))
        def my_function(client, rpc_service):
            # Use client and rpc_service
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            provider = get_service_provider()
            
            # Get the original type hints
            sig = inspect.signature(func)
            
            for param_name, service_info in dependencies.items():
                # Skip if already provided in kwargs
                if param_name in kwargs:
                    continue
                
                service_name = 'default'
                
                if isinstance(service_info, tuple):
                    service_type, service_name = service_info
                else:
                    service_type = service_info
                
                # Inject the dependency into kwargs
                kwargs[param_name] = provider.get_service(service_type, service_name)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def inject_by_type(**name_overrides: str) -> Callable:
    """Decorator to inject dependencies based on parameter types.
    
    This decorator injects dependencies from the service provider
    based on the type annotations of the decorated function.
    
    Args:
        **name_overrides: Mapping of parameter names to service names,
            allowing to override the default service name
    
    Returns:
        The decorated function
    
    Example:
        @inject_by_type(rpc_service='mainnet')
        def my_function(client: SolanaClient, rpc_service: RPCService):
            # Use client and rpc_service
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            provider = get_service_provider()
            
            # Get the parameter types from type hints
            type_hints = get_type_hints(func)
            sig = inspect.signature(func)
            
            for param_name, param in sig.parameters.items():
                # Skip if already provided in args or kwargs
                if param_name in kwargs:
                    continue
                
                # Skip *args and **kwargs
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                
                # Skip if no type hint
                if param_name not in type_hints:
                    continue
                
                # Skip if it's a built-in type
                param_type = type_hints[param_name]
                if param_type in (str, int, float, bool, list, dict, tuple, set):
                    continue
                
                # Check if we have a service of this type
                service_name = name_overrides.get(param_name, 'default')
                
                if provider.has_service(param_type, service_name):
                    # Inject the dependency into kwargs
                    kwargs[param_name] = provider.get_service(param_type, service_name)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator 