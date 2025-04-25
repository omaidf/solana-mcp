"""Solana MCP Package.

This package provides utilities for interacting with the Solana blockchain,
including transaction parsing, program analysis, and more.
"""

import logging
from typing import Dict, List, Optional, Any

from solana_mcp.utils.config import (
    get_settings, 
    get_solana_settings,
    get_app_settings,
    configure_logging
)
from solana_mcp.utils.dependency_injection import ServiceProvider
from solana_mcp.utils.error_handling import ConfigurationError, SolanaMCPError
from solana_mcp.services.rpc_service import RPCService, get_rpc_service

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

def initialize_application(
    config_overrides: Optional[Dict[str, Any]] = None
) -> ServiceProvider:
    """Initialize the Solana MCP application.
    
    This function sets up the application's core components:
    - Configures logging
    - Initializes settings
    - Sets up the dependency injection container
    - Registers core services
    
    Args:
        config_overrides: Optional dictionary of config values to override
        
    Returns:
        Initialized ServiceProvider instance
        
    Raises:
        ConfigurationError: If there's an issue with the configuration
        SolanaMCPError: For other initialization errors
    """
    # Apply config overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            import os
            os.environ[key] = str(value)
    
    # Configure logging first
    try:
        configure_logging()
    except Exception as e:
        raise ConfigurationError(
            f"Failed to configure logging: {str(e)}",
            details={"original_error": str(e)}
        )
    
    logger.info(f"Initializing Solana MCP v{__version__}")
    
    # Get settings
    try:
        settings = get_settings()
        solana_settings = get_solana_settings()
        app_settings = get_app_settings()
    except ConfigurationError:
        logger.exception("Failed to load configuration")
        raise
    
    # Initialize dependency injection container
    provider = ServiceProvider.get_instance()
    
    # Register core services
    try:
        # RPC Service
        rpc_service = RPCService(
            rpc_url=solana_settings.RPC_URL,
            timeout=solana_settings.REQUEST_TIMEOUT,
            max_retries=solana_settings.MAX_RETRIES
        )
        provider.register_singleton(RPCService, rpc_service)
        
        # Register settings objects
        provider.register_singleton("settings", settings)
        provider.register_singleton("solana_settings", solana_settings)
        provider.register_singleton("app_settings", app_settings)
        
        logger.info(f"Core services registered in dependency injection container")
    except Exception as e:
        logger.exception("Failed to register core services")
        raise SolanaMCPError(
            f"Failed to initialize application: {str(e)}",
            details={"original_error": str(e)}
        )
    
    # Log initialization success
    env = app_settings.ENV
    logger.info(f"Solana MCP initialized successfully in {env} environment")
    
    return provider 