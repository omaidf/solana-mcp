"""
Configuration management for the Solana MCP API.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use `solana_mcp.config` instead.
"""

import warnings
import os
from functools import lru_cache

from solana_mcp.config import (
    get_env_var, bool_validator, int_validator, url_validator,
    commitment_validator, log_level_validator, environment_validator,
    SolanaConfig, get_solana_config, ServerConfig, get_server_config,
    CacheConfig, get_cache_config, SessionConfig, APIConfig, AppConfig,
    get_app_config
)

# Show deprecation warning
warnings.warn(
    "The `solana_mcp.utils.config` module is deprecated. "
    "Use `solana_mcp.config` instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility
class Settings:
    """
    Compatibility class for old settings.
    """
    
    def __init__(self):
        """Initialize with values from the new config system."""
        server_config = get_server_config()
        solana_config = get_solana_config()
        cache_config = get_cache_config()
        app_config = get_app_config()
        
        # Core settings
        self.APP_NAME = "Solana MCP API"
        self.APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
        self.DEBUG = server_config.debug
        
        # Solana settings
        self.SOLANA_RPC_URL = solana_config.rpc_url
        self.SOLANA_TIMEOUT = solana_config.timeout
        self.RPC_TIMEOUT = float(solana_config.timeout)
        
        # Cache settings
        self.MAX_CACHE_SIZE = cache_config.metadata_cache_size
        self.CACHE_TTL = cache_config.metadata_cache_ttl
        
        # Server settings
        self.HOST = server_config.host
        self.PORT = server_config.port
        self.ENVIRONMENT = server_config.environment
        self.LOG_LEVEL = server_config.log_level
        self.VERSION = self.APP_VERSION

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching (for backward compatibility).
    
    Returns:
        Settings: Application settings object
    """
    return Settings() 