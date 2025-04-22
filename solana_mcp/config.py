"""Configuration module for the Solana MCP server."""

# Standard library imports
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import lru_cache

# Third-party library imports
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class SolanaConfig:
    """Configuration for Solana RPC connection."""
    
    rpc_url: str
    rpc_user: Optional[str] = None
    rpc_password: Optional[str] = None
    commitment: str = "confirmed"
    timeout: int = 30  # seconds
    
    @property
    def has_auth(self) -> bool:
        """Check if authentication credentials are provided.
        
        Returns:
            True if both username and password are set, False otherwise
        """
        return bool(self.rpc_user and self.rpc_password)


@lru_cache()
def get_solana_config() -> SolanaConfig:
    """Get Solana configuration from environment variables.
    
    Uses cached values for efficiency.
    
    Returns:
        SolanaConfig instance
    """
    return SolanaConfig(
        rpc_url=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
        rpc_user=os.getenv("SOLANA_RPC_USER"),
        rpc_password=os.getenv("SOLANA_RPC_PASSWORD"),
        commitment=os.getenv("SOLANA_COMMITMENT", "confirmed"),
        timeout=int(os.getenv("SOLANA_TIMEOUT", "30"))
    )


@dataclass
class ServerConfig:
    """Configuration for the server."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    
    @property
    def bind_address(self) -> str:
        """Get the bind address for the server.
        
        Returns:
            Formatted bind address
        """
        return f"{self.host}:{self.port}"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.environment not in ("development", "staging", "production"):
            raise ValueError(f"Invalid environment: {self.environment}")
        
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {self.log_level}")


@lru_cache()
def get_server_config() -> ServerConfig:
    """Get server configuration from environment variables.
    
    Returns:
        ServerConfig instance
    """
    return ServerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("DEBUG", "").lower() in ("true", "1", "yes"),
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )


@dataclass
class CacheConfig:
    """Configuration for caching."""
    
    metadata_cache_size: int = 100
    metadata_cache_ttl: int = 300  # seconds
    price_cache_size: int = 500
    price_cache_ttl: int = 60  # seconds


@lru_cache()
def get_cache_config() -> CacheConfig:
    """Get cache configuration from environment variables.
    
    Returns:
        CacheConfig instance
    """
    return CacheConfig(
        metadata_cache_size=int(os.getenv("METADATA_CACHE_SIZE", "100")),
        metadata_cache_ttl=int(os.getenv("METADATA_CACHE_TTL", "300")),
        price_cache_size=int(os.getenv("PRICE_CACHE_SIZE", "500")),
        price_cache_ttl=int(os.getenv("PRICE_CACHE_TTL", "60"))
    )


@dataclass
class SessionConfig:
    """Configuration for session management."""
    
    expiry_minutes: int = int(os.environ.get("SESSION_EXPIRY_MINUTES", "30"))
    cleanup_interval_seconds: int = int(os.environ.get("SESSION_CLEANUP_INTERVAL", "300"))
    storage_type: str = os.environ.get("SESSION_STORAGE", "memory")  # "memory" or "redis"
    redis_url: Optional[str] = os.environ.get("REDIS_URL", None)
    secret_key: str = os.environ.get("SESSION_SECRET_KEY", "solana-mcp-server-secret")


@dataclass
class APIConfig:
    """Configuration for API endpoints."""
    
    cors_origins: List[str] = field(default_factory=lambda: 
        os.environ.get("CORS_ORIGINS", "*").split(","))
    rate_limit_enabled: bool = os.environ.get("RATE_LIMIT_ENABLED", "").lower() in ("true", "1", "yes")
    rate_limit_requests: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window_seconds: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
    default_response_format: str = os.environ.get("DEFAULT_RESPONSE_FORMAT", "standard")
    max_response_size: int = int(os.environ.get("MAX_RESPONSE_SIZE", "1000000"))


@dataclass
class AppConfig:
    """Comprehensive application configuration."""
    
    solana: SolanaConfig = field(default_factory=SolanaConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Feature flags
    enable_websockets: bool = os.environ.get("ENABLE_WEBSOCKETS", "").lower() in ("true", "1", "yes")
    enable_monitoring: bool = os.environ.get("ENABLE_MONITORING", "").lower() in ("true", "1", "yes")
    enable_cache: bool = os.environ.get("ENABLE_CACHE", "").lower() in ("true", "1", "yes")
    
    # General app settings
    environment: str = os.environ.get("ENVIRONMENT", "development")
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    
    # Additional settings as needed
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Perform validation after initialization."""
        # Add any validation logic here
        if self.environment not in ("development", "staging", "production"):
            raise ValueError(f"Invalid environment: {self.environment}")
        
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {self.log_level}")


def get_app_config() -> AppConfig:
    """Get the comprehensive application configuration."""
    return AppConfig() 