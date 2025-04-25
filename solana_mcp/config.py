"""Configuration module for the Solana MCP server."""

# Standard library imports
import os
from typing import Dict, Any, Optional, List, Union, cast
from dataclasses import dataclass, field
from functools import lru_cache
import re
from decimal import Decimal, InvalidOperation

# Third-party library imports
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_var(key: str, default: Any = None, required: bool = False, 
               validator: Optional[callable] = None) -> Any:
    """Get and validate environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not present
        required: If True, raises ValueError when not found
        validator: Optional validation function
        
    Returns:
        The environment variable value or default
        
    Raises:
        ValueError: If required and not found, or fails validation
    """
    value = os.environ.get(key)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' not found")
        return default
    
    if validator and value is not None:
        try:
            return validator(value)
        except Exception as e:
            raise ValueError(f"Invalid value for environment variable '{key}': {str(e)}")
    
    return value


def bool_validator(value: str) -> bool:
    """Validate and convert string to boolean.
    
    Args:
        value: String value to convert
        
    Returns:
        Boolean value
    """
    return value.lower() in ("true", "1", "yes", "y", "on")


def int_validator(value: str) -> int:
    """Validate and convert string to integer.
    
    Args:
        value: String value to convert
        
    Returns:
        Integer value
        
    Raises:
        ValueError: If not a valid integer
    """
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid integer")


def url_validator(value: str) -> str:
    """Validate URL format.
    
    Args:
        value: URL to validate
        
    Returns:
        The validated URL
        
    Raises:
        ValueError: If not a valid URL format
    """
    url_pattern = re.compile(
        r'^(https?):\/\/'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IPv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(value):
        raise ValueError(f"'{value}' is not a valid URL")
    return value


def commitment_validator(value: str) -> str:
    """Validate Solana commitment level.
    
    Args:
        value: Commitment level to validate
        
    Returns:
        The validated commitment level
        
    Raises:
        ValueError: If not a valid commitment level
    """
    valid_commitments = ("processed", "confirmed", "finalized")
    if value.lower() not in valid_commitments:
        raise ValueError(f"Commitment must be one of: {', '.join(valid_commitments)}")
    return value.lower()


def log_level_validator(value: str) -> str:
    """Validate log level.
    
    Args:
        value: Log level to validate
        
    Returns:
        The validated log level
        
    Raises:
        ValueError: If not a valid log level
    """
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    upper_value = value.upper()
    if upper_value not in valid_levels:
        raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
    return upper_value


def environment_validator(value: str) -> str:
    """Validate environment name.
    
    Args:
        value: Environment name to validate
        
    Returns:
        The validated environment name
        
    Raises:
        ValueError: If not a valid environment name
    """
    valid_environments = ("development", "testing", "staging", "production")
    if value.lower() not in valid_environments:
        raise ValueError(f"Environment must be one of: {', '.join(valid_environments)}")
    return value.lower()


def decimal_validator(value: str) -> Decimal:
    """Validate and convert string to Decimal.
    
    Args:
        value: String value to convert
        
    Returns:
        Decimal value
        
    Raises:
        ValueError: If not a valid decimal number
    """
    try:
        return Decimal(value)
    except InvalidOperation:
        raise ValueError(f"'{value}' is not a valid decimal number")


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
        
    Raises:
        ValueError: If environment variables fail validation
    """
    return SolanaConfig(
        rpc_url=get_env_var("SOLANA_RPC_URL", "https://api.devnet.solana.com", 
                            validator=url_validator),
        rpc_user=get_env_var("SOLANA_RPC_USER"),
        rpc_password=get_env_var("SOLANA_RPC_PASSWORD"),
        commitment=get_env_var("SOLANA_COMMITMENT", "confirmed", 
                              validator=commitment_validator),
        timeout=get_env_var("SOLANA_TIMEOUT", 30, validator=int_validator)
    )


@dataclass
class ServerConfig:
    """Configuration for the server."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    transport: str = "stdio"  # "stdio" or "sse"
    
    @property
    def bind_address(self) -> str:
        """Get the bind address for the server.
        
        Returns:
            Formatted bind address
        """
        return f"{self.host}:{self.port}"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.environment not in ("development", "testing", "staging", "production"):
            raise ValueError(f"Invalid environment: {self.environment}")
        
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {self.log_level}")
            
        if self.transport not in ("stdio", "sse"):
            raise ValueError(f"Invalid transport: {self.transport}")


@lru_cache()
def get_server_config() -> ServerConfig:
    """Get server configuration from environment variables.
    
    Returns:
        ServerConfig instance
        
    Raises:
        ValueError: If environment variables fail validation
    """
    return ServerConfig(
        host=get_env_var("HOST", "0.0.0.0"),
        port=get_env_var("PORT", 8000, validator=int_validator),
        debug=get_env_var("DEBUG", False, validator=bool_validator),
        environment=get_env_var("ENVIRONMENT", "development", validator=environment_validator),
        log_level=get_env_var("LOG_LEVEL", "INFO", validator=log_level_validator),
        transport=get_env_var("TRANSPORT", "stdio")
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
class AnalysisConfig:
    """Configuration for Whale/Fresh wallet analysis."""
    whale_holder_limit: int = 50
    whale_total_value_threshold_usd: Decimal = Decimal("50000")
    whale_supply_percentage_threshold: Decimal = Decimal("1.0")
    fresh_wallet_holder_limit: int = 100
    fresh_wallet_tx_limit: int = 10
    fresh_wallet_max_age_days: int = 30
    fresh_wallet_max_tokens_low_diversity: int = 4
    fresh_wallet_max_tokens_new_wallet: int = 5
    min_token_value_usd_threshold: Decimal = Decimal("1.0")
    estimated_sol_price_usd: Decimal = Decimal("150.0") # Fallback if dynamic fetch fails


@lru_cache()
def get_analysis_config() -> AnalysisConfig:
    """Get Analysis configuration from environment variables."""
    return AnalysisConfig(
        whale_holder_limit=get_env_var("WHALE_HOLDER_LIMIT", 50, validator=int_validator),
        whale_total_value_threshold_usd=get_env_var("WHALE_TOTAL_VALUE_THRESHOLD_USD", Decimal("50000"), validator=decimal_validator),
        whale_supply_percentage_threshold=get_env_var("WHALE_SUPPLY_PERCENTAGE_THRESHOLD", Decimal("1.0"), validator=decimal_validator),
        fresh_wallet_holder_limit=get_env_var("FRESH_WALLET_HOLDER_LIMIT", 100, validator=int_validator),
        fresh_wallet_tx_limit=get_env_var("FRESH_WALLET_TX_LIMIT", 10, validator=int_validator),
        fresh_wallet_max_age_days=get_env_var("FRESH_WALLET_MAX_AGE_DAYS", 30, validator=int_validator),
        fresh_wallet_max_tokens_low_diversity=get_env_var("FRESH_WALLET_MAX_TOKENS_LOW_DIVERSITY", 4, validator=int_validator),
        fresh_wallet_max_tokens_new_wallet=get_env_var("FRESH_WALLET_MAX_TOKENS_NEW_WALLET", 5, validator=int_validator),
        min_token_value_usd_threshold=get_env_var("MIN_TOKEN_VALUE_USD_THRESHOLD", Decimal("1.0"), validator=decimal_validator),
        estimated_sol_price_usd=get_env_var("ESTIMATED_SOL_PRICE_USD", Decimal("150.0"), validator=decimal_validator) # Fallback SOL price
    )


@dataclass
class AppConfig:
    """Comprehensive application configuration."""
    
    solana: SolanaConfig = field(default_factory=get_solana_config)
    server: ServerConfig = field(default_factory=get_server_config)
    session: SessionConfig = field(default_factory=SessionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    analysis: AnalysisConfig = field(default_factory=get_analysis_config)
    
    # Feature flags
    enable_monitoring: bool = os.environ.get("ENABLE_MONITORING", "").lower() in ("true", "1", "yes")
    enable_cache: bool = os.environ.get("ENABLE_CACHE", "").lower() in ("true", "1", "yes")
    
    # General app settings
    environment: str = os.environ.get("ENVIRONMENT", "development")
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    
    # Additional settings as needed
    extra: Dict[str, Any] = field(default_factory=dict)


def get_app_config() -> AppConfig:
    """Get the comprehensive application configuration."""
    return AppConfig() 