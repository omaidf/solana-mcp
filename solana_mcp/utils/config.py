"""
Configuration management for Solana MCP.

This module provides utilities for loading and accessing configuration settings
from various sources (environment variables, config files, etc.).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from solana_mcp.utils.error_handling import ConfigurationError, ErrorCode

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class SolanaSettings:
    """Solana blockchain connection settings."""
    
    # RPC endpoint URLs
    RPC_URL: str
    WEBSOCKET_URL: Optional[str] = None
    
    # Connection settings
    REQUEST_TIMEOUT: float = 30.0
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Commitment level
    COMMITMENT: str = "confirmed"
    
    # Rate limiting
    RATE_LIMIT: Optional[int] = None
    
    def validate(self) -> None:
        """Validate Solana settings.
        
        Raises:
            ConfigurationError: If settings are invalid
        """
        if not self.RPC_URL:
            raise ConfigurationError(
                "Solana RPC URL is required",
                details={"setting": "RPC_URL"}
            )
        
        if self.REQUEST_TIMEOUT <= 0:
            raise ConfigurationError(
                "Request timeout must be positive",
                details={"setting": "REQUEST_TIMEOUT", "value": self.REQUEST_TIMEOUT}
            )
        
        if self.MAX_RETRIES < 0:
            raise ConfigurationError(
                "Max retries must be non-negative",
                details={"setting": "MAX_RETRIES", "value": self.MAX_RETRIES}
            )
        
        valid_commitments = ["processed", "confirmed", "finalized"]
        if self.COMMITMENT not in valid_commitments:
            raise ConfigurationError(
                f"Invalid commitment level: {self.COMMITMENT}",
                details={
                    "setting": "COMMITMENT",
                    "value": self.COMMITMENT,
                    "valid_values": valid_commitments
                }
            )


@dataclass
class DatabaseSettings:
    """Database connection settings."""
    
    # Connection URL or parts
    DATABASE_URL: Optional[str] = None
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "solana_mcp"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    
    # Connection pool settings
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    
    def get_connection_url(self) -> str:
        """Get database connection URL.
        
        Returns:
            Database connection URL
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    def validate(self) -> None:
        """Validate database settings.
        
        Raises:
            ConfigurationError: If settings are invalid
        """
        if not self.DATABASE_URL and not all([self.DB_HOST, self.DB_NAME, self.DB_USER]):
            raise ConfigurationError(
                "Database connection information is incomplete",
                details={
                    "DATABASE_URL": self.DATABASE_URL,
                    "DB_HOST": self.DB_HOST,
                    "DB_NAME": self.DB_NAME
                }
            )
        
        if self.DB_PORT <= 0 or self.DB_PORT > 65535:
            raise ConfigurationError(
                f"Invalid database port: {self.DB_PORT}",
                details={"setting": "DB_PORT", "value": self.DB_PORT}
            )
        
        if self.DB_POOL_SIZE <= 0:
            raise ConfigurationError(
                f"Invalid pool size: {self.DB_POOL_SIZE}",
                details={"setting": "DB_POOL_SIZE", "value": self.DB_POOL_SIZE}
            )


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: Optional[str] = None
    LOG_TO_CONSOLE: bool = True
    
    def validate(self) -> None:
        """Validate logging settings.
        
        Raises:
            ConfigurationError: If settings are invalid
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.LOG_LEVEL}",
                details={
                    "setting": "LOG_LEVEL",
                    "value": self.LOG_LEVEL,
                    "valid_values": valid_levels
                }
            )


@dataclass
class AppSettings:
    """Application settings."""
    
    # Environment
    ENV: str = "development"
    DEBUG: bool = False
    
    # Feature flags
    FEATURES: Dict[str, bool] = field(default_factory=dict)
    
    # Performance tuning
    WORKER_COUNT: int = 4
    BATCH_SIZE: int = 100
    
    # Monitoring
    ENABLE_METRICS: bool = False
    METRICS_PORT: int = 9090
    
    def validate(self) -> None:
        """Validate application settings.
        
        Raises:
            ConfigurationError: If settings are invalid
        """
        valid_envs = ["development", "testing", "production"]
        if self.ENV not in valid_envs:
            raise ConfigurationError(
                f"Invalid environment: {self.ENV}",
                details={
                    "setting": "ENV",
                    "value": self.ENV,
                    "valid_values": valid_envs
                }
            )
        
        if self.WORKER_COUNT <= 0:
            raise ConfigurationError(
                f"Worker count must be positive: {self.WORKER_COUNT}",
                details={"setting": "WORKER_COUNT", "value": self.WORKER_COUNT}
            )
        
        if self.BATCH_SIZE <= 0:
            raise ConfigurationError(
                f"Batch size must be positive: {self.BATCH_SIZE}",
                details={"setting": "BATCH_SIZE", "value": self.BATCH_SIZE}
            )


@dataclass
class Settings:
    """Global application settings."""
    
    # Component settings
    solana: SolanaSettings
    database: DatabaseSettings
    logging: LoggingSettings
    app: AppSettings
    
    # Allow additional custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate all settings.
        
        Raises:
            ConfigurationError: If any settings are invalid
        """
        self.solana.validate()
        self.database.validate()
        self.logging.validate()
        self.app.validate()


def load_from_env() -> Settings:
    """Load settings from environment variables.
    
    Returns:
        Settings object with values from environment variables
    """
    # Solana settings
    solana = SolanaSettings(
        RPC_URL=os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
        WEBSOCKET_URL=os.environ.get("SOLANA_WEBSOCKET_URL"),
        REQUEST_TIMEOUT=float(os.environ.get("SOLANA_REQUEST_TIMEOUT", "30.0")),
        MAX_RETRIES=int(os.environ.get("SOLANA_MAX_RETRIES", "3")),
        RETRY_DELAY=float(os.environ.get("SOLANA_RETRY_DELAY", "1.0")),
        COMMITMENT=os.environ.get("SOLANA_COMMITMENT", "confirmed"),
        RATE_LIMIT=int(os.environ.get("SOLANA_RATE_LIMIT", "0")) or None,
    )
    
    # Database settings
    database = DatabaseSettings(
        DATABASE_URL=os.environ.get("DATABASE_URL"),
        DB_HOST=os.environ.get("DB_HOST", "localhost"),
        DB_PORT=int(os.environ.get("DB_PORT", "5432")),
        DB_NAME=os.environ.get("DB_NAME", "solana_mcp"),
        DB_USER=os.environ.get("DB_USER", "postgres"),
        DB_PASSWORD=os.environ.get("DB_PASSWORD", ""),
        DB_POOL_SIZE=int(os.environ.get("DB_POOL_SIZE", "5")),
        DB_MAX_OVERFLOW=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
        DB_POOL_TIMEOUT=int(os.environ.get("DB_POOL_TIMEOUT", "30")),
    )
    
    # Logging settings
    logging = LoggingSettings(
        LOG_LEVEL=os.environ.get("LOG_LEVEL", "INFO"),
        LOG_FORMAT=os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        LOG_DATE_FORMAT=os.environ.get("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        LOG_FILE=os.environ.get("LOG_FILE"),
        LOG_TO_CONSOLE=os.environ.get("LOG_TO_CONSOLE", "true").lower() == "true",
    )
    
    # App settings
    app = AppSettings(
        ENV=os.environ.get("ENV", "development"),
        DEBUG=os.environ.get("DEBUG", "false").lower() == "true",
        WORKER_COUNT=int(os.environ.get("WORKER_COUNT", "4")),
        BATCH_SIZE=int(os.environ.get("BATCH_SIZE", "100")),
        ENABLE_METRICS=os.environ.get("ENABLE_METRICS", "false").lower() == "true",
        METRICS_PORT=int(os.environ.get("METRICS_PORT", "9090")),
    )
    
    # Feature flags
    features = {}
    for key, value in os.environ.items():
        if key.startswith("FEATURE_"):
            feature_name = key[8:].lower()
            features[feature_name] = value.lower() == "true"
    app.FEATURES = features
    
    # Custom settings
    custom = {}
    for key, value in os.environ.items():
        if key.startswith("CUSTOM_"):
            custom_name = key[7:].lower()
            custom[custom_name] = value
    
    return Settings(
        solana=solana,
        database=database,
        logging=logging,
        app=app,
        custom=custom
    )


# Global settings instance
_settings: Optional[Settings] = None


def initialize_settings() -> Settings:
    """Initialize settings from all available sources.
    
    This function loads settings from the environment and validates them.
    
    Returns:
        Validated Settings object
    """
    global _settings
    
    if _settings is None:
        # Load settings from environment
        settings = load_from_env()
        
        # Validate settings
        try:
            settings.validate()
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        
        _settings = settings
    
    return _settings


def get_settings() -> Settings:
    """Get the current settings.
    
    If settings have not been initialized, this will initialize them.
    
    Returns:
        Current settings object
    """
    global _settings
    
    if _settings is None:
        return initialize_settings()
    
    return _settings


def get_solana_settings() -> SolanaSettings:
    """Get Solana-specific settings.
    
    Returns:
        Solana settings
    """
    return get_settings().solana


def get_database_settings() -> DatabaseSettings:
    """Get database-specific settings.
    
    Returns:
        Database settings
    """
    return get_settings().database


def get_logging_settings() -> LoggingSettings:
    """Get logging-specific settings.
    
    Returns:
        Logging settings
    """
    return get_settings().logging


def get_app_settings() -> AppSettings:
    """Get application-specific settings.
    
    Returns:
        Application settings
    """
    return get_settings().app


def configure_logging() -> None:
    """Configure logging based on current settings."""
    settings = get_logging_settings()
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(settings.LOG_FORMAT, settings.LOG_DATE_FORMAT)
    
    # Add console handler if enabled
    if settings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if settings.LOG_FILE:
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level {settings.LOG_LEVEL}") 