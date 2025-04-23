"""Logging configuration for the Solana MCP server."""

import logging
import sys
import os
from typing import Dict, Any, Optional

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure default logging
def configure_logging(log_level: str = "INFO", log_format: Optional[str] = None):
    """Configure global logging settings.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format string
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        stream=sys.stdout
    )
    
    # Set third-party loggers to a higher level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, 
                     level: str, 
                     message: str, 
                     **context) -> None:
    """Log a message with additional context information.
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        context: Additional context information as keyword arguments
    """
    # Add file, function, and line number if not provided
    frame = sys._getframe(1)
    if "module" not in context:
        context["module"] = frame.f_globals["__name__"]
    if "function" not in context:
        context["function"] = frame.f_code.co_name
    if "line" not in context:
        context["line"] = frame.f_lineno
    
    # Format context as JSON-like string
    context_str = ", ".join(f"{k}={repr(v)}" for k, v in context.items())
    
    # Log with the specified level
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"{message}")


class RequestIdMiddleware:
    """FastAPI middleware to add request ID to each request."""
    
    def __init__(self, app):
        """Initialize middleware.
        
        Args:
            app: FastAPI application
        """
        self.app = app
        self.logger = get_logger("middleware")
        
    async def __call__(self, scope, receive, send):
        """Process request with added request ID.
        
        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to scope for use in route handlers
        scope["request_id"] = request_id
        
        # Log request
        path = scope.get("path", "unknown")
        method = scope.get("method", "unknown")
        log_with_context(
            self.logger,
            "info",
            f"Request received: {method} {path}",
            request_id=request_id,
            method=method,
            path=path
        )
        
        # Process request
        start_time = datetime.now()
        
        # Custom send to track response
        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                # Get status code
                status = message.get("status", 0)
                
                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log response
                log_with_context(
                    self.logger,
                    "info",
                    f"Response: {status} - {duration:.2f}ms",
                    request_id=request_id,
                    status=status,
                    duration=duration,
                    method=method,
                    path=path
                )
                
            await send(message)
            
        await self.app(scope, receive, wrapped_send)


def setup_logging():
    """Configure application-wide logging settings."""
    # Set default level for root logger
    logging.basicConfig(
        level=getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    ) 