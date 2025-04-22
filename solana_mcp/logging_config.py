"""Logging configuration for the Solana MCP Server."""

# Standard library imports
import logging
import sys
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Default log level from environment
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure log format with JSON for structured logging
class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields from record
        if hasattr(record, "props"):
            log_record.update(record.props)
            
        return json.dumps(log_record)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with the given name.
    
    Args:
        name: The logger name, typically __name__
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # Choose formatter based on environment
        if os.getenv("LOG_FORMAT", "json").lower() == "json":
            handler.setFormatter(JsonFormatter())
        else:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            handler.setFormatter(logging.Formatter(log_format))
            
        logger.addHandler(handler)
        
        # Set log level
        level = getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO)
        logger.setLevel(level)
        
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    request_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log a message with context information.
    
    Args:
        logger: The logger instance
        level: Log level (info, error, warning, debug)
        message: The log message
        request_id: Optional request ID for tracing
        **kwargs: Additional context to include in log
    """
    if not hasattr(logging, level.upper()):
        level = "info"
        
    # Generate request ID if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())
        
    # Create context with request ID
    context = {"request_id": request_id, **kwargs}
    
    # Add context to record
    extra = {"props": context}
    
    # Log with proper level
    log_method = getattr(logger, level.lower())
    log_method(message, extra=extra)


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