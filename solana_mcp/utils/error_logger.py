"""Centralized error logging utility for Solana MCP.

This module provides standardized error logging functionality for the application.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Type, TypeVar

from solana_mcp.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

# Type variable for the exception type
E = TypeVar('E', bound=Exception)

def log_error(
    error: Exception, 
    context: Optional[str] = None, 
    exc_info: bool = True,
    level: str = "error",
    **kwargs
) -> Exception:
    """Log an error with optional context.
    
    Args:
        error: The exception to log
        context: Optional context message
        exc_info: Whether to include exception info
        level: Log level to use
        **kwargs: Additional context data to include in the log
        
    Returns:
        The original exception for fluent API support
    """
    # Get the appropriate log function
    log_func = getattr(logger, level, logger.error)
    
    # Create log message
    if context:
        message = f"{context}: {str(error)}"
    else:
        message = str(error)
    
    # Log the error
    if kwargs:
        log_func(message, exc_info=exc_info, extra={"context": kwargs})
    else:
        log_func(message, exc_info=exc_info)
    
    # Return the error to allow for reraising
    return error

def catch_and_log(
    exc_types: Type[E] = Exception,
    context: Optional[str] = None,
    level: str = "error",
    reraise: bool = True,
    **kwargs
):
    """Decorator to catch and log exceptions.
    
    Args:
        exc_types: Exception types to catch
        context: Optional context message
        level: Log level to use
        reraise: Whether to reraise the exception after logging
        **kwargs: Additional context data to include in the log
        
    Returns:
        Decorator function
    """
    def decorator(func):
        async def async_wrapper(*args, **wkwargs):
            try:
                return await func(*args, **wkwargs)
            except exc_types as e:
                log_error(e, context, level=level, **kwargs)
                if reraise:
                    raise
                return None
                
        def sync_wrapper(*args, **wkwargs):
            try:
                return func(*args, **wkwargs)
            except exc_types as e:
                log_error(e, context, level=level, **kwargs)
                if reraise:
                    raise
                return None
                
        # Use appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator 