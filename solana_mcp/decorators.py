"""Decorators for use throughout the solana_mcp package."""

import functools
import logging
from typing import Callable, Any, Optional

from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.solana_client import InvalidPublicKeyError, SolanaRpcError

logger = get_logger(__name__)

def validate_solana_key(func: Callable) -> Callable:
    """Decorator to validate Solana public key and handle exceptions.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with Solana key validation
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract the mint/address parameter
        mint = kwargs.get('mint')
        if mint is None and len(args) > 1:
            # Assume the first arg after self is the mint address
            mint = args[1]
            
        request_id = kwargs.get('request_id')
            
        try:
            return await func(*args, **kwargs)
        except InvalidPublicKeyError as e:
            log_with_context(
                logger,
                "error",
                f"Invalid Solana key: {mint} - {str(e)}",
                request_id=request_id,
                mint=mint
            )
            # Re-raise to be handled by the API error handler
            raise
            
    return wrapper

def handle_errors(func: Callable) -> Callable:
    """Decorator to handle common errors in Solana operations.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with error handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = kwargs.get('request_id')
        
        try:
            return await func(*args, **kwargs)
        except SolanaRpcError as e:
            log_with_context(
                logger,
                "error",
                f"Solana RPC error in {func.__name__}: {str(e)}",
                request_id=request_id,
                function=func.__name__
            )
            # Re-raise to be handled by the API error handler
            raise
        except Exception as e:
            log_with_context(
                logger,
                "error",
                f"Unexpected error in {func.__name__}: {str(e)}",
                request_id=request_id,
                function=func.__name__
            )
            # Re-raise to be handled by the API error handler
            raise
            
    return wrapper 