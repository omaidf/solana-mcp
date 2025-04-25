"""Reusable decorators for the API.

This module provides decorators for common functionality like error handling.
"""

import functools
from typing import Any, Callable, TypeVar, cast

from fastapi import Response
from starlette.responses import JSONResponse

from solana_mcp.solana_client import InvalidPublicKeyError, SolanaRpcError
from solana_mcp.models.errors import ErrorResponse, explain_solana_error
from solana_mcp.utils.error_logger import log_error

# Type variable for the decorated function
F = TypeVar('F', bound=Callable[..., Any])

def handle_api_errors(func: F) -> F:
    """Decorator to handle common API errors.
    
    Catches and handles common exceptions, returning appropriate error responses.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except InvalidPublicKeyError as e:
            log_error(e, context="Invalid public key error")
            return JSONResponse(
                ErrorResponse.from_invalid_public_key(e).dict(),
                status_code=400
            )
        except SolanaRpcError as e:
            log_error(e, context="Solana RPC error")
            return JSONResponse(
                ErrorResponse.from_rpc_error(e).dict(),
                status_code=500
            )
        except Exception as e:
            log_error(e, context="Unexpected error in API endpoint", exc_info=True)
            return JSONResponse(
                ErrorResponse.from_exception(
                    e, 
                    explanation="An unexpected error occurred while processing your request."
                ).dict(),
                status_code=500
            )
    
    return cast(F, wrapper)

def validate_solana_key(func: F) -> F:
    """Decorator to validate Solana public keys.
    
    Validates that string arguments matching keys are valid Solana public keys.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        from solana_mcp.solana_client import validate_public_key
        
        # Check keyword arguments for public keys
        for key, value in kwargs.items():
            if key in ["address", "mint", "owner", "program_id", "delegate"] and isinstance(value, str):
                if not validate_public_key(value):
                    raise InvalidPublicKeyError(value)
        
        return await func(*args, **kwargs)
    
    return cast(F, wrapper)

def with_solana_client(func: F) -> F:
    """Decorator to inject a Solana client.
    
    Provides a Solana client to the decorated function.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        from solana_mcp.solana_client import get_solana_client
        
        async with get_solana_client() as solana_client:
            kwargs["solana_client"] = solana_client
            return await func(*args, **kwargs)
    
    return cast(F, wrapper) 