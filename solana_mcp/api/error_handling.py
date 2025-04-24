"""Error handling utilities for API endpoints.

DEPRECATED: This module contains duplicate functionality with solana_mcp/decorators.py.
For better code organization, these utilities should be consolidated into a single location.
New code should use decorators from solana_mcp/decorators.py instead.
"""

import functools
import logging
import traceback
import warnings
from typing import Callable, Any, Dict, Awaitable

from starlette.responses import JSONResponse

from solana_mcp.solana_client import SolanaRpcError, InvalidPublicKeyError
from solana_mcp.decorators import (
    api_error_handler as _api_error_handler,
    rate_limit as _rate_limit,
)

# Set up logger
logger = logging.getLogger(__name__)

# Show deprecation warning
warnings.warn(
    "solana_mcp.api.error_handling is deprecated. "
    "Please use solana_mcp.decorators instead.",
    DeprecationWarning,
    stacklevel=2
)


def explain_solana_error(error_message: str) -> str:
    """Provide a human-readable explanation for a Solana error.
    
    Args:
        error_message: The error message
        
    Returns:
        Human-readable explanation
    """
    # Import from a central location for error explanations
    from solana_mcp.utils.error_codes import SOLANA_ERROR_EXPLANATIONS
    
    for error_type, explanation in SOLANA_ERROR_EXPLANATIONS.items():
        if error_type in error_message:
            return explanation
    
    # Default explanation if specific error not found
    return "An error occurred when interacting with the Solana blockchain. Check your inputs and try again."


def with_error_handling(func: Callable[..., Awaitable[JSONResponse]]) -> Callable[..., Awaitable[JSONResponse]]:
    """Decorator to handle common errors in REST API endpoints.
    
    DEPRECATED: Use solana_mcp.decorators.api_error_handler instead.
    
    Args:
        func: The endpoint function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    warnings.warn(
        "with_error_handling is deprecated. "
        "Please use solana_mcp.decorators.api_error_handler instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> JSONResponse:
        try:
            return await func(*args, **kwargs)
        except InvalidPublicKeyError as e:
            logger.warning(f"Invalid public key: {str(e)}")
            return JSONResponse({
                "error": str(e),
                "error_explanation": explain_solana_error(str(e))
            }, status_code=400)
        except SolanaRpcError as e:
            logger.error(f"Solana RPC error: {str(e)}")
            return JSONResponse({
                "error": str(e),
                "error_explanation": explain_solana_error(str(e)),
                "details": getattr(e, "error_data", {})
            }, status_code=500)
        except ValueError as e:
            logger.warning(f"Value error: {str(e)}")
            return JSONResponse({
                "error": str(e),
                "error_explanation": "Invalid input value provided."
            }, status_code=400)
        except Exception as e:
            # Log the full traceback for unexpected errors
            logger.error(f"Unexpected error: {str(e)}")
            logger.debug(traceback.format_exc())
            return JSONResponse({
                "error": f"Unexpected error: {str(e)}",
                "error_explanation": "An unexpected error occurred while processing your request."
            }, status_code=500)
    
    return wrapper


def with_validation(validation_class: Any) -> Callable:
    """Decorator to validate request body against a model.
    
    DEPRECATED: Use solana_mcp.decorators equivalent instead.
    
    Args:
        validation_class: Pydantic model class for validation
        
    Returns:
        Decorator function
    """
    warnings.warn(
        "with_validation is deprecated. "
        "Please use solana_mcp.decorators equivalent instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request, *args, **kwargs):
            try:
                # Extract body
                body = await request.json()
                
                # Validate with model
                validated_data = validation_class(**body)
                
                # Add validated data to kwargs
                kwargs['validated_data'] = validated_data
                
                return await func(request, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Validation error: {str(e)}")
                return JSONResponse({
                    "error": f"Validation error: {str(e)}",
                    "error_explanation": "The request data failed validation requirements."
                }, status_code=400)
        
        return wrapper
    
    return decorator


def rate_limit(requests_per_minute: int = 60) -> Callable:
    """Decorator to apply rate limiting to an endpoint.
    
    DEPRECATED: Use solana_mcp.decorators.rate_limit instead.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        
    Returns:
        Decorator function
    """
    warnings.warn(
        "rate_limit is deprecated. "
        "Please use solana_mcp.decorators.rate_limit instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return _rate_limit(limit=requests_per_minute, period=60)


# Re-export from decorators.py with a deprecation warning
def api_error_handler(func: Callable) -> Callable:
    """Decorator to handle common exceptions in API endpoints.
    
    DEPRECATED: Use solana_mcp.decorators.api_error_handler instead.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    warnings.warn(
        "api_error_handler is deprecated. "
        "Please use solana_mcp.decorators.api_error_handler instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    
    return _api_error_handler(func) 