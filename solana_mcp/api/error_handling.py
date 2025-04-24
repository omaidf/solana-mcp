"""Error handling utilities for API endpoints."""

import functools
import logging
import traceback
from typing import Callable, Any, Dict, Awaitable

from starlette.responses import JSONResponse

from solana_mcp.solana_client import SolanaRpcError, InvalidPublicKeyError

# Set up logger
logger = logging.getLogger(__name__)


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
    
    Args:
        func: The endpoint function to wrap
        
    Returns:
        Wrapped function with error handling
    """
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
    
    Args:
        validation_class: Pydantic model class for validation
        
    Returns:
        Decorator function
    """
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
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        
    Returns:
        Decorator function
    """
    # This is a placeholder for a more sophisticated rate limiting implementation
    # In a real application, you would use Redis or another distributed store
    rate_limit_store: Dict[str, Dict[str, Any]] = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request, *args, **kwargs):
            # Get client identifier (IP address or API key)
            client_id = request.client.host
            
            # Check rate limit
            from time import time
            current_time = time()
            
            if client_id in rate_limit_store:
                client_data = rate_limit_store[client_id]
                # Reset count if window has passed
                if current_time - client_data['window_start'] > 60:
                    client_data['count'] = 1
                    client_data['window_start'] = current_time
                else:
                    client_data['count'] += 1
                    
                # Check if limit exceeded
                if client_data['count'] > requests_per_minute:
                    logger.warning(f"Rate limit exceeded for {client_id}")
                    return JSONResponse({
                        "error": "Rate limit exceeded",
                        "error_explanation": f"You have exceeded the limit of {requests_per_minute} requests per minute."
                    }, status_code=429)
            else:
                # First request from this client
                rate_limit_store[client_id] = {
                    'count': 1,
                    'window_start': current_time
                }
                
            # Proceed with the request
            return await func(request, *args, **kwargs)
        
        return wrapper
    
    return decorator

def api_error_handler(func: Callable) -> Callable:
    """Decorator to handle common exceptions in API endpoints.
    
    This decorator is specifically designed for FastAPI endpoint functions.
    It catches exceptions and converts them to appropriate HTTP responses.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Execute the function
            return await func(*args, **kwargs)
        except InvalidPublicKeyError as e:
            logger.warning(f"Invalid public key: {str(e)}")
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"Invalid Solana public key: {str(e)}")
        except SolanaRpcError as e:
            logger.error(f"Solana RPC error: {str(e)}")
            from fastapi import HTTPException
            status_code = getattr(e, "status_code", 500)
            raise HTTPException(status_code=status_code, detail=f"Solana RPC error: {str(e)}")
        except ValueError as e:
            logger.warning(f"Value error: {str(e)}")
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    return wrapper 