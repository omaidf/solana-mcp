"""Decorators for use throughout the solana_mcp package.

This module contains reusable decorators organized by category:
1. Error Handling - Decorators for handling exceptions and errors
2. Performance - Decorators for measuring and optimizing performance
3. Validation - Decorators for validating inputs
4. Request Processing - Decorators for processing API requests
"""

import functools
import logging
import time
import asyncio
import uuid
import json
from enum import Enum
from typing import Callable, Any, Dict, Optional, Union, Type, List, TypeVar

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.solana_client import InvalidPublicKeyError, SolanaRpcError

# Type variables for better type hints
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Set up logger
logger = get_logger(__name__)

# ===============================================================
# ERROR CODES AND UTILITIES
# ===============================================================

class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""
    
    # General errors
    INVALID_INPUT = "INVALID_INPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Solana-specific errors
    INVALID_PUBLIC_KEY = "INVALID_PUBLIC_KEY"
    SOLANA_RPC_ERROR = "SOLANA_RPC_ERROR"
    SOLANA_TRANSACTION_ERROR = "SOLANA_TRANSACTION_ERROR"


def format_error_response(
    error_message: str, 
    error_code: ErrorCode, 
    details: Optional[Dict[str, Any]] = None,
    explanation: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized error response format.
    
    Args:
        error_message: The error message
        error_code: The error code
        details: Additional error details
        explanation: Human-readable explanation of the error
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "error": error_message,
        "error_code": error_code,
    }
    
    if explanation:
        response["error_explanation"] = explanation
    
    if details:
        response["details"] = details
        
    return response


def create_error_response(
    error_message: str,
    error_code: ErrorCode,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None,
    explanation: Optional[str] = None
) -> JSONResponse:
    """Create a JSONResponse with standardized error format.
    
    Args:
        error_message: The error message
        error_code: The error code
        status_code: HTTP status code
        details: Additional error details
        explanation: Human-readable explanation of the error
        
    Returns:
        JSONResponse with error details
    """
    return JSONResponse(
        format_error_response(
            error_message=error_message,
            error_code=error_code,
            details=details,
            explanation=explanation
        ),
        status_code=status_code
    )


# ===============================================================
# ERROR HANDLING DECORATORS
# ===============================================================

def handle_errors(func: F) -> F:
    """Decorator to handle common errors in Solana operations.
    
    This decorator catches and logs errors but re-raises them for further
    handling by API error handlers. It is designed for internal service
    functions that aren't directly exposed as API endpoints.
    
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
                function=func.__name__,
                error_code=ErrorCode.SOLANA_RPC_ERROR
            )
            # Re-raise to be handled by the API error handler
            raise
        except Exception as e:
            log_with_context(
                logger,
                "error",
                f"Unexpected error in {func.__name__}: {str(e)}",
                request_id=request_id,
                function=func.__name__,
                error_code=ErrorCode.INTERNAL_ERROR
            )
            # Re-raise to be handled by the API error handler
            raise
            
    return wrapper


def validate_solana_key(func: F) -> F:
    """Decorator to validate Solana public key and handle exceptions.
    
    This decorator checks if the provided Solana public key is valid.
    It extracts the key from function arguments and validates it before
    proceeding with the function execution.
    
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
                mint=mint,
                error_code=ErrorCode.INVALID_PUBLIC_KEY
            )
            # Re-raise to be handled by the API error handler
            raise
            
    return wrapper


def api_error_handler(
    response_format: str = "fastapi"
) -> Callable[[F], F]:
    """Decorator to handle common exceptions in API endpoints.
    
    This decorator provides standardized error handling for API endpoints.
    It can return errors in FastAPI HTTPException format or as JSONResponse
    objects, based on the response_format parameter.
    
    Args:
        response_format: Format for error responses - "fastapi" or "json"
        
    Returns:
        Decorator function for error handling
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request if present to get request_id
            request = None
            request_id = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    # Get request_id from state or generate new one
                    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
                    break
                    
            if not request_id:
                request_id = str(uuid.uuid4())
                
            log_with_context(
                logger,
                "debug",
                f"Executing API endpoint: {func.__name__}",
                request_id=request_id,
                endpoint=func.__name__
            )
            
            try:
                # Add request_id to kwargs if not already present
                if "request_id" not in kwargs:
                    kwargs["request_id"] = request_id
                
                # Execute the endpoint function
                result = await func(*args, **kwargs)
                
                log_with_context(
                    logger,
                    "debug",
                    f"API endpoint {func.__name__} completed successfully",
                    request_id=request_id,
                    endpoint=func.__name__
                )
                
                return result
            except InvalidPublicKeyError as e:
                log_with_context(
                    logger,
                    "error",
                    f"Invalid Solana public key: {str(e)}",
                    request_id=request_id,
                    endpoint=func.__name__,
                    error_type="InvalidPublicKeyError",
                    error=str(e),
                    error_code=ErrorCode.INVALID_PUBLIC_KEY
                )
                
                if response_format == "json":
                    return create_error_response(
                        error_message=f"Invalid Solana public key: {str(e)}",
                        error_code=ErrorCode.INVALID_PUBLIC_KEY,
                        status_code=400,
                        explanation="The provided Solana address is not a valid public key."
                    )
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid Solana public key: {str(e)}"
                    )
                    
            except SolanaRpcError as e:
                log_with_context(
                    logger,
                    "error",
                    f"Solana RPC error: {str(e)}",
                    request_id=request_id,
                    endpoint=func.__name__,
                    error_type="SolanaRpcError",
                    error=str(e),
                    error_code=ErrorCode.SOLANA_RPC_ERROR
                )
                
                status_code = getattr(e, "status_code", 500)
                details = getattr(e, "error_data", {})
                
                if response_format == "json":
                    return create_error_response(
                        error_message=f"Solana RPC error: {str(e)}",
                        error_code=ErrorCode.SOLANA_RPC_ERROR,
                        status_code=status_code,
                        details=details,
                        explanation="An error occurred when interacting with the Solana blockchain."
                    )
                else:
                    raise HTTPException(
                        status_code=status_code, 
                        detail=f"Solana RPC error: {str(e)}"
                    )
                    
            except NotImplementedError as e:
                log_with_context(
                    logger,
                    "error",
                    f"Feature not implemented: {str(e)}",
                    request_id=request_id,
                    endpoint=func.__name__,
                    error_type="NotImplementedError",
                    error=str(e),
                    error_code=ErrorCode.NOT_IMPLEMENTED
                )
                
                if response_format == "json":
                    return create_error_response(
                        error_message=f"Feature not implemented: {str(e)}",
                        error_code=ErrorCode.NOT_IMPLEMENTED,
                        status_code=501,
                        explanation="This feature is not yet implemented."
                    )
                else:
                    raise HTTPException(status_code=501, detail=str(e))
                    
            except ValueError as e:
                log_with_context(
                    logger,
                    "error",
                    f"Invalid input: {str(e)}",
                    request_id=request_id,
                    endpoint=func.__name__,
                    error_type="ValueError",
                    error=str(e),
                    error_code=ErrorCode.INVALID_INPUT
                )
                
                if response_format == "json":
                    return create_error_response(
                        error_message=f"Invalid input: {str(e)}",
                        error_code=ErrorCode.INVALID_INPUT,
                        status_code=400,
                        explanation="The provided input values are invalid."
                    )
                else:
                    raise HTTPException(status_code=400, detail=str(e))
                    
            except HTTPException:
                # Re-raise FastAPI HTTP exceptions without modification
                raise
                
            except Exception as e:
                log_with_context(
                    logger,
                    "error",
                    f"Error in API endpoint {func.__name__}: {str(e)}",
                    request_id=request_id,
                    endpoint=func.__name__,
                    error_type=type(e).__name__,
                    error=str(e),
                    error_code=ErrorCode.INTERNAL_ERROR
                )
                
                if response_format == "json":
                    return create_error_response(
                        error_message=f"Internal server error: {str(e)}",
                        error_code=ErrorCode.INTERNAL_ERROR,
                        status_code=500,
                        explanation="An unexpected error occurred while processing your request."
                    )
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Error processing request: {str(e)}"
                    )
        
        return wrapper
    
    # Handle case when decorator is used without parentheses
    if callable(response_format):
        func = response_format
        response_format = "fastapi"
        return decorator(func)
    
    return decorator


# ===============================================================
# PERFORMANCE DECORATORS
# ===============================================================

def measure_execution_time(func: F) -> F:
    """Decorator to measure and log execution time of a function.
    
    This decorator times the execution of the wrapped function and logs
    the duration. Useful for performance monitoring and optimization.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with execution time measurement
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # Add timing info to result if it's a dict and doesn't already have timing
        if isinstance(result, dict) and "execution_time_ms" not in result:
            result["execution_time_ms"] = round(execution_time * 1000, 2)
            
        return result
    
    return wrapper


def retry_on_failure(
    max_retries: int = 3, 
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_retry: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        exceptions_to_retry: List of exception types to retry on (all exceptions if None)
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = kwargs.get("request_id")
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Skip retry if exception type is not in the retry list
                    if exceptions_to_retry and not any(
                        isinstance(e, exc_type) for exc_type in exceptions_to_retry
                    ):
                        log_with_context(
                            logger,
                            "error",
                            f"Exception not in retry list, re-raising: {str(e)}",
                            request_id=request_id,
                            function=func.__name__,
                            attempt=attempt,
                            max_retries=max_retries
                        )
                        raise
                    
                    if attempt == max_retries:
                        log_with_context(
                            logger,
                            "error",
                            f"Failed after {max_retries} retries: {str(e)}",
                            request_id=request_id,
                            function=func.__name__
                        )
                        raise
                    else:
                        # Calculate backoff delay
                        wait_time = delay * (backoff_factor ** attempt)
                        
                        log_with_context(
                            logger,
                            "warning",
                            f"Retry attempt {attempt+1}/{max_retries} after error: {str(e)}. Waiting {wait_time:.2f}s",
                            request_id=request_id,
                            function=func.__name__,
                            attempt=attempt,
                            max_retries=max_retries,
                            wait_time=wait_time
                        )
                        await asyncio.sleep(wait_time)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    
    return decorator


# ===============================================================
# VALIDATION DECORATORS
# ===============================================================

def validate_input(validation_class: Any) -> Callable[[F], F]:
    """Decorator to validate request body against a Pydantic model.
    
    Args:
        validation_class: Pydantic model class for validation
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(request, *args, **kwargs):
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            
            try:
                # Extract body
                body = await request.json()
                
                # Validate with model
                validated_data = validation_class(**body)
                
                # Add validated data to kwargs
                kwargs['validated_data'] = validated_data
                
                return await func(request, *args, **kwargs)
            except Exception as e:
                log_with_context(
                    logger,
                    "warning",
                    f"Validation error: {str(e)}",
                    request_id=request_id,
                    endpoint=func.__name__,
                    error_type="ValidationError",
                    error=str(e)
                )
                
                return create_error_response(
                    error_message=f"Validation error: {str(e)}",
                    error_code=ErrorCode.INVALID_INPUT,
                    status_code=400,
                    explanation="The request data failed validation requirements."
                )
        
        return wrapper
    
    return decorator


# ===============================================================
# REQUEST PROCESSING DECORATORS
# ===============================================================

def rate_limit(
    limit: int = 100, 
    period: int = 60,
    error_format: str = "fastapi"
) -> Callable[[F], F]:
    """Decorator to implement rate limiting on API endpoints.
    
    In a production environment, this should use a distributed cache like Redis.
    This implementation is for demonstration and development only.
    
    Args:
        limit: Maximum number of requests allowed in the period
        period: Time period in seconds
        error_format: Format for error responses - "fastapi" or "json"
        
    Returns:
        Decorator function
    """
    # In a real implementation, would use Redis or another shared cache for rate limits
    # This is a simplified in-memory version for demonstration
    request_times = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract client IP from request
            client_ip = "unknown"
            request_id = None
            
            for arg in args:
                if isinstance(arg, Request):
                    client_ip = arg.client.host
                    request_id = getattr(arg.state, "request_id", str(uuid.uuid4()))
                    break
            
            current_time = time.time()
            
            # Initialize or clean up old request times
            if client_ip not in request_times:
                request_times[client_ip] = []
            
            # Remove request times older than the period
            request_times[client_ip] = [t for t in request_times[client_ip] if current_time - t < period]
            
            # Check if rate limit is exceeded
            if len(request_times[client_ip]) >= limit:
                log_with_context(
                    logger,
                    "warning",
                    f"Rate limit exceeded for {client_ip}",
                    request_id=request_id,
                    client_ip=client_ip,
                    limit=limit,
                    period=period,
                    error_code=ErrorCode.RATE_LIMITED
                )
                
                error_message = f"Rate limit exceeded. Maximum {limit} requests allowed per {period} seconds."
                
                if error_format == "json":
                    return create_error_response(
                        error_message=error_message,
                        error_code=ErrorCode.RATE_LIMITED,
                        status_code=429,
                        explanation="You have sent too many requests in a short period of time."
                    )
                else:
                    raise HTTPException(
                        status_code=429, 
                        detail=error_message
                    )
            
            # Record this request
            request_times[client_ip].append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator 