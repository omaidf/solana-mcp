"""Decorators for use throughout the solana_mcp package."""

import functools
import logging
import time
import asyncio
import uuid
from typing import Callable, Any, Dict, Optional

from solana_mcp.logging_config import get_logger, log_with_context
from solana_mcp.solana_client import InvalidPublicKeyError, SolanaRpcError
from fastapi import HTTPException, Request

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

def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure and log execution time of a function.
    
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
        
        return result
    
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0) -> Callable:
    """Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    else:
                        logger.warning(f"Retry attempt {attempt+1}/{max_retries} after error: {str(e)}")
                        await asyncio.sleep(delay)
        
        return wrapper
    
    return decorator

def api_error_handler(func: Callable) -> Callable:
    """Decorator to handle common exceptions in API endpoints and return appropriate HTTP responses.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling specifically for API endpoints
    """
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
                error=str(e)
            )
            raise HTTPException(status_code=400, detail=f"Invalid Solana public key: {str(e)}")
        except SolanaRpcError as e:
            log_with_context(
                logger,
                "error",
                f"Solana RPC error: {str(e)}",
                request_id=request_id,
                endpoint=func.__name__,
                error_type="SolanaRpcError",
                error=str(e)
            )
            status_code = e.status_code if hasattr(e, 'status_code') else 500
            raise HTTPException(status_code=status_code, detail=f"Solana RPC error: {str(e)}")
        except NotImplementedError as e:
            log_with_context(
                logger,
                "error",
                f"Feature not implemented: {str(e)}",
                request_id=request_id,
                endpoint=func.__name__,
                error_type="NotImplementedError",
                error=str(e)
            )
            raise HTTPException(status_code=501, detail=str(e))
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
                error=str(e)
            )
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    return wrapper

def rate_limit(limit: int = 100, period: int = 60) -> Callable:
    """Decorator to implement rate limiting on API endpoints.
    
    Args:
        limit: Maximum number of requests allowed in the period
        period: Time period in seconds
        
    Returns:
        Decorator function
    """
    # In a real implementation, would use Redis or another shared cache for rate limits
    # This is a simplified in-memory version for demonstration
    request_times = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract client IP from request
            client_ip = "unknown"
            for arg in args:
                if isinstance(arg, Request):
                    client_ip = arg.client.host
                    break
            
            current_time = time.time()
            
            # Initialize or clean up old request times
            if client_ip not in request_times:
                request_times[client_ip] = []
            
            # Remove request times older than the period
            request_times[client_ip] = [t for t in request_times[client_ip] if current_time - t < period]
            
            # Check if rate limit is exceeded
            if len(request_times[client_ip]) >= limit:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                raise HTTPException(
                    status_code=429, 
                    detail="Too many requests. Please try again later."
                )
            
            # Record this request
            request_times[client_ip].append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator 