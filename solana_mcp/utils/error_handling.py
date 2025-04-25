"""Error handling utilities for Solana MCP.

This module provides standardized error handling patterns and exception
classes to maintain consistent error handling across the codebase.
"""

import asyncio
import functools
import inspect
import logging
import traceback
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast
from dataclasses import dataclass

import aiohttp

# Get logger
logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Base exception classes
class SolanaMCPError(Exception):
    """Base exception class for all Solana MCP errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConnectionError(SolanaMCPError):
    """Error related to connection issues."""
    
    def __init__(
        self, 
        message: str, 
        endpoint: Optional[str] = None, 
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the connection error.
        
        Args:
            message: Error message
            endpoint: The RPC endpoint that failed
            original_error: The original exception that caused this error
        """
        self.endpoint = endpoint
        self.original_error = original_error
        super().__init__(message)


class RPCError(SolanaMCPError):
    """Exception for RPC-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None, 
        rpc_error_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RPC exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            rpc_error_code: RPC-specific error code
            endpoint: The RPC endpoint that returned the error
            details: Additional error details
        """
        self.status_code = status_code
        self.rpc_error_code = rpc_error_code
        self.endpoint = endpoint
        
        error_details = details or {}
        if status_code:
            error_details["status_code"] = status_code
        if rpc_error_code:
            error_details["rpc_error_code"] = rpc_error_code
        if endpoint:
            error_details["endpoint"] = endpoint
            
        super().__init__(message, error_details)


class ValidationError(SolanaMCPError):
    """Error related to validation failures."""
    pass


class ConfigurationError(SolanaMCPError):
    """Error related to configuration issues."""
    pass


class ParseError(SolanaMCPError):
    """Error related to parsing failures."""
    pass


class ProgramError(SolanaMCPError):
    """Error related to Solana programs."""
    
    def __init__(self, program_id: str, error_code: int, message: Optional[str] = None):
        self.program_id = program_id
        self.error_code = error_code
        self.message = message or f"Program error code: {error_code}"
        super().__init__(f"Program error in {program_id}: {self.message}")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    retry_delay: float = 0.5
    backoff_factor: float = 2.0
    retry_on_exceptions: List[Type[Exception]] = None
    retry_on_status_codes: List[int] = None
    
    def __post_init__(self):
        """Initialize default values for retry configuration."""
        if self.retry_on_exceptions is None:
            self.retry_on_exceptions = [
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError
            ]
        
        if self.retry_on_status_codes is None:
            self.retry_on_status_codes = [408, 429, 500, 502, 503, 504]


# Error handling decorators
def handle_errors(
    retries: int = 0,
    retry_delay: float = 1.0,
    retry_on: Optional[List[Type[Exception]]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator for standardized error handling.
    
    This decorator provides common error handling patterns including
    logging, retries, and error transformation.
    
    Args:
        retries: Number of retry attempts (0 = no retries)
        retry_delay: Delay between retry attempts in seconds
        retry_on: List of exception types to retry on
        logger_instance: Logger instance to use (defaults to module logger)
    
    Returns:
        Decorated function with error handling
    
    Example:
        @handle_errors(retries=3, retry_on=[RPCError])
        async def get_account_info(pubkey: str):
            # This function will retry up to 3 times on RPCError
            ...
    """
    log = logger_instance or logger
    retry_exceptions = retry_on or [RPCError]
    
    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                attempts = 0
                last_exception = None
                
                while attempts <= retries:
                    try:
                        return await func(*args, **kwargs)
                    except tuple(retry_exceptions) as e:
                        last_exception = e
                        attempts += 1
                        
                        if attempts <= retries:
                            retry_msg = (
                                f"Retrying {func.__name__} after error: {str(e)}. "
                                f"Attempt {attempts}/{retries}"
                            )
                            log.warning(retry_msg)
                            await asyncio.sleep(retry_delay * attempts)  # Exponential backoff
                        else:
                            log.error(
                                f"Failed after {retries} retries in {func.__name__}: {str(e)}"
                            )
                            raise
                    except Exception as e:
                        # Log non-retry exceptions
                        log.error(f"Error in {func.__name__}: {str(e)}")
                        raise
                
                # If we got here, we've exhausted retries
                assert last_exception is not None
                raise last_exception
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                attempts = 0
                last_exception = None
                
                while attempts <= retries:
                    try:
                        return func(*args, **kwargs)
                    except tuple(retry_exceptions) as e:
                        last_exception = e
                        attempts += 1
                        
                        if attempts <= retries:
                            retry_msg = (
                                f"Retrying {func.__name__} after error: {str(e)}. "
                                f"Attempt {attempts}/{retries}"
                            )
                            log.warning(retry_msg)
                            time.sleep(retry_delay * attempts)  # Exponential backoff
                        else:
                            log.error(
                                f"Failed after {retries} retries in {func.__name__}: {str(e)}"
                            )
                            raise
                    except Exception as e:
                        # Log non-retry exceptions
                        log.error(f"Error in {func.__name__}: {str(e)}")
                        raise
                
                # If we got here, we've exhausted retries
                assert last_exception is not None
                raise last_exception
            
            return cast(F, sync_wrapper)
    
    return decorator


def validate_input(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """
    Decorator for validating function inputs.
    
    This decorator checks function arguments against validator functions.
    
    Args:
        **validators: Map of parameter names to validator functions
    
    Returns:
        Decorated function with input validation
    
    Example:
        @validate_input(
            pubkey=lambda x: isinstance(x, str) and len(x) == 44,
            limit=lambda x: isinstance(x, int) and 0 < x <= 1000
        )
        def get_transaction_history(pubkey, limit=100):
            ...
    """
    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind args and kwargs to function parameters
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Invalid value for parameter '{param_name}': {value}"
                        )
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def transform_exceptions(
    mapping: Dict[Type[Exception], Type[Exception]]
) -> Callable[[F], F]:
    """
    Decorator for transforming exceptions to standardized types.
    
    This decorator catches specified exception types and transforms them 
    into different exception types.
    
    Args:
        mapping: Dictionary mapping source exception types to target exception types
    
    Returns:
        Decorated function with exception transformation
    
    Example:
        @transform_exceptions({
            KeyError: NotFoundError,
            ValueError: InvalidInputError
        })
        def process_data(data_id):
            ...
    """
    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if the exception type should be transformed
                    for source_type, target_type in mapping.items():
                        if isinstance(e, source_type):
                            # Transform to the target exception type
                            raise target_type(str(e)) from e
                    # If not in mapping, re-raise the original exception
                    raise
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if the exception type should be transformed
                    for source_type, target_type in mapping.items():
                        if isinstance(e, source_type):
                            # Transform to the target exception type
                            raise target_type(str(e)) from e
                    # If not in mapping, re-raise the original exception
                    raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def validate_public_key(public_key: str) -> None:
    """
    Validate that a string is a valid Solana public key.
    
    Args:
        public_key: The public key string to validate
    
    Raises:
        ValidationError: If the public key is invalid
    """
    if not public_key:
        raise ValidationError("Public key cannot be empty")
    
    # Basic validation for public key format (44 characters, base58)
    if len(public_key) != 44 or not all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in public_key):
        raise ValidationError(f"Invalid public key format: {public_key}")


def validate_signature(signature: str) -> None:
    """
    Validate that a string is a valid Solana transaction signature.
    
    Args:
        signature: The signature string to validate
    
    Raises:
        ValidationError: If the signature is invalid
    """
    if not signature:
        raise ValidationError("Transaction signature cannot be empty")
    
    # Basic validation for signature format (88 characters, base58)
    if len(signature) != 88 or not all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in signature):
        raise ValidationError(f"Invalid transaction signature format: {signature}")


def catch_and_log_exceptions(
    reraise: bool = True,
    log_level: int = logging.ERROR
) -> Callable[[F], F]:
    """
    Decorator that catches and logs exceptions.
    
    Args:
        reraise: Whether to reraise the exception after logging
        log_level: The logging level to use
        
    Returns:
        Decorated function with exception handling
    """
    def decorator(func: F) -> F:
        """Decorator function that adds exception handling."""
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Async wrapper for the decorated function."""
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Exception in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                
                if reraise:
                    raise
                
                return None
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync wrapper for the decorated function."""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    log_level,
                    f"Exception in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                
                if reraise:
                    raise
                
                return None
        
        # Return the appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator 