"""
Error handling utilities for Solana MCP.

This module provides standardized error handling mechanisms including:
- Custom exception classes
- Decorators for common error handling patterns
- Utilities for error logging and reporting
"""

import asyncio
import functools
import inspect
import logging
import traceback
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast, Awaitable
from dataclasses import dataclass

import aiohttp
from solana.rpc.core import RPCException
# Remove solana import and define our own RPCException class
# from solana.rpc.core import RPCException

# Define our own RPCException class
class RPCException(Exception):
    """Exception raised when an RPC request fails."""
    def __init__(self, message: str, code: int = 0, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data

# Get logger
logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])


class ErrorCode(Enum):
    """Error codes for Solana MCP."""
    # General errors
    UNKNOWN_ERROR = 1000
    CONFIGURATION_ERROR = 1001
    VALIDATION_ERROR = 1002
    NOT_IMPLEMENTED_ERROR = 1003
    
    # Network errors
    NETWORK_ERROR = 2000
    CONNECTION_ERROR = 2001
    TIMEOUT_ERROR = 2002
    REQUEST_ERROR = 2003
    
    # RPC errors
    RPC_ERROR = 3000
    RPC_RATE_LIMIT_ERROR = 3001
    RPC_INVALID_PARAMETER_ERROR = 3002
    RPC_METHOD_NOT_FOUND_ERROR = 3003
    
    # Data errors
    DATA_ERROR = 4000
    PARSING_ERROR = 4001
    SERIALIZATION_ERROR = 4002
    DESERIALIZATION_ERROR = 4003
    
    # Transaction errors
    TRANSACTION_ERROR = 5000
    TRANSACTION_CONFIRMATION_ERROR = 5001
    TRANSACTION_SIMULATION_ERROR = 5002
    TRANSACTION_SIGNING_ERROR = 5003


# Base exception classes
class SolanaMCPError(Exception):
    """Base exception class for all Solana MCP errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize a new SolanaMCPError.
        
        Args:
            message: Error message
            error_code: Error code from ErrorCode enum
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
        # Format the error message
        formatted_message = f"[{error_code.name}] {message}"
        if details:
            formatted_message += f" - Details: {details}"
        
        super().__init__(formatted_message)


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


class NetworkError(SolanaMCPError):
    """Error related to network communication issues."""
    
    def __init__(
        self, 
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the network error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorCode.NETWORK_ERROR, details)


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
            
        super().__init__(message, error_details=error_details)


class ValidationError(SolanaMCPError):
    """Error related to validation failures."""
    pass


class ConfigurationError(SolanaMCPError):
    """Error related to configuration issues."""
    pass


class ParseError(SolanaMCPError):
    """Error related to parsing failures."""
    pass


class DataError(SolanaMCPError):
    """Error related to data handling or processing failures."""
    
    def __init__(
        self, 
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorCode.DATA_ERROR, details)


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


def handle_exceptions(
    *exception_mappings: Union[Type[Exception], tuple[Type[Exception], Type[SolanaMCPError]]],
    log_level: int = logging.ERROR,
    reraise: bool = True,
    default_error: Optional[Type[SolanaMCPError]] = None
) -> Callable[[F], F]:
    """Decorator for handling exceptions.
    
    This decorator catches specified exceptions and optionally maps them to custom exceptions.
    
    Args:
        *exception_mappings: Exception types or tuples of (caught_exception, mapped_exception)
        log_level: Logging level for caught exceptions
        reraise: Whether to reraise caught exceptions
        default_error: Default error type to map to if no mapping exists
        
    Returns:
        Decorated function
        
    Example:
        @handle_exceptions(
            (ValueError, ValidationError),
            TimeoutError,
            log_level=logging.WARNING
        )
        def process_data(data):
            # process data
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build exception mapping
            mappings = {}
            caught_exceptions = []
            
            for item in exception_mappings:
                if isinstance(item, tuple) and len(item) == 2:
                    caught_exc, mapped_exc = item
                    mappings[caught_exc] = mapped_exc
                    caught_exceptions.append(caught_exc)
                else:
                    caught_exc = item
                    caught_exceptions.append(caught_exc)
            
            try:
                return func(*args, **kwargs)
            except tuple(caught_exceptions) as e:
                # Get the exception details
                exc_type = type(e)
                exc_msg = str(e)
                exc_trace = traceback.format_exc()
                
                # Log the exception
                logger.log(log_level, f"Caught exception in {func.__name__}: {exc_type.__name__}: {exc_msg}")
                if log_level == logging.DEBUG:
                    logger.debug(f"Traceback: {exc_trace}")
                
                # Map the exception if a mapping exists
                if exc_type in mappings:
                    mapped_exc_type = mappings[exc_type]
                    if issubclass(mapped_exc_type, SolanaMCPError):
                        raise mapped_exc_type(exc_msg, details={"original_exception": exc_type.__name__})
                elif default_error and issubclass(default_error, SolanaMCPError):
                    raise default_error(exc_msg, details={"original_exception": exc_type.__name__})
                
                # Reraise the original exception if requested
                if reraise:
                    raise
                
                # Return None if not reraising
                return None
        
        return cast(F, wrapper)
    
    return decorator


def handle_async_exceptions(
    *exception_mappings: Union[Type[Exception], tuple[Type[Exception], Type[SolanaMCPError]]],
    log_level: int = logging.ERROR,
    reraise: bool = True,
    default_error: Optional[Type[SolanaMCPError]] = None
) -> Callable[[AsyncF], AsyncF]:
    """Decorator for handling exceptions in async functions.
    
    This decorator catches specified exceptions and optionally maps them to custom exceptions.
    
    Args:
        *exception_mappings: Exception types or tuples of (caught_exception, mapped_exception)
        log_level: Logging level for caught exceptions
        reraise: Whether to reraise caught exceptions
        default_error: Default error type to map to if no mapping exists
        
    Returns:
        Decorated async function
        
    Example:
        @handle_async_exceptions(
            (aiohttp.ClientError, NetworkError),
            asyncio.TimeoutError,
            log_level=logging.WARNING
        )
        async def fetch_data(url):
            # fetch data
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build exception mapping
            mappings = {}
            caught_exceptions = []
            
            for item in exception_mappings:
                if isinstance(item, tuple) and len(item) == 2:
                    caught_exc, mapped_exc = item
                    mappings[caught_exc] = mapped_exc
                    caught_exceptions.append(caught_exc)
                else:
                    caught_exc = item
                    caught_exceptions.append(caught_exc)
            
            try:
                return await func(*args, **kwargs)
            except tuple(caught_exceptions) as e:
                # Get the exception details
                exc_type = type(e)
                exc_msg = str(e)
                exc_trace = traceback.format_exc()
                
                # Log the exception
                logger.log(log_level, f"Caught exception in {func.__name__}: {exc_type.__name__}: {exc_msg}")
                if log_level == logging.DEBUG:
                    logger.debug(f"Traceback: {exc_trace}")
                
                # Map the exception if a mapping exists
                if exc_type in mappings:
                    mapped_exc_type = mappings[exc_type]
                    if issubclass(mapped_exc_type, SolanaMCPError):
                        raise mapped_exc_type(exc_msg, details={"original_exception": exc_type.__name__})
                elif default_error and issubclass(default_error, SolanaMCPError):
                    raise default_error(exc_msg, details={"original_exception": exc_type.__name__})
                
                # Reraise the original exception if requested
                if reraise:
                    raise
                
                # Return None if not reraising
                return None
        
        return cast(AsyncF, wrapper)
    
    return decorator


def retry(
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: List[Type[Exception]] = None,
    log_level: int = logging.WARNING
) -> Callable[[F], F]:
    """Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay between retries
        retryable_exceptions: List of exceptions that trigger a retry
        log_level: Logging level for retry attempts
        
    Returns:
        Decorated function
        
    Example:
        @retry(
            max_attempts=5,
            retry_delay=0.5,
            backoff_factor=1.5,
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
        def fetch_data(url):
            # fetch data
    """
    if retryable_exceptions is None:
        retryable_exceptions = [Exception]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.log(
                            log_level, 
                            f"Retry {attempt}/{max_attempts} for {func.__name__} after error: {str(e)}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.log(
                            log_level, 
                            f"Max retries ({max_attempts}) reached for {func.__name__} with error: {str(e)}"
                        )
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            
            # This should never happen, but just in case
            raise RuntimeError(f"All retries failed for {func.__name__} but no exception was caught")
        
        return cast(F, wrapper)
    
    return decorator


def async_retry(
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: List[Type[Exception]] = None,
    log_level: int = logging.WARNING
) -> Callable[[AsyncF], AsyncF]:
    """Decorator to retry an async function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay between retries
        retryable_exceptions: List of exceptions that trigger a retry
        log_level: Logging level for retry attempts
        
    Returns:
        Decorated async function
        
    Example:
        @async_retry(
            max_attempts=5,
            retry_delay=0.5,
            backoff_factor=1.5,
            retryable_exceptions=[aiohttp.ClientError, asyncio.TimeoutError]
        )
        async def fetch_data(url):
            # fetch data
    """
    if retryable_exceptions is None:
        retryable_exceptions = [Exception]
    
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.log(
                            log_level, 
                            f"Retry {attempt}/{max_attempts} for {func.__name__} after error: {str(e)}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.log(
                            log_level, 
                            f"Max retries ({max_attempts}) reached for {func.__name__} with error: {str(e)}"
                        )
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            
            # This should never happen, but just in case
            raise RuntimeError(f"All retries failed for {func.__name__} but no exception was caught")
        
        return cast(AsyncF, wrapper)
    
    return decorator


def map_rpc_errors(func: AsyncF) -> AsyncF:
    """Decorator to map RPC errors to custom exceptions.
    
    This decorator catches RPCException and maps it to appropriate custom exceptions
    based on the error code.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RPCException as e:
            # Extract error details
            error_data = getattr(e, "data", None) or {}
            error_code = error_data.get("code", 0)
            error_message = error_data.get("message", str(e))
            
            # Map to specific error types based on error code
            if error_code == -32002:
                raise RPCError(
                    f"RPC rate limit exceeded: {error_message}", 
                    ErrorCode.RPC_RATE_LIMIT_ERROR, 
                    details=error_data
                )
            elif error_code == -32602:
                raise RPCError(
                    f"Invalid RPC parameters: {error_message}", 
                    ErrorCode.RPC_INVALID_PARAMETER_ERROR, 
                    details=error_data
                )
            elif error_code == -32601:
                raise RPCError(
                    f"RPC method not found: {error_message}", 
                    ErrorCode.RPC_METHOD_NOT_FOUND_ERROR, 
                    details=error_data
                )
            else:
                # Generic RPC error
                raise RPCError(
                    f"RPC error: {error_message}", 
                    ErrorCode.RPC_ERROR, 
                    details=error_data
                )
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error: {str(e)}", details={"original_exception": type(e).__name__})
        except asyncio.TimeoutError:
            raise NetworkError("Request timed out", details={"error_code": ErrorCode.TIMEOUT_ERROR.value})
    
    return cast(AsyncF, wrapper)


def validate_parameters(func: F) -> F:
    """Decorator to validate function parameters.
    
    This decorator checks for None values in required parameters and raises
    ValidationError if any are found.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the signature
        sig = inspect.signature(func)
        
        # Bind the arguments
        bound_args = sig.bind(*args, **kwargs)
        
        # Check for None values in required parameters
        for param_name, param in sig.parameters.items():
            if param.default is param.empty and param_name in bound_args.arguments:
                if bound_args.arguments[param_name] is None:
                    raise ValidationError(
                        f"Required parameter '{param_name}' cannot be None in {func.__name__}",
                        details={"parameter": param_name}
                    )
        
        return func(*args, **kwargs)
    
    return cast(F, wrapper)


def validate_async_parameters(func: AsyncF) -> AsyncF:
    """Decorator to validate async function parameters.
    
    This decorator checks for None values in required parameters and raises
    ValidationError if any are found.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Get the signature
        sig = inspect.signature(func)
        
        # Bind the arguments
        bound_args = sig.bind(*args, **kwargs)
        
        # Check for None values in required parameters
        for param_name, param in sig.parameters.items():
            if param.default is param.empty and param_name in bound_args.arguments:
                if bound_args.arguments[param_name] is None:
                    raise ValidationError(
                        f"Required parameter '{param_name}' cannot be None in {func.__name__}",
                        details={"parameter": param_name}
                    )
        
        return await func(*args, **kwargs)
    
    return cast(AsyncF, wrapper)


# Common exception mappings
COMMON_NETWORK_EXCEPTIONS = [
    (aiohttp.ClientError, NetworkError),
    (asyncio.TimeoutError, NetworkError),
    (ConnectionError, NetworkError),
    (TimeoutError, NetworkError),
]

COMMON_DATA_EXCEPTIONS = [
    (ValueError, DataError),
    (TypeError, DataError),
    (KeyError, DataError),
    (AttributeError, DataError),
]

# Specialized decorators using the handle_exceptions decorator
handle_network_exceptions = functools.partial(
    handle_exceptions,
    *COMMON_NETWORK_EXCEPTIONS,
    log_level=logging.WARNING,
    default_error=NetworkError
)

handle_data_exceptions = functools.partial(
    handle_exceptions,
    *COMMON_DATA_EXCEPTIONS,
    log_level=logging.WARNING,
    default_error=DataError
)

handle_async_network_exceptions = functools.partial(
    handle_async_exceptions,
    *COMMON_NETWORK_EXCEPTIONS,
    log_level=logging.WARNING,
    default_error=NetworkError
)

handle_async_data_exceptions = functools.partial(
    handle_async_exceptions,
    *COMMON_DATA_EXCEPTIONS,
    log_level=logging.WARNING,
    default_error=DataError
)

# Ensure NetworkError and DataError are properly defined
def _ensure_error_classes():
    """Ensure that all required error classes are defined."""
    global NetworkError, DataError
    
    # Check if NetworkError exists in the global namespace
    if 'NetworkError' not in globals():
        class NetworkError(SolanaMCPError):
            """Error related to network communication issues."""
            
            def __init__(
                self, 
                message: str,
                details: Optional[Dict[str, Any]] = None
            ):
                super().__init__(message, ErrorCode.NETWORK_ERROR, details)
        
        globals()['NetworkError'] = NetworkError
    
    # Check if DataError exists in the global namespace
    if 'DataError' not in globals():
        class DataError(SolanaMCPError):
            """Error related to data handling or processing failures."""
            
            def __init__(
                self, 
                message: str,
                details: Optional[Dict[str, Any]] = None
            ):
                super().__init__(message, ErrorCode.DATA_ERROR, details)
        
        globals()['DataError'] = DataError

# Call the function to ensure classes are defined
_ensure_error_classes() 