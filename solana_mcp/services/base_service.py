"""
Base service class for Solana MCP services.

This module provides a base class for all services in the Solana MCP,
with common functionality for error handling, timeouts, and logging.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, cast, Awaitable

from solana_mcp.utils.errors import (
    RpcConnectionError,
    RpcError,
    RpcTimeoutError,
    ServiceUnavailableError,
    TimeoutError
)
from solana_mcp.utils.api_response import RPCError, SolanaMCPError

T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)


def handle_errors(error_type: type[SolanaMCPError] = RPCError):
    """
    Decorator to handle errors in service methods.
    
    Args:
        error_type: The type of error to raise if an exception occurs
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except SolanaMCPError as e:
                # Preserve original error if it's already a SolanaMCPError
                logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise
            except asyncio.TimeoutError as e:
                logger.exception(f"Timeout in {func.__name__}: {str(e)}")
                raise TimeoutError(f"Operation timed out: {func.__name__}")
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise error_type(f"Error in {func.__name__}: {str(e)}")
        return wrapper
    return decorator


class BaseService:
    """
    Base service class with common functionality.
    
    This class provides:
    - Error handling
    - Timeout management
    - Logging
    - Performance tracking
    """
    
    def __init__(self, timeout: float = 30.0, logger: Optional[logging.Logger] = None):
        """
        Initialize the base service.
        
        Args:
            timeout: Default timeout for service operations in seconds
            logger: Optional logger instance
        """
        self.timeout = timeout
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    async def with_timeout(self, coro: Callable[..., T], timeout: Optional[float] = None, **kwargs: Any) -> T:
        """
        Execute a coroutine with a timeout.
        
        Args:
            coro: The coroutine to execute
            timeout: Optional custom timeout in seconds
            **kwargs: Arguments to pass to the coroutine
            
        Returns:
            The result of the coroutine
            
        Raises:
            TimeoutError: If the operation times out
        """
        timeout_value = timeout or self.timeout
        try:
            return await asyncio.wait_for(coro(**kwargs), timeout=timeout_value)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_value}s")
    
    async def measure_performance(self, coro: Callable[..., T], **kwargs: Any) -> T:
        """
        Measure the performance of a coroutine.
        
        Args:
            coro: The coroutine to execute
            **kwargs: Arguments to pass to the coroutine
            
        Returns:
            The result of the coroutine
        """
        start_time = time.time()
        result = await coro(**kwargs)
        elapsed_time = time.time() - start_time
        
        self.logger.debug(
            f"Performance: {coro.__name__} took {elapsed_time:.4f}s"
        )
        
        return result
    
    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        max_attempts: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        operation_name: str = "operation"
    ) -> Any:
        """
        Execute an operation with retries.
        
        Args:
            operation: Operation to execute
            max_attempts: Maximum number of attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay by after each attempt
            operation_name: Name of the operation for error reporting
            
        Returns:
            Result of the operation
            
        Raises:
            ServiceUnavailableError: If all attempts fail
        """
        last_exception: Optional[Exception] = None
        current_delay = retry_delay
        
        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except (RpcError, RpcTimeoutError, RpcConnectionError) as e:
                last_exception = e
                if attempt < max_attempts:
                    # Log the retry
                    self.logger.warning(
                        f"{operation_name} failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {current_delay:.2f}s: {str(e)}"
                    )
                    # Wait before retrying
                    await asyncio.sleep(current_delay)
                    # Increase delay for next attempt
                    current_delay *= backoff_factor
                else:
                    # Log the final failure
                    self.logger.error(
                        f"{operation_name} failed after {max_attempts} attempts: {str(e)}"
                    )
        
        # If we get here, all attempts failed
        raise ServiceUnavailableError(
            message=f"{operation_name} failed after {max_attempts} attempts",
            service=self.__class__.__name__,
            retry_after=int(current_delay)
        ) from last_exception
    
    async def gather_with_concurrency(
        self,
        concurrency_limit: int,
        *tasks: Awaitable[Any]
    ) -> List[Any]:
        """
        Execute tasks with a concurrency limit.
        
        Args:
            concurrency_limit: Maximum number of tasks to run concurrently
            tasks: Tasks to execute
            
        Returns:
            List of results from the tasks
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def _wrapped_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(
            *[_wrapped_task(task) for task in tasks],
            return_exceptions=False
        )
    
    def log_timing(self, operation_name: str) -> "TimingContextManager":
        """
        Create a context manager to log timing information.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Timing context manager
        """
        return TimingContextManager(operation_name, self.logger)


class TimingContextManager:
    """Context manager to log timing information."""
    
    def __init__(self, operation_name: str, logger: logging.Logger):
        """
        Initialize the timing context manager.
        
        Args:
            operation_name: Name of the operation
            logger: Logger to use for logging
        """
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = 0.0
    
    async def __aenter__(self) -> "TimingContextManager":
        """
        Enter the async context.
        
        Returns:
            Self
        """
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the async context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        elapsed = time.time() - self.start_time
        if exc_val is not None:
            self.logger.error(
                f"{self.operation_name} failed after {elapsed:.2f}s: {str(exc_val)}"
            )
        else:
            self.logger.info(f"{self.operation_name} completed in {elapsed:.2f}s")
    
    def __enter__(self) -> "TimingContextManager":
        """
        Enter the context.
        
        Returns:
            Self
        """
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        elapsed = time.time() - self.start_time
        if exc_val is not None:
            self.logger.error(
                f"{self.operation_name} failed after {elapsed:.2f}s: {str(exc_val)}"
            )
        else:
            self.logger.info(f"{self.operation_name} completed in {elapsed:.2f}s") 