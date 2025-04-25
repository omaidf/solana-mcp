"""Batching utilities for Solana MCP.

This module provides utilities for efficient batching of requests to optimize performance.
It includes both simple function-based batching and more advanced class-based batch processors.
"""

import asyncio
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, cast

# Type variables for generic types
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


async def batch_process_requests(
    processor: Callable[[T], Awaitable[R]],
    items: List[T],
    batch_size: int = 10,
    concurrency: int = 5
) -> List[R]:
    """
    Process a list of items in batches with limited concurrency.
    
    Args:
        processor: Async function to process each item
        items: List of items to process
        batch_size: Maximum number of items per batch
        concurrency: Maximum number of concurrent tasks
        
    Returns:
        List of results in the same order as the input items
    """
    results: List[R] = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_semaphore(item: T) -> R:
        async with semaphore:
            return await processor(item)
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[process_with_semaphore(item) for item in batch], return_exceptions=True)
        results.extend(batch_results)
    
    # Filter out exceptions or handle them as needed
    return [result for result in results if not isinstance(result, Exception)]


class BatchProcessor(Generic[T, R]):
    """
    Class for processing items in batches with configurable parameters.
    
    This class provides more flexibility and state management compared
    to the batch_process_requests function.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        concurrency: int = 5,
        auto_retry: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_wait_time: float = 0.1,
        max_batch_execution_time: float = 2.0
    ):
        """
        Initialize a batch processor.
        
        Args:
            batch_size: Maximum number of items per batch
            concurrency: Maximum number of concurrent tasks
            auto_retry: Whether to automatically retry failed items
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_wait_time: Maximum time to wait before processing a non-full batch
            max_batch_execution_time: Maximum time for batch execution before timeout
        """
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_wait_time = max_wait_time
        self.max_batch_execution_time = max_batch_execution_time
        self.semaphore = asyncio.Semaphore(concurrency)
        self.queues: Dict[str, List[Tuple[Any, asyncio.Future]]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Awaitable[R]]
    ) -> List[R]:
        """
        Process a batch of items.
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            
        Returns:
            List of results in the same order as the input items
        """
        results: List[Optional[R]] = [None] * len(items)
        retry_indices: List[int] = list(range(len(items)))
        
        # Retry loop
        for attempt in range(self.max_retries + 1):
            if not retry_indices:
                break
                
            # Process items that need retry
            current_items = [(i, items[i]) for i in retry_indices]
            current_tasks = [self._process_item(idx, item, processor) for idx, item in current_items]
            
            # Wait for all tasks to complete
            batch_results = await asyncio.gather(*current_tasks, return_exceptions=True)
            
            # Process results and collect indices for retry
            new_retry_indices = []
            for i, result in enumerate(batch_results):
                idx = retry_indices[i]
                
                if isinstance(result, Exception):
                    if attempt < self.max_retries and self.auto_retry:
                        new_retry_indices.append(idx)
                    else:
                        # Store the exception or handle it as needed
                        results[idx] = cast(R, None)  # Type ignore for failed result
                else:
                    results[idx] = result
            
            # Update retry indices for next attempt
            retry_indices = new_retry_indices
            
            # Wait before retrying
            if retry_indices and attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (1 << attempt))  # Exponential backoff
        
        # Filter out None values (failed items)
        return [r for r in results if r is not None]
    
    async def _process_item(
        self,
        idx: int,
        item: T,
        processor: Callable[[T], Awaitable[R]]
    ) -> R:
        """
        Process a single item with concurrency control.
        
        Args:
            idx: Index of the item in the original list
            item: Item to process
            processor: Async function to process the item
            
        Returns:
            Processing result
        """
        async with self.semaphore:
            return await processor(item)
            
    async def add_to_batch(self, queue_name: str, item: Any) -> Any:
        """
        Add an item to a processing batch.
        
        Args:
            queue_name: Name of the batch queue
            item: Item to process
            
        Returns:
            Processed result
        """
        # Create queue if it doesn't exist
        if queue_name not in self.queues:
            self.queues[queue_name] = []
            self.processing_tasks[queue_name] = asyncio.create_task(
                self._process_queue(queue_name)
            )
        
        # Create a future to wait for result
        future = asyncio.get_running_loop().create_future()
        
        # Add item to queue
        self.queues[queue_name].append((item, future))
        
        # Return the result when ready
        return await future
    
    async def _process_queue(self, queue_name: str) -> None:
        """
        Process items in a queue when batch is ready.
        
        Args:
            queue_name: Name of the batch queue
        """
        while True:
            # Wait for items to accumulate or timeout
            start_time = time.time()
            while (
                len(self.queues[queue_name]) < self.batch_size
                and time.time() - start_time < self.max_wait_time
            ):
                # Check if we have any items at all
                if not self.queues[queue_name]:
                    await asyncio.sleep(0.01)  # Small delay to avoid busy waiting
                    continue
                    
                # Wait a bit for more items
                await asyncio.sleep(0.01)
            
            # Process current batch if we have items
            if self.queues[queue_name]:
                batch = self.queues[queue_name]
                self.queues[queue_name] = []
                
                # Start processing in a separate task to avoid blocking
                asyncio.create_task(self._execute_batch(queue_name, batch))
            else:
                # No items, sleep to avoid busy loop
                await asyncio.sleep(0.1)
    
    async def _execute_batch(
        self,
        queue_name: str,
        batch: List[Tuple[Any, asyncio.Future]]
    ) -> None:
        """
        Execute a batch of items.
        
        This method should be overridden by subclasses.
        
        Args:
            queue_name: Name of the batch queue
            batch: List of (item, future) tuples to process
        """
        # The base implementation just resolves futures with the original items
        # Subclasses should override this with actual processing logic
        for item, future in batch:
            if not future.done():
                future.set_result(item)
    
    def cancel_all(self) -> None:
        """Cancel all pending tasks."""
        for queue_name in list(self.processing_tasks.keys()):
            task = self.processing_tasks.pop(queue_name, None)
            if task:
                task.cancel()
            
            # Resolve any pending futures with exceptions
            queue = self.queues.pop(queue_name, [])
            for _, future in queue:
                if not future.done():
                    future.set_exception(asyncio.CancelledError())


class SolanaRpcBatchProcessor(BatchProcessor):
    """Batch processor for Solana RPC requests."""
    
    def __init__(
        self,
        process_batch_func: Callable[[List[Any]], Awaitable[List[Any]]],
        batch_size: int = 10,
        max_wait_time: float = 0.1,
        max_batch_execution_time: float = 2.0,
        concurrency: int = 5,
        auto_retry: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Solana RPC batch processor.
        
        Args:
            process_batch_func: Function to process a batch of items
            batch_size: Maximum number of items to process in a single batch
            max_wait_time: Maximum time to wait before processing a non-full batch
            max_batch_execution_time: Maximum time for a batch to execute
            concurrency: Maximum number of concurrent tasks
            auto_retry: Whether to automatically retry failed items
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            batch_size=batch_size,
            concurrency=concurrency,
            auto_retry=auto_retry,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_wait_time=max_wait_time,
            max_batch_execution_time=max_batch_execution_time
        )
        self.process_batch_func = process_batch_func
    
    async def _execute_batch(
        self,
        queue_name: str,
        batch: List[Tuple[Any, asyncio.Future]]
    ) -> None:
        """
        Execute a batch of Solana RPC requests.
        
        Args:
            queue_name: Name of the batch queue
            batch: List of (item, future) tuples to process
        """
        try:
            # Extract items from the batch
            items = [item for item, _ in batch]
            
            # Process the batch
            with asyncio.timeout(self.max_batch_execution_time):
                results = await self.process_batch_func(items)
            
            # Set results for futures
            for (_, future), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)
                    
        except asyncio.TimeoutError:
            # Handle timeout by setting exceptions on futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(asyncio.TimeoutError("Batch processing timed out"))
        except Exception as e:
            # Handle other exceptions
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)


def batch_requests(
    batch_processor: BatchProcessor,
    queue_name: str
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for batching similar requests.
    
    Args:
        batch_processor: The batch processor to use
        queue_name: Name of the batch queue
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Build a request context
            request_context = {
                "args": args,
                "kwargs": kwargs
            }
            
            # Add to batch and get result
            result = await batch_processor.add_to_batch(queue_name, request_context)
            
            # Return result
            return result
        return wrapper
    return decorator 