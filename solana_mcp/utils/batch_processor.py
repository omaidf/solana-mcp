"""
Batch processing utilities for Solana MCP.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use `solana_mcp.utils.batching` instead.
"""

import warnings
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

from solana_mcp.utils.batching import (
    BatchProcessor, SolanaRpcBatchProcessor, batch_process_requests,
    batch_requests
)

# Show deprecation warning
warnings.warn(
    "The `solana_mcp.utils.batch_processor` module is deprecated. "
    "Use `solana_mcp.utils.batching` instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = [
    'BatchProcessor',
    'SolanaRpcBatchProcessor',
    'batch_process_requests',
    'batch_requests'
]

# Type variables for backward compatibility
T = TypeVar('T')
R = TypeVar('R')


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
        retry_delay: float = 1.0
    ):
        """
        Initialize a batch processor.
        
        Args:
            batch_size: Maximum number of items per batch
            concurrency: Maximum number of concurrent tasks
            auto_retry: Whether to automatically retry failed items
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrency)
    
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