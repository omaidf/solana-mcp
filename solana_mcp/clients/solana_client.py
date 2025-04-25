"""Facade client for Solana MCP.

This module provides a unified client interface composing all specialized Solana client functionality.
"""

from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

from solana_mcp.config import SolanaConfig, get_solana_config
from solana_mcp.clients.base_client import BaseSolanaClient
from solana_mcp.clients.account_client import AccountClient
from solana_mcp.clients.token_client import TokenClient
from solana_mcp.clients.transaction_client import TransactionClient
from solana_mcp.clients.market_client import MarketClient
from solana_mcp.utils.batching import BatchProcessor
from solana_mcp.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

class SolanaClient(BaseSolanaClient):
    """Unified client for Solana blockchain interactions.
    
    This class composes specialized clients for different types of operations,
    providing a unified interface for the application.
    """
    
    def __init__(self, config: Optional[SolanaConfig] = None):
        """Initialize the Solana client.
        
        Args:
            config: Solana configuration. Defaults to environment-based config.
        """
        super().__init__(config)
        
        # Create specialized clients
        self.account = AccountClient(config)
        self.token = TokenClient(config)
        self.transaction = TransactionClient(config)
        self.market = MarketClient(config)
        
        # Create batching processor
        self.batch_size = 10
        self.batch_processor = BatchProcessor(batch_size=self.batch_size)
    
    async def close(self):
        """Close the client and all specialized clients."""
        await self.account.close()
        await self.token.close()
        await self.transaction.close()
        await self.market.close()
        await super().close()


@asynccontextmanager
async def get_solana_client():
    """Get a Solana client as an async context manager.
    
    Yields:
        SolanaClient: An initialized Solana client.
    """
    client = SolanaClient()
    try:
        yield client
    finally:
        await client.close() 