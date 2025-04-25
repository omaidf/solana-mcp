"""Common test fixtures for Solana MCP tests.

This module provides fixtures that can be reused across different test modules.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from solana_mcp.solana_client import SolanaClient
from solana_mcp.services.cache_service import CacheService
from solana_mcp.services.account_service import AccountService
from solana_mcp.services.token_service import TokenService
from solana_mcp.services.transaction_service import TransactionService
from solana_mcp.services.analysis_service import AnalysisService
from solana_mcp.services.market_service import MarketService


@pytest.fixture
def event_loop():
    """Create an event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_solana_client():
    """Create a mock Solana client."""
    client = AsyncMock(spec=SolanaClient)
    
    # Common mock responses
    client.get_balance.return_value = 1000000000  # 1 SOL in lamports
    client.get_transaction.return_value = {
        "slot": 12345,
        "blockTime": 1628000000,
        "meta": {"fee": 5000},
        "transaction": {
            "signatures": ["test_signature"],
            "message": {
                "accountKeys": ["account1", "account2"],
                "recentBlockhash": "test_blockhash",
                "instructions": [
                    {
                        "programIdIndex": 0,
                        "accounts": [0, 1],
                        "data": "test_data"
                    }
                ]
            }
        }
    }
    client.get_signatures_for_address.return_value = [
        {
            "signature": "test_signature",
            "slot": 12345,
            "blockTime": 1628000000,
            "err": None
        }
    ]
    client.get_account_info.return_value = {
        "lamports": 1000000000,
        "owner": "11111111111111111111111111111111",
        "executable": False,
        "rentEpoch": 123
    }
    client.get_token_metadata.return_value = {
        "name": "Test Token",
        "symbol": "TEST",
        "uri": "https://test.com/metadata.json"
    }
    client.get_token_supply.return_value = {
        "amount": "1000000000",
        "decimals": 9,
        "uiAmount": 1000.0
    }
    client.get_market_price.return_value = {
        "price_usd": 1.0,
        "price_sol": 0.01,
        "liquidity_usd": 1000000,
        "market_cap": 10000000,
        "volume_24h": 500000,
        "price_change_24h": 5.0
    }
    
    return client


@pytest.fixture
def mock_cache_service():
    """Create a mock cache service."""
    cache = MagicMock(spec=CacheService)
    
    # Mock cache methods
    cache.get.return_value = None  # Default to cache miss
    cache.set.return_value = None
    
    return cache


@pytest.fixture
def account_service(mock_solana_client, mock_cache_service):
    """Create an AccountService with mock dependencies."""
    return AccountService(mock_solana_client)


@pytest.fixture
def token_service(mock_solana_client, mock_cache_service):
    """Create a TokenService with mock dependencies."""
    return TokenService(mock_solana_client, mock_cache_service)


@pytest.fixture
def transaction_service(mock_solana_client, mock_cache_service):
    """Create a TransactionService with mock dependencies."""
    return TransactionService(mock_solana_client, mock_cache_service)


@pytest.fixture
def market_service(mock_solana_client, mock_cache_service):
    """Create a MarketService with mock dependencies."""
    return MarketService(mock_solana_client, mock_cache_service)


@pytest.fixture
def analysis_service(mock_solana_client, mock_cache_service, transaction_service, token_service):
    """Create an AnalysisService with mock dependencies."""
    return AnalysisService(
        mock_solana_client, 
        transaction_service, 
        token_service, 
        mock_cache_service
    )


# Test data fixtures
@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for tests."""
    return {
        "signature": "test_signature",
        "slot": 12345,
        "blockTime": 1628000000,
        "meta": {"fee": 5000},
        "transaction": {
            "signatures": ["test_signature"],
            "message": {
                "accountKeys": ["account1", "account2"],
                "recentBlockhash": "test_blockhash",
                "instructions": [
                    {
                        "programIdIndex": 0,
                        "accounts": [0, 1],
                        "data": "test_data"
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_token_data():
    """Sample token data for tests."""
    return {
        "mint": "test_mint",
        "metadata": {
            "name": "Test Token",
            "symbol": "TEST",
            "uri": "https://test.com/metadata.json"
        },
        "supply": {
            "amount": "1000000000",
            "decimals": 9,
            "uiAmount": 1000.0
        },
        "price_data": {
            "price_usd": 1.0,
            "price_sol": 0.01,
            "liquidity_usd": 1000000,
            "market_cap_usd": 10000000,
            "volume_24h_usd": 500000,
            "change_24h_percent": 5.0
        }
    } 