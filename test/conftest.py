"""
Pytest configuration for Solana MCP Server tests
"""
import os
import sys
import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator, Generator
import inspect
from fastapi.testclient import TestClient
import aiohttp
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app
from main import app

# Import analyzer for mocking
from core.analyzer import SolanaAnalyzer, TokenInfo

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create a TestClient instance for testing the API"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
async def async_client():
    """Create an AsyncClient instance for testing the API asynchronously"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def mock_analyzer():
    """Create a mock SolanaAnalyzer for testing"""
    analyzer = AsyncMock(spec=SolanaAnalyzer)
    
    # Configure token info mock response
    token_info = AsyncMock(spec=TokenInfo)
    token_info.symbol = "MOCK"
    token_info.name = "Mock Token"
    token_info.decimals = 9
    token_info.price = 1.23
    token_info.mint = "MoCKtokEn111111111111111111111111111111111"
    token_info.market_cap = 1000000.0
    token_info.volume_24h = 500000.0
    token_info.supply = 1000000000.0
    analyzer.get_token_info.return_value = token_info
    
    # Configure account info mock
    account_info = AsyncMock()
    account_info.lamports = 123456789
    account_info.owner = "11111111111111111111111111111111"
    account_info.executable = False
    account_info.data = {"mock": "data"}
    analyzer.get_account_info.return_value = account_info
    
    # Configure whale data mock
    whale_result = {
        "token_info": token_info, 
        "whales": [
            {
                "address": "Wha1e111111111111111111111111111111111111",
                "token_balance": 1000000.0,
                "usd_value": 1230000.0,
                "percentage": 10.0
            },
            {
                "address": "Wha1e222222222222222222222222222222222222",
                "token_balance": 500000.0,
                "usd_value": 615000.0,
                "percentage": 5.0
            }
        ],
        "stats": {
            "total_value": 1845000.0,
            "whale_count": 2,
            "timestamp": "2023-01-01T00:00:00"
        }
    }
    analyzer.find_whales.return_value = whale_result
    
    # Configure token accounts mock
    token_accounts = [
        {
            "data": {
                "parsed": {
                    "info": {
                        "mint": "MoCKtokEn111111111111111111111111111111111",
                        "owner": "OwNer111111111111111111111111111111111111",
                        "tokenAmount": {
                            "amount": "100000000",
                            "decimals": 9,
                            "uiAmount": 100.0
                        }
                    }
                }
            }
        }
    ]
    analyzer.get_token_accounts_by_owner.return_value = token_accounts
    
    return analyzer

@pytest.fixture
def sample_addresses():
    """Sample addresses for testing"""
    return {
        "wallet": "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg",
        "token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "transaction": "5KtPn1LGuxhHrKNJL1VJ4zNm2fwJGALwJJAJBSqADY8tzZVSKHbsCqP5fm4fXXjCYy5xJgJZqTsK5kK3PSQvRd5B"
    } 