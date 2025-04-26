"""
Tests for the SolanaAnalyzer core functionality
"""
import pytest
import asyncio
import json
from typing import Dict, Any, AsyncGenerator
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from core.analyzer import (
    SolanaAnalyzer, 
    TokenInfo, 
    Whale, 
    AccountInfo, 
    TransactionInfo
)

# Mock aiohttp ClientSession responses
class MockResponse:
    def __init__(self, data: Dict[str, Any], status: int = 200):
        self.data = data
        self.status = status
        
    async def json(self) -> Dict[str, Any]:
        return self.data
        
    async def __aenter__(self) -> 'MockResponse':
        return self
        
    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

# Helper function to print JSON output
def print_test_result(name: str, result: Any):
    """Print JSON output for test result"""
    output = {
        "test": name,
        "result": result
    }
    print(f"TEST_JSON_OUTPUT: {json.dumps(output)}")

# Test token info retrieval
@pytest.mark.asyncio
async def test_get_token_info(mock_analyzer: SolanaAnalyzer, sample_addresses: Dict[str, str]) -> None:
    """Test retrieving token information"""
    # Set up return values
    token_info = TokenInfo(
        symbol="USDC",
        name="USD Coin",
        decimals=6,
        price=1.0,
        mint=sample_addresses["token"],
        market_cap=56000000000,
        volume_24h=12500000,
        supply=10000000.0
    )
    
    # Configure mock
    mock_analyzer.get_token_info.return_value = token_info
    
    # Call the method
    result = await mock_analyzer.get_token_info(sample_addresses["token"])
    
    # Print JSON output
    print_test_result("get_token_info", {
        "symbol": result.symbol,
        "name": result.name,
        "decimals": result.decimals,
        "price": result.price,
        "mint": result.mint
    })
    
    # Verify results
    assert result == token_info
    assert result.symbol == "USDC"
    assert result.name == "USD Coin"
    assert result.decimals == 6
    assert result.price == 1.0
    assert result.mint == sample_addresses["token"]
    assert result.market_cap == 56000000000
    assert result.volume_24h == 12500000
    assert result.supply == 10000000.0
    assert mock_analyzer.get_token_info.called

# Test account info retrieval
@pytest.mark.asyncio
async def test_get_account_info(mock_analyzer: SolanaAnalyzer, sample_addresses: Dict[str, str]) -> None:
    """Test retrieving account information"""
    # Setup mock response
    account_info = AccountInfo(
        lamports=12345678,
        owner="11111111111111111111111111111111",
        executable=False,
        data=["base64data", "base64"],
        rent_epoch=123
    )
    
    # Configure mock
    mock_analyzer.get_account_info.return_value = account_info
    
    # Call the method
    result = await mock_analyzer.get_account_info(sample_addresses["wallet"])
    
    # Print JSON output
    print_test_result("get_account_info", {
        "lamports": result.lamports,
        "owner": result.owner,
        "executable": result.executable,
        "rent_epoch": result.rent_epoch
    })
    
    # Verify results
    assert result == account_info
    assert result.lamports == 12345678
    assert result.owner == "11111111111111111111111111111111"
    assert result.executable is False
    assert result.rent_epoch == 123
    assert result.data == ["base64data", "base64"]
    assert mock_analyzer.get_account_info.called

# Test finding whales
@pytest.mark.asyncio
async def test_find_whales(mock_analyzer: SolanaAnalyzer, sample_addresses: Dict[str, str]) -> None:
    """Test finding whale holders"""
    # Mock token info result
    token_info = TokenInfo(
        symbol="USDC",
        name="USD Coin",
        decimals=6,
        price=1.0,
        mint=sample_addresses["token"],
        market_cap=56000000000,
        volume_24h=12500000,
        supply=10000000.0
    )
    
    # Create whales for the result
    whale1 = Whale(
        address="Whale1111111111111111111111111111111111111",
        token_balance=1000000.0,
        usd_value=1000000.0,
        percentage=10.0,
        last_active="2023-01-01T00:00:00Z"
    )
    
    whale2 = Whale(
        address="Whale2222222222222222222222222222222222222",
        token_balance=500000.0,
        usd_value=500000.0,
        percentage=5.0,
        last_active="2023-02-01T00:00:00Z"
    )
    
    # Create the expected result
    expected_result = {
        "token_info": token_info,
        "whales": [whale1, whale2],
        "stats": {
            "total_value": 1500000.0,
            "whale_count": 2,
            "timestamp": "2023-01-01T00:00:00"
        }
    }
    
    # Configure mock
    mock_analyzer.find_whales.return_value = expected_result
    
    # Call the method
    result = await mock_analyzer.find_whales(
        sample_addresses["token"], 
        min_usd_value=100000,
        max_holders=10
    )
    
    # Print JSON output
    print_test_result("find_whales", {
        "whale_count": len(result["whales"]),
        "total_value": result["stats"]["total_value"],
        "token_symbol": result["token_info"].symbol
    })
    
    # Verify results
    assert result == expected_result
    assert len(result["whales"]) == 2
    assert result["stats"]["whale_count"] == 2
    assert result["token_info"] == token_info
    assert mock_analyzer.find_whales.called
    assert mock_analyzer.find_whales.call_args[0][0] == sample_addresses["token"]
    assert mock_analyzer.find_whales.call_args[1]["min_usd_value"] == 100000
    assert mock_analyzer.find_whales.call_args[1]["max_holders"] == 10

# Test token account retrieval
@pytest.mark.asyncio
async def test_get_token_accounts_by_owner(mock_analyzer: SolanaAnalyzer, sample_addresses: Dict[str, str]) -> None:
    """Test retrieving token accounts"""
    # Setup mock response
    token_accounts = [
        {
            "lamports": 2039280,
            "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "data": ["encoded-data", "base64"],
            "executable": False,
            "rentEpoch": 123
        },
        {
            "lamports": 2039280,
            "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "data": ["encoded-data-2", "base64"],
            "executable": False,
            "rentEpoch": 123
        }
    ]
    
    # Configure mock
    mock_analyzer.get_token_accounts_by_owner.return_value = token_accounts
    
    # Call the method
    result = await mock_analyzer.get_token_accounts_by_owner(sample_addresses["wallet"])
    
    # Print JSON output
    print_test_result("get_token_accounts_by_owner", {
        "account_count": len(result),
        "first_account_lamports": result[0]["lamports"],
        "first_account_owner": result[0]["owner"]
    })
    
    # Verify results
    assert result == token_accounts
    assert len(result) == 2
    assert all(isinstance(account, dict) for account in result)
    assert result[0]["lamports"] == 2039280
    assert result[0]["owner"] == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    assert mock_analyzer.get_token_accounts_by_owner.called
    assert mock_analyzer.get_token_accounts_by_owner.call_args[0][0] == sample_addresses["wallet"]
    
# Test cache functionality
@pytest.mark.asyncio
async def test_cache_mechanism(mock_analyzer: SolanaAnalyzer, sample_addresses: Dict[str, str]) -> None:
    """Test the caching mechanism"""
    # Set up test data
    test_data = {"test": "data"}
    cache_key = f"test_key_{sample_addresses['token']}"
    cache_type = "token_metadata"
    
    # Configure mocks
    mock_analyzer._get_cached.return_value = None  # First call returns None (cache miss)
    mock_analyzer._set_cache.return_value = None
    
    # Call set cache
    await mock_analyzer._set_cache(cache_key, test_data, cache_type)
    
    # Now simulate a cache hit on the second call
    mock_analyzer._get_cached.return_value = test_data  # Second call returns data (cache hit)
    
    # Retrieve from cache
    cached_data = await mock_analyzer._get_cached(cache_key, cache_type)
    
    # Print JSON output
    print_test_result("cache_mechanism", {
        "cache_key": cache_key,
        "cache_type": cache_type,
        "cached_data": cached_data
    })
    
    # Verify cache operations
    assert cached_data == test_data
    assert mock_analyzer._set_cache.called
    assert mock_analyzer._get_cached.called 