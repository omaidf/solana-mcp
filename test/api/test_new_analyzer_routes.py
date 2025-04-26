"""
Tests for the analyzer API routes
"""
import pytest
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from core.analyzer import TokenInfo, Whale
# No need to import TestClient here, it's provided by the conftest.py fixture

# Helper function to print JSON output
def print_test_result(name: str, result: Any):
    """Print JSON output for test result"""
    output = {
        "test": name,
        "result": result
    }
    print(f"TEST_JSON_OUTPUT: {json.dumps(output)}")

# Test the analyzer status endpoint
def test_analyzer_status(client):
    """Test the analyzer status endpoint"""
    response = client.get("/api/analyzer/status")
    data = response.json()
    
    # Print JSON output
    print_test_result("analyzer_status", data)
    
    assert response.status_code == 200
    assert "status" in data
    assert "features" in data
    assert isinstance(data["features"], list)

# Test token info endpoint with proper mocking
@pytest.mark.asyncio
async def test_token_info(client, mock_analyzer, sample_addresses):
    """Test getting token info"""
    # Mock token info
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
    
    # Configure the mock
    mock_analyzer.get_token_info.return_value = token_info
    
    # Patch the API route to use our mock
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
        response = client.post(
            "/api/analyzer/token",
            json={"mint_address": sample_addresses["token"]}
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("get_token_info", data)
        
        assert response.status_code == 200
        assert data["symbol"] == "USDC"
        assert data["name"] == "USD Coin"
        assert data["decimals"] == 6
        assert data["price"] == 1.0
        assert data["mint"] == sample_addresses["token"]
        assert mock_analyzer.get_token_info.called

# Test whales endpoint with proper mocking
@pytest.mark.asyncio
async def test_find_whales(client, mock_analyzer, sample_addresses):
    """Test finding whales for a token"""
    # Create a mock whale
    whale1 = {
        "address": "Whale111111111111111111111111111111111111",
        "token_balance": 1000000.0,
        "usd_value": 1000000.0,
        "percentage": 10.0
    }
    
    # Create token info
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
    
    # Set up the whale result
    whale_result = {
        "token_info": token_info,
        "whales": [whale1],
        "stats": {
            "total_value": 1000000.0,
            "whale_count": 1,
            "timestamp": "2023-01-01T00:00:00"
        }
    }
    
    # Configure the mock
    mock_analyzer.find_whales.return_value = whale_result
    
    # Patch the API route to use our mock
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
        # Make the request
        response = client.post(
            "/api/analyzer/whales",
            json={
                "mint_address": sample_addresses["token"],
                "min_usd_value": 100000
            }
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("find_whales", data)
        
        assert response.status_code == 200
        assert "token_info" in data
        assert "whales" in data
        assert "stats" in data
        assert len(data["whales"]) > 0
        assert mock_analyzer.find_whales.called

# Test enhanced account info endpoint
@pytest.mark.asyncio
async def test_get_enhanced_account_info(client, mock_analyzer, sample_addresses):
    """Test getting enhanced account info"""
    # Create a mock account info response
    account_info = {
        "lamports": 12345678,
        "owner": "11111111111111111111111111111111",
        "executable": False,
        "data": "base64 encoded data",
        "rent_epoch": 123
    }
    
    # Configure the mock
    mock_analyzer.get_account_info.return_value = account_info
    
    # Patch the API route to use our mock
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
        # Make the request
        response = client.post(
            "/api/analyzer/account",
            json={
                "address": sample_addresses["wallet"],
                "encoding": "jsonParsed"
            }
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("get_enhanced_account_info", data)
        
        assert response.status_code == 200
        assert "lamports" in data
        assert data["lamports"] == 12345678
        assert data["owner"] == "11111111111111111111111111111111"
        assert mock_analyzer.get_account_info.called

# Test token accounts endpoint
@pytest.mark.asyncio
async def test_get_token_accounts(client, mock_analyzer, sample_addresses):
    """Test getting token accounts"""
    # Configure the mock
    token_accounts = [
        {
            "data": {
                "parsed": {
                    "info": {
                        "mint": "MoCKtokEn111111111111111111111111111111111",
                        "owner": sample_addresses["wallet"],
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
    mock_analyzer.get_token_accounts_by_owner.return_value = token_accounts
    
    # Patch the API route to use our mock
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
        # Make the request
        response = client.post(
            "/api/analyzer/token-accounts",
            json={"address": sample_addresses["wallet"]}
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("get_token_accounts", data)
        
        assert response.status_code == 200
        assert "owner" in data
        assert "accounts" in data
        assert "count" in data
        assert data["owner"] == sample_addresses["wallet"]
        assert isinstance(data["accounts"], list)
        assert mock_analyzer.get_token_accounts_by_owner.called

# Test error handling
@pytest.mark.asyncio
async def test_token_info_error(client, mock_analyzer):
    """Test error handling in token info endpoint"""
    # Configure mock to raise an exception
    mock_analyzer.get_token_info.side_effect = ValueError("Invalid token address")
    
    # Patch the API route to use our mock
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
        # Make the request
        response = client.post(
            "/api/analyzer/token",
            json={"mint_address": "invalid-token"}
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("token_info_error", data)
        
        assert response.status_code == 500
        assert "error" in data
        assert "Invalid token address" in data["error"]
        assert mock_analyzer.get_token_info.called 