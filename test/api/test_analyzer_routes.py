"""
Tests for the Solana Analyzer API routes
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import aiohttp
import json
from typing import Dict, Any

from core.analyzer import TokenInfo, Whale
from api.analyzer_routes import get_analyzer, get_global_analyzer
from main import app

# Create a test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Set up test environment and clean up after tests"""
    # Setup: Create a mock analyzer and override the dependency
    mock_analyzer = AsyncMock()
    
    # Configure token info mock response
    token_info = TokenInfo(
        symbol="MOCK",
        name="Mock Token",
        decimals=9,
        price=1.23,
        mint="MoCKtokEn111111111111111111111111111111111",
        market_cap=1000000.0,
        volume_24h=500000.0,
        supply=1000000000.0
    )
    mock_analyzer.get_token_info.return_value = token_info
    
    # Configure account info mock
    account_info = AsyncMock()
    account_info.lamports = 123456789
    account_info.owner = "11111111111111111111111111111111"
    account_info.executable = False
    account_info.data = {"mock": "data"}
    mock_analyzer.get_account_info.return_value = account_info
    
    # Configure whale data mock with proper Whale objects
    whale1 = Whale(
        address="Wha1e111111111111111111111111111111111111",
        token_balance=1000000.0,
        usd_value=1230000.0,
        percentage=10.0
    )
    
    whale2 = Whale(
        address="Wha1e222222222222222222222222222222222222",
        token_balance=500000.0,
        usd_value=615000.0,
        percentage=5.0
    )
    
    whale_result = {
        "token_info": token_info, 
        "whales": [whale1, whale2],
        "stats": {
            "total_value": 1845000.0,
            "whale_count": 2,
            "timestamp": "2023-01-01T00:00:00"
        }
    }
    mock_analyzer.find_whales.return_value = whale_result
    
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
    mock_analyzer.get_token_accounts_by_owner.return_value = token_accounts

    # Override both dependency functions
    async def override_get_analyzer():
        return mock_analyzer
        
    async def override_get_global_analyzer():
        return mock_analyzer
    
    # Set the dependency overrides
    app.dependency_overrides[get_analyzer] = override_get_analyzer
    app.dependency_overrides[get_global_analyzer] = override_get_global_analyzer
    
    yield mock_analyzer  # This provides the mock to the test
    
    # Teardown: Clear dependency overrides after tests
    app.dependency_overrides.clear()

# Sample wallet addresses
@pytest.fixture
def sample_addresses():
    """Sample addresses for testing"""
    return {
        "wallet": "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg",
        "token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "transaction": "5KtPn1LGuxhHrKNJL1VJ4zNm2fwJGALwJJAJBSqADY8tzZVSKHbsCqP5fm4fXXjCYy5xJgJZqTsK5kK3PSQvRd5B"
    }

# Helper function to print JSON output
def print_test_result(name: str, result: Any):
    """Print JSON output for test result"""
    output = {
        "test": name,
        "result": result
    }
    print(f"TEST_JSON_OUTPUT: {json.dumps(output)}")

# Test the analyzer status endpoint
def test_get_analyzer_status():
    """Test the analyzer status endpoint"""
    response = client.get("/api/analyzer/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "available"
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0

# Test the token info endpoint
def test_get_token_info(setup_and_teardown, sample_addresses):
    """Test getting token information"""
    mock_analyzer = setup_and_teardown
    
    response = client.post(
        "/api/analyzer/token",
        json={"mint_address": sample_addresses["token"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "MOCK"
    assert data["name"] == "Mock Token"
    assert data["decimals"] == 9
    assert data["price"] == 1.23
    assert "market_cap" in data
    assert mock_analyzer.get_token_info.called
    assert mock_analyzer.get_token_info.call_args[0][0] == sample_addresses["token"]

# Test the whales endpoint
def test_find_token_whales(setup_and_teardown, sample_addresses):
    """Test finding token whales"""
    mock_analyzer = setup_and_teardown
    
    response = client.post(
        "/api/analyzer/whales",
        json={
            "mint_address": sample_addresses["token"],
            "min_usd_value": 100000,
            "max_holders": 500
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "token_info" in data
    assert "whales" in data
    assert "stats" in data
    assert len(data["whales"]) == 2
    assert data["stats"]["whale_count"] == 2
    assert mock_analyzer.find_whales.called
    assert mock_analyzer.find_whales.call_args[0][0] == sample_addresses["token"]

# Test the account info endpoint
def test_get_account_info(setup_and_teardown, sample_addresses):
    """Test getting account info"""
    mock_analyzer = setup_and_teardown
    
    response = client.post(
        "/api/analyzer/account",
        json={"address": sample_addresses["wallet"]}
    )
    assert response.status_code == 200
    assert mock_analyzer.get_account_info.called
    assert mock_analyzer.get_account_info.call_args[0][0] == sample_addresses["wallet"]

# Test the token accounts endpoint
def test_get_token_accounts(setup_and_teardown, sample_addresses):
    """Test getting token accounts"""
    mock_analyzer = setup_and_teardown
    
    response = client.post(
        "/api/analyzer/token-accounts",
        json={"address": sample_addresses["wallet"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "owner" in data
    assert "accounts" in data
    assert data["owner"] == sample_addresses["wallet"]
    assert mock_analyzer.get_token_accounts_by_owner.called
    assert mock_analyzer.get_token_accounts_by_owner.call_args[0][0] == sample_addresses["wallet"]

# Test error handling
def test_token_info_error_handling(setup_and_teardown):
    """Test error handling in token info endpoint"""
    mock_analyzer = setup_and_teardown
    
    # Test various error types that could occur in real-world scenarios
    error_cases = [
        # ValueError for invalid data
        {"error_type": ValueError, "message": "Token not found", "expected_status": 500},
        
        # ConnectionError for network issues
        {"error_type": aiohttp.ClientConnectionError, "message": "Connection refused", "expected_status": 500},
        
        # JSONDecodeError for invalid API responses
        {"error_type": json.JSONDecodeError, "message": "Invalid JSON", "expected_status": 500, 
         "extra_args": ["", 0, 0]},  # JSONDecodeError requires additional arguments
    ]
    
    for case in error_cases:
        # Configure the mock to raise the specified exception
        if case.get("extra_args"):
            error = case["error_type"](case["message"], *case.get("extra_args", []))
        else:
            error = case["error_type"](case["message"])
            
        mock_analyzer.get_token_info.side_effect = error
        
        response = client.post(
            "/api/analyzer/token",
            json={"mint_address": "InvalidToken"}
        )
        
        assert response.status_code == case["expected_status"]
        data = response.json()
        assert "detail" in data
        assert case["message"] in data["detail"]
    
    # Reset side effect for other tests
    mock_analyzer.get_token_info.side_effect = None

# Test with missing parameters
def test_missing_parameters():
    """Test API with missing required parameters"""
    response = client.post("/api/analyzer/token", json={})
    assert response.status_code == 422  # Unprocessable Entity
    
    response = client.post("/api/analyzer/whales", json={})
    assert response.status_code == 422 

# Test token info endpoint
@pytest.mark.asyncio
async def test_get_token_info_async(client, sample_addresses, mock_analyzer) -> None:
    """Test getting token info"""
    # Set up return value
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
    
    # Patch the analyzer
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

# Test whales endpoint
@pytest.mark.asyncio
async def test_find_whales_async(client, sample_addresses, mock_analyzer) -> None:
    """Test finding whales for a token"""
    # Set up return value - use the same structure from conftest.py
    whale_result = mock_analyzer.find_whales.return_value
    
    # Patch the analyzer
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
        response = client.post(
            "/api/analyzer/whales",
            json={
                "mint_address": sample_addresses["token"],
                "min_usd_value": 100000,
                "max_holders": 10,
                "batch_size": 100,
                "concurrency": 5
            }
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("find_whales", data)
        
        assert response.status_code == 200
        assert "token_info" in data
        assert "whales" in data
        assert "stats" in data
        assert isinstance(data["whales"], list)
        assert len(data["whales"]) > 0
        assert mock_analyzer.find_whales.called

# Test enhanced account info endpoint
@pytest.mark.asyncio
async def test_get_enhanced_account_info_async(client, sample_addresses, mock_analyzer) -> None:
    """Test getting enhanced account info"""
    # Configure mock
    account_info = mock_analyzer.get_account_info.return_value
    
    # Patch the analyzer
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
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
        assert mock_analyzer.get_account_info.called

# Test token accounts endpoint
@pytest.mark.asyncio
async def test_get_token_accounts_async(client, sample_addresses, mock_analyzer) -> None:
    """Test getting token accounts"""
    # Mock token info
    token_info = TokenInfo(
        symbol="MOCK",
        name="Mock Token", 
        decimals=9,
        price=1.23,
        mint="MoCKtokEn111111111111111111111111111111111"
    )
    
    # Configure mocks
    mock_analyzer.get_token_info.return_value = token_info
    
    # Patch the analyzer
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
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
async def test_token_info_error_async(client, sample_addresses, mock_analyzer) -> None:
    """Test error handling in token info endpoint"""
    # Configure mock to raise an exception
    mock_analyzer.get_token_info.side_effect = ValueError("Invalid token address")
    
    # Patch the analyzer
    with patch('api.analyzer_routes.get_analyzer', return_value=mock_analyzer):
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