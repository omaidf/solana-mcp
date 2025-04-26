"""
Tests for the core API routes
"""
import pytest
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

# Helper function to print JSON output
def print_test_result(name: str, result: Any):
    """Print JSON output for test result"""
    output = {
        "test": name,
        "result": result
    }
    print(f"TEST_JSON_OUTPUT: {json.dumps(output)}")

# Test the API status endpoint
def test_get_status(client) -> None:
    """Test the API status endpoint"""
    response = client.get("/api/status")
    data = response.json()
    
    # Print JSON output
    print_test_result("get_status", data)
    
    assert response.status_code == 200
    assert data["status"] == "online"
    assert data["blockchain"] == "solana"
    assert "version" in data

# Test the health check endpoint
def test_health_check(client) -> None:
    """Test the health check endpoint"""
    response = client.get("/health")
    data = response.json()
    
    # Print JSON output
    print_test_result("health_check", data)
    
    assert response.status_code == 200
    assert data["status"] == "healthy"

# Test the models endpoint
def test_list_models(client) -> None:
    """Test listing available models"""
    response = client.get("/api/models")
    data = response.json()
    
    # Print JSON output
    print_test_result("list_models", data)
    
    assert response.status_code == 200
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0

# Test the account info endpoint
@pytest.mark.asyncio
async def test_get_account_info(client, sample_addresses) -> None:
    """Test getting account info from the core API"""
    # Mock the SolanaClient
    mock_client = AsyncMock()
    mock_client.get_account_info.return_value = {
        "address": sample_addresses["wallet"],
        "lamports": 123456789,
        "owner": "11111111111111111111111111111111",
        "executable": False,
        "data_size": 100
    }
    
    # Patch the SolanaClient constructor
    with patch('api.routes.SolanaClient', return_value=mock_client):
        response = client.post(
            "/api/account",
            json={"address": sample_addresses["wallet"]}
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("get_account_info", data)
        
        assert response.status_code == 200
        assert data["address"] == sample_addresses["wallet"]
        assert "lamports" in data
        assert mock_client.get_account_info.called

# Test the transaction endpoint
@pytest.mark.asyncio
async def test_get_transaction(client, sample_addresses) -> None:
    """Test getting transaction from the core API"""
    # Mock the SolanaClient
    mock_client = AsyncMock()
    mock_client.get_transaction.return_value = {
        "signature": sample_addresses["transaction"],
        "slot": 12345,
        "block_time": 1643673600,
        "success": True,
        "fee": 5000
    }
    
    # Patch the SolanaClient constructor
    with patch('api.routes.SolanaClient', return_value=mock_client):
        response = client.post(
            "/api/transaction",
            json={"signature": sample_addresses["transaction"]}
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("get_transaction", data)
        
        assert response.status_code == 200
        assert data["signature"] == sample_addresses["transaction"]
        assert "slot" in data
        assert mock_client.get_transaction.called

# Test the context generation endpoint
@pytest.mark.asyncio
async def test_generate_context(client, sample_addresses) -> None:
    """Test generating model context"""
    # Mock the ModelContext
    mock_context = AsyncMock()
    mock_context.generate.return_value = {
        "context_id": "test-context-id",
        "address": sample_addresses["wallet"],
        "model_type": "transaction-history",
        "timestamp": 1643673600,
        "data": {
            "summary": {
                "balance_sol": 1.5,
                "transaction_count": 10
            }
        }
    }
    
    # Patch the ModelContext constructor
    with patch('api.routes.ModelContext', return_value=mock_context):
        response = client.post(
            "/api/context",
            json={
                "address": sample_addresses["wallet"],
                "model_type": "transaction-history"
            }
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("generate_context", data)
        
        assert response.status_code == 200
        assert data["context_id"] == "test-context-id"
        assert data["model_type"] == "transaction-history"
        assert "data" in data
        assert mock_context.generate.called

# Test error handling for invalid address
@pytest.mark.asyncio
async def test_invalid_address_error(client) -> None:
    """Test error handling for invalid address"""
    # Mock the SolanaClient
    mock_client = AsyncMock()
    mock_client.get_account_info.side_effect = Exception("Invalid address format")
    
    # Patch the SolanaClient constructor
    with patch('api.routes.SolanaClient', return_value=mock_client):
        response = client.post(
            "/api/account",
            json={"address": "invalid-address"}
        )
        data = response.json()
        
        # Print JSON output
        print_test_result("invalid_address_error", data)
        
        assert response.status_code == 500
        assert "error" in data
        assert "Invalid address" in data["error"] 