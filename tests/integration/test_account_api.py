"""
Integration tests for the account API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock

from app import app
from solana_mcp.models.api_models import ApiResponse

client = TestClient(app)


@pytest.fixture
def mock_account_service(monkeypatch):
    """Mock the AccountService for testing."""
    mock_service = MagicMock()
    
    # Set up mock responses
    mock_service.get_account_info.return_value = {
        "address": "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB",
        "lamports": 1000000000,  # 1 SOL
        "owner": "11111111111111111111111111111111",
        "executable": False,
        "rent_epoch": 275,
        "data": "base64_encoded_data"
    }
    
    mock_service.get_account_balance.return_value = {
        "address": "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB",
        "lamports": 1000000000,
        "sol": 1.0
    }
    
    mock_service.get_tokens_for_owner.return_value = [
        {
            "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "address": "TokenAccountAddress1",
            "amount": "100000000",
            "decimals": 6,
            "ui_amount": 100.0,
            "token_name": "USD Coin",
            "token_symbol": "USDC"
        },
        {
            "mint": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "address": "TokenAccountAddress2",
            "amount": "50000000",
            "decimals": 6,
            "ui_amount": 50.0,
            "token_name": "USDT",
            "token_symbol": "USDT"
        }
    ]
    
    mock_service.is_executable.return_value = False
    
    # Patch the get_account_service dependency in FastAPI
    with patch("solana_mcp.routes.accounts.get_account_service", return_value=mock_service):
        yield mock_service


def test_get_account_info(mock_account_service):
    """Test getting account information."""
    account_address = "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB"
    response = client.get(f"/api/accounts/{account_address}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["address"] == account_address
    assert data["data"]["lamports"] == 1000000000
    assert data["data"]["owner"] == "11111111111111111111111111111111"
    assert data["data"]["executable"] == False


def test_get_account_balance(mock_account_service):
    """Test getting account balance."""
    account_address = "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB"
    response = client.get(f"/api/accounts/{account_address}/balance")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["address"] == account_address
    assert data["data"]["lamports"] == 1000000000
    assert data["data"]["sol"] == 1.0


def test_get_account_tokens(mock_account_service):
    """Test getting tokens owned by an account."""
    account_address = "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB"
    response = client.get(f"/api/accounts/{account_address}/tokens")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert len(data["data"]) == 2
    assert data["data"][0]["mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    assert data["data"][0]["ui_amount"] == 100.0
    assert data["data"][1]["mint"] == "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
    assert data["data"][1]["ui_amount"] == 50.0


def test_is_executable(mock_account_service):
    """Test checking if an account is executable."""
    account_address = "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB"
    response = client.get(f"/api/accounts/{account_address}/is-executable")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"] == False
    
    # Test with a program account
    mock_account_service.is_executable.return_value = True
    program_address = "11111111111111111111111111111111"
    response = client.get(f"/api/accounts/{program_address}/is-executable")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"] == True
    
    # Reset for other tests
    mock_account_service.is_executable.return_value = False


def test_nonexistent_account(mock_account_service):
    """Test error handling for non-existent account."""
    # Configure mock to raise exception
    mock_account_service.get_account_info.side_effect = Exception("Account not found")
    
    response = client.get("/api/accounts/NonExistentAccount")
    
    # API should return a 404 or appropriate error status
    assert response.status_code in [404, 400, 500]
    data = response.json()
    
    # Reset the side effect for other tests
    mock_account_service.get_account_info.side_effect = None


def test_invalid_account_address(mock_account_service):
    """Test error handling for invalid account address."""
    response = client.get("/api/accounts/invalid-address")
    
    # API should return a 400 Bad Request
    assert response.status_code == 400
    data = response.json() 