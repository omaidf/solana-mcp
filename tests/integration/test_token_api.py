"""
Integration tests for the token API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock

from app import app
from solana_mcp.models.token import TokenInfo, TokenMetadata
from solana_mcp.models.api_models import ApiResponse

client = TestClient(app)


@pytest.fixture
def mock_token_service(monkeypatch):
    """Mock the TokenService for testing."""
    mock_service = MagicMock()
    
    # Set up mock responses
    mock_service.list_tokens.return_value = [
        TokenInfo(
            address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            name="USD Coin",
            symbol="USDC",
            decimals=6,
            total_supply="10000000000000",
            is_nft=False,
            metadata=TokenMetadata(
                name="USD Coin",
                symbol="USDC",
                logo="https://example.com/usdc.png",
                description="Stablecoin pegged to USD",
                is_nft=False
            )
        ),
        TokenInfo(
            address="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            name="USDT",
            symbol="USDT",
            decimals=6,
            total_supply="20000000000000",
            is_nft=False,
            metadata=TokenMetadata(
                name="USDT",
                symbol="USDT",
                logo="https://example.com/usdt.png",
                description="Stablecoin pegged to USD",
                is_nft=False
            )
        )
    ]
    
    mock_service.get_token_metadata.return_value = {
        "name": "USD Coin",
        "symbol": "USDC",
        "logo": "https://example.com/usdc.png",
        "description": "Stablecoin pegged to USD",
        "external_url": "https://www.circle.com/usdc",
        "image": "https://example.com/usdc.png",
        "is_nft": False
    }
    
    mock_service.get_token_supply.return_value = {
        "total_supply": "10000000000000",
        "circulating_supply": "9500000000000",
        "decimals": 6,
        "total_holders": 1000000,
        "max_supply": "10000000000000"
    }
    
    mock_service.get_token_largest_accounts.return_value = [
        {
            "address": "Holder1Address",
            "amount": "1000000000000",
            "amount_decimal": 1000000.0,
            "percentage": 10.0
        },
        {
            "address": "Holder2Address",
            "amount": "500000000000",
            "amount_decimal": 500000.0,
            "percentage": 5.0
        }
    ]
    
    mock_service.get_multiple_token_metadata.return_value = {
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": {
            "name": "USD Coin",
            "symbol": "USDC",
            "logo": "https://example.com/usdc.png",
            "description": "Stablecoin pegged to USD"
        },
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": {
            "name": "USDT",
            "symbol": "USDT",
            "logo": "https://example.com/usdt.png",
            "description": "Stablecoin pegged to USD"
        }
    }
    
    # Patch the get_token_service dependency in FastAPI
    with patch("solana_mcp.routes.tokens.get_token_service", return_value=mock_service):
        yield mock_service


def test_list_tokens(mock_token_service):
    """Test listing tokens with pagination."""
    response = client.get("/api/tokens?limit=10&offset=0")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert "tokens" in data["data"]
    assert len(data["data"]["tokens"]) == 2
    assert data["data"]["tokens"][0]["name"] == "USD Coin"
    assert data["data"]["tokens"][1]["name"] == "USDT"
    
    # Check pagination info
    assert "pagination" in data
    assert data["pagination"]["limit"] == 10
    assert data["pagination"]["offset"] == 0


def test_get_token_metadata(mock_token_service):
    """Test getting token metadata."""
    token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    response = client.get(f"/api/tokens/{token_address}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["name"] == "USD Coin"
    assert data["data"]["symbol"] == "USDC"
    assert data["data"]["description"] == "Stablecoin pegged to USD"
    assert data["data"]["is_nft"] == False


def test_get_token_supply(mock_token_service):
    """Test getting token supply information."""
    token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    response = client.get(f"/api/tokens/{token_address}/supply")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["total_supply"] == "10000000000000"
    assert data["data"]["decimals"] == 6


def test_get_token_largest_accounts(mock_token_service):
    """Test getting largest accounts for a token."""
    token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    response = client.get(f"/api/tokens/{token_address}/largest_accounts?limit=10")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert len(data["data"]) == 2
    assert data["data"][0]["address"] == "Holder1Address"
    assert data["data"][0]["percentage"] == 10.0


def test_batch_get_token_metadata(mock_token_service):
    """Test batch getting token metadata."""
    token_addresses = [
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
    ]
    
    response = client.post("/api/tokens/batch", json=token_addresses)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in data["data"]
    assert "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" in data["data"]
    assert data["data"]["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]["symbol"] == "USDC"
    assert data["data"]["Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]["symbol"] == "USDT"


def test_get_nonexistent_token(mock_token_service):
    """Test error handling for non-existent token."""
    # Configure mock to raise exception
    mock_token_service.get_token_metadata.side_effect = Exception("Token not found")
    
    response = client.get("/api/tokens/NonExistentToken")
    
    # API should return a 404 or appropriate error status
    assert response.status_code in [404, 400, 500]
    data = response.json()
    
    # Reset the side effect for other tests
    mock_token_service.get_token_metadata.side_effect = None 