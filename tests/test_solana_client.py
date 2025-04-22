"""Tests for Solana client functionality."""

import os
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from solana_mcp.solana_client import (
    SolanaClient, SolanaRpcError, InvalidPublicKeyError,
    validate_public_key
)
from solana_mcp.config import SolanaConfig


@pytest.fixture
def mock_response():
    """Create a mock HTTP response."""
    mock = MagicMock()
    mock.raise_for_status = AsyncMock()
    mock.json = AsyncMock(return_value={"result": {"test": "data"}})
    return mock


@pytest.fixture
def mock_httpx_client(mock_response):
    """Create a mock httpx client."""
    with patch("httpx.AsyncClient") as mock_client:
        instance = mock_client.return_value
        instance.__aenter__.return_value = instance
        instance.__aexit__.return_value = None
        instance.post = AsyncMock(return_value=mock_response)
        yield instance


@pytest.fixture
def solana_config():
    """Create a test Solana configuration."""
    return SolanaConfig(
        rpc_url="https://api.testnet.solana.com",
        commitment="confirmed",
        timeout=10
    )


@pytest.fixture
def solana_client(solana_config):
    """Create a Solana client for testing."""
    return SolanaClient(config=solana_config)


def test_validate_public_key_valid():
    """Test validating valid public keys."""
    # Valid public key examples
    valid_keys = [
        "11111111111111111111111111111111",
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        "So11111111111111111111111111111111111111112"
    ]
    
    for key in valid_keys:
        assert validate_public_key(key) is True


def test_validate_public_key_invalid():
    """Test validating invalid public keys."""
    # Invalid public key examples
    invalid_keys = [
        "",  # Empty
        "not-a-key",  # Invalid characters
        "11111111",  # Too short
        "1111111111111111111111111111111111111111111111111111111111111111111111111111"  # Too long
    ]
    
    for key in invalid_keys:
        assert validate_public_key(key) is False


@pytest.mark.asyncio
async def test_make_request_success(solana_client, mock_httpx_client, mock_response):
    """Test making a successful RPC request."""
    result = await solana_client._make_request("testMethod", ["param1", "param2"])
    
    # Check the result
    assert result == {"test": "data"}
    
    # Verify the request was made correctly
    mock_httpx_client.post.assert_called_once()
    args, kwargs = mock_httpx_client.post.call_args
    
    # Check URL
    assert args[0] == "https://api.testnet.solana.com"
    
    # Check headers
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    
    # Check payload
    payload = kwargs["json"]
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "testMethod"
    assert payload["params"] == ["param1", "param2", "confirmed"]


@pytest.mark.asyncio
async def test_make_request_error(solana_client, mock_httpx_client):
    """Test handling RPC errors."""
    # Mock an error response
    error_response = MagicMock()
    error_response.raise_for_status = AsyncMock()
    error_response.json = AsyncMock(return_value={
        "error": {
            "code": -32000,
            "message": "Test error"
        }
    })
    
    mock_httpx_client.post.return_value = error_response
    
    # Test that an RPC error is raised
    with pytest.raises(SolanaRpcError) as excinfo:
        await solana_client._make_request("testMethod", ["param1"])
    
    # Check the error details
    assert "Test error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_account_info_invalid_key(solana_client):
    """Test get_account_info with an invalid key."""
    with pytest.raises(InvalidPublicKeyError):
        await solana_client.get_account_info("invalid-key")


@pytest.mark.asyncio
async def test_get_account_info(solana_client, mock_httpx_client):
    """Test get_account_info with a valid key."""
    valid_key = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    result = await solana_client.get_account_info(valid_key)
    
    # Check the payload from the mock call
    args, kwargs = mock_httpx_client.post.call_args
    payload = kwargs["json"]
    
    assert payload["method"] == "getAccountInfo"
    assert payload["params"][0] == valid_key
    assert payload["params"][1]["encoding"] == "base64"
    assert payload["params"][1]["commitment"] == "confirmed"


@pytest.mark.asyncio
async def test_get_balance(solana_client, mock_httpx_client):
    """Test get_balance with a valid key."""
    valid_key = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    result = await solana_client.get_balance(valid_key)
    
    # Check the payload from the mock call
    args, kwargs = mock_httpx_client.post.call_args
    payload = kwargs["json"]
    
    assert payload["method"] == "getBalance"
    assert payload["params"][0] == valid_key 