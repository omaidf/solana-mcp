"""Unit tests for the decorators module."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from solana_mcp.decorators import (
    # Error handling
    api_error_handler,
    handle_errors,
    validate_solana_key,
    ErrorCode,
    
    # Performance
    measure_execution_time,
    retry_on_failure,
    
    # Validation
    validate_input,
    
    # Request processing
    rate_limit,
)
from solana_mcp.solana_client import InvalidPublicKeyError, SolanaRpcError


# Helper functions and fixtures
def create_mock_request(client_ip="127.0.0.1", request_id=None):
    """Create a mock request for testing."""
    mock_request = MagicMock(spec=Request)
    mock_client = MagicMock()
    mock_client.host = client_ip
    mock_request.client = mock_client
    
    # Set up state with request_id
    mock_state = MagicMock()
    if request_id:
        mock_state.request_id = request_id
    mock_request.state = mock_state
    
    # Mock json method
    mock_request.json = AsyncMock(return_value={})
    
    return mock_request


@pytest.fixture
def mock_request():
    """Fixture for a mock request object."""
    return create_mock_request()


# ===============================================================
# Error Handling Decorator Tests
# ===============================================================

@pytest.mark.asyncio
async def test_api_error_handler_success():
    """Test api_error_handler with successful function execution."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = api_error_handler(mock_func)
    
    result = await decorated_func(create_mock_request())
    
    assert result == {"result": "success"}
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_api_error_handler_invalid_key_fastapi():
    """Test api_error_handler with InvalidPublicKeyError in FastAPI mode."""
    mock_func = AsyncMock(side_effect=InvalidPublicKeyError("Invalid public key"))
    decorated_func = api_error_handler(mock_func)
    
    with pytest.raises(HTTPException) as exc_info:
        await decorated_func(create_mock_request())
    
    assert exc_info.value.status_code == 400
    assert "Invalid Solana public key" in exc_info.value.detail


@pytest.mark.asyncio
async def test_api_error_handler_invalid_key_json():
    """Test api_error_handler with InvalidPublicKeyError in JSON mode."""
    mock_func = AsyncMock(side_effect=InvalidPublicKeyError("Invalid public key"))
    decorated_func = api_error_handler(response_format="json")(mock_func)
    
    result = await decorated_func(create_mock_request())
    
    assert isinstance(result, JSONResponse)
    assert result.status_code == 400
    
    response_data = json.loads(result.body)
    assert response_data["error_code"] == ErrorCode.INVALID_PUBLIC_KEY
    assert "Invalid Solana public key" in response_data["error"]


@pytest.mark.asyncio
async def test_api_error_handler_solana_rpc_error():
    """Test api_error_handler with SolanaRpcError."""
    error = SolanaRpcError("RPC error")
    error.status_code = 503
    error.error_data = {"code": -32005, "message": "Node is behind"}
    
    mock_func = AsyncMock(side_effect=error)
    decorated_func = api_error_handler(response_format="json")(mock_func)
    
    result = await decorated_func(create_mock_request())
    
    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    
    response_data = json.loads(result.body)
    assert response_data["error_code"] == ErrorCode.SOLANA_RPC_ERROR
    assert "RPC error" in response_data["error"]
    assert response_data["details"] == {"code": -32005, "message": "Node is behind"}


@pytest.mark.asyncio
async def test_api_error_handler_value_error():
    """Test api_error_handler with ValueError."""
    mock_func = AsyncMock(side_effect=ValueError("Invalid value"))
    decorated_func = api_error_handler(mock_func)
    
    with pytest.raises(HTTPException) as exc_info:
        await decorated_func(create_mock_request())
    
    assert exc_info.value.status_code == 400
    assert "Invalid value" in exc_info.value.detail


@pytest.mark.asyncio
async def test_api_error_handler_unexpected_error():
    """Test api_error_handler with an unexpected error."""
    mock_func = AsyncMock(side_effect=RuntimeError("Something went wrong"))
    decorated_func = api_error_handler(mock_func)
    
    with pytest.raises(HTTPException) as exc_info:
        await decorated_func(create_mock_request())
    
    assert exc_info.value.status_code == 500
    assert "Error processing request" in exc_info.value.detail


@pytest.mark.asyncio
async def test_handle_errors_success():
    """Test handle_errors with successful function execution."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = handle_errors(mock_func)
    
    result = await decorated_func()
    
    assert result == {"result": "success"}
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_handle_errors_with_solana_rpc_error():
    """Test handle_errors with SolanaRpcError."""
    mock_func = AsyncMock(side_effect=SolanaRpcError("RPC error"))
    decorated_func = handle_errors(mock_func)
    
    with pytest.raises(SolanaRpcError):
        await decorated_func()
    
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_validate_solana_key_success():
    """Test validate_solana_key with successful validation."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = validate_solana_key(mock_func)
    
    result = await decorated_func(None, "valid_mint_address")
    
    assert result == {"result": "success"}
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_validate_solana_key_with_kwargs():
    """Test validate_solana_key with mint in kwargs."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = validate_solana_key(mock_func)
    
    result = await decorated_func(None, mint="valid_mint_address")
    
    assert result == {"result": "success"}
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_validate_solana_key_with_invalid_key():
    """Test validate_solana_key with invalid key."""
    mock_func = AsyncMock(side_effect=InvalidPublicKeyError("Invalid public key"))
    decorated_func = validate_solana_key(mock_func)
    
    with pytest.raises(InvalidPublicKeyError):
        await decorated_func(None, "invalid_mint_address")


# ===============================================================
# Performance Decorator Tests
# ===============================================================

@pytest.mark.asyncio
async def test_measure_execution_time():
    """Test measure_execution_time decorator."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = measure_execution_time(mock_func)
    
    result = await decorated_func()
    
    assert "result" in result
    assert "execution_time_ms" in result
    assert isinstance(result["execution_time_ms"], float)
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_retry_on_failure_success_first_try():
    """Test retry_on_failure with success on first try."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = retry_on_failure()(mock_func)
    
    result = await decorated_func()
    
    assert result == {"result": "success"}
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_retry_on_failure_success_after_retry():
    """Test retry_on_failure with success after one retry."""
    side_effects = [ValueError("First attempt fails"), {"result": "success"}]
    mock_func = AsyncMock(side_effect=side_effects)
    
    # Use small delay for testing
    decorated_func = retry_on_failure(max_retries=2, delay=0.01)(mock_func)
    
    result = await decorated_func()
    
    assert result == {"result": "success"}
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_failure_all_attempts_fail():
    """Test retry_on_failure when all attempts fail."""
    mock_func = AsyncMock(side_effect=ValueError("All attempts fail"))
    
    # Use small delay for testing
    decorated_func = retry_on_failure(max_retries=2, delay=0.01)(mock_func)
    
    with pytest.raises(ValueError) as exc_info:
        await decorated_func()
    
    assert "All attempts fail" in str(exc_info.value)
    assert mock_func.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_retry_on_failure_specific_exceptions():
    """Test retry_on_failure with specific exception types."""
    mock_func = AsyncMock(side_effect=KeyError("Specific error"))
    
    # Only retry on ValueError, not KeyError
    decorated_func = retry_on_failure(
        max_retries=2, 
        delay=0.01,
        exceptions_to_retry=[ValueError]
    )(mock_func)
    
    with pytest.raises(KeyError) as exc_info:
        await decorated_func()
    
    assert "Specific error" in str(exc_info.value)
    assert mock_func.call_count == 1  # No retries for KeyError


# ===============================================================
# Validation Decorator Tests
# ===============================================================

class MockValidationModel:
    """Mock Pydantic model for validation testing."""
    
    def __init__(self, **kwargs):
        """Initialize with kwargs checking for required fields."""
        if "required_field" not in kwargs:
            raise ValueError("required_field is required")
        self.__dict__.update(kwargs)


@pytest.mark.asyncio
async def test_validate_input_success():
    """Test validate_input with valid input."""
    mock_request = create_mock_request()
    mock_request.json = AsyncMock(return_value={"required_field": "value"})
    
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = validate_input(MockValidationModel)(mock_func)
    
    result = await decorated_func(mock_request)
    
    assert result == {"result": "success"}
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_validate_input_missing_required():
    """Test validate_input with missing required field."""
    mock_request = create_mock_request()
    mock_request.json = AsyncMock(return_value={"optional_field": "value"})
    
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = validate_input(MockValidationModel)(mock_func)
    
    result = await decorated_func(mock_request)
    
    assert isinstance(result, JSONResponse)
    assert result.status_code == 400
    
    response_data = json.loads(result.body)
    assert response_data["error_code"] == ErrorCode.INVALID_INPUT
    assert "Validation error" in response_data["error"]


# ===============================================================
# Request Processing Decorator Tests
# ===============================================================

@pytest.mark.asyncio
async def test_rate_limit_under_limit():
    """Test rate_limit when under the limit."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = rate_limit(limit=5)(mock_func)
    
    # Make 3 requests (under the limit of 5)
    for _ in range(3):
        result = await decorated_func(create_mock_request())
        assert result == {"result": "success"}
    
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_rate_limit_exceeds_limit_fastapi():
    """Test rate_limit when exceeding the limit with FastAPI error format."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = rate_limit(limit=2, period=10)(mock_func)
    
    # First two requests should succeed
    for _ in range(2):
        result = await decorated_func(create_mock_request(client_ip="test_ip"))
        assert result == {"result": "success"}
    
    # Third request should be rate limited
    with pytest.raises(HTTPException) as exc_info:
        await decorated_func(create_mock_request(client_ip="test_ip"))
    
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exceeds_limit_json():
    """Test rate_limit when exceeding the limit with JSON error format."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = rate_limit(limit=2, period=10, error_format="json")(mock_func)
    
    # First two requests should succeed
    for _ in range(2):
        result = await decorated_func(create_mock_request(client_ip="test_ip2"))
        assert result == {"result": "success"}
    
    # Third request should be rate limited
    result = await decorated_func(create_mock_request(client_ip="test_ip2"))
    
    assert isinstance(result, JSONResponse)
    assert result.status_code == 429
    
    response_data = json.loads(result.body)
    assert response_data["error_code"] == ErrorCode.RATE_LIMITED
    assert "Rate limit exceeded" in response_data["error"]
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_different_ips():
    """Test rate_limit with different client IPs."""
    mock_func = AsyncMock(return_value={"result": "success"})
    decorated_func = rate_limit(limit=2)(mock_func)
    
    # Two requests from first IP
    for _ in range(2):
        result = await decorated_func(create_mock_request(client_ip="ip1"))
        assert result == {"result": "success"}
    
    # Two requests from second IP
    for _ in range(2):
        result = await decorated_func(create_mock_request(client_ip="ip2"))
        assert result == {"result": "success"}
    
    # Third request from first IP should be rate limited
    with pytest.raises(HTTPException) as exc_info:
        await decorated_func(create_mock_request(client_ip="ip1"))
    
    assert exc_info.value.status_code == 429
    assert mock_func.call_count == 4  # 2 successful calls per IP


if __name__ == "__main__":
    pytest.main() 