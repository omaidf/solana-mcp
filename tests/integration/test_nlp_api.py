"""
Integration tests for the NLP API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock

from app import app
from solana_mcp.models.api_models import ApiResponse

client = TestClient(app)


@pytest.fixture
def mock_nlp_service(monkeypatch):
    """Mock the NLPService for testing."""
    mock_service = MagicMock()
    
    # Set up mock responses for process_query
    mock_service.process_query.return_value = {
        "query": "What is the price of USDC?",
        "intent": "token_price",
        "entities": {"mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        "result": {
            "price_usd": 1.0,
            "price_sol": 0.01,
            "change_24h": "+0.01%"
        },
        "timestamp": "2023-01-01T00:00:00Z",
        "session_id": "test-session-123"
    }
    
    # Set up mock responses for suggest_queries
    mock_service.suggest_queries.return_value = [
        "What is the price of USDC?",
        "Show me USDC token info",
        "Who are the largest USDC holders?"
    ]
    
    # Patch the get_nlp_service dependency in FastAPI
    with patch("solana_mcp.routes.nlp.get_nlp_service", return_value=mock_service):
        yield mock_service


def test_process_query(mock_nlp_service):
    """Test processing a natural language query."""
    query_data = {
        "query": "What is the price of USDC?",
        "session_id": "test-session-123",
        "format_level": "standard"
    }
    
    response = client.post("/api/nlp/process", json=query_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["query"] == "What is the price of USDC?"
    assert data["data"]["intent"] == "token_price"
    assert data["data"]["entities"]["mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    assert "result" in data["data"]
    assert data["data"]["result"]["price_usd"] == 1.0
    
    # Verify the mock was called with correct parameters
    mock_nlp_service.process_query.assert_called_once_with(
        query="What is the price of USDC?",
        session_id="test-session-123",
        format_level="standard"
    )


def test_get_suggestions(mock_nlp_service):
    """Test getting query suggestions."""
    response = client.get("/api/nlp/suggest?input=usdc&limit=3")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert len(data["data"]) == 3
    assert "What is the price of USDC?" in data["data"]
    assert "Show me USDC token info" in data["data"]
    
    # Verify the mock was called with correct parameters
    mock_nlp_service.suggest_queries.assert_called_once_with("usdc", 3)


def test_process_invalid_query(mock_nlp_service):
    """Test error handling for invalid query format."""
    # Missing required query field
    query_data = {
        "session_id": "test-session-123",
        "format_level": "standard"
    }
    
    response = client.post("/api/nlp/process", json=query_data)
    
    # Should return 422 Unprocessable Entity for validation errors
    assert response.status_code == 422
    data = response.json()
    
    assert "detail" in data


def test_process_error_query(mock_nlp_service):
    """Test error handling for query processing errors."""
    # Configure mock to raise exception
    mock_nlp_service.process_query.side_effect = Exception("Processing error")
    
    query_data = {
        "query": "invalid query that causes error",
        "session_id": "test-session-123",
        "format_level": "standard"
    }
    
    response = client.post("/api/nlp/process", json=query_data)
    
    # API should return an error status
    assert response.status_code in [400, 500]
    data = response.json()
    
    # Reset the side effect for other tests
    mock_nlp_service.process_query.side_effect = None 