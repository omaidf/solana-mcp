"""Unit tests for TransactionService.

This module tests the transaction service functionality.
"""

import pytest
from unittest.mock import patch, AsyncMock

from solana_mcp.services.transaction_service import TransactionService


@pytest.mark.asyncio
async def test_get_transaction(transaction_service, mock_solana_client, sample_transaction_data):
    """Test retrieving a transaction by signature."""
    # Setup
    signature = "test_signature"
    mock_solana_client.get_transaction.return_value = sample_transaction_data
    
    # Execute
    result = await transaction_service.get_transaction(signature)
    
    # Verify
    assert result is not None
    assert mock_solana_client.get_transaction.called
    mock_solana_client.get_transaction.assert_called_with(signature)
    assert result == sample_transaction_data


@pytest.mark.asyncio
async def test_get_transaction_from_cache(transaction_service, mock_solana_client, mock_cache_service, sample_transaction_data):
    """Test retrieving a transaction from cache."""
    # Setup
    signature = "cached_signature"
    mock_cache_service.get.return_value = sample_transaction_data
    
    # Execute
    result = await transaction_service.get_transaction(signature)
    
    # Verify
    assert result is not None
    assert mock_cache_service.get.called
    mock_cache_service.get.assert_called_with(f"tx:{signature}")
    assert not mock_solana_client.get_transaction.called
    assert result == sample_transaction_data


@pytest.mark.asyncio
async def test_get_transactions_for_address(transaction_service, mock_solana_client):
    """Test getting transaction history for an address."""
    # Setup
    address = "test_address"
    limit = 10
    mock_solana_client.get_signatures_for_address.return_value = [
        {"signature": "sig1", "slot": 100, "blockTime": 1000, "err": None},
        {"signature": "sig2", "slot": 101, "blockTime": 1001, "err": None}
    ]
    
    # Execute
    result = await transaction_service.get_transactions_for_address(
        address, limit, parsed_details=False
    )
    
    # Verify
    assert result is not None
    assert mock_solana_client.get_signatures_for_address.called
    mock_solana_client.get_signatures_for_address.assert_called_with(
        address, before=None, until=None, limit=limit
    )
    assert "address" in result
    assert "transactions" in result
    assert "count" in result
    assert result["address"] == address
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_get_transactions_for_address_with_details(transaction_service, mock_solana_client, mock_cache_service):
    """Test getting transaction history with detailed parsing."""
    # Setup
    address = "test_address"
    limit = 10
    signatures = [
        {"signature": "sig1", "slot": 100, "blockTime": 1000, "err": None},
        {"signature": "sig2", "slot": 101, "blockTime": 1001, "err": None}
    ]
    tx_details = {
        "slot": 100,
        "blockTime": 1000,
        "meta": {"fee": 5000},
        "transaction": {"signatures": ["sig1"]}
    }
    
    mock_solana_client.get_signatures_for_address.return_value = signatures
    mock_solana_client.get_transaction.return_value = tx_details
    
    # Execute
    result = await transaction_service.get_transactions_for_address(
        address, limit, parsed_details=True
    )
    
    # Verify
    assert result is not None
    assert mock_solana_client.get_signatures_for_address.called
    assert mock_solana_client.get_transaction.called
    assert "address" in result
    assert "transactions" in result
    assert "count" in result
    assert result["address"] == address
    assert result["count"] == 2
    
    # Verify detailed transactions
    transactions = result["transactions"]
    assert len(transactions) == 2
    assert "details" in transactions[0]
    assert transactions[0]["details"] == tx_details


@pytest.mark.asyncio
async def test_get_recent_transactions(transaction_service, mock_solana_client):
    """Test getting recent transactions."""
    # Setup
    limit = 5
    mock_solana_client.get_recent_transaction_signatures.return_value = ["sig1", "sig2", "sig3"]
    tx_details = {
        "slot": 100,
        "blockTime": 1000,
        "meta": {"fee": 5000},
        "transaction": {"signatures": ["sig1"]}
    }
    mock_solana_client.get_transaction.return_value = tx_details
    
    # Execute
    result = await transaction_service.get_recent_transactions(limit)
    
    # Verify
    assert result is not None
    assert mock_solana_client.get_recent_transaction_signatures.called
    assert mock_solana_client.get_transaction.called
    assert len(result) == 3
    assert result[0] == tx_details


@pytest.mark.asyncio
async def test_parse_transaction(transaction_service, sample_transaction_data):
    """Test parsing a transaction into a more user-friendly format."""
    # Execute
    result = await transaction_service.parse_transaction(sample_transaction_data)
    
    # Verify
    assert result is not None
    assert "signature" in result
    assert "slot" in result
    assert "block_time" in result
    assert "fee" in result
    assert "instructions" in result
    assert result["signature"] == "test_signature"
    assert result["slot"] == 12345
    assert len(result["instructions"]) == 1
    

@pytest.mark.asyncio
async def test_parse_transaction_empty_data(transaction_service):
    """Test parsing an empty transaction."""
    # Execute
    result = await transaction_service.parse_transaction(None)
    
    # Verify
    assert result == {}


@pytest.mark.asyncio
async def test_parse_transaction_error_handling(transaction_service):
    """Test error handling in transaction parsing."""
    # Setup - create invalid transaction data that will cause an exception
    invalid_tx = {"transaction": {"message": "not_a_dict"}}
    
    # Execute
    result = await transaction_service.parse_transaction(invalid_tx)
    
    # Verify
    assert "error" in result
    assert result["error"] == "Error parsing transaction data"


def test_get_program_name(transaction_service):
    """Test mapping program IDs to human-readable names."""
    # Test known program IDs
    assert transaction_service._get_program_name("11111111111111111111111111111111") == "System Program"
    assert transaction_service._get_program_name("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA") == "Token Program"
    
    # Test unknown program ID
    assert transaction_service._get_program_name("unknown_program_id") == "Unknown Program" 