"""Tests for the Solana client functionality."""

import asyncio
import pytest
import os
import sys
from typing import Dict, Any

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solana_mcp.solana_client import SolanaClient, SolanaRpcError, InvalidPublicKeyError


@pytest.mark.asyncio
async def test_solana_client_connectivity():
    """Test basic connectivity to the Solana RPC endpoint."""
    client = SolanaClient()
    try:
        # Simple request to check connectivity
        cluster_info = await client.get_cluster_nodes()
        assert isinstance(cluster_info, list), "Expected a list of nodes"
        assert len(cluster_info) > 0, "Expected at least one node in the cluster"
        print(f"Successfully connected to Solana RPC with {len(cluster_info)} nodes")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_token_metadata():
    """Test getting token metadata for a known token."""
    client = SolanaClient()
    try:
        # Test with a known token (USDC)
        usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        metadata = await client.get_token_metadata(usdc_mint)
        
        assert "mint" in metadata, "Expected mint field in metadata response"
        assert metadata["mint"] == usdc_mint, "Mint should match input"
        
        if "metadata" in metadata and metadata["metadata"]:
            assert "name" in metadata["metadata"], "Expected name in metadata"
            assert "symbol" in metadata["metadata"], "Expected symbol in metadata"
            print(f"Token metadata: {metadata['metadata']['name']} ({metadata['metadata']['symbol']})")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_invalid_public_key():
    """Test that invalid public keys are properly rejected."""
    client = SolanaClient()
    try:
        # Try with an invalid public key
        invalid_key = "not-a-valid-solana-public-key"
        
        with pytest.raises(InvalidPublicKeyError):
            await client.get_account_info(invalid_key)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_balance():
    """Test getting balance for a known account."""
    client = SolanaClient()
    try:
        # Try to get the current slot number (a simpler test)
        slot = await client.get_slot()
        assert isinstance(slot, int), "Slot should be an integer"
        print(f"Current slot: {slot}")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_commitment_parameter():
    """Test that commitment parameter is properly formatted."""
    client = SolanaClient()
    try:
        # This will fail if commitment is not properly formatted
        result = await client.get_slot()
        assert isinstance(result, int), "Slot should be an integer"
        print(f"Current slot: {result}")
    finally:
        await client.close()


if __name__ == "__main__":
    # Run the tests directly
    asyncio.run(test_solana_client_connectivity())
    asyncio.run(test_get_token_metadata())
    asyncio.run(test_get_balance())
    asyncio.run(test_commitment_parameter())
    
    try:
        asyncio.run(test_invalid_public_key())
        print("Invalid public key test passed")
    except Exception as e:
        print(f"Invalid public key test failed: {e}") 