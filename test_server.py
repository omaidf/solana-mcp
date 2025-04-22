#!/usr/bin/env python
"""Test script for the Solana MCP server."""

import asyncio
import json
import sys
from solana_mcp.solana_client import SolanaClient, get_solana_client


async def test_client():
    """Test the Solana client implementation."""
    print("Testing Solana client...")
    
    # Test account public key - Solana Program Library (SPL) token program
    test_account = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    
    async with get_solana_client() as client:
        # Test connection
        try:
            # Get account info
            print(f"\nFetching account info for {test_account}...")
            account_info = await client.get_account_info(test_account)
            print(f"✅ Account info retrieved: {json.dumps(account_info, indent=2)[:200]}...")
            
            # Get balance
            print(f"\nFetching balance for {test_account}...")
            balance = await client.get_balance(test_account)
            print(f"✅ Balance retrieved: {balance} lamports")
            
            # Get recent blockhash
            print(f"\nFetching recent blockhash...")
            blockhash = await client.get_recent_blockhash()
            print(f"✅ Recent blockhash retrieved: {json.dumps(blockhash, indent=2)}")
            
        except Exception as e:
            print(f"❌ Error testing client: {str(e)}")
            return False
    
    print("\n✅ All Solana client tests passed!")
    return True


def check_installation():
    """Check if all required packages are installed."""
    try:
        import mcp
        import starlette
        import uvicorn
        import click
        import httpx
        import anyio
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install all required packages with: pip install -e .")
        return False


async def main():
    """Run all tests."""
    print("=== Solana MCP Server Test ===\n")
    
    # Check installation
    if not check_installation():
        sys.exit(1)
    
    # Test Solana client
    if not await test_client():
        sys.exit(1)
    
    print("\n=== All tests passed! ===")
    print("\nYou can now run the server with:")
    print("  python -m solana_mcp")
    print("  # or")
    print("  solana-mcp --transport sse --port 8000")


if __name__ == "__main__":
    asyncio.run(main()) 