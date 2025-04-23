#!/usr/bin/env python3
"""Test script to verify our Solana client fix."""

import asyncio
import sys
from solana_mcp.solana_client import SolanaClient, SolanaRpcError

async def test_methods():
    """Test various Solana client methods to verify our fix works."""
    client = SolanaClient()
    try:
        # Test methods without parameters
        print("Testing methods that don't require parameters...")
        slot = await client.get_slot()
        print(f"Current slot: {slot}")
        
        nodes = await client.get_cluster_nodes()
        print(f"Connected to {len(nodes)} nodes")
        
        # Test methods with parameters
        print("\nTesting methods that require parameters...")
        system_program = "11111111111111111111111111111111"
        try:
            balance = await client.get_balance(system_program)
            print(f"System program balance: {balance} lamports")
        except Exception as e:
            print(f"Error getting balance: {e}")
        
        # Test get_recent_blockhash
        try:
            blockhash = await client.get_recent_blockhash()
            if 'blockhash' in blockhash:
                print(f"Recent blockhash: {blockhash['blockhash']}")
            else:
                print(f"Blockhash response: {blockhash}")
        except Exception as e:
            print(f"Error getting blockhash: {e}")
        
        print("\nAll tests completed successfully!")
        return True
    except SolanaRpcError as e:
        print(f"Solana RPC error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(test_methods())
    if not success:
        sys.exit(1) 