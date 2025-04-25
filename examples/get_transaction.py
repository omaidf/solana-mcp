#!/usr/bin/env python3
"""
Example script for fetching and parsing a Solana transaction.

This script demonstrates how to use the Solana MCP library to:
1. Initialize the application with dependency injection
2. Fetch transaction details
3. Parse transaction information
4. Handle errors properly
"""

import asyncio
import logging
import sys
from pprint import pprint
from typing import Dict, Any, Optional

from solana_mcp import initialize_application
from solana_mcp.clients.transaction_client import TransactionClient
from solana_mcp.utils.error_handling import SolanaMCPError, TransactionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("transaction_example")

async def fetch_transaction(
    signature: str, 
    rpc_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch and parse a transaction from the Solana blockchain.
    
    Args:
        signature: Transaction signature
        rpc_url: Optional custom RPC URL
        
    Returns:
        Parsed transaction data
    """
    # Override RPC URL if provided
    config_overrides = {}
    if rpc_url:
        config_overrides["SOLANA_RPC_URL"] = rpc_url
    
    # Initialize the application with dependency injection
    service_provider = initialize_application(config_overrides)
    
    # Get the transaction client from the service provider
    tx_client = TransactionClient()
    
    # Fetch and parse the transaction
    try:
        tx = await tx_client.get_transaction(signature)
        
        # Return a dictionary representation
        return {
            "signature": tx.signature,
            "slot": tx.slot,
            "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
            "success": tx.success,
            "fee": tx.fee,
            "sol_transfers": [
                {
                    "from_account": str(transfer.from_account) if transfer.from_account else None,
                    "to_account": str(transfer.to_account) if transfer.to_account else None,
                    "amount": transfer.amount,
                    "amount_sol": transfer.amount / 1_000_000_000  # Convert lamports to SOL
                }
                for transfer in tx.sol_transfers
            ],
            "token_transfers": [
                {
                    "token_account": str(transfer.token_account) if transfer.token_account else None,
                    "owner": str(transfer.owner) if transfer.owner else None,
                    "mint": str(transfer.mint) if transfer.mint else None,
                    "amount": transfer.amount,
                    "decimals": transfer.decimals
                }
                for transfer in tx.token_transfers
            ]
        }
    except TransactionError as e:
        logger.error(f"Transaction error: {e}")
        logger.error(f"Details: {e.details}")
        raise
    except SolanaMCPError as e:
        logger.error(f"Solana MCP error: {e}")
        logger.error(f"Details: {e.details}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        # Clean up any async resources
        if hasattr(tx_client, 'close') and callable(tx_client.close):
            await tx_client.close()

def usage():
    """Print usage information."""
    print(f"Usage: {sys.argv[0]} <transaction_signature> [rpc_url]")
    print("\nExample:")
    print(f"  {sys.argv[0]} 5hrqyQAaS4KYmzvGjD6mTW5jFoF1MQz4nJeQYuCsQAQHUxzHXXQbHFwHbYg6XpLQJPwCpNFCT8NQ89w2q4DTvJZa")
    sys.exit(1)

async def main():
    """Script entry point."""
    # Check arguments
    if len(sys.argv) < 2:
        usage()
    
    # Get transaction signature
    signature = sys.argv[1]
    
    # Optional RPC URL
    rpc_url = None
    if len(sys.argv) >= 3:
        rpc_url = sys.argv[2]
    
    try:
        # Fetch transaction
        logger.info(f"Fetching transaction: {signature}")
        tx_data = await fetch_transaction(signature, rpc_url)
        
        # Print results
        print("\n=== Transaction Information ===")
        print(f"Signature: {tx_data['signature']}")
        print(f"Slot: {tx_data['slot']}")
        print(f"Timestamp: {tx_data['timestamp']}")
        print(f"Success: {tx_data['success']}")
        print(f"Fee: {tx_data['fee']} lamports")
        
        # Print SOL transfers
        if tx_data['sol_transfers']:
            print("\n=== SOL Transfers ===")
            for transfer in tx_data['sol_transfers']:
                print(f"  {transfer['from_account']} -> {transfer['to_account']}: {transfer['amount_sol']:.9f} SOL")
        
        # Print token transfers
        if tx_data['token_transfers']:
            print("\n=== Token Transfers ===")
            for transfer in tx_data['token_transfers']:
                amount = transfer['amount'] / (10 ** transfer['decimals'])
                print(f"  {transfer['owner']} (account: {transfer['token_account']})")
                print(f"  Token: {transfer['mint']}")
                print(f"  Amount: {amount}")
                print("")
        
        print("\n=== Full Transaction Data ===")
        pprint(tx_data)
        
    except SolanaMCPError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 