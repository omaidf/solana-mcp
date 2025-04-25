#!/usr/bin/env python3
"""
CLI tool to fetch comprehensive Solana token information.

Usage:
    python -m solana_mcp.scripts.token_info_cli <token_address>
"""

import sys
import json
import asyncio
from typing import Dict, Any

from solana_mcp.solana_client import get_solana_client
from solana_mcp.clients.token_client import TokenClient


async def get_token_info(token_address: str) -> Dict[str, Any]:
    """Get comprehensive token information.
    
    Args:
        token_address: Solana token mint address
        
    Returns:
        Dictionary with token information
    """
    async with get_solana_client() as solana_client:
        # Create token client
        token_client = TokenClient(solana_client.config)
        
        # Fetch all token data
        return await token_client.get_all_token_data(token_address)


def display_token_info(data: Dict[str, Any]) -> None:
    """Display token information in a human-readable format.
    
    Args:
        data: Token data dictionary
    """
    if 'errors' in data:
        print("\nWarnings/Errors:")
        for error in data['errors']:
            for key, value in error.items():
                print(f"  {key}: {value}")
        print()
    
    print(f"Token Name: {data['name']}")
    print(f"Token Symbol: {data['symbol']}")
    print(f"Decimals: {data['decimals']}")
    print(f"Mint Address: {data['token_mint']}\n")
    
    print("Supply Info:")
    print(f"  Amount: {data['supply']['amount']}")
    print(f"  UI Amount: {data['supply']['ui_amount_string']}\n")
    
    print("Price Info:")
    print(f"  Price: ${data['price']['current_price_usd']:.8f}")
    print(f"  24h Change: {data['price']['price_change_24h']:.2f}%")
    print(f"  Last Updated: {data['price']['last_updated']}")
    print(f"  Source: {data['price']['source']}\n")
    
    print("Holder Info:")
    print(f"  Total Holders: {data['holders']['total_holders']}")
    
    print("\nTop Holders:")
    for i, holder in enumerate(data['holders']['top_holders'][:5], 1):
        print(f"  {i}. Address: {holder['address']}")
        print(f"     Amount: {holder['amount']}")
        if 'percentage' in holder:
            print(f"     Percentage: {holder['percentage']:.2f}%")
        print()
    
    print(f"Last Updated: {data['last_updated']}")


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <token_address>")
        sys.exit(1)
    
    token_address = sys.argv[1].strip()
    print(f"\nFetching token data for {token_address}...\n")
    
    try:
        data = await get_token_info(token_address)
        
        # Pretty display
        display_token_info(data)
        
        # Option to save as JSON
        save = input("\nSave results to JSON file? (y/N): ").strip().lower()
        if save == 'y':
            filename = f"token_info_{token_address[:8]}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Data saved to {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 