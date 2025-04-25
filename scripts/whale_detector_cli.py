#!/usr/bin/env python3
"""
Whale detector CLI for Solana tokens.
Uses the enhanced whale detection algorithm to identify wallets with significant holdings.
"""

import argparse
import asyncio
import json
import sys
import os

# Add the parent directory to the path so we can import the Solana MCP package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_mcp.solana_client import SolanaClient, SolanaRpcError
from solana_mcp.token_analyzer import TokenAnalyzer
from solana_mcp.config import SolanaConfig

DEFAULT_RPC_URL = "https://mainnet.helius-rpc.com/?api-key=4ffc1228-f093-4974-ad2d-3edd8e5f7c03"


async def detect_whales(token_address, rpc_url, threshold_usd=50000.0, max_accounts=100, json_output=False):
    """Detect whale wallets for a token mint."""
    print(f"Finding whale wallets for token: {token_address}", file=sys.stderr)
    print(f"Using threshold: ${threshold_usd:,.2f} USD", file=sys.stderr)
    
    try:
        # Initialize client and analyzer
        config = SolanaConfig(rpc_url=rpc_url)
        client = SolanaClient(config)
        analyzer = TokenAnalyzer(client)
        
        try:
            # Get token metadata for display
            token_metadata = await analyzer.get_token_metadata(token_address)
            token_name = token_metadata.get("name", "Unknown")
            token_symbol = token_metadata.get("symbol", "UNKNOWN")
            
            print(f"Analyzing token: {token_name} ({token_symbol})", file=sys.stderr)
        except Exception as e:
            print(f"Error fetching metadata for {token_address}: {str(e)}", file=sys.stderr)
            token_name = "Unknown"
            token_symbol = "UNKNOWN"
        
        # Get whale data with the max_accounts parameter
        whale_data = await analyzer.get_whale_holders(
            token_address, 
            threshold_usd=threshold_usd,
            max_accounts=max_accounts
        )
        
        if json_output:
            # Return raw JSON
            print(json.dumps(whale_data, indent=2))
        else:
            # Print human-readable output
            print("\n" + "=" * 80)
            print(f"WHALE ANALYSIS FOR {token_name} ({token_symbol})")
            print("=" * 80)
            print(f"Token address: {token_address}")
            print(f"Total holders analyzed: {whale_data.get('total_holders_analyzed', 0)}")
            print(f"Total holders: {whale_data.get('total_holders', 0)}")
            print(f"Whale count: {whale_data.get('whale_count', 0)}")
            print(f"Whale percentage: {whale_data.get('whale_percentage', 0):.2f}%")
            print(f"Whale holdings percentage: {whale_data.get('whale_holdings_percentage', 0):.2f}%")
            print(f"Whale holdings USD total: ${whale_data.get('whale_holdings_usd_total', 0):,.2f}")
            
            # Print warnings if any
            if "warnings" in whale_data and whale_data["warnings"]:
                print("\nWARNINGS:")
                for warning in whale_data["warnings"]:
                    print(f"- {warning}")
            print("=" * 80)
            
            # Print whale details
            if whale_data.get("whale_holders"):
                print("\nWHALE DETAILS:")
                print("-" * 80)
                for i, whale in enumerate(whale_data.get("whale_holders", []), 1):
                    print(f"{i}. Wallet: {whale.get('address')}")
                    print(f"   Token Balance: {whale.get('token_balance'):,.4f} {token_symbol} " 
                         f"(${whale.get('usd_value', 0):,.2f} USD, {whale.get('percentage_of_supply', 0):.2f}% of supply)")
                    print(f"   Total Wallet Value: ${whale.get('total_wallet_value_usd', 0):,.2f} USD")
                    print(f"   Token Count: {whale.get('token_count', 0)}")
                    
                    # Print top 3 tokens by value if available
                    if 'top_tokens' in whale and whale['top_tokens']:
                        print(f"   Top Tokens: ", end="")
                        for j, token in enumerate(whale['top_tokens'][:3]):
                            if j > 0:
                                print(", ", end="")
                            print(f"{token['token']}: ${token['value_usd']:,.2f}", end="")
                        print()
                    
                    print()
        
        # Clean up
        await client.close()
        return whale_data
    
    except SolanaRpcError as e:
        error_message = f"Solana RPC error: {str(e)}"
        print(f"Error: {error_message}", file=sys.stderr)
        if json_output:
            print(json.dumps({
                "success": False,
                "error": error_message,
                "token_mint": token_address
            }, indent=2))
        return {"success": False, "error": error_message, "token_mint": token_address}
    
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        print(f"Error: {error_message}", file=sys.stderr)
        if json_output:
            print(json.dumps({
                "success": False,
                "error": error_message,
                "token_mint": token_address
            }, indent=2))
        return {"success": False, "error": error_message, "token_mint": token_address}


def main():
    parser = argparse.ArgumentParser(description='Find Solana whale wallets for a specific token')
    parser.add_argument('token', help='Solana token address')
    parser.add_argument('--rpc', default=DEFAULT_RPC_URL, help='Solana RPC URL')
    parser.add_argument('--threshold', type=float, default=50000.0, 
                        help='Minimum USD value to consider a holder a whale (default: $50K)')
    parser.add_argument('--max-accounts', type=int, default=100,
                        help='Maximum number of token accounts to analyze (default: 100)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(detect_whales(args.token, args.rpc, args.threshold, args.max_accounts, args.json))


if __name__ == "__main__":
    main() 