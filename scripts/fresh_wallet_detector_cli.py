#!/usr/bin/env python3
"""
Fresh wallet detector CLI for Solana tokens.
Uses the enhanced fresh wallet detection algorithm to identify new wallets created for a specific token.
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


async def detect_fresh_wallets(token_address, rpc_url, max_accounts=100, json_output=False):
    """Detect fresh wallets for a token mint."""
    print(f"Finding fresh wallets for token: {token_address}", file=sys.stderr)
    
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
        
        # Get fresh wallet data with the max_accounts parameter
        fresh_wallet_data = await analyzer.get_fresh_wallets(
            token_address,
            max_accounts=max_accounts
        )
        
        if json_output:
            # Return raw JSON
            print(json.dumps(fresh_wallet_data, indent=2))
        else:
            # Print human-readable output
            print("\n" + "=" * 80)
            print(f"FRESH WALLET ANALYSIS FOR {token_name} ({token_symbol})")
            print("=" * 80)
            print(f"Token address: {token_address}")
            print(f"Total holders analyzed: {fresh_wallet_data.get('total_holders_analyzed', 0)}")
            print(f"Fresh wallet count: {fresh_wallet_data.get('fresh_wallet_count', 0)}")
            print(f"Fresh wallet percentage: {fresh_wallet_data.get('fresh_wallet_percentage', 0):.2f}%")
            print(f"Fresh wallet holdings percentage: {fresh_wallet_data.get('fresh_wallet_holdings_percentage', 0):.2f}%")
            print(f"Token price USD: ${fresh_wallet_data.get('token_price_usd', 0):.6f}")
            
            # Print warnings if any
            if "warnings" in fresh_wallet_data and fresh_wallet_data["warnings"]:
                print("\nWARNINGS:")
                for warning in fresh_wallet_data["warnings"]:
                    print(f"- {warning}")
            print("=" * 80)
            
            # Print fresh wallet details
            if fresh_wallet_data.get("fresh_wallets"):
                print("\nFRESH WALLET DETAILS:")
                print("-" * 80)
                for i, wallet in enumerate(fresh_wallet_data.get("fresh_wallets", []), 1):
                    age_str = f"{wallet.get('wallet_age_days')} days old" if wallet.get('wallet_age_days') is not None else "Unknown age"
                    
                    print(f"{i}. Wallet: {wallet.get('wallet_address')}")
                    print(f"   Token Balance: {wallet.get('token_balance'):,.4f} {token_symbol} "
                         f"(${wallet.get('token_value_usd', 0):,.2f} USD, {wallet.get('percentage_of_supply', 0):.4f}% of supply)")
                    print(f"   Wallet Age: {age_str}")
                    print(f"   Token TX Ratio: {wallet.get('token_tx_ratio', 0):.2f}")
                    print(f"   Freshness Score: {wallet.get('freshness_score', 0):.2f}")
                    print()
        
        # Clean up
        await client.close()
        return fresh_wallet_data
    
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
    parser = argparse.ArgumentParser(description='Find fresh wallets for a specific Solana token')
    parser.add_argument('token', help='Solana token address')
    parser.add_argument('--rpc', default=DEFAULT_RPC_URL, help='Solana RPC URL')
    parser.add_argument('--max-accounts', type=int, default=100,
                       help='Maximum number of token accounts to analyze (default: 100)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(detect_fresh_wallets(args.token, args.rpc, args.max_accounts, args.json))


if __name__ == "__main__":
    main() 