#!/usr/bin/env python3
"""
Simple test script for Solana token whale detection.

Usage:
    python -m solana_mcp.scripts.simple_whale_test <token_address>
"""

import asyncio
import json
import sys
from typing import Dict, Any, List
from decimal import Decimal

from solana_mcp.solana_client import get_solana_client


async def get_token_metadata(client, token_mint: str) -> Dict[str, Any]:
    """Get token metadata."""
    try:
        return await client.get_token_metadata(token_mint)
    except Exception as e:
        print(f"Error getting token metadata: {str(e)}")
        return {"name": "Unknown", "symbol": token_mint[:6]}


async def get_token_supply(client, token_mint: str) -> Dict[str, Any]:
    """Get token supply."""
    try:
        return await client.get_token_supply(token_mint)
    except Exception as e:
        print(f"Error getting token supply: {str(e)}")
        return {"supply": {"amount": "0", "decimals": 9}}


async def get_token_price(client, token_mint: str) -> float:
    """Get token price."""
    try:
        price_data = await client.get_market_price(token_mint)
        return float(price_data.get("price_data", {}).get("price_usd", 0.01))
    except Exception as e:
        print(f"Error getting token price: {str(e)}")
        return 0.01  # Default fallback price


async def get_token_accounts(client, token_mint: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get top token accounts."""
    try:
        largest_accounts = await client.get_token_largest_accounts(token_mint)
        
        accounts = []
        for i, account in enumerate(largest_accounts):
            if i >= limit:
                break
                
            # Get account info
            account_info = await client.get_account_info(account["address"], encoding="jsonParsed")
            
            if "parsed" in account_info.get("data", {}):
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data:
                    info = parsed_data["info"]
                    accounts.append({
                        "owner": info.get("owner"),
                        "address": account["address"],
                        "amount": info.get("tokenAmount", {}).get("amount", "0"),
                        "decimals": info.get("tokenAmount", {}).get("decimals", 0),
                        "ui_amount": info.get("tokenAmount", {}).get("uiAmount", 0)
                    })
        
        return accounts
    except Exception as e:
        print(f"Error getting token accounts: {str(e)}")
        return []


async def analyze_whales(token_mint: str) -> Dict[str, Any]:
    """Analyze whale wallets for a given token.
    
    Args:
        token_mint: Token mint address
        
    Returns:
        Analysis data
    """
    try:
        async with get_solana_client() as client:
            # Get token metadata
            metadata = await get_token_metadata(client, token_mint)
            
            # Get token supply
            supply_data = await get_token_supply(client, token_mint)
            
            # Get token price
            price = await get_token_price(client, token_mint)
            
            # Get top token accounts
            accounts = await get_token_accounts(client, token_mint)
            
            # Calculate total supply
            decimals = supply_data.get("supply", {}).get("decimals", 0)
            total_supply = float(supply_data.get("supply", {}).get("amount", "0")) / (10 ** decimals)
            
            # Identify whales (accounts holding > 1% of supply or worth > $50,000)
            WHALE_THRESHOLD_PERCENTAGE = 1.0  # 1% of total supply
            WHALE_THRESHOLD_USD = 50000.0     # $50,000 USD
            
            whales = []
            for account in accounts:
                amount = float(account.get("amount", "0"))
                ui_amount = float(account.get("ui_amount", 0))
                
                # Calculate percentage of total supply
                percentage = (ui_amount / total_supply) * 100 if total_supply > 0 else 0
                
                # Calculate USD value
                usd_value = ui_amount * price
                
                # Check if this is a whale
                is_whale = percentage >= WHALE_THRESHOLD_PERCENTAGE or usd_value >= WHALE_THRESHOLD_USD
                
                if is_whale:
                    whales.append({
                        "address": account.get("owner", ""),
                        "token_amount": ui_amount,
                        "percentage": percentage,
                        "usd_value": usd_value
                    })
            
            # Sort whales by percentage
            whales.sort(key=lambda x: x["percentage"], reverse=True)
            
            # Prepare result
            result = {
                "token_address": token_mint,
                "token_name": metadata.get("name", "Unknown"),
                "token_symbol": metadata.get("symbol", "UNKNOWN"),
                "token_price_usd": price,
                "total_supply": total_supply,
                "total_accounts_analyzed": len(accounts),
                "whale_count": len(whales),
                "whale_threshold_percentage": WHALE_THRESHOLD_PERCENTAGE,
                "whale_threshold_usd": WHALE_THRESHOLD_USD,
                "whales": whales
            }
            
            return result
    except Exception as e:
        print(f"Error analyzing whales: {str(e)}")
        return {"error": str(e)}


def display_whale_result(result: Dict[str, Any]) -> None:
    """Display whale analysis results in a readable format."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("=" * 80)
    print(f"WHALE ANALYSIS FOR {result.get('token_name', '')} ({result.get('token_symbol', '')})")
    print("=" * 80)
    print(f"Token address: {result.get('token_address', '')}")
    print(f"Token price: ${result.get('token_price_usd', 0):,.8f}")
    print(f"Total supply: {result.get('total_supply', 0):,.2f}")
    print(f"Accounts analyzed: {result.get('total_accounts_analyzed', 0)}")
    print(f"Number of whales detected: {result.get('whale_count', 0)}")
    print(f"Whale thresholds: {result.get('whale_threshold_percentage', 0)}% of supply or ${result.get('whale_threshold_usd', 0):,.2f}")
    
    # Display whale details
    if "whales" in result and result["whales"]:
        print("\nWHALE DETAILS:")
        print("-" * 80)
        
        for i, whale in enumerate(result["whales"], 1):
            print(f"{i}. Address: {whale.get('address', '')}")
            print(f"   Amount: {whale.get('token_amount', 0):,.2f} {result.get('token_symbol', '')}")
            print(f"   Percentage: {whale.get('percentage', 0):,.2f}% of supply")
            print(f"   USD Value: ${whale.get('usd_value', 0):,.2f}")
            print()


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <token_address>")
        sys.exit(1)
    
    token_mint = sys.argv[1].strip()
    print(f"\nAnalyzing whale wallets for token: {token_mint}\n")
    
    try:
        # Run the analysis
        result = await analyze_whales(token_mint)
        
        # Display results
        display_whale_result(result)
        
        # Option to save as JSON
        save = input("\nSave results to JSON file? (y/N): ").strip().lower()
        if save == 'y':
            filename = f"whale_analysis_{token_mint[:8]}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Data saved to {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 