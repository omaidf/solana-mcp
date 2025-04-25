#!/usr/bin/env python3
"""
Token whale analyzer - Identifies whale wallets based on token holdings.

Usage:
    python -m solana_mcp.scripts.token_whale_analyzer <token_address>
"""

import asyncio
import json
import sys
from typing import Dict, Any, List

from solana_mcp.solana_client import get_solana_client
from solana_mcp.clients.token_client import TokenClient


async def analyze_whales(token_address: str) -> Dict[str, Any]:
    """Analyze token holdings to identify whales.
    
    Args:
        token_address: Solana token mint address
        
    Returns:
        Whale analysis results
    """
    print(f"\nAnalyzing whale wallets for token: {token_address}\n")
    
    try:
        async with get_solana_client() as solana_client:
            # Create token client
            token_client = TokenClient(solana_client.config)
            
            # Get comprehensive token data with holders
            token_data = await token_client.get_all_token_data(token_address, holder_limit=50)
            
            # Extract key information
            token_name = token_data.get("name", "Unknown")
            token_symbol = token_data.get("symbol", "UNKNOWN")
            token_price = token_data.get("price", {}).get("current_price_usd", 0.01)
            total_supply = float(token_data.get("supply", {}).get("ui_amount", 0))
            
            # Holders and total holders
            holders = token_data.get("holders", {}).get("top_holders", [])
            total_holders = token_data.get("holders", {}).get("total_holders", 0)
            
            # Identify whales (wallets with > 1% of total supply or value > $50,000)
            WHALE_THRESHOLD_PERCENTAGE = 1.0  # 1% of supply
            WHALE_THRESHOLD_USD = 50000.0     # $50,000 USD
            
            whales = []
            for holder in holders:
                try:
                    # Get holder details
                    address = holder.get("address", "")
                    amount = float(holder.get("amount", "0")) / (10 ** token_data.get("decimals", 0))
                    percentage = float(holder.get("percentage", 0))
                    
                    # Calculate USD value
                    usd_value = amount * token_price
                    
                    # Check if this is a whale
                    is_whale = percentage >= WHALE_THRESHOLD_PERCENTAGE or usd_value >= WHALE_THRESHOLD_USD
                    
                    if is_whale:
                        whales.append({
                            "address": address,
                            "amount": amount,
                            "percentage": percentage,
                            "usd_value": usd_value
                        })
                except Exception as e:
                    print(f"Error processing holder {holder.get('address', '')}: {str(e)}")
            
            # Sort whales by percentage
            whales.sort(key=lambda x: x["percentage"], reverse=True)
            
            # Prepare analysis result
            result = {
                "token_address": token_address,
                "token_name": token_name,
                "token_symbol": token_symbol,
                "token_price_usd": token_price,
                "total_supply": total_supply,
                "total_holders": total_holders,
                "holders_analyzed": len(holders),
                "whale_count": len(whales),
                "whale_threshold_percentage": WHALE_THRESHOLD_PERCENTAGE,
                "whale_threshold_usd": WHALE_THRESHOLD_USD,
                "whales": whales
            }
            
            return result
    except Exception as e:
        print(f"Error in whale analysis: {str(e)}")
        return {"error": str(e)}


def display_whale_analysis(result: Dict[str, Any]) -> None:
    """Display whale analysis in a human-readable format.
    
    Args:
        result: Whale analysis data
    """
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("=" * 80)
    print(f"WHALE ANALYSIS FOR {result.get('token_name', '')} ({result.get('token_symbol', '')})")
    print("=" * 80)
    print(f"Token address: {result.get('token_address', '')}")
    print(f"Token price: ${result.get('token_price_usd', 0):,.8f}")
    print(f"Total supply: {result.get('total_supply', 0):,.2f}")
    print(f"Total holders: {result.get('total_holders', 0):,}")
    print(f"Holders analyzed: {result.get('holders_analyzed', 0)}")
    print(f"Number of whales detected: {result.get('whale_count', 0)}")
    print(f"Whale thresholds: {result.get('whale_threshold_percentage', 0)}% of supply or ${result.get('whale_threshold_usd', 0):,.2f}")
    
    # Display whale details
    if "whales" in result and result["whales"]:
        print("\nWHALE DETAILS:")
        print("-" * 80)
        
        for i, whale in enumerate(result["whales"], 1):
            print(f"{i}. Address: {whale.get('address', '')}")
            print(f"   Amount: {whale.get('amount', 0):,.2f} {result.get('token_symbol', '')}")
            print(f"   Percentage: {whale.get('percentage', 0):,.2f}% of supply")
            print(f"   USD Value: ${whale.get('usd_value', 0):,.2f}")
            print()
    else:
        print("\nNo whales detected.")


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <token_address>")
        sys.exit(1)
    
    token_address = sys.argv[1].strip()
    
    try:
        # Run the analysis
        result = await analyze_whales(token_address)
        
        # Display results
        display_whale_analysis(result)
        
        # Option to save as JSON
        save = input("\nSave results to JSON file? (y/N): ").strip().lower()
        if save == 'y':
            filename = f"whale_analysis_{token_address[:8]}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Data saved to {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 