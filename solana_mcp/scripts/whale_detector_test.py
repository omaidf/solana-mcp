#!/usr/bin/env python3
"""
Test script for the Solana whale detector.

Usage:
    python -m solana_mcp.scripts.whale_detector_test <token_address>
"""

import asyncio
import json
import sys
from typing import Dict, Any

from solana_mcp.solana_client import get_solana_client
from solana_mcp.services.whale_detector import detect_whale_wallets

async def test_whale_detector(token_address: str) -> Dict[str, Any]:
    """Test the whale detector for a given token.
    
    Args:
        token_address: The Solana token mint address
        
    Returns:
        Whale detection result
    """
    print(f"\nAnalyzing whale wallets for token: {token_address}\n")
    
    try:
        async with get_solana_client() as solana_client:
            # Run the whale detection
            result = await detect_whale_wallets(token_address, solana_client)
            return result
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

def display_whale_result(result: Dict[str, Any]) -> None:
    """Display whale detection results in a readable format.
    
    Args:
        result: Whale detection result
    """
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("=" * 80)
    print(f"WHALE ANALYSIS FOR {result.get('token_name', '')} ({result.get('token_symbol', '')})")
    print("=" * 80)
    print(f"Token address: {result.get('token_address', '')}")
    print(f"Token price: ${result.get('token_price_usd', 0):,.8f}")
    print(f"Whale threshold: ${result.get('whale_threshold_usd', 0):,.2f}")
    print(f"Total holders analyzed: {result.get('total_holders_analyzed', 0)}")
    print(f"Total holders: {result.get('total_holders', 0)}")
    print(f"Number of whales detected: {result.get('whale_count', 0)}")
    
    # Display warnings if any
    if "warnings" in result and result["warnings"]:
        print("\nWARNINGS:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    
    # Display whale details
    if "whales" in result and result["whales"]:
        print("\nWHALE DETAILS:")
        print("-" * 80)
        
        for i, whale in enumerate(result["whales"], 1):
            print(f"{i}. Wallet: {whale.get('wallet', '')}")
            print(f"   Token Balance: {whale.get('target_token_amount', 0):,.4f} " 
                 f"(${whale.get('target_token_value_usd', 0):,.2f}, "
                 f"{whale.get('target_token_supply_percentage', 0):.2f}% of supply)")
            print(f"   Total Wallet Value: ${whale.get('total_value_usd', 0):,.2f}")
            print(f"   Token Count: {whale.get('token_count', 0)}")
            
            # Display top tokens
            if "top_tokens" in whale and whale["top_tokens"]:
                print("   Top Tokens:")
                for token in whale["top_tokens"][:3]:  # Show top 3 tokens
                    print(f"     - {token.get('token', '')}: ${token.get('value_usd', 0):,.2f}")
            
            print()

async def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <token_address>")
        sys.exit(1)
    
    token_address = sys.argv[1].strip()
    
    try:
        # Run the test
        result = await test_whale_detector(token_address)
        
        # Display results
        display_whale_result(result)
        
        # Option to save as JSON
        save = input("\nSave results to JSON file? (y/N): ").strip().lower()
        if save == 'y':
            filename = f"whale_detection_{token_address[:8]}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Data saved to {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 