#!/usr/bin/env python3
"""
Test script for token metadata extraction debugging.
"""

import sys
import os
import asyncio
import base64
import base58
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_mcp.solana_client import SolanaClient
from solana_mcp.config import SolanaConfig


async def test_token_metadata(token_mint: str, rpc_url: Optional[str] = None, verbose: bool = False,
                               output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Test token metadata extraction with detailed diagnostic information.
    
    Args:
        token_mint: The token mint address to analyze
        rpc_url: Optional custom RPC URL
        verbose: Whether to print verbose output
        output_file: Optional path to save results as JSON
        
    Returns:
        Dictionary with test results
    """
    # Initialize results dictionary
    results = {
        "token_mint": token_mint,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Setup client with custom RPC if provided
    rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
    config = SolanaConfig(rpc_url=rpc_url)
    client = SolanaClient(config)
    
    print(f"Testing token metadata extraction for: {token_mint}")
    print(f"Using RPC URL: {config.rpc_url}")
    print("=" * 80)
    
    # Test 1: Basic metadata extraction
    print("\n🔍 Test 1: Basic Metadata Extraction")
    try:
        metadata = await client.get_token_metadata(token_mint)
        results["tests"]["basic_metadata"] = {
            "success": True,
            "result": metadata
        }
        print(f"✅ Name: {metadata['metadata']['name']}")
        print(f"✅ Symbol: {metadata['metadata']['symbol']}")
        print(f"✅ URI: {metadata['metadata']['uri']}")
        
        if verbose:
            print(f"Raw metadata: {json.dumps(metadata, indent=2)}")
    except Exception as e:
        results["tests"]["basic_metadata"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Error: {str(e)}")
    
    # Test 2: Token Supply Information
    print("\n🔍 Test 2: Token Supply Information")
    try:
        supply_info = await client.get_token_supply(token_mint)
        results["tests"]["token_supply"] = {
            "success": True,
            "result": supply_info
        }
        print(f"✅ Supply: {supply_info.get('result', {}).get('value', {}).get('amount', 'Unknown')}")
        print(f"✅ Decimals: {supply_info.get('result', {}).get('value', {}).get('decimals', 'Unknown')}")
        
        if verbose:
            print(f"Raw supply info: {json.dumps(supply_info, indent=2)}")
    except Exception as e:
        results["tests"]["token_supply"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Error: {str(e)}")
    
    # Test 3: Mint Account Info
    print("\n🔍 Test 3: Mint Account Info")
    try:
        account_info = await client.get_account_info(token_mint)
        mint_data_exists = account_info and "result" in account_info and account_info["result"] and account_info["result"]["value"]
        results["tests"]["mint_account_info"] = {
            "success": mint_data_exists,
            "result": {
                "exists": mint_data_exists,
                "owner": account_info.get("result", {}).get("value", {}).get("owner", "Unknown") if mint_data_exists else "Unknown"
            }
        }
        if mint_data_exists:
            print(f"✅ Account exists: Yes")
            print(f"✅ Owner program: {account_info['result']['value']['owner']}")
            
            if verbose:
                print(f"Raw account info: {json.dumps(account_info, indent=2)}")
        else:
            print(f"❌ Account does not exist or is not a valid token mint")
    except Exception as e:
        results["tests"]["mint_account_info"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Error: {str(e)}")
    
    # Test 4: Metaplex PDA Derivation
    print("\n🔍 Test 4: Metaplex PDA Derivation")
    try:
        import hashlib
        
        # Convert string mint address to bytes
        mint_bytes = base58.b58decode(token_mint)
        
        # Calculate the PDA seeds
        metadata_seed = "metadata"
        metadata_seed_bytes = metadata_seed.encode('utf-8')
        metadata_program_id = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"  # Metaplex Token Metadata Program ID
        metadata_program_id_bytes = base58.b58decode(metadata_program_id)
        
        # Proper seed concatenation for PDA finding
        seeds = bytearray()
        seeds.extend(metadata_seed_bytes)
        seeds.extend(metadata_program_id_bytes)
        seeds.extend(mint_bytes)
        
        # Calculate the PDA hash
        metadata_address_hash = hashlib.sha256(seeds).digest()
        metadata_pubkey = base58.b58encode(metadata_address_hash[:32]).decode('utf-8')
        
        # Get the account info for the derived PDA
        metadata_account = await client.get_account_info(metadata_pubkey)
        account_exists = metadata_account and "result" in metadata_account and metadata_account["result"] and metadata_account["result"]["value"]
        
        results["tests"]["metadata_pda"] = {
            "success": True,
            "result": {
                "derived_pda": metadata_pubkey,
                "pda_exists": account_exists
            }
        }
        
        print(f"✅ Derived PDA: {metadata_pubkey}")
        print(f"✅ PDA account exists: {'Yes' if account_exists else 'No'}")
        
        if account_exists and verbose:
            print(f"Raw PDA account info: {json.dumps(metadata_account, indent=2)}")
    except Exception as e:
        results["tests"]["metadata_pda"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Error: {str(e)}")
    
    # Test 5: Market Price
    print("\n🔍 Test 5: Market Price")
    try:
        price_info = await client.get_market_price(token_mint)
        results["tests"]["market_price"] = {
            "success": True,
            "result": price_info
        }
        print(f"✅ USD Price: ${price_info.get('usd_price', 'Unknown')}")
        print(f"✅ Price Source: {price_info.get('source', 'Unknown')}")
        
        if verbose:
            print(f"Raw price info: {json.dumps(price_info, indent=2)}")
    except Exception as e:
        results["tests"]["market_price"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Error: {str(e)}")
    
    # Save results to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Test results saved to {output_file}")
        except Exception as e:
            print(f"\n❌ Error saving results to {output_file}: {str(e)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test token metadata extraction")
    parser.add_argument("token_mint", help="Token mint address to test")
    parser.add_argument("--rpc", "-r", help="Custom RPC URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    parser.add_argument("--output", "-o", help="Output file path for JSON results")
    
    args = parser.parse_args()
    
    asyncio.run(test_token_metadata(
        args.token_mint,
        rpc_url=args.rpc,
        verbose=args.verbose,
        output_file=args.output
    ))


if __name__ == "__main__":
    main() 