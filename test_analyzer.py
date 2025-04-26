#!/usr/bin/env python3
"""
Test script for the SolanaAnalyzer class
This script demonstrates the functionality of the SolanaAnalyzer class
"""
import asyncio
import logging
import time
import json
from datetime import datetime
from core.analyzer import SolanaAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_step(message):
    """Print a step with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] ▶ {message}")

def print_result(success, message=""):
    """Print a test result with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"[{timestamp}] {status} {message}")

def print_section(title):
    """Print a section header"""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"   {title}")
    print(f"{separator}")

async def test_get_token_info():
    """Test the get_token_info method"""
    print_section("Testing get_token_info")
    async with SolanaAnalyzer() as analyzer:
        # Test with USDC token
        usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        print_step(f"Getting info for USDC token: {usdc_mint}")
        try:
            start_time = time.time()
            token_info = await analyzer.get_token_info(usdc_mint)
            elapsed = time.time() - start_time
            
            print_step(f"API call completed in {elapsed:.2f} seconds")
            print(f"\nToken Info for {token_info.symbol}:")
            print(f"  Name: {token_info.name}")
            print(f"  Decimals: {token_info.decimals}")
            print(f"  Price: ${token_info.price}")
            print(f"  Market Cap: ${token_info.market_cap if token_info.market_cap else 'N/A'}")
            print(f"  Volume 24h: ${token_info.volume_24h if token_info.volume_24h else 'N/A'}")
            print(f"  Supply: {token_info.supply}")
            
            success = token_info is not None
            print_result(success, f"Retrieved token info for {token_info.symbol}")
            return success
        except Exception as e:
            print_result(False, f"Error: {str(e)}")
            return False

async def test_find_whales():
    """Test the find_whales method"""
    print_section("Testing find_whales")
    async with SolanaAnalyzer() as analyzer:
        try:
            print_step("Starting whale detection...")
            
            # Instead of JUP token, we're now looking for whale wallets
            # For testing, we'll use a few known Solana wallets
            test_wallets = [
                "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg",  # Solana core dev
                "3dAfLu6KK6QfQj1f2GYqR2ZjXg6dPR1gBNq1rY7VQ8Vw",
                "HN7Zk4QuVQGSGYrKEJ2HqV5i9UaAGbUbjTQBJXqXkAeg"  # Jupiter aggregator
            ]
            
            print_step(f"Analyzing {len(test_wallets)} wallets for whale activity")
            start_time = time.time()
            
            # Call the new find_whales method
            # Use lower threshold for testing
            results = await analyzer.find_whales(
                candidate_wallets=test_wallets,
                min_usd_value=1000,  # Lower for testing
                max_wallets=3
            )
            
            elapsed = time.time() - start_time
            print_step(f"Whale detection completed in {elapsed:.2f} seconds")
            
            # Print results
            stats = results["stats"]
            whales = results["whales"]
            
            print(f"\nWallet Analysis Results:")
            print(f"  Analyzed {stats['wallets_analyzed']} wallets")
            print(f"  Found {stats['whale_count']} whales (${stats['min_usd_threshold']:,.2f}+ in value)")
            print(f"  Whale percentage: {stats['whale_percentage']:.2f}%")
            
            # Display whale details if any were found
            if whales:
                print("\nTop Whales by Total Value:")
                for i, whale in enumerate(whales[:3], 1):
                    print(f"\n{i}. {whale['address']}")
                    print(f"   Total Value: ${whale['total_value']:,.2f}")
                    
                    # Show top tokens by value
                    top_tokens = whale['token_holdings'][:3]  # Top 3 tokens
                    if top_tokens:
                        print(f"   Top Holdings:")
                        for token in top_tokens:
                            print(f"     {token['symbol']}: {token['balance']:,.2f} (${token['value']:,.2f})")
            else:
                print("\nNo whales found matching the criteria")
            
            # Test passes if we successfully retrieved the data, even if there are no whales
            success = True
            print_result(success, f"Successfully analyzed {stats['wallets_analyzed']} wallets")
            return success
        except Exception as e:
            print_result(False, f"Error: {str(e)}")
            return False

async def test_account_info():
    """Test the get_account_info method"""
    print_section("Testing get_account_info")
    async with SolanaAnalyzer() as analyzer:
        # Test with a known Solana address
        address = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"  # Jupiter Program
        print_step(f"Getting account info for: {address}")
        try:
            start_time = time.time()
            account_info = await analyzer.get_account_info(address)
            elapsed = time.time() - start_time
            
            print_step(f"Account info retrieved in {elapsed:.2f} seconds")
            print(f"\nAccount Info:")
            print(f"  Owner: {account_info.owner}")
            print(f"  Lamports: {account_info.lamports} ({analyzer.lamports_to_sol(account_info.lamports)} SOL)")
            print(f"  Executable: {account_info.executable}")
            
            # Try to print some data details if available
            if isinstance(account_info.data, dict) and account_info.data.get('parsed'):
                print("  Data: Contains parsed program data")
            elif isinstance(account_info.data, list) and len(account_info.data) > 0:
                print(f"  Data: Binary data of length {len(account_info.data[0])}")
            else:
                print(f"  Data: {type(account_info.data)}")
                
            success = account_info is not None
            print_result(success, f"Retrieved account info for {address}")
            return success
        except Exception as e:
            print_result(False, f"Error: {str(e)}")
            return False
            
async def run_tests():
    """Run all tests"""
    print_section("SOLANA ANALYZER TEST SUITE")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tests = [
        ("Token Info", test_get_token_info()),
        ("Whale Detection", test_find_whales()),
        ("Account Info", test_account_info())
    ]
    
    results = []
    for name, test in tests:
        try:
            result = await test
            results.append((name, result))
        except Exception as e:
            print_result(False, f"Test {name} failed with unexpected error: {e}")
            results.append((name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed successfully! 🎉")
    else:
        print("⚠️  Some tests failed - please check the logs above")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests()) 