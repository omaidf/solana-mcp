#!/usr/bin/env python3
"""
Test script for the wallet-based whale detection in the SolanaAnalyzer class
This demonstrates analyzing wallets based on their total token value
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

async def test_wallet_whale_detection():
    """Test the wallet-based whale detection"""
    print_section("Testing Wallet-Based Whale Detection")
    
    # Example wallets to test (replace with real ones for better testing)
    # These are some known large wallets on Solana
    test_wallets = [
        "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg",  # Solana core dev
        "FnCFNcFnCFnCFnCFnCFnCFNcFnCFnCFNcfNCmTcFNc",   # Binance?
        "HN7Zk4QuVQGSGYrKEJ2HqV5i9UaAGbUbjTQBJXqXkAeg", # Jupiter aggregator
        "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1"  # JUP whale
    ]
    
    print_step(f"Analyzing {len(test_wallets)} candidate wallets for whale activity")
    
    async with SolanaAnalyzer() as analyzer:
        try:
            print_step("Starting wallet value calculation...")
            start_time = time.time()
            
            # Set threshold to 1000 for testing purposes - you'd use 50000 in production
            results = await analyzer.find_whales(
                candidate_wallets=test_wallets,
                min_usd_value=1000,  # Lower for testing
                max_wallets=4
            )
            
            elapsed = time.time() - start_time
            print_step(f"Wallet analysis completed in {elapsed:.2f} seconds")
            
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
                for i, whale in enumerate(whales[:5], 1):
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
            
            # Output results as JSON
            print_step("Generating JSON output")
            json_output = {
                "timestamp": datetime.now().isoformat(),
                "analysis_duration_seconds": elapsed,
                "stats": stats,
                "whales": whales
            }
            
            # Save to file
            output_file = "whale_analysis_results.json"
            with open(output_file, 'w') as f:
                json.dump(json_output, f, indent=2)
            print_step(f"Results saved to {output_file}")
            
            # Also print JSON to console if needed
            print("\nJSON Output Sample (truncated):")
            print(json.dumps(json_output, indent=2)[:500] + "...\n")
            
            success = True
            print_result(success, f"Successfully analyzed {stats['wallets_analyzed']} wallets")
            return success
        except Exception as e:
            import traceback
            print("\n===== ERROR DETAILS =====")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            print("========================\n")
            print_result(False, f"Error: {str(e)}")
            return False

async def run_test():
    """Run the test"""
    print_section("SOLANA WALLET WHALE ANALYSIS TEST")
    print(f"Starting test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    result = await test_wallet_whale_detection()
    
    # Print summary
    print_section("TEST SUMMARY")
    status = "✅ PASSED" if result else "❌ FAILED"
    print(f"{status} - Wallet Whale Detection Test")
    
    print("\n" + "=" * 60)
    if result:
        print("🎉 Test passed successfully! 🎉")
    else:
        print("⚠️  Test failed - please check the logs above")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_test()) 