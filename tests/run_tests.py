#!/usr/bin/env python3
"""Run all tests for the Solana MCP API."""

import os
import sys
import subprocess
import time
import argparse


def run_client_tests():
    """Run the Solana client tests."""
    print("\n=== Running Solana client tests ===")
    subprocess.run(["python3", "tests/test_solana_client.py"], check=False)


def run_api_tests():
    """Run the API route tests."""
    print("\n=== Running API route tests ===")
    subprocess.run(["python3", "tests/test_api_routes.py"], check=False)


def run_http_tests(base_url):
    """Run the HTTP API tests.
    
    Args:
        base_url: API base URL to test
    """
    print(f"\n=== Running HTTP API tests against {base_url} ===")
    # Set the API base URL via environment variable
    env = os.environ.copy()
    env["API_BASE_URL"] = base_url
    subprocess.run(["python3", "tests/test_api_http.py"], env=env, check=False)


def test_solana_rpc_connection():
    """Test the connection to the Solana RPC endpoint."""
    print("\n=== Testing Solana RPC connection ===")
    try:
        import asyncio
        from solana_mcp.solana_client import SolanaClient
        
        async def test_connection():
            client = SolanaClient()
            try:
                # Try to get slot (simple request)
                slot = await client.get_slot()
                print(f"Connection successful! Current slot: {slot}")
                return True
            except Exception as e:
                print(f"Connection failed: {str(e)}")
                return False
            finally:
                await client.close()
        
        success = asyncio.run(test_connection())
        return success
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Run tests for Solana MCP API")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="API base URL for HTTP tests (default: http://localhost:8000)")
    parser.add_argument("--skip-client", action="store_true", 
                       help="Skip Solana client tests")
    parser.add_argument("--skip-api", action="store_true", 
                       help="Skip API route tests")
    parser.add_argument("--skip-http", action="store_true", 
                       help="Skip HTTP API tests")
    args = parser.parse_args()
    
    # Check if we can connect to Solana RPC
    rpc_ok = test_solana_rpc_connection()
    if not rpc_ok:
        print("WARNING: Solana RPC connection failed. Tests may fail.")
        response = input("Continue with tests? (y/n): ")
        if response.lower() != "y":
            print("Tests aborted.")
            return
    
    # Run the tests
    if not args.skip_client:
        run_client_tests()
    
    if not args.skip_api:
        run_api_tests()
    
    if not args.skip_http:
        run_http_tests(args.api_url)
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main() 