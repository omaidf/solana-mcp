#!/usr/bin/env python
"""
Script to manually test token risk analysis API endpoints.
Run this script after starting the API server to verify endpoints are working correctly.
"""

import requests
import json
import sys
from typing import Dict, Any
import argparse
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init()

# Constants
BASE_URL = "http://localhost:8000"  # Default API server URL
TEST_TOKEN_MINT = "ENxauXrtBtnFH1aJAFYZnxVVM4rGLkXJi3bUhT7tpump"  # SOLMCP token


def print_response(endpoint: str, response: requests.Response, test_passed: bool = None) -> None:
    """Print API response in a readable format with pass/fail status.
    
    Args:
        endpoint: API endpoint being tested
        response: Response object from the request
        test_passed: Whether the test passed (True/False/None)
    """
    if test_passed is True:
        status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
    elif test_passed is False:
        status = f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
    else:
        status = f"{Fore.YELLOW}• INFO{Style.RESET_ALL}"
    
    print(f"\n===== Testing endpoint: {endpoint} [{status}] =====")
    print(f"Status code: {response.status_code}")
    
    # Always try to print some data, even for non-200 responses
    try:
        if response.status_code == 200:
            data = response.json()
            print(f"Response data (pretty):\n{json.dumps(data, indent=2)}")
        else:
            print(f"Error response: {response.text}")
            
            # Try to parse error response if it's JSON
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    print(f"Parsed error data:\n{json.dumps(error_data, indent=2)}")
            except:
                # Not JSON or couldn't parse
                pass
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        
    # Print separator for better readability
    print(f"{'-' * 80}")


def test_analyze_token_risks(base_url: str, token_mint: str) -> bool:
    """Test the token risk analysis endpoint.
    
    Args:
        base_url: Base URL of the API
        token_mint: Token mint address to test
        
    Returns:
        True if test passed, False otherwise
    """
    endpoint = f"/token-risk/analyze/{token_mint}"
    response = requests.get(f"{base_url}{endpoint}")
    test_passed = response.status_code == 200
    
    if test_passed:
        data = response.json()
        # Additional validation
        test_passed = all(key in data for key in ["name", "symbol", "risk_level", "risk_score"])
    
    print_response(endpoint, response, test_passed)
    return test_passed


def test_get_liquidity_locks(base_url: str, token_mint: str) -> bool:
    """Test the liquidity locks endpoint.
    
    Args:
        base_url: Base URL of the API
        token_mint: Token mint address to test
        
    Returns:
        True if test passed, False otherwise
    """
    endpoint = f"/token-risk/liquidity-locks/{token_mint}"
    response = requests.get(f"{base_url}{endpoint}")
    test_passed = response.status_code == 200
    
    if test_passed:
        data = response.json()
        # Additional validation
        test_passed = all(key in data for key in ["token_mint", "token_name", "has_locked_liquidity"])
    
    print_response(endpoint, response, test_passed)
    return test_passed


def test_get_tokenomics(base_url: str, token_mint: str) -> bool:
    """Test the tokenomics endpoint.
    
    Args:
        base_url: Base URL of the API
        token_mint: Token mint address to test
        
    Returns:
        True if test passed, False otherwise
    """
    endpoint = f"/token-risk/tokenomics/{token_mint}"
    response = requests.get(f"{base_url}{endpoint}")
    test_passed = response.status_code == 200
    
    if test_passed:
        data = response.json()
        # Additional validation
        test_passed = all(key in data for key in ["token_mint", "token_name", "supply", "authorities"])
    
    print_response(endpoint, response, test_passed)
    return test_passed


def test_get_token_category(base_url: str, token_mint: str) -> bool:
    """Test the token category endpoint.
    
    Args:
        base_url: Base URL of the API
        token_mint: Token mint address to test
        
    Returns:
        True if test passed, False otherwise
    """
    endpoint = f"/token-risk/meme-category/{token_mint}"
    response = requests.get(f"{base_url}{endpoint}")
    test_passed = response.status_code == 200
    
    if test_passed:
        data = response.json()
        # Additional validation
        test_passed = all(key in data for key in ["token_mint", "category", "is_meme_token"])
    
    print_response(endpoint, response, test_passed)
    return test_passed


def test_get_meme_tokens(base_url: str) -> bool:
    """Test the meme tokens listing endpoint.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if test passed, False otherwise
    """
    endpoint = "/token-risk/meme-tokens"
    response = requests.get(f"{base_url}{endpoint}")
    test_passed = response.status_code == 200
    
    if test_passed:
        data = response.json()
        # Additional validation
        test_passed = all(key in data for key in ["tokens", "total_count", "category", "limit"])
    
    print_response(endpoint, response, test_passed)
    return test_passed


def test_get_meme_tokens_with_filters(base_url: str) -> bool:
    """Test the meme tokens listing endpoint with filters.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        True if test passed, False otherwise
    """
    endpoint = "/token-risk/meme-tokens?category=Meme&limit=5&min_holders=100&min_liquidity=10000"
    response = requests.get(f"{base_url}{endpoint}")
    test_passed = response.status_code == 200
    
    if test_passed:
        data = response.json()
        # Additional validation
        test_passed = all(key in data for key in ["tokens", "total_count", "category", "limit"])
        if test_passed:
            # Check if category filter was applied
            test_passed = data["category"] == "Meme" and data["limit"] == 5
    
    print_response(endpoint, response, test_passed)
    return test_passed


def main() -> None:
    """Run all tests with command line arguments."""
    parser = argparse.ArgumentParser(description="Test token risk analysis API endpoints")
    parser.add_argument("--url", default=BASE_URL, help=f"API server URL (default: {BASE_URL})")
    parser.add_argument("--token", default=TEST_TOKEN_MINT, help=f"Token mint address (default: {TEST_TOKEN_MINT})")
    parser.add_argument("--endpoint", help="Test only a specific endpoint (analyze, liquidity, tokenomics, category, meme, meme-filters)")
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}Testing token risk API endpoints{Style.RESET_ALL}")
    print(f"API URL: {args.url}")
    print(f"Token: {args.token}\n")
    
    try:
        # Dictionary of test functions
        tests = {
            "analyze": lambda: test_analyze_token_risks(args.url, args.token),
            "liquidity": lambda: test_get_liquidity_locks(args.url, args.token),
            "tokenomics": lambda: test_get_tokenomics(args.url, args.token),
            "category": lambda: test_get_token_category(args.url, args.token),
            "meme": lambda: test_get_meme_tokens(args.url),
            "meme-filters": lambda: test_get_meme_tokens_with_filters(args.url)
        }
        
        # Run specific test or all tests
        results = {}
        if args.endpoint:
            if args.endpoint in tests:
                results[args.endpoint] = tests[args.endpoint]()
            else:
                print(f"{Fore.RED}Error: Unknown endpoint '{args.endpoint}'{Style.RESET_ALL}")
                print(f"Available endpoints: {', '.join(tests.keys())}")
                sys.exit(1)
        else:
            # Run all tests
            for name, test_func in tests.items():
                results[name] = test_func()
        
        # Print summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        print(f"\n{Fore.CYAN}===== Test Summary ====={Style.RESET_ALL}")
        print(f"Total tests: {total}")
        print(f"Passed: {Fore.GREEN}{passed}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{total - passed}{Style.RESET_ALL}")
        
        if passed == total:
            print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}Some tests failed. Check the output above for details.{Style.RESET_ALL}")
            # List failed tests
            failed_tests = [name for name, result in results.items() if not result]
            print(f"Failed tests: {', '.join(failed_tests)}")
        
    except requests.RequestException as e:
        print(f"\n{Fore.RED}Error connecting to API: {e}{Style.RESET_ALL}")
        print("Make sure the API server is running at the specified URL.")
        sys.exit(1)


if __name__ == "__main__":
    main() 