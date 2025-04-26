#!/usr/bin/env python3
"""
Live API tests for Solana Analyzer - makes real requests to the running server
"""
import requests
import json
import sys
import time

# Server URL
BASE_URL = "http://localhost:8000/api/analyzer"

# Sample addresses for testing
SAMPLE_ADDRESSES = {
    "wallet": "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg",
    "token": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "transaction": "5KtPn1LGuxhHrKNJL1VJ4zNm2fwJGALwJJAJBSqADY8tzZVSKHbsCqP5fm4fXXjCYy5xJgJZqTsK5kK3PSQvRd5B"
}

def print_test_result(test_name, response):
    """Print the test result in a formatted way"""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    
    print(f"Status Code: {response.status_code}")
    print("Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    
    print("\nResponse Body:")
    try:
        # Try to format as JSON
        json_response = response.json()
        print(json.dumps(json_response, indent=2))
    except:
        # If not JSON, print raw text
        print(response.text)
    
    print("-" * 80)

def test_analyzer_status():
    """Test the analyzer status endpoint"""
    response = requests.get(f"{BASE_URL}/status")
    print_test_result("Analyzer Status", response)
    return response

def test_get_token_info():
    """Test getting token information"""
    response = requests.post(
        f"{BASE_URL}/token",
        json={"mint_address": SAMPLE_ADDRESSES["token"]}
    )
    print_test_result("Token Info", response)
    return response

def test_find_whales():
    """Test finding whales for a token"""
    response = requests.post(
        f"{BASE_URL}/whales",
        json={
            "mint_address": SAMPLE_ADDRESSES["token"],
            "min_usd_value": 100000,
            "max_holders": 10
        }
    )
    print_test_result("Find Whales", response)
    return response

def test_get_account_info():
    """Test getting account information"""
    response = requests.post(
        f"{BASE_URL}/account",
        json={
            "address": SAMPLE_ADDRESSES["wallet"],
            "encoding": "jsonParsed"
        }
    )
    print_test_result("Account Info", response)
    return response

def test_get_token_accounts():
    """Test getting token accounts"""
    response = requests.post(
        f"{BASE_URL}/token-accounts",
        json={"address": SAMPLE_ADDRESSES["wallet"]}
    )
    print_test_result("Token Accounts", response)
    return response

def test_token_info_error():
    """Test error handling in token info endpoint"""
    response = requests.post(
        f"{BASE_URL}/token",
        json={"mint_address": "invalid-token"}
    )
    print_test_result("Token Info Error", response)
    return response

def check_server_available():
    """Check if the server is available"""
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def main():
    """Run all tests"""
    print("Checking if server is running...")
    if not check_server_available():
        print("ERROR: Server is not running. Please start the server first with:")
        print("python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    print("Server is running. Starting tests...\n")
    
    # Run all tests
    test_analyzer_status()
    test_get_token_info()
    test_find_whales()  
    test_get_account_info()
    test_get_token_accounts()
    test_token_info_error()

if __name__ == "__main__":
    main() 