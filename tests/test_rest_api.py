#!/usr/bin/env python3
"""
Comprehensive tests for all REST API endpoints in the Solana MCP server.
This script tests all the API endpoints defined in server.py.
"""

import os
import sys
import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import requests
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Default API URL if not set in environment
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Test data - valid Solana addresses
VALID_ADDRESS = "9hDpVnYqPAokHL1LNtxUwvD9ymL8FTQ3wMfM7JrNP1qB"  # Example wallet
VALID_TOKEN_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
VALID_NFT_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Use a known token as NFT for testing
VALID_TX_SIG = "5Kr8iHFNxDQYz7XkW5xcqZ1roARMuJ8Z6BQsrQLpCyQrAA68Jck2ocGiNhdbNVnZDoYMUzL1Ry5x7GfJh8yJGqdf"

# Invalid test data
INVALID_ADDRESS = "invalid-address"
INVALID_TOKEN_MINT = "invalid-token-mint"
INVALID_TX_SIG = "invalid-transaction-signature"


class TestSolanaMCPRestAPI:
    """Test class for Solana MCP REST API endpoints."""

    # ============ Account API Tests ============

    def test_get_account_info(self):
        """Test getting account information."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/account/{VALID_ADDRESS}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "pubkey" in data, "Response missing pubkey field"
        assert data["pubkey"] == VALID_ADDRESS, f"Expected pubkey {VALID_ADDRESS}, got {data.get('pubkey')}"
        print(f"✓ GET /api/account/{VALID_ADDRESS} - Success")

    def test_get_account_info_invalid(self):
        """Test getting information for an invalid account address."""
        response = requests.get(f"{API_BASE_URL}/api/account/{INVALID_ADDRESS}")
        assert response.status_code in [400, 404], f"Expected status 400/404, got {response.status_code}"
        print(f"✓ GET /api/account/{INVALID_ADDRESS} - Expected error received")

    def test_get_account_balance(self):
        """Test getting account balance."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/balance/{VALID_ADDRESS}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "balance" in data, "Response missing balance field"
        assert isinstance(data["balance"], (int, float)), f"Expected numeric balance, got {type(data['balance'])}"
        print(f"✓ GET /api/balance/{VALID_ADDRESS} - Success")

    def test_get_account_balance_invalid(self):
        """Test getting balance for an invalid account address."""
        response = requests.get(f"{API_BASE_URL}/api/balance/{INVALID_ADDRESS}")
        assert response.status_code in [400, 404], f"Expected status 400/404, got {response.status_code}"
        print(f"✓ GET /api/balance/{INVALID_ADDRESS} - Expected error received")

    # ============ Token API Tests ============

    def test_get_token_info(self):
        """Test getting token information."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/token/{VALID_TOKEN_MINT}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "mint" in data, "Response missing mint field"
        assert data["mint"] == VALID_TOKEN_MINT, f"Expected mint {VALID_TOKEN_MINT}, got {data.get('mint')}"
        print(f"✓ GET /api/token/{VALID_TOKEN_MINT} - Success")

    def test_get_token_info_invalid(self):
        """Test getting information for an invalid token mint."""
        response = requests.get(f"{API_BASE_URL}/api/token/{INVALID_TOKEN_MINT}")
        assert response.status_code in [400, 404], f"Expected status 400/404, got {response.status_code}"
        print(f"✓ GET /api/token/{INVALID_TOKEN_MINT} - Expected error received")
    
    # ============ Transaction API Tests ============
    
    def test_get_transactions_for_address(self):
        """Test getting transaction history for an address."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/transactions/{VALID_ADDRESS}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "address" in data, "Response missing expected address"
        assert "transactions" in data, "Response missing transactions list"
        print(f"✓ GET /api/transactions/{VALID_ADDRESS} - Success")

    def test_get_transactions_with_limit(self):
        """Test getting transaction history with a limit parameter."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        limit = 5
        response = requests.get(f"{API_BASE_URL}/api/transactions/{VALID_ADDRESS}?limit={limit}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        # Checking that we respect the limit (or have fewer transactions)
        assert len(data.get("transactions", [])) <= limit, f"Expected at most {limit} transactions"
        print(f"✓ GET /api/transactions/{VALID_ADDRESS}?limit={limit} - Success")

    # ============ NFT API Tests ============

    def test_get_nft_info(self):
        """Test getting NFT information."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/nft/{VALID_NFT_MINT}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "mint" in data, "Response missing mint field"
        assert data["mint"] == VALID_NFT_MINT, f"Expected mint {VALID_NFT_MINT}, got {data.get('mint')}"
        print(f"✓ GET /api/nft/{VALID_NFT_MINT} - Success")

    # ============ NLP API Tests ============

    def test_nlp_query(self):
        """Test the natural language processing query endpoint."""
        query_data = {
            "query": f"What is the balance of {VALID_ADDRESS}?",
            "format_level": "auto"
        }
        
        response = requests.post(f"{API_BASE_URL}/api/nlp/query", json=query_data)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "result" in data, "Response missing result data"
        assert "session_id" in data, "Response missing session_id"
        print(f"✓ POST /api/nlp/query (balance query) - Success")
    
    def test_nlp_token_query(self):
        """Test the NLP query endpoint with a token-related query."""
        query_data = {
            "query": f"What is the information about token {VALID_TOKEN_MINT}?",
            "format_level": "auto"
        }
        
        response = requests.post(f"{API_BASE_URL}/api/nlp/query", json=query_data)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        print(f"✓ POST /api/nlp/query (token query) - Success")

    def test_nlp_query_invalid(self):
        """Test the NLP query endpoint with an invalid query."""
        query_data = {
            "query": "This is a nonsensical query that doesn't relate to Solana",
            "format_level": "auto"
        }
        
        response = requests.post(f"{API_BASE_URL}/api/nlp/query", json=query_data)
        # Even with invalid queries, we should get a 200 response with an error in the payload
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "result" in data, "Response missing result field"
        # Check for error indication in result
        if "error" in data["result"]:
            print(f"✓ POST /api/nlp/query (invalid query) - Expected error in result")
        else:
            # If no error, it might have been interpreted in some way
            print(f"✓ POST /api/nlp/query (invalid query) - Query was interpreted")

    # ============ Session API Tests ============

    def test_session_history(self):
        """Test getting session history."""
        # First create a session with a query
        query_data = {
            "query": f"What is the balance of {VALID_ADDRESS}?",
            "format_level": "auto"
        }
        
        session_response = requests.post(f"{API_BASE_URL}/api/nlp/query", json=query_data)
        assert session_response.status_code == 200
        
        session_data = session_response.json()
        session_id = session_data.get("session_id")
        assert session_id, "Failed to get session_id from query response"
        
        # Now get the session history
        response = requests.get(f"{API_BASE_URL}/api/session/{session_id}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "session_id" in data, "Response missing session_id"
        assert "queries" in data, "Response missing queries history"
        print(f"✓ GET /api/session/{session_id} - Success")

    def test_session_history_invalid(self):
        """Test getting history for an invalid session ID."""
        invalid_session_id = "nonexistent-session-id"
        
        response = requests.get(f"{API_BASE_URL}/api/session/{invalid_session_id}")
        assert response.status_code == 404, f"Expected status 404, got {response.status_code}"
        print(f"✓ GET /api/session/{invalid_session_id} - Expected error received")

    # ============ Schema API Tests ============

    def test_get_schemas(self):
        """Test getting all available schemas."""
        response = requests.get(f"{API_BASE_URL}/api/schemas")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "available_schemas" in data, "Response missing available_schemas list"
        print(f"✓ GET /api/schemas - Success")

    def test_get_specific_schema(self):
        """Test getting a specific schema."""
        schema_name = "account"
        
        response = requests.get(f"{API_BASE_URL}/api/schemas/{schema_name}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "type" in data, "Response missing schema type"
        assert "properties" in data, "Response missing schema properties"
        print(f"✓ GET /api/schemas/{schema_name} - Success")

    def test_get_dynamic_schema(self):
        """Test getting a dynamically generated schema."""
        schema_name = "token"
        
        response = requests.get(f"{API_BASE_URL}/api/schemas/{schema_name}?dynamic=true")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "type" in data, "Response missing schema type"
        assert "properties" in data, "Response missing schema properties"
        print(f"✓ GET /api/schemas/{schema_name}?dynamic=true - Success")

    # ============ Analysis API Tests ============

    def test_chain_analysis(self):
        """Test the chain analysis endpoint."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/analysis/chain/{VALID_ADDRESS}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "analysis" in data, "Response missing analysis field"
        print(f"✓ GET /api/analysis/chain/{VALID_ADDRESS} - Success")

    def test_token_flow_analysis(self):
        """Test the token flow analysis endpoint."""
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(f"{API_BASE_URL}/api/analysis/token-flow/{VALID_TOKEN_MINT}")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "analysis" in data, "Response missing analysis field"
        print(f"✓ GET /api/analysis/token-flow/{VALID_TOKEN_MINT} - Success")

    # ============ Health Check Test ============

    def test_health_check(self):
        """Test the health check endpoint."""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "status" in data, "Response missing status"
        assert data["status"] == "healthy", f"Expected 'healthy' status, got {data.get('status')}"
        print(f"✓ GET /health - Success")

    # ============ API Docs Test ============

    def test_api_docs(self):
        """Test the API documentation endpoint."""
        response = requests.get(f"{API_BASE_URL}/api/docs")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "openapi" in data, "Response missing OpenAPI version"
        assert "paths" in data, "Response missing API paths"
        print(f"✓ GET /api/docs - Success")


def print_banner(message):
    """Print a banner with the given message."""
    border = "=" * (len(message) + 4)
    print(f"\n{border}")
    print(f"| {message} |")
    print(f"{border}\n")


def run_tests():
    """Run all the API tests."""
    print_banner(f"Running Solana MCP REST API Tests against {API_BASE_URL}")
    
    test_instance = TestSolanaMCPRestAPI()
    
    # Run all test methods
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    success_count = 0
    failure_count = 0
    
    for method_name in test_methods:
        test_method = getattr(test_instance, method_name)
        try:
            test_method()
            success_count += 1
        except Exception as e:
            print(f"✗ {method_name} - FAILED: {str(e)}")
            failure_count += 1
    
    total = success_count + failure_count
    print_banner(f"Test Results: {success_count}/{total} passed, {failure_count}/{total} failed")


if __name__ == "__main__":
    run_tests() 