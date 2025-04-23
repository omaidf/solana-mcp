"""Test script for making HTTP requests to the deployed API."""

import requests
import json
import sys
import time
from typing import Dict, Any, Optional


class APITester:
    """Class for testing the Solana MCP API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester.
        
        Args:
            base_url: The base URL of the API server
        """
        self.base_url = base_url
    
    def make_request(self, 
                    endpoint: str, 
                    method: str = "GET", 
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a request to the API.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body for POST/PUT requests
            headers: Custom headers
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, json=data, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            execution_time = time.time() - start_time
            
            print(f"Request to {endpoint} completed in {execution_time:.2f}s with status {response.status_code}")
            
            if response.status_code >= 400:
                print(f"Error response: {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            return response.json()
            
        except requests.RequestException as e:
            print(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_response": response.text}
    
    def test_token_metadata(self, mint: str) -> Dict[str, Any]:
        """Test the token metadata endpoint.
        
        Args:
            mint: Token mint address
            
        Returns:
            Response data
        """
        return self.make_request(f"/token-analysis/metadata/{mint}")
    
    def test_token_supply(self, mint: str) -> Dict[str, Any]:
        """Test the token supply endpoint.
        
        Args:
            mint: Token mint address
            
        Returns:
            Response data
        """
        return self.make_request(f"/token-analysis/supply/{mint}")
    
    def test_token_holders(self, mint: str) -> Dict[str, Any]:
        """Test the token holders endpoint.
        
        Args:
            mint: Token mint address
            
        Returns:
            Response data
        """
        return self.make_request(f"/token-analysis/holders/{mint}")
    
    def test_token_risk(self, mint: str) -> Dict[str, Any]:
        """Test the token risk analysis endpoint.
        
        Args:
            mint: Token mint address
            
        Returns:
            Response data
        """
        return self.make_request(f"/token-risk/analyze/{mint}")
    
    def test_top_pools(self, limit: int = 5) -> Dict[str, Any]:
        """Test the top pools endpoint.
        
        Args:
            limit: Maximum number of pools to return
            
        Returns:
            Response data
        """
        return self.make_request(f"/liquidity-analysis/top-pools", params={"limit": limit})


def run_tests():
    """Run all API tests."""
    # Set your API base URL here
    tester = APITester("http://localhost:8000")
    
    # USDC token for testing
    usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    
    print("Testing Token Metadata API...")
    metadata_result = tester.test_token_metadata(usdc_mint)
    print(f"Result: {json.dumps(metadata_result, indent=2)}\n")
    
    print("Testing Token Supply API...")
    supply_result = tester.test_token_supply(usdc_mint)
    print(f"Result: {json.dumps(supply_result, indent=2)}\n")
    
    print("Testing Token Holders API...")
    holders_result = tester.test_token_holders(usdc_mint)
    print(f"Result: {json.dumps(holders_result, indent=2)}\n")
    
    print("Testing Token Risk Analysis API...")
    risk_result = tester.test_token_risk(usdc_mint)
    print(f"Result: {json.dumps(risk_result, indent=2)}\n")
    
    print("Testing Top Pools API...")
    pools_result = tester.test_top_pools(3)
    print(f"Result: {json.dumps(pools_result, indent=2)}\n")
    
    # Test with an invalid token
    print("Testing with invalid token...")
    invalid_result = tester.test_token_metadata("invalid-token")
    print(f"Result: {json.dumps(invalid_result, indent=2)}\n")


if __name__ == "__main__":
    run_tests() 