#!/usr/bin/env python3
"""
Main test runner for Solana MCP API.
This script runs all tests for the Solana MCP project.
"""

import os
import sys
import argparse
import subprocess
import time


def print_banner(message):
    """Print a banner with the given message."""
    border = "=" * (len(message) + 4)
    print(f"\n{border}")
    print(f"| {message} |")
    print(f"{border}\n")


def run_unit_tests():
    """Run unit tests using pytest."""
    print_banner("Running Unit Tests")
    
    # Check if the tests/unit directory exists
    if not os.path.isdir("tests/unit"):
        print("⚠️ Unit tests directory (tests/unit) not found. Skipping unit tests.")
        return True
    
    # Run all tests in the tests/unit directory
    try:
        result = subprocess.run(["python", "-m", "pytest", "tests/unit", "-v"], check=False)
        
        # Report result
        if result.returncode == 0:
            print("✓ Unit tests PASSED")
        else:
            print("✗ Unit tests FAILED")
        
        return result.returncode == 0
    except FileNotFoundError:
        print("⚠️ Pytest not found. Make sure it's installed.")
        return False
    except Exception as e:
        print(f"⚠️ Error running unit tests: {str(e)}")
        return False


def run_integration_tests():
    """Run integration tests using pytest."""
    print_banner("Running Integration Tests")
    
    # Check if the tests/integration directory exists
    if not os.path.isdir("tests/integration"):
        print("⚠️ Integration tests directory (tests/integration) not found. Skipping integration tests.")
        return True
    
    # Run all tests in the tests/integration directory
    try:
        result = subprocess.run(["python", "-m", "pytest", "tests/integration", "-v"], check=False)
        
        # Report result
        if result.returncode == 0:
            print("✓ Integration tests PASSED")
        else:
            print("✗ Integration tests FAILED")
        
        return result.returncode == 0
    except FileNotFoundError:
        print("⚠️ Pytest not found. Make sure it's installed.")
        return False
    except Exception as e:
        print(f"⚠️ Error running integration tests: {str(e)}")
        return False


def run_api_tests():
    """Run API route tests."""
    print_banner("Running API Route Tests")
    
    # Check if the test file exists
    if not os.path.isfile("tests/test_api_routes.py"):
        print("⚠️ API route tests file (tests/test_api_routes.py) not found. Skipping API tests.")
        return True
    
    try:
        result = subprocess.run(["python", "tests/test_api_routes.py"], check=False)
        
        # Report result
        if result.returncode == 0:
            print("✓ API route tests PASSED")
        else:
            print("✗ API route tests FAILED")
        
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ Error running API route tests: {str(e)}")
        return False


def run_rest_api_tests(base_url=None):
    """Run REST API tests.
    
    Args:
        base_url: Optional API base URL to test against
    """
    print_banner("Running REST API Tests")
    
    # Check if the test file exists
    if not os.path.isfile("test_rest_api.py"):
        print("⚠️ REST API tests file (test_rest_api.py) not found. Skipping REST API tests.")
        return True
    
    try:
        cmd = ["python", "test_rest_api.py"]
        env = os.environ.copy()
        
        if base_url:
            env["API_BASE_URL"] = base_url
            print(f"Testing against API at: {base_url}")
        
        result = subprocess.run(cmd, env=env, check=False)
        
        # Report result
        if result.returncode == 0:
            print("✓ REST API tests PASSED")
        else:
            print("✗ REST API tests FAILED")
        
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ Error running REST API tests: {str(e)}")
        return False


def run_solana_client_tests():
    """Run Solana client tests."""
    print_banner("Running Solana Client Tests")
    
    # Check if the test file exists
    if not os.path.isfile("tests/test_solana_client.py"):
        print("⚠️ Solana client tests file (tests/test_solana_client.py) not found. Skipping client tests.")
        return True
    
    try:
        result = subprocess.run(["python", "tests/test_solana_client.py"], check=False)
        
        # Report result
        if result.returncode == 0:
            print("✓ Solana client tests PASSED")
        else:
            print("✗ Solana client tests FAILED")
        
        return result.returncode == 0
    except Exception as e:
        print(f"⚠️ Error running Solana client tests: {str(e)}")
        return False


def run_all_tests(args):
    """Run all test suites.
    
    Args:
        args: Command line arguments
    """
    print_banner("SOLANA MCP TEST SUITE")
    
    start_time = time.time()
    success = True
    
    # Run tests based on command line arguments
    if not args.skip_unit:
        unit_success = run_unit_tests()
        success = success and unit_success
    
    if not args.skip_integration:
        integration_success = run_integration_tests()
        success = success and integration_success
    
    if not args.skip_api:
        api_success = run_api_tests()
        success = success and api_success
    
    if not args.skip_client:
        client_success = run_solana_client_tests()
        success = success and client_success
    
    if not args.skip_rest:
        rest_success = run_rest_api_tests(args.api_url)
        success = success and rest_success
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print final results
    print_banner(f"ALL TESTS COMPLETED IN {execution_time:.2f} SECONDS")
    
    if success:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests for Solana MCP")
    
    parser.add_argument("--skip-unit", action="store_true", help="Skip unit tests")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--skip-api", action="store_true", help="Skip API route tests")
    parser.add_argument("--skip-client", action="store_true", help="Skip Solana client tests")
    parser.add_argument("--skip-rest", action="store_true", help="Skip REST API tests")
    parser.add_argument("--api-url", default=None, help="API base URL for REST tests")
    
    args = parser.parse_args()
    
    # Run all tests and exit with appropriate code
    return run_all_tests(args)


if __name__ == "__main__":
    sys.exit(main()) 