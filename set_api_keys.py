#!/usr/bin/env python3
"""
Utility script to set API keys in environment variables and run the server.
"""
import os
import sys
import subprocess

# API Keys
HELIUS_API_KEY = "6b12fb2c-e3e4-4e3d-b3d1-7bd9d54e4c87"
BIRDEYE_API_KEY = "8uW5DQrxs9V389RwnbCSr0fdVGzEZo3y"

def main():
    """Set environment variables and run the server"""
    # Set environment variables
    os.environ["HELIUS_API_KEY"] = HELIUS_API_KEY
    os.environ["BIRDEYE_API_KEY"] = BIRDEYE_API_KEY
    
    print(f"Set HELIUS_API_KEY: {HELIUS_API_KEY[:5]}...")
    print(f"Set BIRDEYE_API_KEY: {BIRDEYE_API_KEY[:5]}...")
    
    # Check if we need to run the server
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        print("Starting server with API keys set...")
        # Run the server
        cmd = ["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
        subprocess.run(cmd)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running API tests with API keys set...")
        # Run the tests
        cmd = ["python", "test/api/live_api_test.py"]
        subprocess.run(cmd)
    else:
        print("Environment variables set. Run your command now.")
        print("Usage:")
        print("  ./set_api_keys.py server   - Start the server with API keys")
        print("  ./set_api_keys.py test     - Run the API tests with API keys")
        print("  ./set_api_keys.py          - Just set the environment variables")

if __name__ == "__main__":
    main() 