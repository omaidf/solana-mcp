#!/usr/bin/env python
"""
Consolidated test script for Solana MCP.
This script provides a unified interface for running all types of tests:
- Unit tests (pytest)
- API tests (with server management)
- Direct analyzer tests (with or without RPC connection)
- Integration tests

Features:
- Better resource management
- Configurable RPC endpoints
- Test selection options
- Detailed reporting
"""

import os
import sys
import subprocess
import time
import argparse
import signal
import json
import logging
import asyncio
import atexit
from typing import List, Dict, Any, Optional, Callable, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("solana-mcp-tests")

# Global variables for resource management
server_processes = []

def cleanup_resources():
    """Clean up all resources on exit."""
    for process in server_processes:
        if process and process.poll() is None:  # If process exists and is running
            logger.info(f"Stopping process PID {process.pid}...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)  # Give it a moment to shut down
            except (ProcessLookupError, OSError) as e:
                logger.warning(f"Error stopping process: {e}")

# Register cleanup function
atexit.register(cleanup_resources)

def check_port_in_use(port: int) -> bool:
    """Check if a port is already in use.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is in use, False otherwise
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int, end_port: int = 65535) -> int:
    """Find an available port within a range.
    
    Args:
        start_port: Starting port number
        end_port: Ending port number
        
    Returns:
        Available port number or -1 if none found
    """
    for port in range(start_port, min(end_port + 1, 65536)):
        if not check_port_in_use(port):
            return port
    return -1

def kill_processes_on_port(port: int):
    """Kill processes using a specific port.
    
    Args:
        port: Port number
    """
    if sys.platform == 'win32':
        # Windows implementation
        try:
            output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True)
            for line in output.decode().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True)
                        logger.info(f"Killed process with PID {pid} using port {port}")
                    except subprocess.SubprocessError:
                        logger.warning(f"Failed to kill process with PID {pid}")
        except subprocess.SubprocessError:
            logger.info(f"No processes found using port {port}")
    else:
        # Unix/Linux/Mac implementation
        try:
            output = subprocess.check_output(f'lsof -i :{port} -t', shell=True)
            for pid in output.decode().strip().split('\n'):
                if pid:
                    try:
                        subprocess.run(f'kill -9 {pid}', shell=True)
                        logger.info(f"Killed process with PID {pid} using port {port}")
                    except subprocess.SubprocessError:
                        logger.warning(f"Failed to kill process with PID {pid}")
        except subprocess.SubprocessError:
            logger.info(f"No processes found using port {port}")

# ==================================================
# Unit Tests
# ==================================================

def run_unit_tests(args: argparse.Namespace) -> bool:
    """Run unit tests using pytest.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info("Running unit tests...")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path if specified, otherwise run all tests
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append("solana_mcp/tests")
    
    # Add verbosity and output options
    cmd.extend(["-v"])
    
    # Show print output if requested
    if args.show_output:
        cmd.append("-s")
    
    # Add specific test markers if provided
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # Set custom pytest args if provided
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error("❌ Unit tests failed!")
        return False
    
    logger.info("✅ Unit tests passed!")
    return True

# ==================================================
# API Server and Tests
# ==================================================

def start_api_server(args: argparse.Namespace) -> Tuple[bool, Optional[subprocess.Popen], int]:
    """Start the API server on an available port.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (success, server_process, port)
    """
    global server_processes
    
    # Find an available port starting from the specified one
    port = args.port
    
    # If port is in use and force flag is set, kill processes using it
    if check_port_in_use(port):
        if args.force:
            logger.warning(f"Port {port} is in use. Attempting to kill processes...")
            kill_processes_on_port(port)
            time.sleep(1)  # Give it a moment to release the port
        else:
            # Try to find an available port
            new_port = find_available_port(port + 1, port + 100)
            if new_port > 0:
                logger.info(f"Port {port} is in use. Using port {new_port} instead.")
                port = new_port
            else:
                logger.error(f"Port {port} is in use and no available ports found in range.")
                return False, None, -1
    
    logger.info(f"Starting API server on port {port}...")
    
    # Set environment variables for port
    os.environ["PORT"] = str(port)
    os.environ["SOLANA_MCP_PORT"] = str(port)
    logger.info(f"Environment variables set: PORT={port}, SOLANA_MCP_PORT={port}")
    
    # Choose the right server script
    if args.server_type == "fastapi":
        cmd = ["python", "solana_mcp/scripts/run_api_server.py", "--port", str(port)]
        # Add port to command line rather than just environment
    elif args.server_type == "fastmcp":
        cmd = ["python", "-m", "solana_mcp.server", "--port", str(port)]
        # Add port to command line rather than just environment
    else:
        cmd = ["python", "-m", "solana_mcp.main", "--port", str(port)]
        # Add port to command line rather than just environment
    
    # Set environment variables for Solana RPC if provided
    if args.rpc_url:
        os.environ["SOLANA_RPC_URL"] = args.rpc_url
        logger.info(f"Using custom RPC URL: {args.rpc_url}")
    
    # Start the server process
    server_process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE if not args.show_server_output else None,
        stderr=subprocess.STDOUT if not args.show_server_output else None,
        text=True,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Add to the list of server processes to be cleaned up on exit
    server_processes.append(server_process)
    
    # Wait for server to start
    logger.info("Waiting for server to start...")
    wait_time = args.server_wait_time
    time.sleep(wait_time)
    
    # Check if server is still running
    if server_process.poll() is not None:
        # Process exited
        if not args.show_server_output:
            stdout, _ = server_process.communicate()
            logger.error(f"Server failed to start:\n{stdout}")
        return False, server_process, -1
    
    # Simple check to see if the server is responsive
    try:
        curl_cmd = ["curl", "-s", f"http://localhost:{port}/health"]
        health_result = subprocess.run(curl_cmd, capture_output=True, text=True)
        if health_result.returncode != 0 or "healthy" not in health_result.stdout.lower():
            logger.warning("Server health check failed, but process is running.")
        else:
            logger.info("Server health check passed.")
    except Exception as e:
        logger.warning(f"Error checking server health: {e}")
    
    logger.info(f"Server started successfully on port {port}")
    return True, server_process, port

def run_api_tests(args: argparse.Namespace, port: int) -> bool:
    """Run API tests against a running server.
    
    Args:
        args: Command line arguments
        port: Port the server is running on
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info(f"Running API tests against server on port {port}...")
    
    # Run the API test script
    if args.api_test_path:
        test_script = args.api_test_path
    else:
        test_script = "solana_mcp.scripts.test_token_risk_apis"
    
    # Set environment variables for test configuration
    os.environ["API_TEST_URL"] = f"http://localhost:{port}"
    
    # Build command
    cmd = ["python", "-m", test_script]
    
    # Add any extra arguments if needed
    if args.api_test_args:
        cmd.extend(args.api_test_args.split())
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Note: we're considering this a success even if the API tests fail
    # if testing not-yet-implemented routes
    if result.returncode != 0:
        if args.ignore_api_failures:
            logger.warning("API tests returned non-zero code but ignoring as requested.")
            return True
        else:
            logger.error("❌ API tests failed!")
            return False
    
    logger.info("✅ API tests passed!")
    return True

# ==================================================
# Direct Analyzer Tests
# ==================================================

def run_direct_analyzer_tests(args: argparse.Namespace) -> bool:
    """Run direct token analyzer tests without API dependency.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info("Running direct analyzer tests...")
    
    # Set up the test token - use provided token or default
    test_token = args.test_token or "ENxauXrtBtnFH1aJAFYZnxVVM4rGLkXJi3bUhT7tpump"
    
    # Build the Python script with all imports
    python_script = f"""
import asyncio
import os
import json
import sys
from solana_mcp.token_risk_analyzer import TokenRiskAnalyzer
from solana_mcp.solana_client import get_solana_client

# Set RPC URL if provided
"""
    
    # Add RPC URL if provided
    if args.rpc_url:
        python_script += f'os.environ["SOLANA_RPC_URL"] = "{args.rpc_url}"\n'
        python_script += f'print("Using custom RPC endpoint: {args.rpc_url}")\n'
    
    # Add test token info
    python_script += f"""
# Test token details
TEST_TOKEN_MINT = "{test_token}"
print(f"Analyzing token: {{TEST_TOKEN_MINT}}")

async def test_analyzer():
    try:
        # Connect to Solana
        async with get_solana_client() as client:
            analyzer = TokenRiskAnalyzer(client)
            
            try:
                # Analyze the token
                result = await analyzer.analyze_token_risks(TEST_TOKEN_MINT)
                
                # Print basic info
                print(f"\\n=== TEST PASSED: Direct Analyzer Test ===")
                print(f"Token: {{result.get('name', 'Unknown')}} ({{result.get('symbol', 'UNKNOWN')}})")
                print(f"Risk Level: {{result.get('risk_level', 'Unknown')}}")
                print(f"Risk Score: {{result.get('overall_risk_score', 0)}}")
                
                # Print risk breakdowns
                print("\\nRisk Breakdown:")
                print(f"- Supply Risk: {{result.get('supply_risk_score', 0)}}")
                print(f"- Authority Risk: {{result.get('authority_risk_score', 0)}}")
                print(f"- Liquidity Risk: {{result.get('liquidity_risk_score', 0)}}")
                print(f"- Ownership Risk: {{result.get('ownership_risk_score', 0)}}")
                
                # Print flags if any
                if result.get('flags'):
                    print("\\nRisk Flags:")
                    for flag in result.get('flags', []):
                        print(f"- {{flag}}")
                    
                # Print token category
                print(f"\\nToken Category: {{result.get('token_category', 'Unknown')}}")
                
                # Print full JSON for reference
                print("\\nFull Analysis:")
                print(json.dumps(result, indent=2))
                
                return True
            except Exception as e:
                print(f"❌ Analyzer test failed: {{str(e)}}")
                import traceback
                traceback.print_exc()
                return False
    except Exception as e:
        print(f"❌ Error initializing client: {{str(e)}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_analyzer())
    sys.exit(0 if success else 1)
"""
    
    # Run the Python script
    cmd = ["python", "-c", python_script]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error("❌ Direct analyzer tests failed!")
        return False
    
    logger.info("✅ Direct analyzer tests passed!")
    return True

# ==================================================
# Mock Tests
# ==================================================

def run_mock_tests(args: argparse.Namespace) -> bool:
    """Run tests with mock data and no RPC dependency.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info("Running mock tests (no RPC dependency)...")
    
    # Build pytest command targeting mock tests
    cmd = ["python", "-m", "pytest", "solana_mcp/tests", "-m", "mock", "-v"]
    
    # Show print output if requested
    if args.show_output:
        cmd.append("-s")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error("❌ Mock tests failed!")
        return False
    
    logger.info("✅ Mock tests passed!")
    return True

# ==================================================
# Integration Tests
# ==================================================

def run_integration_tests(args: argparse.Namespace) -> bool:
    """Run integration tests that test multiple components together.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info("Running integration tests...")
    
    # Start API server if not already running
    if not hasattr(args, 'integration_server_port') or args.integration_server_port is None:
        server_success, server_process, port = start_api_server(args)
        if not server_success:
            logger.error("Failed to start server for integration tests.")
            return False
        args.integration_server_port = port
    else:
        port = args.integration_server_port
        logger.info(f"Using existing server on port {port} for integration tests.")
    
    # Build test command
    cmd = ["python", "-m", "pytest", "solana_mcp/tests", "-m", "integration", "-v"]
    
    # Pass server details to tests
    os.environ["INTEGRATION_TEST_SERVER_URL"] = f"http://localhost:{port}"
    
    # Show print output if requested
    if args.show_output:
        cmd.append("-s")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error("❌ Integration tests failed!")
        return False
    
    logger.info("✅ Integration tests passed!")
    return True

# ==================================================
# Main Test Runner
# ==================================================

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Solana MCP Consolidated Test Runner")
    
    # Test selection options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--analyzer", action="store_true", help="Run direct analyzer tests")
    parser.add_argument("--mock", action="store_true", help="Run mock tests (no RPC dependency)")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    
    # Server options
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server (default: 8000)")
    parser.add_argument("--server-type", choices=["fastapi", "fastmcp", "main"], default="fastapi", 
                        help="Type of server to start (default: fastapi)")
    parser.add_argument("--force", action="store_true", help="Force kill processes using the specified port")
    parser.add_argument("--server-wait-time", type=int, default=5, 
                        help="Seconds to wait for server to start (default: 5)")
    parser.add_argument("--show-server-output", action="store_true", 
                        help="Show server output in console")
    
    # RPC options
    parser.add_argument("--rpc-url", type=str,
                        help="Custom Solana RPC URL (default: uses environment or config)")
    
    # Test options
    parser.add_argument("--test-path", type=str, help="Path to specific test file or directory")
    parser.add_argument("--api-test-path", type=str, help="Path to API test module")
    parser.add_argument("--test-token", type=str, help="Token mint address to test")
    parser.add_argument("--markers", type=str, help="Pytest markers to filter tests")
    parser.add_argument("--pytest-args", type=str, help="Additional arguments to pass to pytest")
    parser.add_argument("--api-test-args", type=str, help="Additional arguments to pass to API tests")
    parser.add_argument("--ignore-api-failures", action="store_true", 
                        help="Continue with success status even if API tests fail")
    parser.add_argument("--show-output", action="store_true", help="Show test output in console")
    
    # Performance options
    parser.add_argument("--parallel", action="store_true", 
                        help="Run tests in parallel where possible")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default to run all tests if no specific test type is selected
    if not any([args.all, args.unit, args.api, args.analyzer, args.mock, args.integration]):
        args.all = True
    
    # Store results
    results = {}
    success = True
    
    try:
        # Determine which tests to run
        run_all = args.all
        
        # Unit Tests
        if run_all or args.unit:
            unit_success = run_unit_tests(args)
            results["Unit Tests"] = "✅ Passed" if unit_success else "❌ Failed"
            success = success and unit_success
        
        # API Tests (need server)
        if run_all or args.api:
            server_success, server_process, port = start_api_server(args)
            if server_success:
                api_success = run_api_tests(args, port)
                results["API Tests"] = "✅ Passed" if api_success else "❌ Failed"
                success = success and api_success
            else:
                results["API Tests"] = "❌ Failed (server start failed)"
                success = False
        
        # Direct Analyzer Tests
        if run_all or args.analyzer:
            analyzer_success = run_direct_analyzer_tests(args)
            results["Analyzer Tests"] = "✅ Passed" if analyzer_success else "❌ Failed"
            success = success and analyzer_success
        
        # Mock Tests
        if run_all or args.mock:
            mock_success = run_mock_tests(args)
            results["Mock Tests"] = "✅ Passed" if mock_success else "❌ Failed"
            success = success and mock_success
        
        # Integration Tests
        if run_all or args.integration:
            integration_success = run_integration_tests(args)
            results["Integration Tests"] = "✅ Passed" if integration_success else "❌ Failed"
            success = success and integration_success
        
        # Print summary
        logger.info("\n===== Test Summary =====")
        for test_name, result in results.items():
            logger.info(f"{test_name}: {result}")
        
        if success:
            logger.info("\n✅ All tests completed successfully!")
        else:
            logger.info("\n❌ Some tests failed. Check the output above for details.")
        
        return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user.")
        return 1
    finally:
        cleanup_resources()

if __name__ == "__main__":
    sys.exit(main()) 