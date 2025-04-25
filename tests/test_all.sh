#!/bin/bash
# Script to run all Solana MCP tests

# Function to print color messages
print_colored() {
  local color=$1
  local message=$2
  
  case $color in
    "green") echo -e "\033[0;32m$message\033[0m" ;;
    "red") echo -e "\033[0;31m$message\033[0m" ;;
    "yellow") echo -e "\033[0;33m$message\033[0m" ;;
    "blue") echo -e "\033[0;34m$message\033[0m" ;;
    *) echo "$message" ;;
  esac
}

# Parse command line arguments
SKIP_SERVER_CHECK=0
API_URL="http://localhost:8000"
SKIP_UNIT=0
SKIP_INTEGRATION=0
SKIP_API=0
SKIP_CLIENT=0
SKIP_REST=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-server-check)
      SKIP_SERVER_CHECK=1
      shift
      ;;
    --api-url)
      API_URL="$2"
      shift
      shift
      ;;
    --skip-unit)
      SKIP_UNIT=1
      shift
      ;;
    --skip-integration)
      SKIP_INTEGRATION=1
      shift
      ;;
    --skip-api)
      SKIP_API=1
      shift
      ;;
    --skip-client)
      SKIP_CLIENT=1
      shift
      ;;
    --skip-rest)
      SKIP_REST=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print banner
print_colored "blue" "========================================"
print_colored "blue" "    SOLANA MCP TESTING SUITE"
print_colored "blue" "========================================"
echo ""

# Check if the server is running
if [ $SKIP_SERVER_CHECK -eq 0 ]; then
  print_colored "yellow" "Checking if Solana MCP server is running..."
  if python ensure_server.py --url "$API_URL" --no-start; then
    print_colored "green" "Server is running. Proceeding with tests."
  else
    print_colored "red" "Error: Server is not running at $API_URL"
    print_colored "yellow" "Would you like to start the server? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
      # Start server in background
      print_colored "yellow" "Starting server..."
      python ensure_server.py --url "$API_URL" &
      SERVER_PID=$!
      sleep 5  # Give some time for the server to start
      # Check if server started successfully
      if ! python ensure_server.py --url "$API_URL" --no-start; then
        print_colored "red" "Error: Failed to start server"
        exit 1
      fi
      print_colored "green" "Server started successfully."
    else
      print_colored "red" "Aborting tests since server is not running."
      exit 1
    fi
  fi
fi

# Build arguments for run_all_tests.py
ARGS=()
if [ $SKIP_UNIT -eq 1 ]; then
  ARGS+=("--skip-unit")
fi
if [ $SKIP_INTEGRATION -eq 1 ]; then
  ARGS+=("--skip-integration")
fi
if [ $SKIP_API -eq 1 ]; then
  ARGS+=("--skip-api")
fi
if [ $SKIP_CLIENT -eq 1 ]; then
  ARGS+=("--skip-client")
fi
if [ $SKIP_REST -eq 1 ]; then
  ARGS+=("--skip-rest")
fi
ARGS+=("--api-url" "$API_URL")

# Run the tests
print_colored "yellow" "Running tests..."
python run_all_tests.py "${ARGS[@]}"
TEST_RESULT=$?

# Print final result
if [ $TEST_RESULT -eq 0 ]; then
  print_colored "green" "✓ All tests completed successfully!"
else
  print_colored "red" "✗ Some tests failed."
fi

# Stop the server if we started it
if [ -n "$SERVER_PID" ]; then
  print_colored "yellow" "Stopping the server..."
  kill $SERVER_PID
  print_colored "green" "Server stopped."
fi

exit $TEST_RESULT 