# API Test Plan for Solana MCP

This document outlines the test coverage needed for each API endpoint in the Solana MCP project.

## Test Structure

We'll organize our tests into the following categories:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **API Tests**: Test the API endpoints directly

## API Endpoints to Test

Based on our routes directory, we need to test the following endpoints:

### 1. Account Endpoints (`accounts.py`)
- `GET /api/accounts/{address}` - Get account info
- `GET /api/accounts/{address}/balance` - Get account balance
- `GET /api/accounts/{address}/tokens` - Get tokens owned by account
- `GET /api/accounts/{address}/is-executable` - Check if account is executable

### 2. Token Endpoints (`tokens.py`)
- `GET /api/tokens` - List tokens
- `GET /api/tokens/{address}` - Get token metadata
- `GET /api/tokens/{address}/supply` - Get token supply
- `GET /api/tokens/{address}/largest_accounts` - Get largest holders
- `POST /api/tokens/batch` - Batch get token metadata

### 3. Transaction Endpoints (`transactions.py`)
- `GET /api/transactions/{signature}` - Get transaction details
- `GET /api/transactions/batch` - Batch get transactions
- `GET /api/accounts/{address}/transactions` - Get transactions for account

### 4. Analysis Endpoints (`analysis.py`)
- `GET /api/analysis/account/{address}` - Account analysis
- `GET /api/analysis/token/{mint_address}` - Token analysis
- `GET /api/analysis/market-overview` - Market overview

### 5. NLP Endpoints (`nlp.py`)
- `POST /api/nlp/process` - Process natural language query
- `GET /api/nlp/suggest` - Get query suggestions

### 6. Task Endpoints (`tasks.py`)
- `GET /api/tasks` - List background tasks
- `GET /api/tasks/{task_id}` - Get task status
- `DELETE /api/tasks/{task_id}` - Cancel task

## Test Scenarios

### Account API Tests
1. Get valid account info
2. Get non-existent account info (should return appropriate error)
3. Get account balance with valid address
4. Get tokens owned by account
5. Check executable status for program and non-program accounts

### Token API Tests
1. List tokens with pagination
2. Get valid token metadata
3. Get non-existent token metadata (should return appropriate error)
4. Get token supply information
5. Get largest token accounts
6. Batch get token metadata for multiple addresses

### Transaction API Tests
1. Get transaction details for valid signature
2. Get transaction details for invalid signature (should return appropriate error)
3. Get paginated transactions for an account
4. Batch get multiple transactions

### Analysis API Tests
1. Get account analysis for a wallet
2. Get token analysis data
3. Get market overview data
4. Test with various analysis parameters

### NLP API Tests
1. Process various query types (token info, price, account info)
2. Get query suggestions based on input text
3. Test error handling for malformed queries

### Task API Tests
1. List all tasks
2. Get status of specific task
3. Cancel a running task
4. Test task completion and error handling

## Mock Dependencies

For testing, we'll use the following mocks:
- `mock_solana_client`: Mock Solana RPC client
- `mock_cache_service`: Mock caching service
- Service mocks for each service class

## Implementation Plan

1. Create fixture files with test data for each endpoint
2. Implement unit tests for service functions
3. Implement API tests using FastAPI TestClient
4. Set up integration tests with mocked dependencies
5. Add test cases for error scenarios

## Test Commands

To run the tests, use:
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run tests for specific module
pytest tests/unit/test_token_service.py

# Run with coverage report
pytest --cov=solana_mcp
``` 