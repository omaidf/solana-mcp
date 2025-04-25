# Solana MCP

Solana Monitoring, Caching, and Processing library for interacting with the Solana blockchain.

## Overview

Solana MCP is a Python library that provides a robust foundation for building Solana blockchain applications. It includes:

- Standardized error handling
- Dependency injection for service management
- Configuration management
- Transaction parsing and analysis
- RPC service integration

## Architecture

The library is built around a modern, extensible architecture designed for reliability and maintainability:

### Core Components

1. **Dependency Injection System**
   - Located in `solana_mcp/utils/dependency_injection.py`
   - Manages service instances across the application
   - Supports singleton and transient service lifetimes
   - Provides decorators for automatic dependency injection

2. **Error Handling System**
   - Located in `solana_mcp/utils/error_handling.py`
   - Standardized exception hierarchy with error codes
   - Decorators for common error handling patterns
   - Retry mechanisms for transient failures
   - Detailed error information with context

3. **Configuration Management**
   - Located in `solana_mcp/utils/config.py`
   - Environment variable-based configuration
   - Settings validation
   - Typed configuration objects

4. **RPC Service**
   - Located in `solana_mcp/services/rpc_service.py`
   - Handles communication with Solana blockchain
   - Automatic retry and error mapping
   - Connection management

5. **Transaction Client**
   - Located in `solana_mcp/clients/transaction_client.py`
   - High-level interface for transaction operations
   - Transaction parsing and analysis

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/solana-mcp.git
cd solana-mcp

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Configuration

Solana MCP can be configured using environment variables:

```bash
# Solana RPC endpoint
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"

# Connection settings
export SOLANA_REQUEST_TIMEOUT=30.0
export SOLANA_MAX_RETRIES=3
export SOLANA_RETRY_DELAY=1.0

# Logging
export LOG_LEVEL=INFO
```

### Basic Usage

Here's a simple example of fetching a transaction:

```python
import asyncio
from solana_mcp import initialize_application
from solana_mcp.clients.transaction_client import TransactionClient

async def get_transaction(signature):
    # Initialize the application
    initialize_application()
    
    # Create transaction client (dependencies injected automatically)
    tx_client = TransactionClient()
    
    # Get transaction
    tx = await tx_client.get_transaction(signature)
    
    # Print basic info
    print(f"Transaction: {tx.signature}")
    print(f"Status: {'Success' if tx.success else 'Failed'}")
    print(f"Fee: {tx.fee} lamports")
    
    # Handle SOL transfers
    for transfer in tx.sol_transfers:
        amount_sol = transfer.amount / 1_000_000_000  # Convert lamports to SOL
        print(f"SOL Transfer: {transfer.from_account} -> {transfer.to_account}: {amount_sol} SOL")

# Run example
asyncio.run(get_transaction("5hrqyQAaS4KYmzvGjD6mTW5jFoF1MQz4nJeQYuCsQAQHUxzHXXQbHFwHbYg6XpLQJPwCpNFCT8NQ89w2q4DTvJZa"))
```

See the `examples` directory for more detailed examples.

## Dependency Injection

The dependency injection system simplifies service management and testing:

```python
from solana_mcp.utils.dependency_injection import inject_by_type, ServiceProvider

# Register a service
provider = ServiceProvider.get_instance()
provider.register_singleton(DatabaseService, DatabaseService())

# Use injection in a class
class UserRepository:
    @inject_by_type
    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service
        
    async def get_user(self, user_id):
        return await self.db_service.query("SELECT * FROM users WHERE id = ?", user_id)
```

## Error Handling

The error handling system provides standardized exceptions and decorators:

```python
from solana_mcp.utils.error_handling import (
    handle_async_exceptions,
    TransactionError,
    ValidationError
)

@handle_async_exceptions(
    (ValueError, ValidationError),
    (ConnectionError, NetworkError),
    log_level=logging.WARNING
)
async def process_data(data_id):
    # This function will have standardized error handling
    if not data_id:
        raise ValueError("Data ID is required")
    
    # Process data...
    return result
```

## Examples

Check the `examples` directory for sample scripts:

- `get_transaction.py` - Demonstrates fetching and parsing a transaction
- `monitor_account.py` - Shows how to monitor an account for changes

## Development

### Project Structure

```
solana-mcp/
├── examples/                  # Example scripts
├── solana_mcp/                # Main package
│   ├── __init__.py            # Package initialization
│   ├── clients/               # Client classes for specific operations
│   ├── models/                # Data models
│   ├── services/              # Service implementations
│   └── utils/                 # Utility modules
│       ├── config.py          # Configuration management
│       ├── dependency_injection.py # Dependency injection system
│       └── error_handling.py  # Error handling utilities
├── tests/                     # Test suite
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

### Testing

Run the test suite using pytest:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 