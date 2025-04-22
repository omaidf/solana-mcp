# Solana Model Context Protocol (MCP)

A comprehensive Solana blockchain interaction server that implements the Model Context Protocol (MCP), optimized for seamless integration with AI tools and interfaces.

## What is Model Context Protocol?

Model Context Protocol (MCP) provides a standardized way for AI tools and language models to interact with blockchain data. This implementation allows AI agents to:

- Query Solana blockchain data using natural language
- Access structured token and account information
- Maintain context across multiple interactions
- Perform semantic searches across blockchain transactions
- Generate human-readable explanations of complex blockchain data

## Why MCP for AI Integration?

MCP creates a bridge between AI agents and blockchain data, enabling:

- **Contextual Understanding**: AI models can maintain conversation history and build context about tokens and accounts
- **Semantic Queries**: Support for natural language processing to translate user queries into blockchain operations
- **Structured Responses**: Data is returned in standardized formats optimized for AI consumption
- **Enhanced Explanations**: Complex blockchain concepts are explained in accessible language

## Features

- **Natural Language Processing**: Query blockchain data using everyday language
- **Token Analysis**: Comprehensive token information and metrics
- **Semantic Search**: Find transactions and activities based on meaning, not just exact matches
- **Context Awareness**: Server maintains session state and understands entity relationships
- **Solana RPC Integration**: Full access to Solana blockchain capabilities
- **RESTful API**: Easy integration with existing systems
- **Docker Support**: Simple deployment with containerization

## Quick Start with Docker

### Option 1: Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/omaidf/solana-mcp.git
cd solana-mcp

# Build and start the container
docker-compose up -d
```

### Option 2: Using Docker directly

```bash
# Build the Docker image
docker build -t solana-mcp .

# Run the container
docker run -p 8000:8000 solana-mcp
```

## Environment Variables

Customize the server by setting the following environment variables:

```
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_COMMITMENT=confirmed
SOLANA_TIMEOUT=30
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json
ENVIRONMENT=production
METADATA_CACHE_SIZE=100
METADATA_CACHE_TTL=300
PRICE_CACHE_SIZE=500
PRICE_CACHE_TTL=60
```

## API Endpoints

### Core MCP Endpoints

- `GET /health` - Health check endpoint
- `GET /version` - Get API version information

### Solana Token Analysis

- `GET /token-analysis/analyze/{mint}` - Get comprehensive token analysis
- `GET /token-analysis/metadata/{mint}` - Get token metadata
- `GET /token-analysis/supply/{mint}` - Get token supply information
- `GET /token-analysis/price/{mint}` - Get token price information
- `GET /token-analysis/holders/{mint}` - Get token holder information

### Natural Language Queries

- `POST /nlp/query` - Submit natural language queries about the Solana blockchain

See `API_DOCUMENTATION.md` for complete API documentation.

## Development

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running locally

```bash
python -m solana_mcp.main
```

The server will be available at http://localhost:8000.

## MCP Integration Examples

### Python

```python
import httpx
import asyncio

async def get_token_analysis(mint_address):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:8000/token-analysis/analyze/{mint_address}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.json().get('detail')}")

# Example usage
async def main():
    try:
        token_data = await get_token_analysis("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        print(f"Token name: {token_data['token_name']}")
        print(f"Current price: ${token_data['current_price_usd']}")
    except Exception as e:
        print(f"Failed to get token data: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## License

See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Created by [omaidf](https://github.com/omaidf) 