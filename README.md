# Solana MCP Server

A comprehensive Solana token analysis server with focus on pumpfun tokens.

## Features

- Token analysis
- Solana RPC client integration
- RESTful API with FastAPI

## Quick Start with Docker

### Option 1: Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/yourusername/solana-mcp-server.git
cd solana-mcp-server

# Build and start the container
docker-compose up -d
```

### Option 2: Using Docker directly

```bash
# Build the Docker image
docker build -t solana-mcp-server .

# Run the container
docker run -p 8000:8000 solana-mcp-server
```

## Environment Variables

You can customize the server by setting the following environment variables:

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

- `GET /health` - Health check endpoint
- `GET /version` - Get API version information
- See API_DOCUMENTATION.md for complete API documentation

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