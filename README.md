# Solana Model Context Protocol (MCP) Server

A Python-based server implementing the Model Context Protocol for the Solana blockchain. This server provides an interface for interacting with Solana blockchain data and models.

## Features

- Real-time Solana blockchain data processing
- Model Context Protocol implementation
- RESTful API endpoints for blockchain interaction
- WebSocket support for real-time updates

## Setup

### Standard Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```
   cp .env.example .env
   ```
4. Edit `.env` file with your Solana RPC node details

### Docker Setup

1. Build the Docker image:
   ```
   docker build -t solana-mcp-server .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 --env-file .env solana-mcp-server
   ```

## Running the Server

### Development mode:
```
python main.py
```

### Production mode:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker mode:
```
docker run -p 8000:8000 --env-file .env solana-mcp-server
```

## API Documentation

Once the server is running, access API documentation at:
```
http://localhost:8000/docs
```

## Environment Variables

The following environment variables can be configured:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Enable debug mode (default: False)
- `LOG_LEVEL`: Logging level (default: INFO)
- `CORS_ORIGINS`: Comma-separated list of allowed CORS origins (default: *)
- `RELOAD`: Enable hot reload for development (default: False)

## Enhanced Solana Analytics API

The server provides enhanced Solana analytics capabilities through the following endpoints:

### Token Information
```
POST /api/analyzer/token
```
Get detailed information about a token including price, market cap, and supply data.

### Whale Detection
```
POST /api/analyzer/whales
```
Identify large holders ("whales") of specific tokens with configurable thresholds.

### Enhanced Account Analysis
```
POST /api/analyzer/account
```
Get detailed parsed account information with additional context.

### Token Accounts with Pricing
```
POST /api/analyzer/token-accounts
```
Get token accounts for an address with enriched pricing and valuation data. 