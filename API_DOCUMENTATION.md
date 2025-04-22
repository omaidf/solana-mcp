# Solana MCP Server API Documentation

This document provides detailed information about the Solana MCP Server API endpoints, including request parameters, response formats, and example usage.

Base URL: `http://localhost:8000` (or your deployed server address)

## Authentication

Currently, the API does not require authentication.

## Error Handling

All endpoints follow a consistent error response format:

```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid parameters (including invalid Solana public keys)
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error
- `501 Not Implemented`: Feature not implemented

## Endpoints

### Token Analysis

#### Analyze Token

Get comprehensive analysis for a specific token.

**Endpoint:** `GET /token-analysis/analyze/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/analyze/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "token_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "token_name": "USD Coin",
  "token_symbol": "USDC",
  "decimals": 6,
  "total_supply": 5000000000000,
  "circulation_supply": 4500000000000,
  "current_price_usd": 1.0,
  "launch_date": "2020-12-15T00:00:00",
  "age_days": 1095,
  "owner_can_mint": true,
  "owner_can_freeze": true,
  "total_holders": 10250,
  "largest_holder_percentage": 12.3,
  "last_updated": "2024-01-15T12:34:56"
}
```

#### Token Metadata

Get metadata information for a token.

**Endpoint:** `GET /token-analysis/metadata/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/metadata/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "name": "USD Coin",
  "symbol": "USDC",
  "uri": "https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v/logo.png",
  "update_authority": "BZsGMwGpLvfgWPaJxpJXUXKXKxuA5d5Kx2F7NG7BXzqL"
}
```

#### Token Supply

Get supply information for a token.

**Endpoint:** `GET /token-analysis/supply/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/supply/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "value": {
    "amount": "5000000000000",
    "decimals": 6,
    "uiAmount": 5000000.0,
    "uiAmountString": "5000000"
  }
}
```

#### Token Price

Get current price information for a token.

**Endpoint:** `GET /token-analysis/price/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/price/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "price": 1.0,
  "id": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "mintSymbol": "USDC",
  "vsToken": "USDC",
  "vsTokenSymbol": "USDC"
}
```

#### Token Holders

Get information about token holders.

**Endpoint:** `GET /token-analysis/holders/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/holders/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "total_holders": 15,
  "largest_holder_percentage": 25.7,
  "accounts": [
    {
      "address": "FG4Y3yX4AAchp1HvNZ7LfzFTew5pRpfwAzCiJ1Z9gwBN",
      "amount": "1285000000",
      "decimals": 6,
      "uiAmount": 1285.0
    },
    {
      "address": "6QuXb6mB6WmRASP2y8AavXh6aabBXEH5ZzrSH5xRrgSm",
      "amount": "985000000",
      "decimals": 6,
      "uiAmount": 985.0
    }
  ]
}
```

#### Token Age

Get age information for a token.

**Endpoint:** `GET /token-analysis/age/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/age/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "launch_date": "2020-12-15T00:00:00",
  "age_days": 1095
}
```

#### Token Authority

Get authority information for a token.

**Endpoint:** `GET /token-analysis/authority/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/authority/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "mint_authority": "BZsGMwGpLvfgWPaJxpJXUXKXKxuA5d5Kx2F7NG7BXzqL",
  "freeze_authority": "BZsGMwGpLvfgWPaJxpJXUXKXKxuA5d5Kx2F7NG7BXzqL",
  "has_mint_authority": true,
  "has_freeze_authority": true
}
```

#### Token Holders Count

Get the count of token holders.

**Endpoint:** `GET /token-analysis/holders-count/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/holders-count/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "count": 10250
}
```

## Rate Limiting

Currently, there are no rate limits implemented, but excessive usage may be throttled in the future to ensure service availability.

## Webhook Notifications

Webhook notifications for token events are not currently implemented, but may be added in future versions.

## Client Libraries

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

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function getTokenAnalysis(mintAddress) {
  try {
    const response = await axios.get(`http://localhost:8000/token-analysis/analyze/${mintAddress}`);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get token data: ${error.response?.data?.detail || error.message}`);
  }
}

// Example usage
async function main() {
  try {
    const tokenData = await getTokenAnalysis('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v');
    console.log(`Token name: ${tokenData.token_name}`);
    console.log(`Current price: $${tokenData.current_price_usd}`);
  } catch (error) {
    console.error(error.message);
  }
}

main();
``` 