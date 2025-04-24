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
  "whale_count": 5,
  "whale_percentage": 0.05,
  "whale_holdings_percentage": 45.2,
  "whale_holdings_usd_total": 225000000.0,
  "fresh_wallet_count": 125,
  "fresh_wallet_percentage": 1.22,
  "fresh_wallet_holdings_percentage": 3.5,
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

#### Whale Holders

Get token holders with balances over the specified USD threshold (whales).

**Endpoint:** `GET /token-analysis/whales/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address
- `threshold_usd` (query parameter, optional): The minimum USD value to consider a holder a whale (default: $50,000)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/whales/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?threshold_usd=100000"
```

**Response:**
```json
{
  "token_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "token_name": "USD Coin",
  "token_symbol": "USDC",
  "threshold_usd": 100000.0,
  "total_holders_analyzed": 100,
  "whale_count": 3,
  "whale_percentage": 3.0,
  "whale_holders": [
    {
      "address": "FG4Y3yX4AAchp1HvNZ7LfzFTew5pRpfwAzCiJ1Z9gwBN",
      "token_balance": 500000.0,
      "usd_value": 500000.0,
      "percentage_of_supply": 10.0
    },
    {
      "address": "6QuXb6mB6WmRASP2y8AavXh6aabBXEH5ZzrSH5xRrgSm",
      "token_balance": 300000.0,
      "usd_value": 300000.0,
      "percentage_of_supply": 6.0
    },
    {
      "address": "9vgYWRnxJ7zK8AzFKuEFfRv3mRK3FJPspZKXAa3XNUJs",
      "token_balance": 150000.0,
      "usd_value": 150000.0,
      "percentage_of_supply": 3.0
    }
  ],
  "whale_holdings_percentage": 19.0,
  "whale_holdings_usd_total": 950000.0,
  "last_updated": "2024-01-15T12:34:56"
}
```

#### Fresh Wallets

Get token holders that only hold this specific token (new wallets with no history of other tokens).

**Endpoint:** `GET /token-analysis/fresh-wallets/{mint}`

**Parameters:**
- `mint` (path parameter, required): The Solana token mint address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/token-analysis/fresh-wallets/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
```

**Response:**
```json
{
  "token_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "token_name": "USD Coin",
  "token_symbol": "USDC",
  "total_holders_analyzed": 100,
  "fresh_wallet_count": 12,
  "fresh_wallet_percentage": 12.0,
  "fresh_wallets": [
    {
      "wallet_address": "FG4Y3yX4AAchp1HvNZ7LfzFTew5pRpfwAzCiJ1Z9gwBN",
      "token_account": "7quA1MV2rnHSDgfkfGaESQ1VjgQTqY9k9QUTaAaLhHrA",
      "token_balance": 5000.0,
      "percentage_of_supply": 0.1
    },
    {
      "wallet_address": "6QuXb6mB6WmRASP2y8AavXh6aabBXEH5ZzrSH5xRrgSm",
      "token_account": "CdKPtCb5fBRaGFS4bJgytfReeHuFyhpe9YUyWHPnEWZG",
      "token_balance": 2500.0,
      "percentage_of_supply": 0.05
    }
  ],
  "fresh_wallet_holdings_percentage": 0.15,
  "fresh_wallet_holdings_token_total": 7500.0,
  "last_updated": "2024-01-15T12:34:56"
}
```

#### Semantic Query

Process a natural language query about tokens to get data based on what's being asked.

**Endpoint:** `POST /token-analysis/query`

**Parameters:**
- `query` (string parameter, required): The natural language query text (must include a valid token address)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/token-analysis/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "show me the whales holding more than $100k of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"}'
```

**Response:**
```json
{
  "query": "show me the whales holding more than $100k of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
  "normalized_query": "show me the whales holding more than $100k of epjfwdd5aufqssqem2qn1xzybapC8g4wegkzwyTdt1v",
  "primary_intent": "whales",
  "all_intents": [
    {"intent": "whales", "confidence": 0.8},
    {"intent": "holders", "confidence": 0.4}
  ],
  "confidence": 0.8,
  "extracted_parameters": {
    "token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "threshold": 100000.0
  },
  "data": {
    "token_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "token_name": "USD Coin",
    "token_symbol": "USDC",
    "threshold_usd": 100000.0,
    "total_holders_analyzed": 100,
    "whale_count": 3,
    "whale_percentage": 3.0,
    "whale_holders": [
      {
        "address": "FG4Y3yX4AAchp1HvNZ7LfzFTew5pRpfwAzCiJ1Z9gwBN",
        "token_balance": 500000.0,
        "usd_value": 500000.0,
        "percentage_of_supply": 10.0
      },
      {
        "address": "6QuXb6mB6WmRASP2y8AavXh6aabBXEH5ZzrSH5xRrgSm",
        "token_balance": 300000.0,
        "usd_value": 300000.0,
        "percentage_of_supply": 6.0
      },
      {
        "address": "9vgYWRnxJ7zK8AzFKuEFfRv3mRK3FJPspZKXAa3XNUJs",
        "token_balance": 150000.0,
        "usd_value": 150000.0,
        "percentage_of_supply": 3.0
      }
    ],
    "whale_holdings_percentage": 19.0,
    "whale_holdings_usd_total": 950000.0,
    "last_updated": "2024-01-15T12:34:56"
  },
  "additional_data": {
    "holders": {
      "total_holders": 15,
      "largest_holder_percentage": 25.7
    }
  },
  "error": null
}
```

**Supported Intents:**

The semantic query system can understand various intents:

1. **General Information:** 
   - "Tell me about token EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
   - "Show me the basic info for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

2. **Price Information:**
   - "What is the price of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "How much is token EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v worth?"

3. **Holder Information:**
   - "Who holds EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "Show me the top holders of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

4. **Whale Analysis:**
   - "Who are the whales for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "Show accounts with more than $75k of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

5. **Fresh Wallet Detection:**
   - "Show me fresh wallets for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
   - "Find new accounts only holding EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

6. **Supply Information:**
   - "What's the supply of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "Tell me about the total supply of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

7. **Authority Information:**
   - "Who has mint authority for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "Can EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v be minted?"

8. **Age Information:**
   - "How old is token EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "When was EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v created?"

9. **Risk Assessment:**
   - "Is EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v a risky token?"
   - "What are the risk factors for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
   - "Is EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v safe to buy?"

10. **Token Comparison:**
    - "Compare EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v with another token"
    - "How does EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v stack up?"

**Multi-Intent Queries:**

The system can understand queries with multiple intents. For example:

- "What's the price and whale distribution of EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"
- "Show me risk assessment and fresh wallets for EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

When multiple intents are detected, the system will respond with both the primary and secondary data.

**Natural Language Features:**

- **Threshold Extraction:** Automatically extracts dollar amounts for whale analysis (e.g., "$50k", "100,000 USD", "more than 75k")
- **Contraction Handling:** Understands common contractions like "what's", "who's", "isn't"
- **Stopword Removal:** Filters common stopwords for better intent detection
- **Confidence Scoring:** Provides confidence levels for detected intents

### Liquidity Pool Analysis

#### Analyze Pool

Get comprehensive analysis for a specific liquidity pool.

**Endpoint:** `GET /liquidity-analysis/pool/{pool_address}`

**Parameters:**
- `pool_address` (path parameter, required): The Solana liquidity pool address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/liquidity-analysis/pool/7quA1MV2rnHSDgfkfGaESQ1VjgQTqY9k9QUTaAaLhHrA"
```

**Response:**
```json
{
  "pool_address": "7quA1MV2rnHSDgfkfGaESQ1VjgQTqY9k9QUTaAaLhHrA",
  "protocol": "raydium",
  "pool_data": {
    "tokenA": {
      "mint": "So11111111111111111111111111111111111111112",
      "symbol": "SOL",
      "amount": 1000000
    },
    "tokenB": {
      "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
      "symbol": "USDC",
      "amount": 2000000
    },
    "fees": {
      "fee_tier": 0.0025
    }
  },
  "metrics": {
    "volume_24h": 1000000,
    "volume_7d": 5000000,
    "fees_24h": 2500,
    "fees_7d": 12500,
    "apy": 12.5,
    "price_impact_1000usd": 0.05,
    "liquidity_usd": 2000000
  },
  "last_updated": "2024-01-15T12:34:56"
}
```

#### User Positions

Get liquidity positions for a specific user wallet.

**Endpoint:** `GET /liquidity-analysis/user-positions/{wallet_address}`

**Parameters:**
- `wallet_address` (path parameter, required): The user's wallet address

**Example Request:**
```bash
curl -X GET "http://localhost:8000/liquidity-analysis/user-positions/FG4Y3yX4AAchp1HvNZ7LfzFTew5pRpfwAzCiJ1Z9gwBN"
```

**Response:**
```json
{
  "wallet_address": "FG4Y3yX4AAchp1HvNZ7LfzFTew5pRpfwAzCiJ1Z9gwBN",
  "positions": {
    "raydium": [
      {
        "pool_address": "7quA1MV2rnHSDgfkfGaESQ1VjgQTqY9k9QUTaAaLhHrA",
        "token_a": "SOL",
        "token_b": "USDC",
        "lp_token_amount": 10.5,
        "share_percentage": 0.01,
        "value_usd": 5000,
        "apy": 15.2,
        "rewards": [
          {
            "token": "RAY",
            "amount": 2.5,
            "value_usd": 50
          }
        ]
      }
    ],
    "orca": [
      {
        "pool_address": "9vgYWRnxJ7zK8AzFKuEFfRv3mRK3FJPspZKXAa3XNUJs",
        "token_a": "SOL",
        "token_b": "USDT",
        "lp_token_amount": 20.2,
        "share_percentage": 0.005,
        "value_usd": 2500,
        "apy": 18.7,
        "rewards": [
          {
            "token": "ORCA",
            "amount": 5,
            "value_usd": 75
          }
        ]
      }
    ],
    "other": []
  },
  "total_value_locked": 7500,
  "position_count": 2,
  "last_updated": "2024-01-15T12:34:56"
}
```

#### Top Pools

Get top liquidity pools by Total Value Locked (TVL).

**Endpoint:** `GET /liquidity-analysis/top-pools`

**Parameters:**
- `limit` (query parameter, optional): Maximum number of pools to return (default: 10)
- `protocol` (query parameter, optional): Filter by protocol (e.g., 'raydium', 'orca')

**Example Request:**
```bash
curl -X GET "http://localhost:8000/liquidity-analysis/top-pools?limit=5&protocol=raydium"
```

**Response:**
```json
{
  "pools": [
    {
      "pool_address": "7quA1MV2rnHSDgfkfGaESQ1VjgQTqY9k9QUTaAaLhHrA",
      "protocol": "raydium",
      "token_a": "SOL",
      "token_b": "USDC",
      "tvl": 50000000,
      "volume_24h": 10000000,
      "apy": 12.5
    },
    {
      "pool_address": "CdKPtCb5fBRaGFS4bJgytfReeHuFyhpe9YUyWHPnEWZG",
      "protocol": "raydium",
      "token_a": "RAY",
      "token_b": "USDC",
      "tvl": 25000000,
      "volume_24h": 5000000,
      "apy": 18.7
    }
  ],
  "total_count": 2,
  "total_tvl": 75000000,
  "last_updated": "2024-01-15T12:34:56"
}
```

#### Calculate Impermanent Loss

Calculate impermanent loss for given price changes.

**Endpoint:** `POST /liquidity-analysis/impermanent-loss`

**Request Body:**
```json
{
  "token_a_price_change": 1.5,
  "token_b_price_change": 1.0
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/liquidity-analysis/impermanent-loss" \
  -H "Content-Type: application/json" \
  -d '{"token_a_price_change": 1.5, "token_b_price_change": 1.0}'
```

**Response:**
```json
{
  "token_a_price_change": 1.5,
  "token_b_price_change": 1.0,
  "price_ratio": 1.5,
  "impermanent_loss": -0.0202,
  "percentage_loss": -2.02,
  "explanation": "If you held the tokens instead of providing liquidity, your portfolio would be 2.02% higher"
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