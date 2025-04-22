# Wallet Classifier API Documentation

The Wallet Classifier module provides tools to identify potentially high-risk wallets based on transaction patterns and behavioral analysis. It can detect suspicious activity patterns and calculate an overall risk score for Solana wallets.

## Classes

### WalletClassifier

The main class that analyzes wallet behavior and classifies them according to risk patterns.

#### Constructor

```python
classifier = WalletClassifier(solana_client)
```

- **solana_client**: An instance of `SolanaClient` used to interact with the Solana blockchain.

## Primary Methods

### classify_wallet

Analyze a wallet's transaction history and classify its behavior patterns.

```python
result = await classifier.classify_wallet(wallet_address, request_id=None)
```

#### Parameters

- **wallet_address** (str): The Solana public key of the wallet to analyze.
- **request_id** (str, optional): An ID for tracking the request in logs.

#### Returns

A dictionary containing:

```json
{
  "wallet_address": "...",
  "classifications": {
    "high_velocity": true|false,
    "whale": true|false,
    "new_wallet": true|false,
    "temporary_holder": true|false
  },
  "risk_score": 42,  // 0-100 (higher = riskier)
  "risk_level": "low|medium|high|very_high",
  "stats": {
    "tx_count_total": 123,
    "tx_count_24h": 10,
    "unique_tokens_count": 5,
    "total_value_usd": 15000.0,
    "largest_transfer_usd": 5000.0,
    "age_days": 30.5,
    "avg_hold_time_hours": 48.5,
    "token_in_out_ratio": 0.75
  }
}
```

### batch_classify_wallets

Analyze multiple wallets simultaneously.

```python
results = await classifier.batch_classify_wallets(wallet_addresses, request_id=None)
```

#### Parameters

- **wallet_addresses** (List[str]): List of Solana wallet addresses to analyze.
- **request_id** (str, optional): An ID for tracking the request in logs.

#### Returns

A dictionary containing:

```json
{
  "wallets": {
    "wallet1": { /* classification result */ },
    "wallet2": { /* classification result */ },
    // ...
  },
  "count": 5,
  "risk_summary": {
    "low": 2,
    "medium": 1,
    "high": 1,
    "very_high": 0,
    "error": 1
  }
}
```

## Risk Classification Descriptions

The classifier identifies the following wallet types:

### High Velocity
Wallets with high transaction volume or interaction with many different tokens. May indicate trading bots, arbitrage operations, or suspicious activity.
- Threshold: 50+ transactions in 24h or 20+ unique tokens transferred

### Whale
Wallets holding or transferring large value. Important to monitor as they can influence market movements.
- Threshold: $100,000+ in assets or $50,000+ transfer

### New Wallet
Recently created wallets with limited transaction history. Higher risk due to lack of established patterns.
- Threshold: Less than 7 days old and fewer than 10 transactions

### Temporary Holder
Wallets that quickly transfer tokens after receiving them. May indicate wash trading or money laundering.
- Threshold: Average hold time less than 6 hours and 90%+ of received tokens are sent out

## Risk Score Calculation

The risk score is calculated on a scale of 0-100, where higher scores indicate higher risk:

- **0-29**: Low risk - Normal wallet behavior
- **30-59**: Medium risk - Some suspicious patterns
- **60-84**: High risk - Multiple suspicious patterns
- **85-100**: Very high risk - Strong indicators of suspicious activity

## Example Usage

```python
from solana_mcp.solana_client import SolanaClient
from solana_mcp.wallet_classifier import WalletClassifier

async def analyze_wallet(wallet_address):
    # Initialize the client
    client = SolanaClient(rpc_url="https://api.mainnet-beta.solana.com")
    
    # Create the classifier
    classifier = WalletClassifier(client)
    
    # Analyze a single wallet
    wallet_analysis = await classifier.classify_wallet(wallet_address)
    print(f"Wallet Risk Level: {wallet_analysis['risk_level']}")
    print(f"Risk Score: {wallet_analysis['risk_score']}/100")
    
    # Analyze multiple wallets
    wallet_addresses = ["Address1", "Address2", "Address3"]
    batch_results = await classifier.batch_classify_wallets(wallet_addresses)
    print(f"Risk summary: {batch_results['risk_summary']}") 