# Solana MCP API Documentation

This directory contains documentation for the Solana MCP (Monitoring and Classification Platform) API.

## Modules

### [Token Categorizer](./token_categorizer.md)
The Token Categorizer module provides functionality to classify Solana tokens into categories based on their characteristics, keywords in their name/symbol, and other metadata.

### [Token Risk Analyzer](./token_risk_analyzer.md)
The Token Risk Analyzer module evaluates Solana tokens for potential risk factors, providing comprehensive risk scores and detailed analysis of various risk dimensions.

### [Wallet Classifier](./wallet_classifier.md)
The Wallet Classifier module identifies potentially high-risk wallet behavior through transaction pattern analysis, categorizing wallets by activity patterns and providing risk scores.

## Using the API

All modules use the Solana Client for RPC communication. Most methods are asynchronous and should be awaited.

Example:

```python
from solana_mcp.solana_client import SolanaClient
from solana_mcp.token_categorizer import TokenCategorizer
from solana_mcp.token_risk_analyzer import TokenRiskAnalyzer
from solana_mcp.wallet_classifier import WalletClassifier

async def main():
    # Initialize the Solana client
    client = SolanaClient("https://api.mainnet-beta.solana.com")
    
    # Analyze a token
    token_mint = "TokenMintAddressHere"
    
    # Categorize the token
    categorizer = TokenCategorizer(client)
    category_info = await categorizer.categorize_token(token_mint)
    print(f"Token Category: {category_info['primary_category']}")
    
    # Analyze token risks
    risk_analyzer = TokenRiskAnalyzer(client)
    risk_analysis = await risk_analyzer.analyze_token_risks(token_mint)
    print(f"Token Risk Level: {risk_analysis['risk_level']}")
    
    # Analyze a wallet
    wallet_address = "WalletAddressHere"
    wallet_classifier = WalletClassifier(client)
    wallet_analysis = await wallet_classifier.classify_wallet(wallet_address)
    print(f"Wallet Risk Level: {wallet_analysis['risk_level']}")
    
    # Batch analyze multiple wallets
    wallet_addresses = ["Address1", "Address2", "Address3"]
    batch_results = await wallet_classifier.batch_classify_wallets(wallet_addresses)
    print(f"Risk summary: {batch_results['risk_summary']}")
```

## Error Handling

All modules use the standard error handling pattern with the `@handle_errors` decorator.
The recommended approach is to use try/except blocks to catch potential errors:

```python
try:
    result = await categorizer.categorize_token(token_mint)
except Exception as e:
    print(f"Error: {str(e)}") 