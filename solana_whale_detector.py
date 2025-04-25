#!/usr/bin/env python3
import argparse
import requests
import json
import sys
from decimal import Decimal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Default RPC URL with Helius API key
DEFAULT_RPC_URL = "https://mainnet.helius-rpc.com/?api-key=4ffc1228-f093-4974-ad2d-3edd8e5f7c03"

# Constants
TOP_HOLDERS_LIMIT = 100
SIGNIFICANT_VALUE_USD = 50000  # Minimum USD value to consider significant

def make_rpc_call(method, params, rpc_url):
    """Make a JSON-RPC call to Solana."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    response = requests.post(rpc_url, json=payload)
    result = response.json()
    
    if "error" in result:
        raise Exception(f"RPC error: {result['error']['message']}")
    
    return result["result"]

def get_token_price(token_address):
    """Get token price from Jupiter API."""
    try:
        # First try Jupiter API
        jupiter_url = f"https://price.jup.ag/v4/price?ids={token_address}"
        response = requests.get(jupiter_url)
        data = response.json()
        
        if data.get("data") and token_address in data["data"]:
            price = data["data"][token_address]["price"]
            return Decimal(str(price))
            
        # If Jupiter doesn't have it, try Birdeye API
        birdeye_url = f"https://public-api.birdeye.so/public/price?address={token_address}"
        headers = {"X-API-KEY": "5c88c2dddead4a1094fbcc5f2d86d6fe"}
        response = requests.get(birdeye_url, headers=headers)
        data = response.json()
        
        if data.get("success") and data.get("data", {}).get("value"):
            price = data["data"]["value"]
            return Decimal(str(price))
            
        print(f"Could not find price for token {token_address}, using default price of $0.01", file=sys.stderr)
        return Decimal("0.01")  # Default fallback price
    except Exception as e:
        print(f"Error fetching token price: {e}", file=sys.stderr)
        print("Using default price of $0.01", file=sys.stderr)
        return Decimal("0.01")  # Default fallback price

def get_token_info(token_address, rpc_url):
    """Fetch token metadata including decimals and symbol using Solana RPC."""
    try:
        # Get token supply which includes decimals
        supply_data = make_rpc_call("getTokenSupply", [token_address], rpc_url)
        decimals = supply_data["value"]["decimals"]
        
        # For token symbol, we need to check its metadata program account
        # First try to get largest token accounts to find the metadata
        accounts = make_rpc_call("getTokenLargestAccounts", [token_address], rpc_url)
        
        # If we can find at least one account, we'll get details
        if accounts["value"] and len(accounts["value"]) > 0:
            account = accounts["value"][0]["address"]
            account_info = make_rpc_call("getAccountInfo", [account, {"encoding": "jsonParsed"}], rpc_url)
            
            if "parsed" in account_info["value"]["data"]:
                parsed_data = account_info["value"]["data"]["parsed"]
                # Try to extract mint and symbol if available
                if "info" in parsed_data and "mint" in parsed_data["info"]:
                    mint_info = parsed_data["info"]["mint"]
                    if mint_info == token_address:
                        symbol = parsed_data.get("info", {}).get("symbol", token_address[:6])
                    else:
                        symbol = token_address[:6]  # Use first 6 chars of token address as symbol
                else:
                    symbol = token_address[:6]
            else:
                symbol = token_address[:6]
        else:
            symbol = token_address[:6]
        
        # Get token price automatically
        price_usd = get_token_price(token_address)
        print(f"Found token {symbol} with price: ${price_usd} USD", file=sys.stderr)
        
        return {
            'decimals': decimals,
            'symbol': symbol,
            'price_usd': price_usd,
            'total_supply': Decimal(supply_data["value"]["amount"]) / Decimal(10 ** decimals)
        }
    except Exception as e:
        print(f"Error fetching token info: {e}", file=sys.stderr)
        sys.exit(1)

def get_wallet_tokens(wallet_address, rpc_url):
    """Get all token balances for a specific wallet."""
    try:
        # Get native SOL balance
        sol_balance_response = make_rpc_call("getBalance", [wallet_address], rpc_url)
        sol_balance = Decimal(sol_balance_response["value"]) / Decimal(10**9)  # SOL has 9 decimals
        
        # Get token accounts owned by the wallet
        token_accounts_response = make_rpc_call(
            "getTokenAccountsByOwner", 
            [
                wallet_address, 
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"}
            ], 
            rpc_url
        )
        
        # Initialize with SOL
        balances = [
            {
                'token': 'SOL',
                'mint': 'So11111111111111111111111111111111111111112',  # Native SOL mint
                'amount': sol_balance,
                'decimals': 9
            }
        ]
        
        # Process all token accounts
        for account in token_accounts_response.get("value", []):
            try:
                token_info = account["account"]["data"]["parsed"]["info"]
                mint = token_info["mint"]
                amount = Decimal(token_info["tokenAmount"]["amount"])
                decimals = token_info["tokenAmount"]["decimals"]
                
                # Skip empty accounts
                if amount == 0:
                    continue
                
                # Calculate actual token amount
                token_amount = amount / Decimal(10 ** decimals)
                
                balances.append({
                    'token': mint[:6],  # Use first 6 chars as shorthand
                    'mint': mint,
                    'amount': token_amount,
                    'decimals': decimals
                })
            except (KeyError, TypeError) as e:
                continue
                
        return balances
    except Exception as e:
        print(f"Error fetching wallet balance for {wallet_address}: {e}", file=sys.stderr)
        return []  # Return empty list on error

def get_token_holders(token_address, rpc_url):
    """Fetch the top holders of a given token using Solana RPC."""
    try:
        # Get all token accounts for the mint
        result = make_rpc_call("getProgramAccounts", [
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program ID
            {
                "filters": [
                    {
                        "dataSize": 165  # Size of token account data
                    },
                    {
                        "memcmp": {
                            "offset": 0,
                            "bytes": token_address
                        }
                    }
                ],
                "encoding": "jsonParsed"
            }
        ], rpc_url)
        
        # Process the accounts
        holders = []
        print(f"Processing {len(result)} token accounts...", file=sys.stderr)
        
        for account in result:
            try:
                parsed_info = account.get('account', {}).get('data', {}).get('parsed', {}).get('info', {})
                owner = parsed_info.get('owner')
                amount_str = parsed_info.get('tokenAmount', {}).get('amount', '0')
                
                # Skip empty accounts
                if amount_str == '0':
                    continue
                    
                amount = Decimal(amount_str)
                
                holders.append({
                    'owner': owner,
                    'amount': amount
                })
            except (KeyError, TypeError) as e:
                continue
                
        # Sort holders by amount (descending) and limit to top holders
        holders.sort(key=lambda x: x['amount'], reverse=True)
        return holders[:TOP_HOLDERS_LIMIT]  # Always return top 100
        
    except Exception as e:
        print(f"Error fetching token holders: {e}", file=sys.stderr)
        print("This may be due to RPC rate limits or token size.", file=sys.stderr)
        sys.exit(1)

def get_token_prices(token_mints, target_token_price, rpc_url):
    """Return price estimations for tokens."""
    prices = {}
    
    # Add default prices for common tokens
    default_prices = {
        "So11111111111111111111111111111111111111112": Decimal("150"),  # SOL - hardcoded estimate
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": Decimal("1"),   # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": Decimal("1"),   # USDT
        "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": Decimal("1"),   # stUSDC
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": Decimal("30"),  # BONK - can update this
    }
    
    # Use the default prices
    prices.update(default_prices)
    
    # For token mints not in defaults, try to get prices
    for mint in token_mints:
        if mint not in prices:
            try:
                price = get_token_price(mint)
                prices[mint] = price
            except:
                # If we can't get price, use a very low default
                prices[mint] = Decimal("0.0001")
    
    return prices

def calculate_wallet_value(token_balances, prices, target_token=None):
    """Calculate the total value of all tokens in a wallet."""
    total_value = Decimal(0)
    target_token_value = Decimal(0)
    token_values = []
    
    for token in token_balances:
        mint = token['mint']
        amount = token['amount']
        price = prices.get(mint, Decimal("0.0001"))  # Default very low price if unknown
        
        value = amount * price
        total_value += value
        
        # Track value of the target token separately
        if target_token and mint == target_token:
            target_token_value = value
            
        token_values.append({
            'mint': mint,
            'token': token.get('token', mint[:6]),
            'amount': float(amount),
            'value_usd': float(value)
        })
    
    # Sort tokens by value (highest first)
    token_values.sort(key=lambda x: x['value_usd'], reverse=True)
    
    return {
        'total_value_usd': float(total_value),
        'target_token_value_usd': float(target_token_value),
        'tokens': token_values[:10]  # Include top 10 tokens by value
    }

def process_holder(holder, token_info, rpc_url, token_prices):
    """Process a token holder to determine if it's a whale wallet based on total holdings."""
    try:
        amount = holder['amount']
        wallet_address = holder['owner']
        
        # Calculate the amount of the specific token
        token_amount = amount / Decimal(10 ** token_info['decimals'])
        token_value = token_amount * token_info['price_usd']
        supply_percentage = (token_amount / token_info['total_supply']) * 100 if token_info['total_supply'] > 0 else 0
        
        # Get all token balances for this wallet
        wallet_tokens = get_wallet_tokens(wallet_address, rpc_url)
        
        # Calculate total wallet value
        wallet_value_data = calculate_wallet_value(wallet_tokens, token_prices, token_info['mint'])
        
        return {
            'wallet': wallet_address,
            'target_token_amount': float(token_amount),
            'target_token_value_usd': float(token_value),
            'target_token_supply_percentage': float(supply_percentage),
            'total_value_usd': wallet_value_data['total_value_usd'],
            'token_count': len(wallet_tokens),
            'top_tokens': wallet_value_data['tokens']
        }
    except Exception as e:
        # Silently handle errors
        return None

def determine_whale_threshold(wallets):
    """Automatically determine a reasonable threshold for whale classification."""
    if not wallets:
        return SIGNIFICANT_VALUE_USD
    
    # Sort wallets by total value
    sorted_wallets = sorted(wallets, key=lambda x: x['total_value_usd'], reverse=True)
    
    # Get values for analysis
    values = [w['total_value_usd'] for w in sorted_wallets]
    
    # Calculate some statistical thresholds
    try:
        # 1. Top 20% percentile of holders are considered whales
        percentile_threshold = values[int(len(values) * 0.2)] if len(values) >= 5 else values[-1]
        
        # 2. Anyone holding more than 1% of supply is a whale
        target_token_threshold = next((w['total_value_usd'] for w in sorted_wallets 
                                if w['target_token_supply_percentage'] >= 1), 0)
        
        # 3. Base minimum threshold for significance
        base_threshold = SIGNIFICANT_VALUE_USD
        
        # Take the minimum of the three thresholds (excluding 0)
        candidates = [t for t in [percentile_threshold, target_token_threshold, base_threshold] if t > 0]
        threshold = min(candidates) if candidates else base_threshold
        
        # Make sure it's at least the base threshold
        return max(threshold, base_threshold)
    except:
        # Fallback to base threshold if statistical calculation fails
        return SIGNIFICANT_VALUE_USD

def find_whale_wallets(token_address, token_info, rpc_url):
    """Find whale wallets holding the token - automatically determining threshold and checking entire wallet value."""
    # Get token holders (always top 100)
    holders = get_token_holders(token_address, rpc_url)
    print(f"Analyzing top {len(holders)} token holders...", file=sys.stderr)
    
    # Get token prices for all the tokens we might encounter
    # First collect a list of token mints we might need prices for
    token_mints = [token_address]  # Start with our target token
    
    # For token prices, we'll use a combination of hardcoded values and api calls
    token_prices = get_token_prices(token_mints, token_info['price_usd'], rpc_url)
    
    # Make sure our target token is in the price dictionary
    token_prices[token_address] = token_info['price_usd']
    
    # Process holders in parallel
    processed_wallets = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks for processing each holder
        futures = {}
        for holder in holders:
            future = executor.submit(process_holder, holder, token_info, rpc_url, token_prices)
            futures[future] = holder
        
        for future in as_completed(futures):
            processed += 1
            
            result = future.result()
            if result:
                processed_wallets.append(result)
            
            # Print progress
            if processed % 20 == 0 or processed == len(holders):
                print(f"Processed {processed}/{len(holders)} wallets.", file=sys.stderr)
    
    # Automatically determine whale threshold based on total wallet value
    whale_threshold = determine_whale_threshold(processed_wallets)
    print(f"Automatically determined whale threshold: ${whale_threshold:,.2f} USD", file=sys.stderr)
    
    # Filter for whale wallets based on total value
    whale_wallets = [w for w in processed_wallets if w['total_value_usd'] >= whale_threshold]
    
    # Sort whale wallets by total USD value (descending)
    whale_wallets.sort(key=lambda x: x['total_value_usd'], reverse=True)
    
    return whale_wallets, whale_threshold

def main():
    parser = argparse.ArgumentParser(description='Find Solana whale wallets for a specific token')
    parser.add_argument('token', help='Solana token address')
    parser.add_argument('--rpc', default=DEFAULT_RPC_URL, help='Solana RPC URL')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    
    args = parser.parse_args()
    
    print(f"Finding whale wallets for token: {args.token}", file=sys.stderr)
    print(f"Analyzing top {TOP_HOLDERS_LIMIT} token holders", file=sys.stderr)
    
    # Get token info
    start_time = time.time()
    token_info = get_token_info(args.token, args.rpc)
    
    # Find whale wallets
    whale_wallets, threshold = find_whale_wallets(args.token, token_info, args.rpc)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    if args.json:
        # Output in JSON format
        result = {
            "token_address": args.token,
            "token_symbol": token_info['symbol'],
            "token_price_usd": float(token_info['price_usd']),
            "whale_threshold_usd": float(threshold),
            "whale_count": len(whale_wallets),
            "execution_time_seconds": execution_time,
            "whales": whale_wallets
        }
        print(json.dumps(result, indent=2))
    else:
        # Output in human-readable format
        print(f"\nFound {len(whale_wallets)} whale wallets with total value of ${threshold:,.2f}+ USD in {execution_time:.2f} seconds:")
        print("-" * 80)
        
        for i, wallet in enumerate(whale_wallets, 1):
            print(f"{i}. Wallet: {wallet['wallet']}")
            print(f"   Target Token: {wallet['target_token_amount']:.4f} {token_info['symbol']} " 
                  f"(${wallet['target_token_value_usd']:,.2f} USD, {wallet['target_token_supply_percentage']:.2f}% of supply)")
            print(f"   Total Wallet Value: ${wallet['total_value_usd']:,.2f} USD")
            print(f"   Token Count: {wallet['token_count']}")
            
            # Print top 3 tokens by value if available
            if wallet['top_tokens']:
                print(f"   Top Tokens: ", end="")
                for j, token in enumerate(wallet['top_tokens'][:3]):
                    if j > 0:
                        print(", ", end="")
                    print(f"{token['token']}: ${token['value_usd']:,.2f}", end="")
                print()
            
            print()
    
if __name__ == "__main__":
    main() 