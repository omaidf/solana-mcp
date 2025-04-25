"""Whale detector service for identifying large token holders."""

import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError
from solana_mcp.services.whale_detector.models import TokenInfo, WhaleWallet, WhaleDetectionResult
from solana_mcp.services.whale_detector.helpers import (
    get_token_price, get_token_prices, calculate_wallet_value, determine_whale_threshold
)

logger = logging.getLogger(__name__)

# Constants
TOP_HOLDERS_LIMIT = 100
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

async def get_token_holders(token_address: str, solana_client: SolanaClient, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch token holders for a given token.
    
    Args:
        token_address: Token mint address
        solana_client: Solana client
        limit: Maximum number of holders to return
        
    Returns:
        List of token holders with their balances
    """
    try:
        # Get all token accounts for the mint
        filters = [
            {"dataSize": 165},  # Size of token account data
            {"memcmp": {"offset": 0, "bytes": token_address}}
        ]
        
        accounts = await solana_client.get_program_accounts(
            TOKEN_PROGRAM_ID,
            filters=filters,
            encoding="jsonParsed"
        )
        
        # Process the accounts
        holders = []
        
        for account in accounts:
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
        return holders[:limit]
        
    except Exception as e:
        logger.error(f"Error fetching token holders: {str(e)}")
        return []


async def get_wallet_tokens(wallet_address: str, solana_client: SolanaClient) -> List[Dict[str, Any]]:
    """Get all token balances for a specific wallet.
    
    Args:
        wallet_address: Wallet address
        solana_client: Solana client
        
    Returns:
        List of token balances
    """
    try:
        # Get native SOL balance
        sol_balance_response = await solana_client.get_balance(wallet_address)
        sol_balance = Decimal(sol_balance_response) / Decimal(10**9)  # SOL has 9 decimals
        
        # Get token accounts owned by the wallet
        token_accounts = await solana_client.get_token_accounts_by_owner(wallet_address)
        
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
        for account in token_accounts:
            try:
                mint = account.get("mint", "")
                amount = Decimal(account.get("amount", 0))
                decimals = account.get("decimals", 0)
                
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
        logger.warning(f"Error fetching wallet balance for {wallet_address}: {str(e)}")
        return []  # Return empty list on error


async def detect_whale_wallets(token_address: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Find whale wallets for a token.
    
    Args:
        token_address: Token mint address
        solana_client: Solana client
        
    Returns:
        Whale wallet information
    """
    try:
        # Validate token address
        try:
            # A simple check before making RPC calls
            if not token_address or len(token_address) < 32 or len(token_address) > 44:
                return WhaleDetectionResult.error("Invalid token address format")
        except Exception:
            return WhaleDetectionResult.error("Invalid token address")
        
        # Get token info
        try:
            # Get token supply
            supply_data = await solana_client._make_request("getTokenSupply", [token_address])
            decimals = supply_data.get("value", {}).get("decimals", 0)
            total_supply = Decimal(supply_data.get("value", {}).get("amount", "0")) / Decimal(10 ** decimals)
            
            # Get token metadata
            metadata = await solana_client.get_token_metadata(token_address)
            symbol = metadata.get("symbol", token_address[:6])
            
            # Get token price
            price_usd = get_token_price(token_address)
            
            # Create token info object
            token_info = TokenInfo(
                mint=token_address,
                decimals=decimals,
                symbol=symbol,
                price_usd=price_usd,
                total_supply=total_supply
            )
            
            # Get token holders
            holders = await get_token_holders(token_address, solana_client, TOP_HOLDERS_LIMIT)
            
            if not holders:
                return WhaleDetectionResult.error("No token holders found")
                
            logger.info(f"Analyzing {len(holders)} token holders for whale detection")
            
            # Get token prices for balance calculations
            token_mints = [token_address]
            token_prices = get_token_prices(token_mints)
            token_prices[token_address] = price_usd  # Ensure our target token price is set
            
            # Process holders to calculate wallet values
            processed_wallets = []
            
            # Process first 20 holders for performance reasons
            for holder in holders[:20]:
                # Process the holder directly
                owner = holder['owner']
                amount = holder['amount']
                
                # Calculate the amount of the specific token
                token_amount = amount / Decimal(10 ** token_info.decimals)
                token_value = token_amount * token_info.price_usd
                supply_percentage = (token_amount / token_info.total_supply) * 100 if token_info.total_supply > 0 else 0
                
                # Get all token balances for this wallet
                wallet_tokens = await get_wallet_tokens(owner, solana_client)
                
                # Calculate total wallet value
                wallet_value_data = calculate_wallet_value(wallet_tokens, token_prices, token_address)
                
                processed_wallet = {
                    'wallet': owner,
                    'target_token_amount': float(token_amount),
                    'target_token_value_usd': float(token_value),
                    'target_token_supply_percentage': float(supply_percentage),
                    'total_value_usd': wallet_value_data.total_value_usd,
                    'token_count': len(wallet_tokens),
                    'top_tokens': wallet_value_data.tokens
                }
                processed_wallets.append(processed_wallet)
            
            # Determine whale threshold
            whale_threshold = determine_whale_threshold(processed_wallets)
            logger.info(f"Determined whale threshold: ${whale_threshold:,.2f} USD")
            
            # Filter for whale wallets based on total value
            whale_wallets = [
                WhaleWallet(
                    wallet=w['wallet'],
                    target_token_amount=w['target_token_amount'],
                    target_token_value_usd=w['target_token_value_usd'],
                    target_token_supply_percentage=w['target_token_supply_percentage'],
                    total_value_usd=w['total_value_usd'],
                    token_count=w['token_count'],
                    top_tokens=w['top_tokens']
                ) 
                for w in processed_wallets if w['total_value_usd'] >= whale_threshold
            ]
            
            # Sort whale wallets by total USD value (descending)
            whale_wallets.sort(key=lambda x: x.total_value_usd, reverse=True)
            
            # Create the result object
            result = WhaleDetectionResult(
                token_address=token_address,
                token_symbol=token_info.symbol,
                token_price_usd=float(token_info.price_usd),
                whale_threshold_usd=float(whale_threshold),
                whale_count=len(whale_wallets),
                whales=whale_wallets
            )
            
            # Convert to dictionary for API response
            return vars(result)
                
        except InvalidPublicKeyError as e:
            logger.error(f"Invalid public key: {str(e)}")
            return WhaleDetectionResult.error(f"Invalid token address: {str(e)}")
        except Exception as e:
            logger.error(f"Error in whale detection: {str(e)}")
            return WhaleDetectionResult.error(f"Error in whale detection: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error in whale detection: {str(e)}")
        return WhaleDetectionResult.error(f"Unexpected error: {str(e)}") 