"""Fresh wallet detector service for identifying new token holders."""

import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal

from solana_mcp.solana_client import SolanaClient, InvalidPublicKeyError
from solana_mcp.services.fresh_wallet.models import TokenInfo, FreshWallet, FreshWalletDetectionResult
from solana_mcp.services.fresh_wallet.helpers import (
    get_wallet_age_days, get_transaction_count, calculate_freshness_score, FRESH_WALLET_AGE_DAYS
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


async def detect_fresh_wallets(token_address: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Find fresh wallets for a token.
    
    Args:
        token_address: Token mint address
        solana_client: Solana client
        
    Returns:
        Fresh wallet information
    """
    try:
        # Validate token address
        try:
            # A simple check before making RPC calls
            if not token_address or len(token_address) < 32 or len(token_address) > 44:
                return FreshWalletDetectionResult.error("Invalid token address format")
        except Exception:
            return FreshWalletDetectionResult.error("Invalid token address")
        
        # Get token info
        try:
            # Get token supply
            supply_data = await solana_client._make_request("getTokenSupply", [token_address])
            decimals = supply_data.get("value", {}).get("decimals", 0)
            
            # Get token metadata
            metadata = await solana_client.get_token_metadata(token_address)
            symbol = metadata.get("symbol", token_address[:6])
            
            # Get token price
            price_data = await solana_client.get_market_price(token_address)
            price_usd = Decimal(str(price_data.get("price", 0.01)))
            
            # Create token info object
            token_info = TokenInfo(
                mint=token_address,
                decimals=decimals,
                symbol=symbol,
                price_usd=price_usd
            )
            
            # Get token holders
            holders = await get_token_holders(token_address, solana_client, 100)
            
            if not holders:
                return FreshWalletDetectionResult.error("No token holders found")
                
            logger.info(f"Analyzing {len(holders)} token holders for fresh wallet detection")
            
            # Process holders to determine freshness
            fresh_wallets = []
            all_wallets = []
            
            # Process first 30 holders for performance reasons
            for holder in holders[:30]:
                wallet_address = holder['owner']
                
                # Get all tokens in the wallet
                tokens = await get_wallet_tokens(wallet_address, solana_client)
                
                # Calculate target token stats
                target_token_amount = Decimal(0)
                token_count = len(tokens)
                non_dust_token_count = 0
                
                for token in tokens:
                    if token['mint'] == token_address:
                        target_token_amount = token['amount']
                    
                    # Count tokens with meaningful value
                    if token['mint'] in ['So11111111111111111111111111111111111111112', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB']:
                        # SOL, USDC, USDT - count if more than $1
                        price = Decimal(150) if token['mint'] == 'So11111111111111111111111111111111111111112' else Decimal(1)
                        if token['amount'] * price > 1:
                            non_dust_token_count += 1
                    else:
                        # For other tokens, assume they have some value
                        non_dust_token_count += 1
                
                # Check if too many tokens (not fresh)
                if non_dust_token_count > 8:
                    freshness_score = 0.1
                    is_fresh = False
                    wallet_age_days = None
                else:
                    # Get wallet age
                    wallet_age_days = await get_wallet_age_days(wallet_address, solana_client)
                    
                    # Calculate token transaction ratio (simplified)
                    token_tx_ratio = 0.5  # Default middle value
                    
                    # Check if wallet is new
                    is_new_wallet = wallet_age_days is not None and wallet_age_days <= FRESH_WALLET_AGE_DAYS
                    
                    # Calculate freshness score
                    is_fresh, freshness_score = calculate_freshness_score(
                        token_count, 
                        non_dust_token_count,
                        is_new_wallet
                    )
                
                # Calculate target token value
                target_token_value = target_token_amount * token_info.price_usd
                
                # Create FreshWallet object
                wallet_result = FreshWallet(
                    wallet=wallet_address,
                    is_fresh=is_fresh,
                    token_count=token_count,
                    non_dust_token_count=non_dust_token_count,
                    token_tx_ratio=float(token_tx_ratio),
                    wallet_age_days=wallet_age_days,
                    target_token_amount=float(target_token_amount),
                    target_token_value_usd=float(target_token_value),
                    freshness_score=float(freshness_score)
                )
                
                all_wallets.append(wallet_result)
                
                if wallet_result.is_fresh:
                    fresh_wallets.append(wallet_result)
            
            # Sort wallets by freshness score
            fresh_wallets.sort(key=lambda x: x.freshness_score, reverse=True)
            all_wallets.sort(key=lambda x: x.freshness_score, reverse=True)
            
            # Create result object
            result = FreshWalletDetectionResult(
                token_address=token_address,
                token_symbol=token_info.symbol,
                token_price_usd=float(token_info.price_usd),
                fresh_wallet_count=len(fresh_wallets),
                total_analyzed_wallets=len(all_wallets),
                fresh_wallets=fresh_wallets
            )
            
            # Convert to dictionary for API response
            return vars(result)
                
        except InvalidPublicKeyError as e:
            logger.error(f"Invalid public key: {str(e)}")
            return FreshWalletDetectionResult.error(f"Invalid token address: {str(e)}")
        except Exception as e:
            logger.error(f"Error in fresh wallet detection: {str(e)}")
            return FreshWalletDetectionResult.error(f"Error in fresh wallet detection: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error in fresh wallet detection: {str(e)}")
        return FreshWalletDetectionResult.error(f"Unexpected error: {str(e)}") 