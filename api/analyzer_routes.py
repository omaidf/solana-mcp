"""
Enhanced Solana analysis API routes
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import asyncio
import logging
import json
import aiohttp

from core.analyzer import SolanaAnalyzer, TokenInfo, Whale

# Setup logging
logger = logging.getLogger(__name__)

# Initialize router
analyzer_router = APIRouter(prefix="/analyzer")

# Request and response models
class TokenRequest(BaseModel):
    mint_address: str

# Make sure our response model matches the returned object structure
class TokenResponse(BaseModel):
    symbol: str
    name: str
    decimals: int
    price: float
    mint: str
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    supply: Optional[float] = None

    # Add a conversion class method for TokenInfo
    @classmethod
    def from_token_info(cls, token_info: TokenInfo) -> 'TokenResponse':
        return cls(
            symbol=token_info.symbol,
            name=token_info.name,
            decimals=token_info.decimals,
            price=token_info.price,
            mint=token_info.mint,
            market_cap=token_info.market_cap,
            volume_24h=token_info.volume_24h,
            supply=token_info.supply
        )

class WhaleRequest(BaseModel):
    mint_address: str
    min_usd_value: float = 50000
    max_holders: int = 1000
    batch_size: int = 100
    concurrency: int = 5

class WhaleResponse(BaseModel):
    address: str
    token_balance: float
    usd_value: float
    percentage: float
    last_active: Optional[str] = None

    @classmethod
    def from_whale(cls, whale: Whale) -> 'WhaleResponse':
        return cls(
            address=whale.address,
            token_balance=whale.token_balance,
            usd_value=whale.usd_value,
            percentage=whale.percentage,
            last_active=whale.last_active
        )

class WhalesResponse(BaseModel):
    token_info: TokenResponse
    whales: List[WhaleResponse]
    stats: Dict[str, Any]

class AccountRequest(BaseModel):
    address: str
    encoding: str = "jsonParsed"

class SemanticSearchRequest(BaseModel):
    prompt: str

# Global analyzer instance for better resource management
_analyzer_instance = None
_analyzer_lock = asyncio.Lock()

async def get_global_analyzer():
    """Get or create a global SolanaAnalyzer instance"""
    global _analyzer_instance
    async with _analyzer_lock:
        if _analyzer_instance is None:
            _analyzer_instance = SolanaAnalyzer()
            await _analyzer_instance.__aenter__()
    return _analyzer_instance

# Improved dependency with better resource management
async def get_analyzer():
    """Get SolanaAnalyzer client instance"""
    return await get_global_analyzer()

# Background task to close the analyzer when shutting down
async def close_analyzer():
    """Close the analyzer when the application shuts down"""
    global _analyzer_instance
    if _analyzer_instance:
        await _analyzer_instance.__aexit__(None, None, None)

# Endpoints
@analyzer_router.get("/status")
async def get_status():
    """Get the status of the analyzer API"""
    return {
        "status": "available",
        "features": [
            "token-info",
            "whale-detection",
            "account-analysis",
            "semantic-search"
        ]
    }

@analyzer_router.post("/token", response_model=TokenResponse)
async def get_token_info(request: TokenRequest, analyzer: SolanaAnalyzer = Depends(get_analyzer)):
    """Get detailed token information"""
    try:
        token_info = await analyzer.get_token_info(request.mint_address)
        return TokenResponse.from_token_info(token_info)
    except ValueError as e:
        # Handle validation errors, missing tokens, etc.
        logger.error(f"Value error getting token info for {request.mint_address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting token info: {str(e)}")
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        # Handle network and connection errors
        logger.error(f"Network error getting token info for {request.mint_address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection error getting token info: {str(e)}")
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        logger.error(f"JSON parsing error for {request.mint_address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing response data: {str(e)}")
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error getting token info for {request.mint_address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting token info: {str(e)}")

@analyzer_router.post("/whales", response_model=WhalesResponse)
async def find_token_whales(
    request: WhaleRequest, 
    analyzer: SolanaAnalyzer = Depends(get_analyzer)
):
    """Find whale holders for a token"""
    try:
        whale_data = await analyzer.find_whales(
            request.mint_address,
            min_usd_value=request.min_usd_value,
            max_holders=request.max_holders,
            batch_size=request.batch_size,
            concurrency=request.concurrency
        )
        
        # Convert to our response model
        return WhalesResponse(
            token_info=TokenResponse.from_token_info(whale_data["token_info"]),
            whales=[WhaleResponse.from_whale(whale) for whale in whale_data["whales"]],
            stats=whale_data["stats"]
        )
    except Exception as e:
        logger.exception(f"Error finding whales for {request.mint_address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding whales: {str(e)}")

@analyzer_router.post("/account")
async def get_enhanced_account_info(
    request: AccountRequest,
    analyzer: SolanaAnalyzer = Depends(get_analyzer)
):
    """Get enhanced account information"""
    try:
        account_info = await analyzer.get_account_info(
            request.address,
            encoding=request.encoding
        )
        return account_info
    except Exception as e:
        logger.exception(f"Error getting account info for {request.address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting account info: {str(e)}")

@analyzer_router.post("/token-accounts")
async def get_token_accounts(
    request: AccountRequest,
    analyzer: SolanaAnalyzer = Depends(get_analyzer)
):
    """Get all token accounts for an address with enhanced data"""
    try:
        token_accounts = await analyzer.get_token_accounts_by_owner(request.address)
        
        # Process token accounts with price data when available
        enhanced_accounts = []
        for account in token_accounts:
            # Guard against incorrect account structure
            if not isinstance(account, dict) or 'data' not in account:
                logger.warning(f"Unexpected account structure: {account}")
                continue
                
            try:
                # Safely extract mint address
                if 'data' in account and 'parsed' in account['data'] and 'info' in account['data']['parsed']:
                    mint = account['data']['parsed']['info'].get('mint')
                    if not mint:
                        enhanced_accounts.append(account['data']['parsed']['info'])
                        continue
                        
                    token_info = await analyzer.get_token_info(mint)
                    
                    # Safely extract token amount
                    amount = 0
                    if 'tokenAmount' in account['data']['parsed']['info']:
                        amount_data = account['data']['parsed']['info']['tokenAmount']
                        if isinstance(amount_data, dict) and 'uiAmount' in amount_data:
                            try:
                                amount = float(amount_data['uiAmount'])
                            except (ValueError, TypeError):
                                amount = 0
                    
                    usd_value = amount * token_info.price
                    
                    enhanced_accounts.append({
                        **account['data']['parsed']['info'],
                        "token_info": {
                            "symbol": token_info.symbol,
                            "name": token_info.name,
                            "price": token_info.price,
                            "market_cap": token_info.market_cap
                        },
                        "usd_value": usd_value
                    })
                else:
                    # Add raw account if parsed info not available
                    enhanced_accounts.append(account)
            except Exception as token_error:
                logger.warning(f"Error processing token account: {str(token_error)}")
                # If we can't get token info, just add the raw account
                enhanced_accounts.append(account)
                
        return {
            "owner": request.address,
            "count": len(enhanced_accounts),
            "accounts": enhanced_accounts
        }
    except Exception as e:
        logger.exception(f"Error getting token accounts for {request.address}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting token accounts: {str(e)}")

@analyzer_router.post("/semantic-search")
async def process_semantic_search(
    request: SemanticSearchRequest,
    analyzer: SolanaAnalyzer = Depends(get_analyzer)
):
    """
    Process natural language queries about Solana blockchain data.
    This endpoint takes a prompt and uses semantic analysis to understand the intent
    and return relevant blockchain data.
    
    Example prompts:
    - "What is the balance of address Gw4nUxMFKaaEZZ1xqQZXJAbrLFvZPdN1jPXQSN3qvXSS?"
    - "Show me tokens owned by Gw4nUxMFKaaEZZ1xqQZXJAbrLFvZPdN1jPXQSN3qvXSS"
    - "Get transaction details for 5kWVvtUDMAKCMfmNNJcZ22UVvezKZ5FJgbpP1LNmQAtSGSnauiEELMGfg8MaYDaSSRQizMwCvwuP9Mb9Zv5X9kNM"
    """
    try:
        search_result = await analyzer.semantic_search(request.prompt)
        return search_result
    except Exception as e:
        logger.exception(f"Error processing semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing semantic search: {str(e)}") 