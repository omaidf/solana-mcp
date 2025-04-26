"""
API Routes for Solana MCP Server
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from core.solana import SolanaClient
from models.context import ModelContext

# Initialize router
api_router = APIRouter(prefix="/api")

# Import analyzer routes
from api.analyzer_routes import analyzer_router

# Include analyzer routes
api_router.include_router(analyzer_router)

# Request and response models
class BlockchainRequest(BaseModel):
    address: str
    network: Optional[str] = "mainnet"  # mainnet, testnet, devnet

class TransactionRequest(BaseModel):
    signature: str
    network: Optional[str] = "mainnet"

class ContextRequest(BaseModel):
    address: str
    model_type: str
    parameters: Optional[Dict[str, Any]] = {}

class ContextResponse(BaseModel):
    context_id: str
    data: Dict[str, Any]
    model_type: str
    timestamp: int

class RpcHealthRequest(BaseModel):
    network: Optional[str] = "mainnet"
    custom_url: Optional[str] = None

# Endpoints
@api_router.get("/status")
async def get_status():
    """Get the current status of the MCP server"""
    return {
        "status": "online",
        "version": "0.1.0",
        "blockchain": "solana"
    }

@api_router.post("/health")
async def check_rpc_health(request: RpcHealthRequest = RpcHealthRequest()):
    """
    Check the health and responsiveness of the Solana RPC endpoint
    
    Args:
        request: Optional network or custom URL to check
        
    Returns:
        Dict containing health status and metrics
    """
    try:
        async with SolanaClient(network=request.network, custom_url=request.custom_url) as client:
            health_data = await client.health_check()
            return health_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@api_router.post("/account")
async def get_account_info(request: BlockchainRequest):
    """
    Get detailed information about a Solana account
    
    Args:
        request: The blockchain request containing address and network
        
    Returns:
        Dict containing detailed account information including lamports, owner, executable status, and data
        
    Raises:
        HTTPException: If the account information cannot be retrieved
    """
    try:
        async with SolanaClient(network=request.network) as client:
            account_info = await client.get_account_info(request.address)
            return account_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/transaction")
async def get_transaction(request: TransactionRequest):
    """
    Get detailed information about a Solana transaction
    
    Args:
        request: The transaction request containing signature and network
        
    Returns:
        Dict containing transaction details including status, fees, and involved accounts
        
    Raises:
        HTTPException: If the transaction information cannot be retrieved
    """
    try:
        async with SolanaClient(network=request.network) as client:
            tx_info = await client.get_transaction(request.signature)
            return tx_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/context", response_model=ContextResponse)
async def generate_context(request: ContextRequest):
    """
    Generate model context for a Solana address
    
    This endpoint processes blockchain data to create context information
    for different model types like transaction history or token holdings.
    
    Args:
        request: The context request with address, model type, and optional parameters
        
    Returns:
        ContextResponse: A structured context response with model-specific data
        
    Raises:
        HTTPException: If context generation fails
    """
    try:
        async with ModelContext(model_type=request.model_type) as context:
            result = await context.generate(request.address, request.parameters)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/models")
async def list_available_models():
    """
    List all available models for context generation
    
    Returns:
        Dict containing a list of available models with their descriptions
    """
    return {
        "models": [
            {
                "id": "transaction-history",
                "name": "Transaction History",
                "description": "Analysis of transaction history for an address"
            },
            {
                "id": "token-holdings",
                "name": "Token Holdings",
                "description": "Current token holdings for an address"
            }
        ]
    } 