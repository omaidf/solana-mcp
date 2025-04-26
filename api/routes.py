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

# Endpoints
@api_router.get("/status")
async def get_status():
    """Get the current status of the MCP server"""
    return {
        "status": "online",
        "version": "0.1.0",
        "blockchain": "solana"
    }

@api_router.post("/account")
async def get_account_info(request: BlockchainRequest):
    """Get Solana account information"""
    try:
        client = SolanaClient(network=request.network)
        account_info = await client.get_account_info(request.address)
        return account_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/transaction")
async def get_transaction(request: TransactionRequest):
    """Get transaction information"""
    try:
        client = SolanaClient(network=request.network)
        tx_info = await client.get_transaction(request.signature)
        return tx_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/context", response_model=ContextResponse)
async def generate_context(request: ContextRequest):
    """Generate model context for a Solana address"""
    try:
        context = ModelContext(model_type=request.model_type)
        result = await context.generate(request.address, request.parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/models")
async def list_available_models():
    """List all available models for context generation"""
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