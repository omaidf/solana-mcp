"""Solana MCP server implementation using FastMCP."""

import anyio
import click
import json
import uuid
import re
import asyncio
import sys
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator, Union, Tuple, Set
import datetime
import threading
import logging
import time
from decimal import Decimal
import random
import requests

from mcp.server.fastmcp import Context, FastMCP
import mcp.types as types
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request

from solana_mcp.config import ServerConfig, get_server_config
from solana_mcp.solana_client import (
    SolanaClient, get_solana_client, SolanaRpcError, InvalidPublicKeyError,
    TOKEN_PROGRAM_ID, METADATA_PROGRAM_ID
)
from solana_mcp.semantic_search import (
    get_account_balance, get_account_details, get_token_accounts_for_owner,
    get_token_details, get_transaction_history_for_address, get_nft_details,
    semantic_transaction_search
)

# Import the refactored modules
from solana_mcp.nlp.parser import parse_natural_language_query
from solana_mcp.nlp.formatter import format_response
from solana_mcp.services.whale_detector.detector import detect_whale_wallets
from solana_mcp.services.fresh_wallet.detector import detect_fresh_wallets

# Import wallet classifier (internal module)
from solana_mcp.wallet_classifier import WalletClassifier

# Global session store with thread safety - replace the existing SESSION_STORE declaration
_session_store_async_lock = asyncio.Lock()
SESSION_STORE = {}
SESSION_EXPIRY = 30  # Session expiry in minutes
_cleanup_task = None  # Task for periodic cleanup

logger = logging.getLogger(__name__)

# Solana public key validation pattern (base58 format)
PUBKEY_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

def validate_public_key(pubkey: str) -> bool:
    """Validate a Solana public key.
    
    Args:
        pubkey: The public key to validate
        
    Returns:
        True if the public key is valid, False otherwise
    """
    if not pubkey or not isinstance(pubkey, str):
        return False
    return bool(PUBKEY_PATTERN.match(pubkey))

@dataclass
class Session:
    """Session to track context across requests."""
    id: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_accessed: datetime.datetime = field(default_factory=datetime.datetime.now)
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def update_access_time(self):
        """Update the last access time."""
        self.last_accessed = datetime.datetime.now()
        
    def add_query(self, query: str, result: Any):
        """Add a query to the history.
        
        Args:
            query: The query string
            result: The query result
        """
        self.query_history.append({
            "query": query,
            "timestamp": datetime.datetime.now().isoformat(),
            "result": result
        })
        
    def get_context_for_entity(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get context for a specific entity.
        
        Args:
            entity_type: Type of entity (account, token, etc.)
            entity_id: ID of the entity
            
        Returns:
            Context data for the entity
        """
        key = f"{entity_type}:{entity_id}"
        return self.context.get(key, {})
    
    def update_context_for_entity(self, entity_type: str, entity_id: str, data: Dict[str, Any]):
        """Update context for a specific entity.
        
        Args:
            entity_type: Type of entity (account, token, etc.)
            entity_id: ID of the entity
            data: Context data to update
        """
        key = f"{entity_type}:{entity_id}"
        if key not in self.context:
            self.context[key] = {}
        self.context[key].update(data)
    
    def is_expired(self) -> bool:
        """Check if the session is expired.
        
        Returns:
            True if expired, False otherwise
        """
        expiry_time = self.last_accessed + datetime.timedelta(minutes=SESSION_EXPIRY)
        return datetime.datetime.now() > expiry_time


async def get_or_create_session(session_id: Optional[str] = None) -> Session:
    """Get an existing session or create a new one.
    
    Args:
        session_id: Optional session ID
        
    Returns:
        The session
    """
    # Clean expired sessions
    await clean_expired_sessions()
    
    async with _session_store_async_lock:
        # Create new session if ID not provided or not found
        if not session_id or session_id not in SESSION_STORE:
            new_session = Session(id=str(uuid.uuid4()))
            SESSION_STORE[new_session.id] = new_session
            return new_session
        
        # Update access time for existing session
        session = SESSION_STORE[session_id]
        session.update_access_time()
        return session


async def clean_expired_sessions():
    """Remove expired sessions from the store.
    
    Returns:
        int: The number of sessions removed
    """
    to_remove = []
    async with _session_store_async_lock:
        # First collect the expired sessions
        for session_id, session in SESSION_STORE.items():
            if session.is_expired():
                to_remove.append(session_id)
        
        # Then remove them - safer than modifying during iteration
        for session_id in to_remove:
            del SESSION_STORE[session_id]
            
    return len(to_remove)


@dataclass
class AppContext:
    """Application context for the Solana MCP server."""
    
    solana_client: SolanaClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context.
    
    Args:
        server: The FastMCP server instance.
        
    Yields:
        The application context.
    """
    # Initialize resources and connections
    solana_client = None
    logger.info("Starting Solana MCP server lifespan...")
    
    try:
        # Initialize Solana client
        async with get_solana_client() as solana_client:
            logger.info("Solana client initialized successfully")
            
            # Start session cleanup task
            await start_session_cleanup_task()
            logger.info("Session management initialized")
            
            # Yield the application context
            yield AppContext(solana_client=solana_client)
            
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        # Re-raise to prevent application from starting with incomplete initialization
        raise
    finally:
        logger.info("Cleaning up application resources...")
        
        # Cancel the session cleanup task if it's running
        if _cleanup_task is not None and not _cleanup_task.done():
            _cleanup_task.cancel()
            try:
                await _cleanup_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Application shutdown complete")


# Create the FastMCP server with Solana lifespan
app = FastMCP(
    "solana-mcp",
    lifespan=app_lifespan,
    dependencies=["solana"]
)


# -------------------------------------------
# Natural Language Query Processing
# -------------------------------------------

# Define common transaction types and their keywords for semantic search
TRANSACTION_CATEGORIES = {
    "token_transfer": ["transfer", "send", "receive", "spl-token", "token program"],
    "nft_mint": ["mint", "nft", "metaplex", "metadata", "master edition"],
    "nft_sale": ["sale", "marketplace", "auction", "bid", "offer", "purchase"],
    "swap": ["swap", "exchange", "trade", "jupiter", "orca", "raydium"],
    "stake": ["stake", "delegate", "staking", "validator", "withdraw stake"],
    "system_transfer": ["system program", "sol transfer", "lamports"],
    "vote": ["vote", "voting", "governance"],
    "program_deploy": ["bpf loader", "deploy", "upgrade", "program"],
    "failed": ["failed", "error", "rejected"]
}

# Basic patterns for NL query understanding
QUERY_PATTERNS = [
    # Balance queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:sol|solana)? ?balance (?:of|for) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_balance",
        "params": lambda match: {"address": match.group(1)}
    },
    # Account info queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:information|info|details) (?:about|for|of) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_account_info",
        "params": lambda match: {"address": match.group(1)}
    },
    # Token queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:token|tokens|token info|token details) (?:of|for|owned by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_accounts",
        "params": lambda match: {"owner": match.group(1)}
    },
    # Token info
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:information|info|details) (?:about|for|of) token (?:with mint )?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_info",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Transaction history
    {
        "pattern": r"(?:what are|get|show|check|find) (?:the )?(?:transactions|tx|transaction history) (?:of|for|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})(?: with limit (\d+))?",
        "intent": "get_transactions",
        "params": lambda match: {"address": match.group(1), "limit": int(match.group(2)) if match.group(2) else 20}
    },
    # NFT queries
    {
        "pattern": r"(?:what is|get|show|check|find) (?:the )?(?:nft|nft info|nft details) (?:with mint )?([a-zA-Z0-9]{32,44})",
        "intent": "get_nft_info",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Whale queries - Find whales (large holders) for a token
    {
        "pattern": r"(?:are there|are there any|do you see|can you find|any) (?:whales|whale|large holder|big investor|big wallet) (?:in|for|holding) (?:this token|this|token|mint)? ?([a-zA-Z0-9]{32,44})",
        "intent": "get_token_whales",
        "params": lambda match: {"mint": match.group(1)}
    },
    # Fresh wallet queries - Find new/fresh wallets for a token
    {
        "pattern": r"(?:are there|are there any|do you see|can you find|any) (?:fresh|new|recent|suspicious) (?:wallets|wallet|holder|holders|account|accounts) (?:in|for|holding) (?:this token|this|token|mint)? ?([a-zA-Z0-9]{32,44})",
        "intent": "get_fresh_wallets",
        "params": lambda match: {"mint": match.group(1)}
    },
]

# This function now delegates to the imported implementation
async def parse_natural_language_query(query: str, solana_client: SolanaClient, session: Session = None) -> Dict[str, Any]:
    """Parse a natural language query into an API call.
    
    Args:
        query: The natural language query
        solana_client: The Solana client
        session: Optional session for context
        
    Returns:
        The query results
    """
    # This function has been moved to solana_mcp/nlp/parser.py
    # Delegating to the imported implementation
    return await parse_natural_language_query(query, solana_client, session)


# -------------------------------------------
# Context-Aware Response Formatting
# -------------------------------------------

# This function now delegates to the imported implementation
def format_response(data: Any, format_level: str = "standard") -> Dict[str, Any]:
    """Format a response based on the requested detail level.
    
    Args:
        data: The data to format
        format_level: The format level (minimal, standard, detailed, auto)
        
    Returns:
        Formatted response
    """
    # This function has been moved to solana_mcp/nlp/formatter.py
    # Delegating to the imported implementation
    return format_response(data, format_level)


# -------------------------------------------
# Utility Functions with Formatting Support
# -------------------------------------------

async def get_account_balance(address: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get account balance with formatting.
    
    Args:
        address: Account address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted balance information
    """
    try:
        balance_lamports = await solana_client.get_balance(address)
        
        # Handle case where balance might be returned as a dictionary rather than an integer
        if isinstance(balance_lamports, dict) and "lamports" in balance_lamports:
            # Extract the lamports value from the dictionary
            balance_lamports = balance_lamports["lamports"]
        elif isinstance(balance_lamports, dict) and "value" in balance_lamports:
            # Alternative format sometimes returned
            balance_lamports = balance_lamports["value"]
            
        # Ensure balance_lamports is an integer before division
        if not isinstance(balance_lamports, (int, float)):
            # If we can't determine the format, return an error
            return {"error": "Invalid balance format returned", "error_explanation": f"Unexpected balance format: {type(balance_lamports)}"}
            
        balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        
        data = {
            "lamports": balance_lamports,
            "sol": balance_sol,
            "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
        }
        
        return format_response(data, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_account_details(address: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get account details with formatting.
    
    Args:
        address: Account address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted account information
    """
    try:
        account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
        
        # Make sure account_info is a dictionary
        if account_info is None:
            account_info = {}
            
        # Always include the address in the response
        account_info["address"] = address
            
        # Add owner program information if available
        if "owner" in account_info:
            owner = account_info["owner"]
            if owner == TOKEN_PROGRAM_ID:
                account_info["owner_program"] = "Token Program"
            elif owner == METADATA_PROGRAM_ID:
                account_info["owner_program"] = "Metadata Program"
        
        return format_response(account_info, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_token_accounts_for_owner(owner: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get token accounts owned by an address with formatting.
    
    Args:
        owner: Owner address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted token account information
    """
    try:
        token_accounts = await solana_client.get_token_accounts_by_owner(owner)
        
        # Add additional information
        result = {
            "owner": owner,
            "token_accounts": token_accounts,
            "token_count": len(token_accounts)
        }
        
        return format_response(result, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_token_details(mint: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get token details with formatting.
    
    Args:
        mint: Token mint address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted token information
    """
    try:
        # Get token supply
        supply = await solana_client.get_token_supply(mint)
        
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get largest token accounts
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Get market price data if available
        price_data = await solana_client.get_market_price(mint)
        
        # Compile all information
        token_info = {
            "mint": mint,
            "supply": supply,
            "metadata": metadata,
            "largest_accounts": largest_accounts,
            "price_data": price_data
        }
        
        return format_response(token_info, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The mint address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_transaction_history_for_address(
    address: str, 
    solana_client: SolanaClient, 
    limit: int = 20,
    before: str = None,
    format_level: str = "standard"
) -> Dict[str, Any]:
    """Get transaction history for an address with formatting.
    
    Args:
        address: Account address
        solana_client: Solana client
        limit: Maximum number of transactions
        before: Signature to search backwards from
        format_level: Response format level
        
    Returns:
        Formatted transaction history
    """
    try:
        # Get signatures - directly pass parameters in the correct order rather than using an options dict
        signatures = await solana_client.get_signatures_for_address(
            address,
            before=before,  # Pass before parameter directly
            limit=limit     # Pass limit parameter directly
        )
        
        result = {
            "address": address,
            "transactions": signatures
        }
        
        return format_response(result, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_nft_details(mint: str, solana_client: SolanaClient, format_level: str = "standard") -> Dict[str, Any]:
    """Get NFT details with formatting.
    
    Args:
        mint: NFT mint address
        solana_client: Solana client
        format_level: Response format level
        
    Returns:
        Formatted NFT information
    """
    try:
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get token account to find the owner
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Get the current owner if possible
        owner = None
        if largest_accounts and len(largest_accounts) > 0:
            # Get the account with the highest balance
            largest_account = largest_accounts[0]["address"]
            account_info = await solana_client.get_account_info(largest_account, encoding="jsonParsed")
            
            if "parsed" in account_info.get("data", {}):
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data:
                    owner = parsed_data["info"].get("owner")
        
        # Compile NFT information
        nft_info = {
            "mint": mint,
            "metadata": metadata,
            "owner": owner,
            "token_standard": "Unknown"  # In a real implementation, determine if it's NFT/SFT
        }
        
        return format_response(nft_info, format_level)
    except InvalidPublicKeyError as e:
        return {"error": str(e), "error_explanation": "The mint address provided is not a valid Solana public key."}
    except SolanaRpcError as e:
        return {"error": str(e), "error_explanation": "Error communicating with the Solana blockchain."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# -------------------------------------------
# REST API Endpoints with NLP and Context-aware formatting
# -------------------------------------------

async def rest_nlp_query(request):
    """Natural language query endpoint."""
    # Get request body
    try:
        body = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON body", "error_explanation": "The request body must be valid JSON"}, status_code=400)
    
    query = body.get("query")
    format_level = body.get("format_level", "auto")
    session_id = body.get("session_id")
    
    if not query:
        return JSONResponse({
            "error": "Missing 'query' field",
            "error_explanation": "The request must include a 'query' field with the natural language query"
        }, status_code=400)
    
    # Get or create session
    session = await get_or_create_session(session_id)
    
    # Check for semantic transaction search patterns
    transaction_search_patterns = [
        r"(?:find|search for|show me|get) (?:transactions|tx) (?:for|from|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44}) (?:with|that are|that have|related to|about) (.*)",
        r"(?:find|search for|show me|get) (.*) (?:transactions|tx) (?:for|from|by) (?:address |wallet |account )?([a-zA-Z0-9]{32,44})"
    ]
    
    for pattern in transaction_search_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Extract address and search terms
            groups = match.groups()
            if len(groups) >= 2:
                # For the first pattern
                address = groups[0]
                search_terms = groups[1]
                
                # Check if the pattern is reversed (second pattern)
                if not re.match(r"[a-zA-Z0-9]{32,44}", address):
                    search_terms = groups[0]
                    address = groups[1]
                
                # Perform semantic search
                solana_client = request.app.state.solana_client
                result = await semantic_transaction_search(address, search_terms, solana_client)
                
                # Update session
                session.add_query(query, result)
                
                # Return results
                return JSONResponse({
                    "result": result,
                    "session_id": session.id,
                    "query_count": len(session.query_history)
                })
    
    # If not a semantic search, process as normal NL query
    solana_client = request.app.state.solana_client
    result = await parse_natural_language_query(query, solana_client, session)
    
    # Return result with session ID
    return JSONResponse({
        "result": result,
        "session_id": session.id,
        "query_count": len(session.query_history)
    })


async def rest_session_history(request):
    """Get session query history."""
    session_id = request.path_params.get("session_id")
    
    if not session_id or session_id not in SESSION_STORE:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    session = SESSION_STORE[session_id]
    
    return JSONResponse({
        "session_id": session.id,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
        "query_count": len(session.query_history),
        "queries": session.query_history
    })


# -------------------------------------------
# Dynamic Schema Documentation Generator
# -------------------------------------------

def infer_schema_from_data(data: Any, schema_name: str = "object") -> Dict[str, Any]:
    """Dynamically infer a JSON schema from example data.
    
    Args:
        data: Example data
        schema_name: Name of the schema
        
    Returns:
        JSON schema
    """
    if data is None:
        return {"type": "null"}
    
    if isinstance(data, bool):
        return {"type": "boolean"}
    
    if isinstance(data, int):
        return {"type": "integer"}
    
    if isinstance(data, float):
        return {"type": "number"}
    
    if isinstance(data, str):
        # Try to detect if string is actually a base58 Solana address
        if len(data) >= 32 and all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in data):
            return {
                "type": "string",
                "description": "Solana public key in base58 encoding",
                "pattern": "^[1-9A-HJ-NP-Za-km-z]{32,44}$"
            }
        return {"type": "string"}
    
    if isinstance(data, list):
        if not data:
            return {"type": "array", "items": {}}
        
        # Check if all items are the same type
        item_types = [infer_schema_from_data(item) for item in data[:5]]  # Sample first 5
        
        if all(item_type == item_types[0] for item_type in item_types):
            return {"type": "array", "items": item_types[0]}
        else:
            # Mixed types, use oneOf schema
            return {
                "type": "array",
                "items": {"oneOf": item_types}
            }
    
    if isinstance(data, dict):
        properties = {}
        required = []
        
        # Build schema for each property, but limit to reasonable size
        for key, value in list(data.items())[:20]:  # Limit to 20 properties
            properties[key] = infer_schema_from_data(value)
            
            # Consider non-null values as required
            if value is not None:
                required.append(key)
        
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
        
        if schema_name != "object":
            schema["title"] = schema_name
        
        return schema
    
    # Default
    return {"type": "string"}


async def generate_schema_from_example(schema_type: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Generate a schema by inferring from example data.
    
    This is a safer implementation that handles potential RPC failures.
    
    Args:
        schema_type: Type of schema to generate
        solana_client: Solana client
        
    Returns:
        Inferred schema
    """
    # Default schemas if live data fetching fails
    default_schemas = {
        "account": {
            "type": "object",
            "title": "AccountInfo",
            "properties": {
                "lamports": {"type": "integer"},
                "owner": {"type": "string"},
                "executable": {"type": "boolean"},
                "data": {"type": "object"}
            }
        },
        "token": {
            "type": "object",
            "title": "TokenInfo",
            "properties": {
                "mint": {"type": "string"},
                "supply": {"type": "object"},
                "decimals": {"type": "integer"},
                "metadata": {"type": "object"}
            }
        },
        "transaction": {
            "type": "object",
            "title": "Transaction",
            "properties": {
                "signature": {"type": "string"},
                "slot": {"type": "integer"},
                "err": {"type": ["object", "null"]},
                "blockTime": {"type": "integer"}
            }
        },
        "nft": {
            "type": "object",
            "title": "NFT",
            "properties": {
                "mint": {"type": "string"},
                "owner": {"type": "string"},
                "metadata": {"type": "object"},
                "token_standard": {"type": "string"}
            }
        }
    }
    
    try:
        # Fetch example data based on schema type
        if schema_type == "account":
            # Use system program as an example account
            data = await solana_client.get_account_info("11111111111111111111111111111111", encoding="jsonParsed")
            if data:
                return infer_schema_from_data(data, "AccountInfo")
            
        elif schema_type == "token":
            # Try multiple token mints in case some are unavailable
            token_mints = [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "So11111111111111111111111111111111111111112",   # Wrapped SOL
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"   # USDT
            ]
            
            for mint in token_mints:
                try:
                    supply = await solana_client.get_token_supply(mint)
                    metadata = await solana_client.get_token_metadata(mint)
                    
                    token_data = {
                        "mint": mint,
                        "supply": supply,
                        "decimals": 6,  # Assuming 6 decimals, could get from metadata
                        "metadata": metadata
                    }
                    return infer_schema_from_data(token_data, "TokenInfo")
                except Exception:
                    # Try next mint
                    continue
            
        elif schema_type == "transaction":
            try:
                # Try to get a recent transaction
                recent_blocks = await solana_client._make_request("getBlocks", [
                    await solana_client.get_slot() - 10,
                    await solana_client.get_slot()
                ])
                
                if recent_blocks and len(recent_blocks) > 0:
                    # Get transaction from recent block
                    block_info = await solana_client._make_request("getBlock", [
                        recent_blocks[0],
                        {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                    ])
                    
                    if block_info and "transactions" in block_info and len(block_info["transactions"]) > 0:
                        return infer_schema_from_data(block_info["transactions"][0], "Transaction")
            except Exception:
                # Fall back to default schema
                pass
                
        # For NFT or any other type, fall back to default schema
        return default_schemas.get(schema_type, {
            "type": "object",
            "title": schema_type,
            "properties": {}
        })
        
    except Exception as e:
        print(f"Error generating schema for {schema_type}: {str(e)}")
        # Return default schema for this type
        return default_schemas.get(schema_type, {
            "type": "object",
            "title": schema_type,
            "properties": {}
        })


# Update the schemas endpoint
async def rest_schemas(request):
    """Get schemas for blockchain data structures."""
    schema_name = request.path_params.get("schema")
    dynamic = request.query_params.get("dynamic", "").lower() in ["true", "1", "yes"]
    
    solana_client = request.app.state.solana_client
    
    # Define base schemas
    base_schemas = {
        "account": {
            "type": "object",
            "properties": {
                "lamports": {"type": "integer", "description": "Account balance in lamports"},
                "owner": {"type": "string", "description": "Program that owns this account"},
                "executable": {"type": "boolean", "description": "Whether the account contains executable code"},
                "rentEpoch": {"type": "integer", "description": "Epoch at which rent was last collected"},
                "data": {"type": "object", "description": "Account data, either as encoded binary or parsed JSON"}
            }
        },
        "token": {
            "type": "object",
            "properties": {
                "mint": {"type": "string", "description": "Token mint address"},
                "owner": {"type": "string", "description": "Token account owner"},
                "amount": {"type": "string", "description": "Token amount as string to handle large numbers"},
                "decimals": {"type": "integer", "description": "Number of decimal places"},
                "uiAmount": {"type": "number", "description": "Token amount in user interface representation"}
            }
        },
        "transaction": {
            "type": "object",
            "properties": {
                "signature": {"type": "string", "description": "Transaction signature"},
                "slot": {"type": "integer", "description": "Slot in which the transaction was processed"},
                "err": {"type": ["object", "null"], "description": "Error if transaction failed, null if successful"},
                "confirmationStatus": {"type": "string", "description": "Transaction confirmation status"},
                "confirmations": {"type": ["integer", "null"], "description": "Number of confirmations"}
            }
        },
        "nft": {
            "type": "object",
            "properties": {
                "mint": {"type": "string", "description": "NFT mint address"},
                "owner": {"type": "string", "description": "Current NFT owner"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "NFT name"},
                        "symbol": {"type": "string", "description": "NFT symbol"},
                        "uri": {"type": "string", "description": "URI to off-chain metadata"},
                        "sellerFeeBasisPoints": {"type": "integer", "description": "Royalty in basis points"}
                    }
                }
            }
        }
    }
    
    if schema_name:
        if schema_name in base_schemas:
            if dynamic:
                # Generate schema dynamically from example data
                schema = await generate_schema_from_example(schema_name, solana_client)
                return JSONResponse(schema)
            else:
                return JSONResponse(base_schemas[schema_name])
        else:
            return JSONResponse({"error": f"Schema '{schema_name}' not found"}, status_code=404)
    
    # Return list of available schemas
    if dynamic:
        # Generate all schemas dynamically
        schemas = {}
        for name in base_schemas.keys():
            schemas[name] = await generate_schema_from_example(name, solana_client)
        
        return JSONResponse({
            "available_schemas": list(schemas.keys()),
            "schemas": schemas,
            "note": "These schemas were dynamically generated from example data"
        })
    else:
        return JSONResponse({
            "available_schemas": list(base_schemas.keys()),
            "schemas": base_schemas,
            "note": "Use ?dynamic=true to generate schemas from real blockchain data"
        })


async def rest_chain_analysis(request):
    """Analyze patterns in token movements or transactions."""
    try:
        body = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    
    analysis_type = body.get("type")
    address = body.get("address")
    
    if not analysis_type or not address:
        return JSONResponse({"error": "Missing required fields"}, status_code=400)
    
    solana_client = request.app.state.solana_client
    
    # Perform analysis based on type
    if analysis_type == "token_flow":
        # Analyze token inflows and outflows
        try:
            # Get transaction history - pass parameters directly
            signatures = await solana_client.get_signatures_for_address(address, limit=50)
            
            # Process transactions to find token movements
            inflows = []
            outflows = []
            
            for tx_info in signatures:
                # Get full transaction details
                tx = await solana_client.get_transaction(tx_info["signature"])
                
                # Analyze token transfers (simplified)
                # In a real implementation, you'd parse program instructions
                # to identify token transfers and their directions
                if "meta" in tx and "postTokenBalances" in tx["meta"] and "preTokenBalances" in tx["meta"]:
                    for pre, post in zip(tx["meta"]["preTokenBalances"], tx["meta"]["postTokenBalances"]):
                        if pre["owner"] == address and pre["uiTokenAmount"]["uiAmount"] > post["uiTokenAmount"]["uiAmount"]:
                            outflows.append({
                                "mint": pre["mint"],
                                "amount": pre["uiTokenAmount"]["uiAmount"] - post["uiTokenAmount"]["uiAmount"],
                                "timestamp": tx_info.get("blockTime")
                            })
                        elif post["owner"] == address and post["uiTokenAmount"]["uiAmount"] > pre["uiTokenAmount"]["uiAmount"]:
                            inflows.append({
                                "mint": post["mint"],
                                "amount": post["uiTokenAmount"]["uiAmount"] - pre["uiTokenAmount"]["uiAmount"],
                                "timestamp": tx_info.get("blockTime")
                            })
            
            return JSONResponse({
                "address": address,
                "analysis_type": "token_flow",
                "inflows": inflows,
                "outflows": outflows,
                "total_inflow_count": len(inflows),
                "total_outflow_count": len(outflows)
            })
        
        except Exception as e:
            return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)
    
    elif analysis_type == "activity_pattern":
        # Analyze activity patterns over time
        try:
            # Get transaction history - pass parameters directly
            signatures = await solana_client.get_signatures_for_address(address, limit=100)
            
            # Group by day
            activity_by_day = {}
            
            for tx in signatures:
                if "blockTime" in tx:
                    # Convert timestamp to date string
                    date = datetime.datetime.fromtimestamp(tx["blockTime"]).strftime("%Y-%m-%d")
                    
                    if date not in activity_by_day:
                        activity_by_day[date] = 0
                    
                    activity_by_day[date] += 1
            
            # Sort by date
            sorted_activity = [{"date": k, "transactions": v} for k, v in sorted(activity_by_day.items())]
            
            return JSONResponse({
                "address": address,
                "analysis_type": "activity_pattern",
                "activity_by_day": sorted_activity,
                "total_days": len(sorted_activity),
                "total_transactions": len(signatures)
            })
        
        except Exception as e:
            return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)
    
    else:
        return JSONResponse({"error": f"Unknown analysis type: {analysis_type}"}, status_code=400)


# -------------------------------------------
# Account Information Resources
# -------------------------------------------

@app.resource("solana://account/{address}")
async def get_account(address: str) -> str:
    """Get Solana account information.
    
    Args:
        address: The account address
        
    Returns:
        Account information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
            return json.dumps(account_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@app.resource("solana://balance/{address}")
async def get_balance(address: str) -> str:
    """Get Solana account balance.
    
    Args:
        address: The account address
        
    Returns:
        Account balance in SOL
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            balance_lamports = await solana_client.get_balance(address)
            balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
            return json.dumps({
                "lamports": balance_lamports,
                "sol": balance_sol,
                "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
            }, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -------------------------------------------
# Token Information Resources
# -------------------------------------------

@app.resource("solana://tokens/{owner}")
async def get_token_accounts(owner: str) -> str:
    """Get token accounts owned by an address.
    
    Args:
        owner: The owner address
        
    Returns:
        Token account information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            token_accounts = await solana_client.get_token_accounts_by_owner(owner)
            return json.dumps(token_accounts, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@app.resource("solana://token/{mint}")
async def get_token_info(mint: str) -> str:
    """Get token information.
    
    Args:
        mint: The token mint address
        
    Returns:
        Token information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            # Get token supply
            supply = await solana_client.get_token_supply(mint)
            
            # Get token metadata
            metadata = await solana_client.get_token_metadata(mint)
            
            # Get largest token accounts
            largest_accounts = await solana_client.get_token_largest_accounts(mint)
            
            # Get market price data if available
            price_data = await solana_client.get_market_price(mint)
            
            # Compile all information
            token_info = {
                "mint": mint,
                "supply": supply,
                "metadata": metadata,
                "largest_accounts": largest_accounts,
                "price_data": price_data
            }
            
            return json.dumps(token_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@app.resource("solana://token/{mint}/holders")
async def get_token_holders(mint: str) -> str:
    """Get token holders for a mint.
    
    Args:
        mint: The token mint address
        
    Returns:
        Token holders information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            # Get largest accounts for this token
            largest_accounts = await solana_client.get_token_largest_accounts(mint)
            
            # For each account, get the owner
            holders = []
            for account in largest_accounts:
                # Get the account info to find the owner
                account_info = await solana_client.get_account_info(
                    account["address"], 
                    encoding="jsonParsed"
                )
                
                # Extract owner and balance information
                if "parsed" in account_info.get("data", {}):
                    parsed_data = account_info["data"]["parsed"]
                    if "info" in parsed_data:
                        info = parsed_data["info"]
                        holders.append({
                            "owner": info.get("owner"),
                            "address": account["address"],
                            "amount": info.get("tokenAmount", {}).get("amount"),
                            "decimals": info.get("tokenAmount", {}).get("decimals"),
                            "uiAmount": info.get("tokenAmount", {}).get("uiAmount")
                        })
            
            return json.dumps({"mint": mint, "holders": holders}, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -------------------------------------------
# Transaction Resources
# -------------------------------------------

@app.resource("solana://transaction/{signature}")
async def get_transaction_details(signature: str) -> str:
    """Get transaction details.
    
    Args:
        signature: The transaction signature
        
    Returns:
        Transaction details as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            tx_details = await solana_client.get_transaction(signature)
            return json.dumps(tx_details, indent=2)
    except ValueError as e:  # Invalid signature format
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@app.resource("solana://transactions/{address}")
async def get_address_transactions(address: str) -> str:
    """Get transaction history for an address.
    
    Args:
        address: The account address
        
    Returns:
        Transaction history as JSON string
    """
    limit = 20  # Default value
    before = None  # Default value
    
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            # Get signatures
            # Create options dictionary
            options = {"limit": limit}
            if before:
                options["before"] = before
                
            signatures = await solana_client.get_signatures_for_address(
                address,
                options
            )
            
            # For detailed view, we could get full transaction details
            # But that would be a lot of RPC calls
            # Instead just return the signatures with metadata
            
            return json.dumps({
                "address": address,
                "transactions": signatures
            }, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -------------------------------------------
# Program Resources
# -------------------------------------------

@app.resource("solana://program/{program_id}")
async def get_program_info(program_id: str) -> str:
    """Get program information.
    
    Args:
        program_id: The program ID
        
    Returns:
        Program information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            # Get the program account itself
            program_account = await solana_client.get_account_info(program_id, encoding="jsonParsed")
            
            # Prepare result
            program_info = {
                "program_id": program_id,
                "account": program_account,
                "is_executable": program_account.get("executable", False),
                "owner": program_account.get("owner"),
                "lamports": program_account.get("lamports", 0)
            }
            
            return json.dumps(program_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@app.resource("solana://program/{program_id}/accounts")
async def get_program_account_list(program_id: str) -> str:
    """Get accounts owned by a program.
    
    Args:
        program_id: The program ID
        
    Returns:
        Program accounts as JSON string
    """
    # Default values
    limit = 10  # Default to 10 to avoid large responses
    offset = 0
    memcmp = None  # JSON-encoded memcmp filter
    datasize = None  # Filter by data size
    
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            filters = []
            
            # Add memcmp filter if provided
            if memcmp:
                try:
                    memcmp_filter = json.loads(memcmp)
                    filters.append({"memcmp": memcmp_filter})
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid memcmp JSON format"})
            
            # Add datasize filter if provided
            if datasize is not None:
                filters.append({"dataSize": datasize})
            
            # Get program accounts with pagination
            accounts = await solana_client.get_program_accounts(
                program_id,
                filters=filters if filters else None,
                encoding="jsonParsed",
                limit=limit,
                offset=offset
            )
            
            # Count total found
            account_count = len(accounts)
            
            return json.dumps({
                "program_id": program_id,
                "count": account_count,
                "limit": limit,
                "offset": offset,
                "accounts": accounts
            }, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -------------------------------------------
# Network Status Resources
# -------------------------------------------

@app.resource("solana://network/epoch")
async def get_network_epoch() -> str:
    """Get current epoch information.
    
    Returns:
        Epoch information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            epoch_info = await solana_client.get_epoch_info()
            inflation_rate = await solana_client.get_inflation_rate()
            
            # Combine information
            network_info = {
                "epoch_info": epoch_info,
                "inflation_rate": inflation_rate
            }
            
            return json.dumps(network_info, indent=2)
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@app.resource("solana://network/validators")
async def get_network_validators() -> str:
    """Get information about validators.
    
    Returns:
        Validator information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            nodes = await solana_client.get_cluster_nodes()
            
            return json.dumps({
                "validators": nodes
            }, indent=2)
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -------------------------------------------
# NFT Resources
# -------------------------------------------

@app.resource("solana://nft/{mint}")
async def get_nft_info(mint: str) -> str:
    """Get NFT information.
    
    Args:
        mint: The NFT mint address
        
    Returns:
        NFT information as JSON string
    """
    try:
        from solana_mcp.solana_client import get_solana_client
        async with get_solana_client() as solana_client:
            # Get token metadata
            metadata = await solana_client.get_token_metadata(mint)
            
            # Get token account to find the owner
            largest_accounts = await solana_client.get_token_largest_accounts(mint)
            
            # Get the current owner if possible
            owner = None
            if largest_accounts and len(largest_accounts) > 0:
                # Get the account with the highest balance
                largest_account = largest_accounts[0]["address"]
                account_info = await solana_client.get_account_info(largest_account, encoding="jsonParsed")
                
                if "parsed" in account_info.get("data", {}):
                    parsed_data = account_info["data"]["parsed"]
                    if "info" in parsed_data:
                        owner = parsed_data["info"].get("owner")
                
            # Compile NFT information
            nft_info = {
                "mint": mint,
                "metadata": metadata,
                "owner": owner,
                "token_standard": "Unknown"  # In a real implementation, determine if it's NFT/SFT
            }
            
            return json.dumps(nft_info, indent=2)
    except InvalidPublicKeyError as e:
        return json.dumps({"error": str(e)})
    except SolanaRpcError as e:
        return json.dumps({"error": str(e), "details": e.error_data})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -------------------------------------------
# Tool Implementations
# -------------------------------------------

@app.tool()
async def get_solana_account(ctx: Context, address: str) -> str:
    """Fetch a Solana account's information.
    
    Args:
        ctx: The request context
        address: The account public key
        
    Returns:
        Formatted account information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
        return json.dumps(account_info, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching account: {str(e)}"


@app.tool()
async def get_solana_balance(ctx: Context, address: str) -> str:
    """Fetch a Solana account's balance.
    
    Args:
        ctx: The request context
        address: The account public key
        
    Returns:
        Account balance in SOL
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        balance_lamports = await solana_client.get_balance(address)
        balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        return f"{balance_sol} SOL ({balance_lamports} lamports)"
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching balance: {str(e)}"


@app.tool()
async def get_token_balance(ctx: Context, token_account: str) -> str:
    """Get the balance of a token account.
    
    Args:
        ctx: The request context
        token_account: The token account address
        
    Returns:
        Token balance information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        response = await solana_client._make_request(
            "getTokenAccountBalance", 
            [token_account]
        )
        return json.dumps(response, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching token balance: {str(e)}"


@app.tool()
async def get_program_accounts(
    ctx: Context, 
    program_id: str,
    limit: int = 10,
    filter_json: str = None
) -> str:
    """Get accounts owned by a program.
    
    Args:
        ctx: The request context
        program_id: The program ID
        limit: Maximum number of accounts to return
        filter_json: JSON string of filters to apply
        
    Returns:
        List of accounts owned by the program
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        filters = None
        if filter_json:
            try:
                filters = json.loads(filter_json)
            except json.JSONDecodeError:
                return "Error: Invalid filter JSON format"
                
        accounts = await solana_client.get_program_accounts(
            program_id, 
            filters=filters,
            encoding="jsonParsed",
            limit=limit
        )
        
        # Count total accounts for reporting
        account_count = len(accounts)
        
        if account_count > limit:
            accounts = accounts[:limit]
            return f"Found {account_count} accounts (showing first {limit}):\n{json.dumps(accounts, indent=2)}"
        else:
            return json.dumps(accounts, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching program accounts: {str(e)}"


@app.tool()
async def get_recent_blockhash(ctx: Context) -> str:
    """Get a recent blockhash.
    
    Args:
        ctx: The request context
        
    Returns:
        Recent blockhash information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        blockhash = await solana_client.get_recent_blockhash()
        return json.dumps(blockhash, indent=2)
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching recent blockhash: {str(e)}"


@app.tool()
async def get_token_supply(ctx: Context, mint: str) -> str:
    """Get token supply for a mint.
    
    Args:
        ctx: The request context
        mint: The mint address
        
    Returns:
        Token supply information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        supply = await solana_client.get_token_supply(mint)
        return json.dumps(supply, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching token supply: {str(e)}"


@app.tool()
async def execute_rpc_method(ctx: Context, method: str, params: str) -> str:
    """Execute a custom Solana RPC method.
    
    Args:
        ctx: The request context
        method: The RPC method name
        params: JSON string of parameters
        
    Returns:
        RPC response
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        # Parse params string to list/dict
        try:
            parsed_params = json.loads(params)
        except json.JSONDecodeError:
            return "Error: params must be valid JSON"
            
        result = await solana_client._make_request(method, parsed_params)
        return json.dumps(result, indent=2)
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error executing RPC method: {str(e)}"


@app.tool()
async def get_token_metadata(ctx: Context, mint: str) -> str:
    """Get token or NFT metadata.
    
    Args:
        ctx: The request context
        mint: The mint address
        
    Returns:
        Token metadata information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        # This calls the centralized implementation in solana_client which delegates to TokenClient
        metadata = await solana_client.get_token_metadata(mint)
        return json.dumps(metadata, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching token metadata: {str(e)}"


@app.tool()
async def get_token_holders(ctx: Context, mint: str, limit: int = 10) -> str:
    """Get holders for a token.
    
    Args:
        ctx: The request context
        mint: The mint address
        limit: Maximum number of holders to return
        
    Returns:
        Token holders information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        # Get largest accounts for this token
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Limit the number of accounts to process
        accounts_to_process = largest_accounts[:limit] if len(largest_accounts) > limit else largest_accounts
        
        # For each account, get the owner
        holders = []
        for account in accounts_to_process:
            # Get the account info to find the owner
            account_info = await solana_client.get_account_info(
                account["address"], 
                encoding="jsonParsed"
            )
            
            # Extract owner and balance information
            if "parsed" in account_info.get("data", {}):
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data:
                    info = parsed_data["info"]
                    holders.append({
                        "owner": info.get("owner"),
                        "address": account["address"],
                        "amount": info.get("tokenAmount", {}).get("amount"),
                        "decimals": info.get("tokenAmount", {}).get("decimals"),
                        "uiAmount": info.get("tokenAmount", {}).get("uiAmount")
                    })
        
        return json.dumps({
            "mint": mint, 
            "total_holders": len(largest_accounts),
            "holders_shown": len(holders),
            "holders": holders
        }, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching token holders: {str(e)}"


@app.tool()
async def get_transaction_history(
    ctx: Context, 
    address: str, 
    limit: int = 10, 
    before: str = None
) -> str:
    """Get transaction history for an address.
    
    Args:
        ctx: The request context
        address: The account address
        limit: Maximum number of transactions to return
        before: Signature to search backwards from
        
    Returns:
        Transaction history
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        signatures = await solana_client.get_signatures_for_address(
            address, 
            before=before, 
            limit=limit
        )
        
        return json.dumps({
            "address": address,
            "transaction_count": len(signatures),
            "transactions": signatures
        }, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: {str(e)}"
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching transaction history: {str(e)}"


@app.tool()
async def get_network_status(ctx: Context) -> str:
    """Get current Solana network status.
    
    Args:
        ctx: The request context
        
    Returns:
        Network status information
    """
    solana_client = ctx.request_context.lifespan_context.solana_client
    try:
        # Get various network stats
        epoch_info = await solana_client.get_epoch_info()
        slot = await solana_client.get_slot()
        recent_blockhash = await solana_client.get_recent_blockhash()
        
        # Combine into status report
        status = {
            "slot": slot,
            "epoch": epoch_info.get("epoch"),
            "epoch_progress": f"{epoch_info.get('slotIndex', 0) / max(1, epoch_info.get('slotsInEpoch', 1)) * 100:.2f}%",
            "recent_blockhash": recent_blockhash.get("blockhash"),
            "last_valid_block_height": recent_blockhash.get("lastValidBlockHeight")
        }
        
        return json.dumps(status, indent=2)
    except SolanaRpcError as e:
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        return f"Error fetching network status: {str(e)}"


# -------------------------------------------
# Prompts
# -------------------------------------------

@app.prompt()
def solana_query() -> types.Prompt:
    """Define a prompt for querying Solana data."""
    return types.Prompt(
        name="solana_query",
        description="Query for Solana blockchain data",
        arguments=[
            types.PromptArgument(
                name="query_type",
                description="Type of query (account, balance, transaction, token, nft, program)",
                required=True,
            ),
            types.PromptArgument(
                name="address",
                description="Solana address to query",
                required=True,
            ),
        ],
    )


@app.prompt()
def token_analysis() -> types.Prompt:
    """Define a prompt for analyzing tokens."""
    return types.Prompt(
        name="token_analysis",
        description="Analyze a Solana token's data",
        arguments=[
            types.PromptArgument(
                name="mint",
                description="Token mint address",
                required=True,
            ),
        ],
    )


# -------------------------------------------
# Server Runner
# -------------------------------------------

# Add REST API handlers

async def rest_get_account(request):
    """REST API endpoint for getting account info"""
    address = request.path_params.get("address")
    if not address:
        return JSONResponse({
            "error": "Missing address parameter",
            "error_explanation": "The account address must be provided in the URL path."
        }, status_code=400)
        
    solana_client = request.app.state.solana_client
    try:
        account_info = await solana_client.get_account_info(address, encoding="jsonParsed")
        return JSONResponse(account_info)
    except InvalidPublicKeyError as e:
        return JSONResponse({
            "error": str(e),
            "error_explanation": explain_solana_error(str(e))
        }, status_code=400)
    except SolanaRpcError as e:
        return JSONResponse({
            "error": str(e),
            "error_explanation": explain_solana_error(str(e)),
            "details": e.error_data
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "error": f"Unexpected error: {str(e)}",
            "error_explanation": "An unexpected error occurred while processing your request."
        }, status_code=500)


async def rest_get_balance(request):
    """REST API endpoint for getting account balance"""
    address = request.path_params.get("address")
    if not address:
        return JSONResponse({
            "error": "Missing address parameter",
            "error_explanation": "The account address must be provided in the URL path."
        }, status_code=400)
        
    solana_client = request.app.state.solana_client
    try:
        balance_lamports = await solana_client.get_balance(address)
        balance_sol = balance_lamports / 1_000_000_000  # Convert lamports to SOL
        return JSONResponse({
            "lamports": balance_lamports,
            "sol": balance_sol,
            "formatted": f"{balance_sol} SOL ({balance_lamports} lamports)"
        })
    except InvalidPublicKeyError as e:
        return JSONResponse({
            "error": str(e),
            "error_explanation": explain_solana_error(str(e))
        }, status_code=400)
    except SolanaRpcError as e:
        return JSONResponse({
            "error": str(e),
            "error_explanation": explain_solana_error(str(e)),
            "details": e.error_data
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "error": f"Unexpected error: {str(e)}",
            "error_explanation": "An unexpected error occurred while processing your request."
        }, status_code=500)


async def rest_get_token_info(request):
    """REST API endpoint for getting token info"""
    mint = request.path_params.get("mint")
    solana_client = request.app.state.solana_client
    try:
        # Get token supply
        supply = await solana_client.get_token_supply(mint)
        
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get largest token accounts
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Get market price data if available
        price_data = await solana_client.get_market_price(mint)
        
        # Compile all information
        token_info = {
            "mint": mint,
            "supply": supply,
            "metadata": metadata,
            "largest_accounts": largest_accounts,
            "price_data": price_data
        }
        
        return JSONResponse(token_info)
    except InvalidPublicKeyError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except SolanaRpcError as e:
        return JSONResponse({"error": str(e), "details": e.error_data}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)


async def rest_get_transactions(request):
    """REST API endpoint for getting transaction history"""
    address = request.path_params.get("address")
    limit = int(request.query_params.get("limit", "20"))
    before = request.query_params.get("before")
    search_query = request.query_params.get("search")
    
    solana_client = request.app.state.solana_client
    
    # Handle semantic search if query provided
    if search_query:
        try:
            results = await semantic_transaction_search(address, search_query, solana_client, limit)
            return JSONResponse(results)
        except InvalidPublicKeyError as e:
            return JSONResponse({
                "error": str(e), 
                "error_explanation": explain_solana_error(str(e))
            }, status_code=400)
        except SolanaRpcError as e:
            return JSONResponse({
                "error": str(e), 
                "error_explanation": explain_solana_error(str(e)), 
                "details": e.error_data
            }, status_code=500)
        except Exception as e:
            return JSONResponse({
                "error": f"Unexpected error: {str(e)}",
                "error_explanation": "An unexpected error occurred while processing your request."
            }, status_code=500)
    
    # Otherwise, get regular transaction history
    try:
        # Get signatures
        # Create options dictionary
        options = {"limit": limit}
        if before:
            options["before"] = before
            
        signatures = await solana_client.get_signatures_for_address(
            address,
            options
        )
        
        return JSONResponse({
            "address": address,
            "transactions": signatures
        })
    except InvalidPublicKeyError as e:
        return JSONResponse({
            "error": str(e), 
            "error_explanation": explain_solana_error(str(e))
        }, status_code=400)
    except SolanaRpcError as e:
        return JSONResponse({
            "error": str(e), 
            "error_explanation": explain_solana_error(str(e)), 
            "details": e.error_data
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "error": f"Unexpected error: {str(e)}",
            "error_explanation": "An unexpected error occurred while processing your request."
        }, status_code=500)


async def rest_get_nft_info(request):
    """REST API endpoint for getting NFT info"""
    mint = request.path_params.get("mint")
    solana_client = request.app.state.solana_client
    try:
        # Get token metadata
        metadata = await solana_client.get_token_metadata(mint)
        
        # Get token account to find the owner
        largest_accounts = await solana_client.get_token_largest_accounts(mint)
        
        # Get the current owner if possible
        owner = None
        if largest_accounts and len(largest_accounts) > 0:
            # Get the account with the highest balance
            largest_account = largest_accounts[0]["address"]
            account_info = await solana_client.get_account_info(largest_account, encoding="jsonParsed")
            
            if "parsed" in account_info.get("data", {}):
                parsed_data = account_info["data"]["parsed"]
                if "info" in parsed_data:
                    owner = parsed_data["info"].get("owner")
        
        # Compile NFT information
        nft_info = {
            "mint": mint,
            "metadata": metadata,
            "owner": owner,
            "token_standard": "Unknown"  # In a real implementation, determine if it's NFT/SFT
        }
        
        return JSONResponse(nft_info)
    except InvalidPublicKeyError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except SolanaRpcError as e:
        return JSONResponse({"error": str(e), "details": e.error_data}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)


async def health_check(request):
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "service": "solana-mcp"})


def explain_solana_error(error_message: str) -> str:
    """Convert Solana error messages to user-friendly explanations.
    
    Args:
        error_message: The error message from the Solana client
        
    Returns:
        A user-friendly explanation of the error
    """
    error_message = error_message.lower()
    
    if "invalid public key" in error_message:
        return "The address provided is not a valid Solana account address."
    elif "not found" in error_message or "does not exist" in error_message:
        return "The requested account or data does not exist on the Solana blockchain."
    elif "insufficient funds" in error_message:
        return "The account does not have enough SOL to perform this operation."
    elif "rate limited" in error_message:
        return "The request was rate limited by the Solana RPC node. Please try again later."
    elif "timed out" in error_message:
        return "The request timed out. The Solana network might be experiencing high load."
    elif "rpc error" in error_message:
        return "There was an error communicating with the Solana blockchain."
    else:
        return "An error occurred while processing your request on the Solana blockchain."


async def api_docs(request):
    """API documentation endpoint"""
    api_paths = {
        "/api/account/{address}": {
            "get": {
                "summary": "Get account information",
                "parameters": [
                    {
                        "name": "address",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Solana account address"
                    }
                ]
            }
        },
        "/api/balance/{address}": {
            "get": {
                "summary": "Get account balance",
                "parameters": [
                    {
                        "name": "address",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Solana account address"
                    }
                ]
            }
        },
        "/api/token/{mint}": {
            "get": {
                "summary": "Get token information",
                "parameters": [
                    {
                        "name": "mint",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Token mint address"
                    }
                ]
            }
        },
        "/api/transactions/{address}": {
            "get": {
                "summary": "Get transaction history",
                "parameters": [
                    {
                        "name": "address",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Solana account address"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer"},
                        "description": "Maximum number of transactions to return"
                    },
                    {
                        "name": "before",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Signature to search backwards from"
                    },
                    {
                        "name": "search",
                        "in": "query",
                        "schema": {"type": "string"},
                        "description": "Semantic search query for transactions"
                    }
                ]
            }
        },
        "/api/nft/{mint}": {
            "get": {
                "summary": "Get NFT information",
                "parameters": [
                    {
                        "name": "mint",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "NFT mint address"
                    }
                ]
            }
        },
        "/api/nlp/query": {
            "post": {
                "summary": "Process natural language queries about Solana",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "format_level": {"type": "string", "enum": ["minimal", "standard", "detailed", "auto"]},
                                    "session_id": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                }
            }
        },
        "/api/session/{session_id}": {
            "get": {
                "summary": "Get session history",
                "parameters": [
                    {
                        "name": "session_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Session ID"
                    }
                ]
            }
        },
        "/api/schemas": {
            "get": {
                "summary": "Get all available schemas"
            }
        },
        "/api/schemas/{schema}": {
            "get": {
                "summary": "Get schema for a specific data type",
                "parameters": [
                    {
                        "name": "schema",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Schema name"
                    }
                ]
            }
        },
        "/api/analysis": {
            "post": {
                "summary": "Analyze blockchain patterns",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["token_flow", "activity_pattern"]},
                                    "address": {"type": "string"}
                                },
                                "required": ["type", "address"]
                            }
                        }
                    }
                }
            }
        },
        "/api/whale-detector": {
            "post": {
                "summary": "Find whale wallets for a specific token",
                "description": "Identifies accounts holding significant amounts of a token and having high overall value",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "token_address": {"type": "string", "description": "Solana token mint address"}
                                },
                                "required": ["token_address"]
                            }
                        }
                    }
                }
            }
        },
        "/api/fresh-wallet-detector": {
            "post": {
                "summary": "Find fresh wallet holders for a specific token",
                "description": "Identifies newly created wallets or wallets primarily holding only the target token",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "token_address": {"type": "string", "description": "Solana token mint address"}
                                },
                                "required": ["token_address"]
                            }
                        }
                    }
                }
            }
        },
        "/health": {
            "get": {
                "summary": "Health check endpoint"
            }
        }
    }
    
    # Build final OpenAPI documentation
    docs = {
        "openapi": "3.0.0",
        "info": {
            "title": "Solana MCP Server API",
            "description": "API for interacting with Solana blockchain, includes whale and fresh wallet detection",
            "version": "1.0.0"
        },
        "paths": api_paths
    }
    
    return JSONResponse(docs)


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    help="Transport type (stdio or sse)",
)
@click.option("--port", type=int, help="Port to listen on for SSE transport")
@click.option("--host", type=str, help="Host to bind to for SSE transport")
def run_server(
    transport: Optional[str] = None,
    port: Optional[int] = None,
    host: Optional[str] = None,
) -> None:
    """Run the Solana MCP server.
    
    Args:
        transport: Transport type. Defaults to environment setting or "stdio".
        port: Port to listen on for SSE transport. Defaults to environment setting or 8000.
        host: Host to bind to for SSE transport. Defaults to environment setting or "0.0.0.0".
    """
    # Get server config with environment defaults
    config = get_server_config()
    
    # Override with CLI options if provided
    if transport:
        config.transport = transport
    if port:
        config.port = port
    if host:
        config.host = host
    
    # Run server with appropriate transport
    if config.transport == "sse":
        # Set up SSE transport
        sse = SseServerTransport("/messages/")
        
        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run_sse_async(streams[0], streams[1])
        
        # Set up REST API endpoints alongside SSE
        # Create Starlette app with CORS middleware
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],  # In production, specify actual origins
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            Middleware(
                SessionMiddleware,
                secret_key="solana-mcp-server-secret",  # Use a proper secret in production
                max_age=60 * 60 * 24  # 1 day
            )
        ]
        
        starlette_app = Starlette(
            debug=config.debug,
            middleware=middleware,
            routes=[
                # SSE routes
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
                
                # REST API endpoints
                Route("/api/account/{address}", rest_get_account),
                Route("/api/balance/{address}", rest_get_balance),
                Route("/api/token/{mint}", rest_get_token_info),
                Route("/api/transactions/{address}", rest_get_transactions),
                Route("/api/nft/{mint}", rest_get_nft_info),
                
                # New LLM-focused endpoints
                Route("/api/nlp/query", rest_nlp_query, methods=["POST"]),
                Route("/api/session/{session_id}", rest_session_history),
                Route("/api/schemas", rest_schemas),
                Route("/api/schemas/{schema}", rest_schemas),
                Route("/api/analysis", rest_chain_analysis, methods=["POST"]),
                
                # Whale and fresh wallet detection endpoints
                Route("/api/whale-detector", rest_whale_detector, methods=["POST"]),
                Route("/api/fresh-wallet-detector", rest_fresh_wallet_detector, methods=["POST"]),
                
                # Service endpoints
                Route("/health", health_check),
                Route("/api/docs", api_docs),
            ],
        )
        
        # Set up shared Solana client for the REST endpoints
        @starlette_app.on_event("startup")
        async def startup():
            """Initialize resources on server startup."""
            try:
                # Create a Solana client for the application
                solana_client = await get_solana_client().__aenter__()
                starlette_app.state.solana_client = solana_client
                
                print("Server startup completed successfully")
            except Exception as e:
                print(f"Error during server startup: {str(e)}")
                # Re-raise to prevent server from starting with incomplete initialization
                raise
                
        @starlette_app.on_event("shutdown")
        async def shutdown():
            """Clean up resources on server shutdown."""
            # Close the client explicitly if needed
            if hasattr(starlette_app.state, "solana_client"):
                try:
                    await starlette_app.state.solana_client.close()
                except Exception as e:
                    print(f"Error closing Solana client: {str(e)}")
            
            # Clean up sessions
            try:
                with _session_store_async_lock:
                    SESSION_STORE.clear()
            except Exception as e:
                print(f"Error clearing sessions: {str(e)}")
            
            print("Server shutdown completed successfully.")
                
        # Start the server
        import uvicorn
        uvicorn.run(
            starlette_app, 
            host=config.host, 
            port=config.port
        )
    else:
        # Default to stdio
        app.run(transport="stdio")
        
        # The following code is not needed anymore since we're using the synchronous run method
        # from mcp.server.stdio import stdio_server
        # 
        # async def arun():
        #     async with stdio_server() as streams:
        #         await app.run(
        #             streams[0], streams[1], app.create_initialization_options()
        #         )
        # 
        # anyio.run(arun)


# -------------------------------------------
# Whale and Fresh Wallet Detection
# -------------------------------------------

# These functions now delegate to the imported implementations
async def detect_whale_wallets(token_address: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Find whale wallets for a token.
    
    Args:
        token_address: Token mint address
        solana_client: Solana client
        
    Returns:
        Whale wallet information
    """
    try:
        if not validate_public_key(token_address):
            return {
                "error": "Invalid token address format",
                "error_explanation": "The provided token address is not a valid Solana public key"
            }
        
        # Get token info
        try:
            supply_data = await solana_client.get_token_supply(token_address)
            decimals = supply_data["value"]["decimals"]
            total_supply = Decimal(supply_data["value"]["amount"]) / Decimal(10 ** decimals)
            
            # Get token metadata
            metadata = await solana_client.get_token_metadata(token_address)
            symbol = metadata.get("symbol", token_address[:6])
            
            # Get token price
            price_data = await solana_client.get_market_price(token_address)
            price_usd = Decimal(str(price_data.get("price", 0.01)))
            
            token_info = {
                'decimals': decimals,
                'symbol': symbol,
                'price_usd': price_usd,
                'total_supply': total_supply
            }
        except SolanaRpcError as e:
            return {
                "error": f"Error fetching token info: {str(e)}",
                "error_explanation": "Failed to get token information from the Solana blockchain"
            }
        
        # Get top token holders
        try:
            holders = await get_token_holders(token_address, solana_client)
            if not holders:
                return {
                    "error": "No token holders found",
                    "error_explanation": "Could not find any holders for this token"
                }
        except SolanaRpcError as e:
            return {
                "error": f"Error fetching token holders: {str(e)}",
                "error_explanation": "Failed to get token holders from the Solana blockchain"
            }
        
        # Process top holders to find whales
        whale_wallets = []
        for holder in holders[:25]:  # Limit to top 25 for API response time
            wallet_address = holder["owner"]
            amount = holder["amount"]
            
            # Calculate token amount and value
            token_amount = amount / Decimal(10 ** token_info['decimals'])
            token_value = token_amount * token_info['price_usd']
            supply_percentage = (token_amount / token_info['total_supply']) * 100 if token_info['total_supply'] > 0 else 0
            
            # Get wallet tokens to calculate total value
            wallet_tokens = await get_wallet_tokens_for_owner(wallet_address, solana_client)
            
            # Calculate total wallet value (simplified for API speed)
            total_value = token_value  # Start with target token value
            token_count = len(wallet_tokens)
            
            # Add value of common tokens (simplified estimate)
            for token in wallet_tokens:
                if token["mint"] == "So11111111111111111111111111111111111111112":  # SOL
                    # Add SOL value (estimated at $150 per SOL)
                    total_value += Decimal(token["amount"]) * Decimal(150)
                elif token["mint"] in ["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]:
                    # USDC, USDT (stablecoins valued at 1:1)
                    total_value += Decimal(token["amount"])
            
            # Determine if this is a whale based on value threshold (simplified)
            # Consider as whale if total value > $50,000 or holding >1% of supply
            is_whale = total_value > 50000 or supply_percentage > 1
            
            if is_whale:
                whale_wallets.append({
                    'wallet': wallet_address,
                    'target_token_amount': float(token_amount),
                    'target_token_value_usd': float(token_value),
                    'target_token_supply_percentage': float(supply_percentage),
                    'total_value_usd': float(total_value),
                    'token_count': token_count
                })
        
        # Sort whale wallets by total value
        whale_wallets.sort(key=lambda x: x['total_value_usd'], reverse=True)
        
        return {
            "token_address": token_address,
            "token_symbol": token_info['symbol'],
            "token_price_usd": float(token_info['price_usd']),
            "whale_count": len(whale_wallets),
            "whales": whale_wallets
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "error_explanation": "An unexpected error occurred during whale detection"
        }


async def detect_fresh_wallets(token_address: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Find fresh wallets for a token.
    
    Args:
        token_address: Token mint address
        solana_client: Solana client
        
    Returns:
        Fresh wallet information
    """
    try:
        if not validate_public_key(token_address):
            return {
                "error": "Invalid token address format",
                "error_explanation": "The provided token address is not a valid Solana public key"
            }
        
        # Get token info
        try:
            supply_data = await solana_client.get_token_supply(token_address)
            decimals = supply_data["value"]["decimals"]
            
            # Get token metadata
            metadata = await solana_client.get_token_metadata(token_address)
            symbol = metadata.get("symbol", token_address[:6])
            
            # Get token price
            price_data = await solana_client.get_market_price(token_address)
            price_usd = Decimal(str(price_data.get("price", 0.01)))
            
            token_info = {
                'decimals': decimals,
                'symbol': symbol,
                'price_usd': price_usd
            }
        except SolanaRpcError as e:
            return {
                "error": f"Error fetching token info: {str(e)}",
                "error_explanation": "Failed to get token information from the Solana blockchain"
            }
        
        # Get top token holders
        try:
            holders = await get_token_holders(token_address, solana_client)
            if not holders:
                return {
                    "error": "No token holders found",
                    "error_explanation": "Could not find any holders for this token"
                }
        except SolanaRpcError as e:
            return {
                "error": f"Error fetching token holders: {str(e)}",
                "error_explanation": "Failed to get token holders from the Solana blockchain"
            }
        
        # Process top holders to find fresh wallets
        fresh_wallets = []
        processed_holders = 0
        
        for holder in holders[:50]:  # Limit to top 50 for API response time
            processed_holders += 1
            wallet_address = holder["owner"]
            amount = holder["amount"]
            
            # Calculate token amount and value
            token_amount = amount / Decimal(10 ** token_info['decimals'])
            token_value = token_amount * token_info['price_usd']
            
            # 1. Check token diversity (main indicator of a fresh wallet)
            wallet_tokens = await get_wallet_tokens_for_owner(wallet_address, solana_client)
            token_count = len(wallet_tokens)
            non_dust_token_count = 0
            
            # Count non-dust tokens
            for token in wallet_tokens:
                if token["mint"] in ["So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]:
                    # SOL, USDC, USDT - count if more than $1
                    price = Decimal(150) if token["mint"] == "So11111111111111111111111111111111111111112" else Decimal(1)
                    if Decimal(token["amount"]) * price > 1:
                        non_dust_token_count += 1
                else:
                    # For other tokens, assume they have some value
                    non_dust_token_count += 1
            
            # 2. Check wallet age
            try:
                signatures = await solana_client.get_signatures_for_address(wallet_address, limit=10)
                wallet_age_days = None
                
                if signatures and len(signatures) > 0:
                    oldest_tx = signatures[-1]
                    if "blockTime" in oldest_tx:
                        created_at = datetime.datetime.fromtimestamp(oldest_tx["blockTime"])
                        wallet_age_days = (datetime.datetime.now() - created_at).days
            except:
                wallet_age_days = None
            
            # 3. Simplified transaction analysis (check if recent txs involve this token)
            token_tx_ratio = 0.0
            if signatures and len(signatures) > 0:
                # For each signature, get the transaction
                token_tx_count = 0
                for sig_info in signatures[:5]:  # Check just a few recent txs
                    try:
                        # Simplified detection - just see if the token address appears in the transaction data
                        tx_data = await solana_client.get_transaction(sig_info["signature"])
                        if token_address in str(tx_data):
                            token_tx_count += 1
                    except:
                        pass
                
                token_tx_ratio = token_tx_count / len(signatures[:5])
            
            # Determine if this is a fresh wallet
            is_new_wallet = wallet_age_days is not None and wallet_age_days <= 30
            is_fresh = (
                (is_new_wallet and non_dust_token_count < 5) or  # New wallet with few tokens
                (non_dust_token_count < 4) or  # Very few tokens
                token_tx_ratio > 0.5  # High ratio of token transactions
            )
            
            # Calculate freshness score
            freshness_score = 0.0
            if is_fresh:
                freshness_score = (1 - (non_dust_token_count / 10)) * 0.5
                if is_new_wallet:
                    freshness_score += 0.3
                freshness_score += min(0.2, token_tx_ratio * 0.3)
                freshness_score = min(0.95, freshness_score)  # Cap at 0.95
            
            if is_fresh:
                fresh_wallets.append({
                    'wallet': wallet_address,
                    'is_fresh': True,
                    'token_count': token_count,
                    'non_dust_token_count': non_dust_token_count,
                    'token_tx_ratio': float(token_tx_ratio),
                    'wallet_age_days': wallet_age_days,
                    'target_token_amount': float(token_amount),
                    'target_token_value_usd': float(token_value),
                    'freshness_score': float(freshness_score)
                })
        
        # Sort fresh wallets by freshness score
        fresh_wallets.sort(key=lambda x: x['freshness_score'], reverse=True)
        
        return {
            "token_address": token_address,
            "token_symbol": token_info['symbol'],
            "token_price_usd": float(token_info['price_usd']),
            "fresh_wallet_count": len(fresh_wallets),
            "total_analyzed_wallets": processed_holders,
            "fresh_wallets": fresh_wallets
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "error_explanation": "An unexpected error occurred during fresh wallet detection"
        }


# Add helper functions for the detectors
async def get_token_holders(token_address: str, solana_client: SolanaClient, limit: int = 100) -> List[Dict[str, Any]]:
    """Get token holders for a mint.
    
    Args:
        token_address: Token mint address
        solana_client: Solana client
        limit: Maximum holders to return
        
    Returns:
        List of token holders
    """
    # Get all token accounts for the mint
    result = await solana_client._make_request("getProgramAccounts", [
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
    ])
    
    # Process the accounts
    holders = []
    
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
                'amount': amount,
                'address': account.get('pubkey')
            })
        except (KeyError, TypeError) as e:
            continue
            
    # Sort holders by amount (descending) and limit to top holders
    holders.sort(key=lambda x: x['amount'], reverse=True)
    return holders[:limit]


async def get_wallet_tokens_for_owner(owner_address: str, solana_client: SolanaClient) -> List[Dict[str, Any]]:
    """Get all tokens owned by a wallet.
    
    Args:
        owner_address: Owner address
        solana_client: Solana client
        
    Returns:
        List of tokens
    """
    # Get native SOL balance
    sol_balance_response = await solana_client.get_balance(owner_address)
    sol_balance = Decimal(sol_balance_response) / Decimal(10**9)  # SOL has 9 decimals
    
    # Get token accounts owned by the wallet
    token_accounts = await solana_client.get_token_accounts_by_owner(owner_address)
    
    # Initialize with SOL
    balances = [
        {
            'token': 'SOL',
            'mint': 'So11111111111111111111111111111111111111112',  # Native SOL mint
            'amount': str(sol_balance),
            'decimals': 9
        }
    ]
    
    # Process all token accounts
    for account in token_accounts:
        try:
            account_data = account.get("data", {})
            if not account_data or not isinstance(account_data, dict):
                continue
                
            parsed_data = account_data.get("parsed", {})
            info = parsed_data.get("info", {})
            
            mint = info.get("mint")
            token_amount = info.get("tokenAmount", {})
            amount = token_amount.get("amount", "0")
            decimals = token_amount.get("decimals", 0)
            
            # Skip empty accounts
            if amount == "0":
                continue
            
            # Calculate actual token amount
            token_amount_decimal = Decimal(amount) / Decimal(10 ** decimals)
            
            balances.append({
                'token': mint[:6],  # Use first 6 chars as shorthand
                'mint': mint,
                'amount': str(token_amount_decimal),
                'decimals': decimals
            })
        except (KeyError, TypeError) as e:
            continue
            
    return balances


# Add the REST API endpoint functions
async def rest_whale_detector(request):
    """REST API endpoint for detecting whale wallets."""
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "error": "Invalid JSON body",
            "error_explanation": "The request body must be valid JSON"
        }, status_code=400)
    
    token_address = body.get("token_address")
    
    if not token_address:
        return JSONResponse({
            "error": "Missing token_address parameter",
            "error_explanation": "Please provide a token_address in the request body"
        }, status_code=400)
    
    solana_client = request.app.state.solana_client
    result = await detect_whale_wallets(token_address, solana_client)
    
    # Return error status code if result contains an error
    if "error" in result:
        return JSONResponse(result, status_code=400)
    
    return JSONResponse(result)


async def rest_fresh_wallet_detector(request):
    """REST API endpoint for detecting fresh wallets."""
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "error": "Invalid JSON body",
            "error_explanation": "The request body must be valid JSON"
        }, status_code=400)
    
    token_address = body.get("token_address")
    
    if not token_address:
        return JSONResponse({
            "error": "Missing token_address parameter",
            "error_explanation": "Please provide a token_address in the request body"
        }, status_code=400)
    
    solana_client = request.app.state.solana_client
    result = await detect_fresh_wallets(token_address, solana_client)
    
    # Return error status code if result contains an error
    if "error" in result:
        return JSONResponse(result, status_code=400)
    
    return JSONResponse(result)


# Add a new function for starting the cleanup task
async def start_session_cleanup_task():
    """Start a background task to periodically clean up expired sessions."""
    global _cleanup_task
    
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_periodic_session_cleanup())
        logger.info("Session cleanup task started")


async def _periodic_session_cleanup():
    """Periodically clean up expired sessions in the background."""
    try:
        while True:
            await asyncio.sleep(60)  # Check every minute
            removed_count = await clean_expired_sessions()
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired sessions")
    except asyncio.CancelledError:
        logger.info("Session cleanup task cancelled")
    except Exception as e:
        logger.error(f"Error in session cleanup task: {str(e)}", exc_info=True)


if __name__ == "__main__":
    run_server() 