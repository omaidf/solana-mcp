"""Solana MCP server implementation using FastMCP."""

import click
import json
import uuid
import re
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator, Union, Tuple, Set
import datetime
import logging
import time
from decimal import Decimal, InvalidOperation

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
from starlette.exceptions import HTTPException

from solana_mcp.config import ServerConfig, get_server_config, get_app_config
from solana_mcp.solana_client import (
    SolanaClient, get_solana_client, SolanaRpcError, InvalidPublicKeyError,
    TOKEN_PROGRAM_ID, METADATA_PROGRAM_ID, SOL_MINT
)
from solana_mcp.semantic_search import (
    get_account_balance, get_account_details, get_token_accounts_for_owner,
    get_token_details, get_transaction_history_for_address, get_nft_details,
    semantic_transaction_search
)

# Import the refactored modules
from solana_mcp.nlp.parser import parse_natural_language_query
from solana_mcp.nlp.formatter import format_response

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
        # Use get_token_price which primarily uses Birdeye
        price_data_result = await solana_client.get_token_price(mint)
        # Extract relevant price data structure if needed, handle potential errors
        price_data = {
            "price": price_data_result.get("price"),
            "source": price_data_result.get("source"),
            "last_updated": price_data_result.get("last_updated")
        }
        
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
    
    async with _session_store_async_lock:
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
                # TODO: Enhance token flow analysis to handle complex transactions (e.g., swaps, multi-sends)
                # The current logic only compares pre/post balances and may misattribute flows.
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
# Tool Implementations (Simplified Error Handling)
# -------------------------------------------

@app.tool()
async def get_solana_account(ctx: Context, address: str) -> str:
    """Fetch a Solana account's information (MCP Tool). Uses utility function."""
    solana_client = ctx.request_context.lifespan_context.solana_client
    # Use utility function
    try:
        # Request detailed format, let utility handle errors
        account_info = await get_account_details(address, solana_client, format_level="detailed")
        # Return JSON string representation of the result
        return json.dumps(account_info, indent=2, default=str)
    except Exception as e:
        # Catch unexpected errors not handled by utility func
        logger.error(f"Unexpected error in get_solana_account tool for {address}: {e}", exc_info=True)
        # Tools typically return strings, format error message
        return f"Unexpected Error: {str(e)}"


@app.tool()
async def get_solana_balance(ctx: Context, address: str) -> str:
    """Fetch a Solana account's balance (MCP Tool). Uses utility function."""
    solana_client = ctx.request_context.lifespan_context.solana_client
    # Use utility function
    try:
        balance_info = await get_account_balance(address, solana_client, format_level="detailed")
        # Return JSON string representation of the result
        return json.dumps(balance_info, indent=2, default=str)
    except Exception as e:
        logger.error(f"Unexpected error in get_solana_balance tool for {address}: {e}", exc_info=True)
        return f"Unexpected Error: {str(e)}"


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
    """Get holders for a token (MCP Tool - uses efficient internal helper)."""
    solana_client = ctx.request_context.lifespan_context.solana_client
    # Use efficient internal helper function
    try:
        holders_data = await _get_token_holders_internal(mint, solana_client, limit=limit)

        # Convert Decimal amounts to string/float for JSON serialization in tool response
        response_holders = []
        for holder in holders_data:
            try:
                 # Calculate uiAmount if decimals are available
                 ui_amount = None
                 if holder.get('decimals') is not None:
                     ui_amount = float(holder['amount'] / (Decimal(10)**holder['decimals']))

                 response_holders.append({
                     "owner": holder["owner"],
                     "address": holder["address"],
                     "amount": str(holder['amount']), # Raw amount as string
                     "decimals": holder.get('decimals'),
                     "uiAmount": ui_amount # Calculated uiAmount
                 })
            except Exception as e:
                 logger.warning(f"Error processing holder data for JSON response in tool: {holder}, Error: {e}")
                 response_holders.append({
                     "owner": holder.get("owner"),
                     "address": holder.get("address"),
                     "amount": "Error processing amount",
                 })

        return json.dumps({
            "mint": mint,
            "holders_returned": len(response_holders),
            "limit_applied": limit,
            "holders": response_holders
            # Note: Total holder count is not efficiently available via getProgramAccounts
        }, indent=2)
    except InvalidPublicKeyError as e:
        return f"Error: Invalid mint address: {str(e)}"
    except SolanaRpcError as e:
        logger.error(f"Solana RPC Error in get_token_holders tool for {mint}: {e} - Details: {e.error_data}", exc_info=True)
        return f"Solana RPC Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in get_token_holders tool for {mint}: {e}", exc_info=True)
        return f"Unexpected Error: {str(e)}"


@app.tool()
async def get_transaction_history(
    ctx: Context,
    address: str,
    limit: int = 10,
    before: str = None
) -> str:
    """Get transaction history for an address (MCP Tool). Uses utility function."""
    solana_client = ctx.request_context.lifespan_context.solana_client
    # Use utility function
    try:
        tx_history = await get_transaction_history_for_address(
            address,
            solana_client,
            limit=limit,
            before=before,
            format_level="detailed" # Request detailed format from utility
        )
        # Return JSON string representation of the result
        return json.dumps(tx_history, indent=2, default=str)
    except Exception as e:
        # Catch unexpected errors not handled by utility func
        logger.error(f"Unexpected error in get_transaction_history tool for {address}: {e}", exc_info=True)
        return f"Unexpected Error: {str(e)}"


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
    """REST API endpoint for getting account info. Uses utility function."""
    address = request.path_params.get("address")
    format_level = request.query_params.get("format", "standard") # Allow format control

    if not address:
        return JSONResponse({
            "error": "Missing address parameter",
            "error_explanation": "The account address must be provided in the URL path."
        }, status_code=400)

    if not validate_public_key(address):
         return JSONResponse({
            "error": "Invalid address format",
            "error_explanation": explain_solana_error("Invalid public key")
        }, status_code=400)

    solana_client = request.app.state.solana_client
    try:
        # Use utility function
        account_info = await get_account_details(address, solana_client, format_level=format_level)
        # Utility function returns dict with potential 'error' key
        if "error" in account_info:
            status_code = 400 if "Invalid public key" in account_info.get("error","") else 500
            return JSONResponse(account_info, status_code=status_code)
        return JSONResponse(account_info)
    except Exception as e:
        logger.error(f"Unexpected error in rest_get_account for {address}: {e}", exc_info=True)
        return JSONResponse({
            "error": f"Unexpected server error",
            "error_explanation": "An unexpected internal error occurred while fetching account details."
        }, status_code=500)


async def rest_get_balance(request):
    """REST API endpoint for getting account balance. Uses utility function."""
    address = request.path_params.get("address")
    format_level = request.query_params.get("format", "standard") # Allow format control

    if not address:
        return JSONResponse({
            "error": "Missing address parameter",
            "error_explanation": "The account address must be provided in the URL path."
        }, status_code=400)

    if not validate_public_key(address):
         return JSONResponse({
            "error": "Invalid address format",
            "error_explanation": explain_solana_error("Invalid public key")
        }, status_code=400)

    solana_client = request.app.state.solana_client
    try:
        # Use utility function
        balance_info = await get_account_balance(address, solana_client, format_level=format_level)
        # Utility function returns dict with potential 'error' key
        if "error" in balance_info:
            status_code = 400 if "Invalid public key" in balance_info.get("error","") else 500
            # Consider 404 if account not found (utility might need to return specific error)
            return JSONResponse(balance_info, status_code=status_code)
        return JSONResponse(balance_info)
    except Exception as e:
        logger.error(f"Unexpected error in rest_get_balance for {address}: {e}", exc_info=True)
        return JSONResponse({
            "error": f"Unexpected server error",
            "error_explanation": "An unexpected internal error occurred while fetching balance."
        }, status_code=500)


async def rest_get_token_info(request):
    """REST API endpoint for getting token info. Uses utility function."""
    mint = request.path_params.get("mint")
    format_level = request.query_params.get("format", "standard") # Allow format control

    if not mint:
        return JSONResponse({"error": "Missing mint parameter"}, status_code=400)

    if not validate_public_key(mint):
         return JSONResponse({
            "error": "Invalid mint address format",
            "error_explanation": explain_solana_error("Invalid public key")
        }, status_code=400)

    solana_client = request.app.state.solana_client
    try:
         # Use utility function
        token_details = await get_token_details(mint, solana_client, format_level=format_level)
        # Utility function returns dict with potential 'error' key
        if "error" in token_details:
            status_code = 400 if "Invalid public key" in token_details.get("error","") else 500
            # Consider 404 if mint not found
            return JSONResponse(token_details, status_code=status_code)
        # Convert Decimal fields for JSON response if format_level isn't minimal
        if format_level != "minimal":
            if token_details.get('supply',{}).get('value',{}).get('amount'):
                try:
                    token_details['supply']['value']['amount'] = str(Decimal(token_details['supply']['value']['amount']))
                except: pass # Ignore conversion errors
            if token_details.get('price_data',{}).get('price'):
                try:
                    token_details['price_data']['price'] = float(Decimal(token_details['price_data']['price']))
                except: pass
            # Convert largest_accounts amounts?

        return JSONResponse(token_details)
    except Exception as e:
        logger.error(f"Unexpected error in rest_get_token_info for {mint}: {e}", exc_info=True)
        return JSONResponse({
            "error": f"Unexpected server error",
            "error_explanation": "An unexpected internal error occurred while fetching token details."
        }, status_code=500)


async def rest_get_transactions(request):
    """REST API endpoint for getting transaction history. Uses utility function for non-search."""
    address = request.path_params.get("address")
    limit = int(request.query_params.get("limit", "20"))
    before = request.query_params.get("before")
    search_query = request.query_params.get("search")
    format_level = request.query_params.get("format", "standard") # Allow format control

    if not address:
        return JSONResponse({"error": "Missing address parameter"}, status_code=400)

    if not validate_public_key(address):
         return JSONResponse({
            "error": "Invalid address format",
            "error_explanation": explain_solana_error("Invalid public key")
        }, status_code=400)

    solana_client = request.app.state.solana_client

    try:
        # Handle semantic search if query provided
        if search_query:
            # Assume semantic_transaction_search handles errors internally or raises
            # It should ideally return a dict with an error key on failure
            results = await semantic_transaction_search(address, search_query, solana_client, limit)
            status_code = 200
            if isinstance(results, dict) and "error" in results:
                 status_code = 400 if "Invalid public key" in results.get("error", "") else 500
            return JSONResponse(results, status_code=status_code)

        # Otherwise, get regular transaction history using utility function
        else:
            tx_history = await get_transaction_history_for_address(
                address,
                solana_client,
                limit=limit,
                before=before,
                format_level=format_level
            )
            # Utility function returns dict with potential 'error' key
            if "error" in tx_history:
                status_code = 400 if "Invalid public key" in tx_history.get("error","") else 500
                return JSONResponse(tx_history, status_code=status_code)
            return JSONResponse(tx_history)

    except Exception as e:
        logger.error(f"Unexpected error in rest_get_transactions for {address}: {e}", exc_info=True)
        return JSONResponse({
            "error": f"Unexpected server error",
            "error_explanation": "An unexpected internal error occurred while fetching transactions."
        }, status_code=500)


async def rest_get_nft_info(request):
    """REST API endpoint for getting NFT info. Uses utility function."""
    mint = request.path_params.get("mint")
    format_level = request.query_params.get("format", "standard") # Allow format control

    if not mint:
        return JSONResponse({"error": "Missing mint parameter"}, status_code=400)

    if not validate_public_key(mint):
         return JSONResponse({
            "error": "Invalid mint address format",
            "error_explanation": explain_solana_error("Invalid public key")
        }, status_code=400)

    solana_client = request.app.state.solana_client
    try:
        # Use utility function
        nft_details = await get_nft_details(mint, solana_client, format_level=format_level)
        # Utility function returns dict with potential 'error' key
        if "error" in nft_details:
             status_code = 400 if "Invalid public key" in nft_details.get("error","") else 500
             return JSONResponse(nft_details, status_code=status_code)
        return JSONResponse(nft_details)
    except Exception as e:
        logger.error(f"Unexpected error in rest_get_nft_info for {mint}: {e}", exc_info=True)
        return JSONResponse({
            "error": f"Unexpected server error",
            "error_explanation": "An unexpected internal error occurred while fetching NFT details."
        }, status_code=500)


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
                # Use async context manager for asyncio.Lock
                async with _session_store_async_lock:
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


# -------------------------------------------
# Whale and Fresh Wallet Detection (Refactored for Performance)
# -------------------------------------------

async def detect_whale_wallets(token_address: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Find whale wallets for a token (Refactored for Performance).

    Identifies wallets holding a significant amount of the target token
    AND/OR having a high estimated total portfolio value.
    """
    start_time = time.monotonic()
    # Load configuration
    config = get_app_config()
    analysis_config = config.analysis
    logger.info(f"Starting whale detection for token: {token_address}")

    if not validate_public_key(token_address):
        return {
            "success": False,
            "error": "Invalid token address format",
            "error_explanation": "The provided token address is not a valid Solana public key"
        }

    try:
        # 1. Fetch basic token info (supply, decimals, metadata, price) sequentially
        logger.debug(f"[{token_address}] Fetching token supply, metadata, price, and SOL price sequentially...")
        error_details = []
        
        try:
            supply_data = await solana_client.get_token_supply(token_address)
        except Exception as e:
            supply_data = e 
            logger.error(f"Failed get_token_supply for {token_address}: {e}")

        try:
            metadata = await solana_client.get_token_metadata(token_address)
        except Exception as e:
             metadata = e
             logger.error(f"Failed get_token_metadata for {token_address}: {e}")

        try:
            target_price_data = await solana_client.get_token_price(token_address)
        except Exception as e:
             target_price_data = e
             logger.error(f"Failed get_token_price for {token_address}: {e}")

        try:
            sol_price_data = await solana_client.get_token_price(SOL_MINT)
        except Exception as e:
             sol_price_data = e
             logger.error(f"Failed get_token_price for SOL: {e}")

        # --- Process Token Info Results ---
        # supply_data = token_info_results.get("supply") # OLD concurrent way
        # metadata = token_info_results.get("metadata")
        # target_price_data = token_info_results.get("target_price")
        # sol_price_data = token_info_results.get("sol_price")
        # error_details = [] # Defined above

        # Validate supply data
        if isinstance(supply_data, Exception) or not supply_data or "value" not in supply_data:
            error_msg = f"Failed to get valid supply for {token_address}: {supply_data}"
            logger.error(error_msg)
            return {"success": False, "error": "Error fetching token supply", "details": str(supply_data)}

        # Handle potential errors for metadata and prices
        if isinstance(metadata, Exception): logger.warning(f"[{token_address}] Failed to get metadata: {metadata}"); error_details.append(f"Metadata failed: {metadata}"); metadata = {}
        if isinstance(target_price_data, Exception): logger.warning(f"[{token_address}] Failed to get target price: {target_price_data}"); error_details.append(f"Target price failed: {target_price_data}"); target_price_data = {}
        if isinstance(sol_price_data, Exception): logger.warning(f"[{token_address}] Failed to get SOL price: {sol_price_data}"); error_details.append(f"SOL price failed: {sol_price_data}"); sol_price_data = {}

        # Determine SOL price to use (dynamic or fallback from config)
        current_sol_price_usd = analysis_config.estimated_sol_price_usd # Default to config fallback
        if sol_price_data and "price" in sol_price_data:
            try:
                current_sol_price_usd = Decimal(str(sol_price_data["price"]))
                logger.debug(f"[{token_address}] Using dynamic SOL price: {current_sol_price_usd}")
            except (InvalidOperation, TypeError):
                logger.warning(f"[{token_address}] Invalid dynamic SOL price value ({sol_price_data['price']}), using fallback: {current_sol_price_usd}")
        else:
            logger.warning(f"[{token_address}] Failed to fetch dynamic SOL price, using fallback: {current_sol_price_usd}")

        # Extract token details
        decimals = supply_data.get("value", {}).get("decimals")
        total_supply_lamports = Decimal(supply_data.get("value", {}).get("amount", '0'))
        if decimals is None: return {"success": False, "error": "Could not determine token decimals"}

        total_supply = total_supply_lamports / Decimal(10 ** decimals) if decimals >= 0 else Decimal(0)
        symbol = metadata.get("symbol", token_address[:6])
        name = metadata.get("name", "Unknown Token")
        target_price_usd = Decimal(str(target_price_data.get("price", 0.0))) # Default to 0

        token_info = {
            'decimals': decimals,
            'symbol': symbol,
            'name': name,
            'price_usd': target_price_usd,
            'total_supply': total_supply,
            'current_sol_price_usd': current_sol_price_usd # Include SOL price used
        }
        logger.debug(f"[{token_address}] Token info processed: {token_info}")

        # 2. Fetch top N holders using the internal helper (which now uses SolanaFM)
        whale_holder_limit = analysis_config.whale_holder_limit
        logger.debug(f"[{token_address}] Fetching top {whale_holder_limit} holders via _get_token_holders_internal...")
        try:
            # Ensure we are calling the internal helper function
            holders = await _get_token_holders_internal(token_address, solana_client, limit=whale_holder_limit)
        except SolanaRpcError as e:
            logger.error(f"Failed to get holders for {token_address}: {e}")
            return {"success": False, "error": "Error fetching token holders", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error fetching holders for {token_address}: {e}", exc_info=True)
            return {"success": False, "error": "Unexpected error fetching holders", "details": str(e)}

        if not holders:
            logger.warning(f"[{token_address}] No holders found or processed.")
            # ... (rest of the no holders return logic as before)
            return {
                "success": True, "warning": "No token holders found.",
                "token_address": token_address, "token_info": final_token_info,
                "whale_count": 0, "whales": []
            }

        logger.debug(f"[{token_address}] Processing {len(holders)} potential whale holders...")
        # Extract owner addresses from the holders list returned by the internal helper
        holder_owners = [h['owner'] for h in holders if h.get('owner')]
        # Create map from owner to their target token amount (already fetched, assuming lamports)
        owner_target_holdings = {h['owner']: h['amount'] for h in holders if h.get('owner') and h.get('amount') is not None}

        if not holder_owners:
            logger.warning(f"Could not extract owners from holder data for {token_address}")
            # Return simplified success if no owners found
            # ... (similar return logic as no holders) ...
            return {"success": True, "warning": "Could not extract owners from holder data.", "token_address": token_address, "token_info": final_token_info, "whale_count": 0, "whales": []}

        unique_owners = list(owner_target_holdings.keys())
        logger.debug(f"[{token_address}] Found {len(unique_owners)} unique owners to check SOL balance.")

        # 3. Fetch SOL balances for unique owners using getMultipleAccounts (batched)
        owner_sol_balances = {}
        batch_size = 100 # getMultipleAccounts limit
        
        # Add delay before fetching SOL balances
        await asyncio.sleep(0.5) 
        
        for i in range(0, len(unique_owners), batch_size):
            batch_owners = unique_owners[i:i + batch_size]
            logger.debug(f"Fetching SOL balances for owners batch {i//batch_size + 1}")
            try:
                # Fetch SOL balances (lamports)
                multiple_balances_info = await solana_client._make_request(
                    "getMultipleAccounts", 
                    [batch_owners] # Default encoding is base64, only need lamports
                )
                if multiple_balances_info and isinstance(multiple_balances_info.get("value"), list):
                    balances_data = multiple_balances_info["value"]
                    for idx, balance_data in enumerate(balances_data):
                        if balance_data and isinstance(balance_data.get("lamports"), int):
                            owner_addr = batch_owners[idx]
                            owner_sol_balances[owner_addr] = Decimal(balance_data["lamports"])
                await asyncio.sleep(0.1) # Small delay between batches
            except SolanaRpcError as e:
                logger.warning(f"Failed getMultipleAccounts batch for SOL balances: {e}")
                error_details.append(f"SOL balance fetch failed for batch {i//batch_size + 1}: {e}")
            except Exception as e:
                 logger.warning(f"Unexpected error in getMultipleAccounts batch for SOL balances: {e}")
                 error_details.append(f"Unexpected SOL balance fetch error for batch {i//batch_size + 1}: {e}")

        # 4. Process owners and identify whales based on simplified value
        logger.debug(f"[{token_address}] Analyzing {len(unique_owners)} owners for whale status (simplified value)...")
        whale_wallets = []
        processed_holder_count = 0
        # Use thresholds from config
        whale_value_threshold = analysis_config.whale_total_value_threshold_usd
        whale_supply_threshold = analysis_config.whale_supply_percentage_threshold

        for owner_address, total_target_lamports in owner_target_holdings.items():
            processed_holder_count += 1
            sol_balance_lamports = owner_sol_balances.get(owner_address, Decimal(0))

            # Convert amounts using fetched decimals/prices
            try: 
                token_amount = total_target_lamports / Decimal(10 ** token_info['decimals'])
                sol_amount = sol_balance_lamports / Decimal(10**9)
            except (ZeroDivisionError, TypeError):
                 logger.warning(f"Error converting amounts for owner {owner_address}")
                 continue # Skip holder if conversion fails

            target_token_value_usd = token_amount * token_info['price_usd']
            sol_value_usd = sol_amount * token_info['current_sol_price_usd']
            supply_percentage = (token_amount / token_info['total_supply']) * 100 if token_info['total_supply'] > 0 else Decimal(0)

            # Simplified total value (Target Token + SOL only)
            estimated_total_value_usd = target_token_value_usd + sol_value_usd
            # Token count is unknown with this simplified approach
            token_count = None 

            # Use thresholds from config
            is_whale = (estimated_total_value_usd > whale_value_threshold or
                        supply_percentage > whale_supply_threshold)

            if is_whale:
                whale_wallets.append({
                    'wallet': owner_address,
                    'target_token_amount': float(token_amount),
                    'target_token_value_usd': float(target_token_value_usd),
                    'target_token_supply_percentage': float(supply_percentage),
                    'estimated_total_value_usd': float(estimated_total_value_usd),
                    'sol_balance': float(sol_amount),
                    'token_count': token_count # Indicate count is unavailable
                })

        whale_wallets.sort(key=lambda x: x['estimated_total_value_usd'], reverse=True)

        end_time = time.monotonic()
        duration = end_time - start_time
        logger.info(f"[{token_address}] Whale detection completed in {duration:.2f}s. Found {len(whale_wallets)} whales out of {processed_holder_count} processed.")

        final_token_info = token_info.copy()
        final_token_info['price_usd'] = float(final_token_info['price_usd'])
        final_token_info['total_supply'] = float(final_token_info['total_supply'])
        final_token_info['current_sol_price_usd'] = float(final_token_info['current_sol_price_usd'])

        result = {
            "success": True,
            "token_address": token_address,
            "token_info": final_token_info,
            "analysis_config": {
                "holder_limit": whale_holder_limit,
                "value_threshold_usd": float(whale_value_threshold),
                "supply_percentage_threshold": float(whale_supply_threshold),
            },
            "analysis_duration_seconds": round(duration, 2),
            "holders_analyzed": processed_holder_count,
            "whale_count": len(whale_wallets),
            "whales": whale_wallets
        }
        if error_details:
            result["warnings"] = error_details
        return result

    except Exception as e:
        logger.error(f"[{token_address}] Unexpected error during whale detection: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected server error: {str(e)}",
            "error_explanation": "An unexpected error occurred during whale detection."
        }


async def detect_fresh_wallets(token_address: str, solana_client: SolanaClient) -> Dict[str, Any]:
    """Find fresh wallets for a token (Refactored for Performance).

    Identifies wallets that hold the target token and are likely new
    or have very low token diversity.
    """
    start_time = time.monotonic()
    # Load configuration
    config = get_app_config()
    analysis_config = config.analysis
    logger.info(f"Starting fresh wallet detection for token: {token_address}")

    if not validate_public_key(token_address):
         return {
             "success": False,
             "error": "Invalid token address format",
             "error_explanation": "The provided token address is not a valid Solana public key"
         }

    try:
        # 1. Fetch basic token info (decimals, metadata, price) + SOL price concurrently
        logger.debug(f"[{token_address}] Fetching token supply, metadata, price, and SOL price...")
        token_info_tasks = {
            "supply": solana_client.get_token_supply(token_address),
            "metadata": solana_client.get_token_metadata(token_address),
            "target_price": solana_client.get_token_price(token_address), # Use get_token_price for target
            "sol_price": solana_client.get_token_price(SOL_MINT) # Use get_token_price for SOL
        }
        results = await asyncio.gather(*token_info_tasks.values(), return_exceptions=True)
        token_info_results = dict(zip(token_info_tasks.keys(), results))

        # --- Process Token Info --- #
        supply_data = token_info_results.get("supply")
        metadata = token_info_results.get("metadata")
        target_price_data = token_info_results.get("target_price")
        sol_price_data = token_info_results.get("sol_price")
        error_details = []

        if isinstance(supply_data, Exception) or not supply_data or "value" not in supply_data: return {"success": False, "error": "Error fetching token supply", "details": str(supply_data)}
        if isinstance(metadata, Exception): logger.warning(f"[{token_address}] Metadata failed: {metadata}"); error_details.append(f"Metadata failed: {metadata}"); metadata = {}
        if isinstance(target_price_data, Exception): logger.warning(f"[{token_address}] Target price failed: {target_price_data}"); error_details.append(f"Target price failed: {target_price_data}"); target_price_data = {}
        if isinstance(sol_price_data, Exception): logger.warning(f"[{token_address}] SOL price failed: {sol_price_data}"); error_details.append(f"SOL price failed: {sol_price_data}"); sol_price_data = {}

        # Determine SOL price
        current_sol_price_usd = analysis_config.estimated_sol_price_usd
        if sol_price_data and "price" in sol_price_data:
             try: current_sol_price_usd = Decimal(str(sol_price_data["price"]))
             except (InvalidOperation, TypeError): logger.warning(f"[{token_address}] Invalid dynamic SOL price, using fallback: {current_sol_price_usd}")
        else: logger.warning(f"[{token_address}] Failed dynamic SOL price, using fallback: {current_sol_price_usd}")

        # Extract token details
        decimals = supply_data.get("value", {}).get("decimals")
        if decimals is None: return {"success": False, "error": "Could not determine token decimals"}
        symbol = metadata.get("symbol", token_address[:6])
        name = metadata.get("name", "Unknown Token")
        target_price_usd = Decimal(str(target_price_data.get("price", 0.0)))

        token_info = {
            'decimals': decimals,
            'symbol': symbol,
            'name': name,
            'price_usd': target_price_usd,
            'current_sol_price_usd': current_sol_price_usd
        }
        logger.debug(f"[{token_address}] Token info processed: {token_info}")

        # 2. Fetch top N holders efficiently (using config limit)
        fresh_holder_limit = analysis_config.fresh_wallet_holder_limit
        logger.debug(f"[{token_address}] Fetching top {fresh_holder_limit} holders...")
        holders = await _get_token_holders_internal(token_address, solana_client, limit=fresh_holder_limit)
        if not holders:
            logger.warning(f"[{token_address}] No holders found.")
            final_token_info = token_info.copy()
            final_token_info['price_usd'] = float(final_token_info['price_usd'])
            final_token_info['current_sol_price_usd'] = float(final_token_info['current_sol_price_usd'])
            return {
                 "success": True, "warning": "No token holders found.",
                 "token_address": token_address, "token_info": final_token_info,
                 "fresh_wallet_count": 0, "fresh_wallets": []
             }

        logger.debug(f"[{token_address}] Processing {len(holders)} potential fresh holders...")
        fresh_wallets = []
        holder_owners = [h['owner'] for h in holders]

        # 3. Fetch wallet token balances AND recent transaction signatures concurrently
        logger.debug(f"[{token_address}] Fetching balances and tx signatures for {len(holder_owners)} wallets...")
        tx_limit = analysis_config.fresh_wallet_tx_limit
        tasks = []
        for owner in holder_owners:
            tasks.append(_get_wallet_token_balances_internal(owner, solana_client))
            tasks.append(solana_client.get_signatures_for_address(owner, limit=tx_limit))

        holder_data_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Process holders with their fetched data
        logger.debug(f"[{token_address}] Analyzing holders for freshness...")
        processed_holders_count = 0
        # Use thresholds from config
        max_age_days = analysis_config.fresh_wallet_max_age_days
        max_tokens_low_diversity = analysis_config.fresh_wallet_max_tokens_low_diversity
        max_tokens_new_wallet = analysis_config.fresh_wallet_max_tokens_new_wallet
        min_value_threshold = analysis_config.min_token_value_usd_threshold

        for i, holder in enumerate(holders):
            processed_holders_count += 1
            wallet_address = holder["owner"]
            amount = holder["amount"]
            holder_decimals = holder.get("decimals")

            if holder_decimals is None or holder_decimals != token_info['decimals']:
                logger.warning(f"[{token_address}] Skipping fresh holder {wallet_address} due to decimals mismatch")
                continue

            balances_result = holder_data_results[i * 2]
            signatures_result = holder_data_results[i * 2 + 1]

            if isinstance(balances_result, Exception):
                logger.warning(f"[{token_address}] Failed balances for fresh candidate {wallet_address}: {balances_result}")
                error_details.append(f"Balance fetch failed for {wallet_address[:6]}...: {balances_result}")
                continue

            signatures = []
            if isinstance(signatures_result, Exception):
                logger.warning(f"[{token_address}] Failed signatures for fresh candidate {wallet_address}: {signatures_result}")
                error_details.append(f"Signature fetch failed for {wallet_address[:6]}...: {signatures_result}")
            elif signatures_result: signatures = signatures_result

            wallet_balances = balances_result
            token_count = len(wallet_balances)

            try: token_amount = amount / Decimal(10 ** token_info['decimals'])
            except ZeroDivisionError: continue
            target_token_value_usd = token_amount * token_info['price_usd']

            # --- Freshness Criteria ---
            non_dust_token_count = 0
            for balance_info in wallet_balances:
                 mint = balance_info['mint']
                 bal_amount = balance_info['amount_decimal']
                 value = Decimal(0)
                 # TODO: Refine non-dust calculation. Currently counts any non-target token with balance > 0
                 # if value < min_threshold, potentially misclassifying wallets with many dust balances.
                 if mint == token_address: value = target_token_value_usd if token_info['price_usd'] > 0 else Decimal(0)
                 elif mint == "So11111111111111111111111111111111111111112": value = bal_amount * current_sol_price_usd # Use fetched/fallback
                 elif mint in ["EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]: value = bal_amount

                 if value >= min_value_threshold:
                     non_dust_token_count += 1
                 elif mint != token_address and bal_amount > 0:
                     non_dust_token_count += 1

            wallet_age_days = None
            first_tx_timestamp = None
            now_aware = datetime.datetime.now(datetime.timezone.utc)
            if signatures:
                try:
                    timestamps = [sig.get("blockTime") for sig in signatures if sig.get("blockTime") is not None]
                    if timestamps:
                        first_tx_timestamp = min(timestamps)
                        created_at_utc = datetime.datetime.fromtimestamp(first_tx_timestamp, tz=datetime.timezone.utc)
                        wallet_age_days = (now_aware - created_at_utc).days
                except Exception as e: logger.warning(f"[{token_address}] Error calculating wallet age for {wallet_address}: {e}")

            # --- Determine Freshness --- Use thresholds from config
            is_new_wallet = wallet_age_days is not None and wallet_age_days <= max_age_days
            is_low_diversity = non_dust_token_count < max_tokens_low_diversity
            is_fresh = (is_new_wallet and non_dust_token_count < max_tokens_new_wallet) or is_low_diversity

            freshness_score = 0.0
            if is_fresh:
                diversity_score = max(0.0, 1.0 - (non_dust_token_count / 10.0))
                freshness_score += diversity_score * 0.7
                if is_new_wallet:
                     age_score = max(0.0, 1.0 - (wallet_age_days / max_age_days))
                     freshness_score += age_score * 0.3
                freshness_score = min(1.0, max(0.0, freshness_score))

            if is_fresh:
                fresh_wallets.append({
                    'wallet': wallet_address,
                    'is_fresh': True,
                    'freshness_score': round(freshness_score, 3),
                    'criteria': {
                         'is_new': is_new_wallet,
                         'is_low_diversity': is_low_diversity,
                         'calculated_age_days': wallet_age_days,
                         'non_dust_token_count': non_dust_token_count,
                    },
                    'details': {
                        'token_count': token_count,
                        'first_tx_timestamp': first_tx_timestamp,
                        'target_token_amount': float(token_amount),
                        'target_token_value_usd': float(target_token_value_usd),
                    }
                })

        fresh_wallets.sort(key=lambda x: x['freshness_score'], reverse=True)

        end_time = time.monotonic()
        duration = end_time - start_time
        logger.info(f"[{token_address}] Fresh wallet detection completed in {duration:.2f}s. Found {len(fresh_wallets)} fresh wallets out of {processed_holders_count} processed.")

        final_token_info = token_info.copy()
        final_token_info['price_usd'] = float(final_token_info['price_usd'])
        final_token_info['current_sol_price_usd'] = float(final_token_info['current_sol_price_usd'])

        result = {
            "success": True,
            "token_address": token_address,
            "token_info": final_token_info,
             "analysis_config": {
                "holder_limit": fresh_holder_limit,
                "tx_limit_for_age": tx_limit,
                "max_age_days": max_age_days,
                "low_diversity_threshold": max_tokens_low_diversity,
                "new_wallet_diversity_threshold": max_tokens_new_wallet,
                "min_token_value_usd": float(min_value_threshold)
            },
            "analysis_duration_seconds": round(duration, 2),
            "holders_analyzed": processed_holders_count,
            "fresh_wallet_count": len(fresh_wallets),
            "fresh_wallets": fresh_wallets
        }
        if error_details:
            result["warnings"] = error_details
        return result

    except Exception as e:
        logger.error(f"[{token_address}] Unexpected error during fresh wallet detection: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected server error: {str(e)}",
            "error_explanation": "An unexpected error occurred during fresh wallet detection."
        }


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
    
    # Ensure token_name is present
    if "token_name" not in result:
        # Get token metadata
        try:
            metadata = await solana_client.get_token_metadata(token_address)
            result["token_name"] = metadata.get("name", "Unknown Token")
        except Exception as e:
            result["token_name"] = "Unknown Token"
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append(f"Failed to fetch token name: {str(e)}")
    
    # Ensure success flag is included
    result["success"] = True
    
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


# -------------------------------------------
# Utility Functions (Add Internal Helpers Here)
# -------------------------------------------

# Renamed from get_token_holders to avoid conflicts and clarify internal use
async def _get_token_holders_internal(token_address: str, solana_client: SolanaClient, limit: int = 100) -> List[Dict[str, Any]]:
    """(Internal) Get token holders for a mint using efficient getProgramAccounts.

    Args:
        token_address: Token mint address
        solana_client: Solana client
        limit: Maximum holders to return

    Returns:
        List of token holders with owner, amount (Decimal), address
        Raises SolanaRpcError or Exception on failure.
    """
    logger.debug(f"Fetching top {limit} holders for mint: {token_address} via SolanaFM")
    try:
        # Use the SolanaClient method which calls SolanaFM API
        solanafm_response = await solana_client.get_token_holders(token_address, limit=limit)
        
        # Check if the API call itself indicated an error
        if solanafm_response.get("error") or solanafm_response.get("source") == "fallback":
            error_detail = solanafm_response.get("error", "Fallback response returned")
            logger.warning(f"SolanaFM API failed or returned fallback for {token_address}: {error_detail}")
            # Raise an exception to signal failure to the caller (detect_whale_wallets)
            raise SolanaRpcError(f"Failed to get holders from SolanaFM: {error_detail}")

        raw_holders = solanafm_response.get("holders", [])
        holders = []
        if not raw_holders:
            logger.warning(f"SolanaFM returned empty holder list for {token_address}")
            return []

        processed_count = 0
        for holder_data in raw_holders:
            # Limit processing in case SolanaFM returns more than requested limit slightly
            if processed_count >= limit:
                break 
            try:
                # SolanaFM provides owner wallet address in the 'address' field
                owner = holder_data.get('address') 
                # Amount is provided as a string by SolanaFM (representing lamports)
                amount_str = holder_data.get('amount')
                
                if not owner or amount_str is None: # Check for None explicitly for amount
                    logger.warning(f"Skipping holder data from SolanaFM with missing address or amount: {holder_data}")
                    continue

                # Convert raw amount string to Decimal
                try:
                    amount_decimal = Decimal(amount_str) 
                except InvalidOperation:
                    logger.warning(f"Skipping holder data with invalid amount string '{amount_str}': {holder_data}")
                    continue
                
                # Skip zero balance accounts if needed (SolanaFM might include them)
                if amount_decimal <= 0:
                    continue 

                holders.append({
                    # Key required by detect_whale_wallets
                    'owner': owner, 
                    # Store the raw decimal amount (lamports)
                    'amount': amount_decimal, 
                    # Use owner for the 'address' key for compatibility with detect_whale_wallets
                    'address': owner, 
                    # Decimals are not available from this SolanaFM endpoint.
                    'decimals': None 
                })
                processed_count += 1
            except (KeyError, TypeError) as e:
                logger.warning(f"Error processing holder data from SolanaFM for mint {token_address}: {e} - Data: {holder_data}", exc_info=True)
                continue
        
        # Sort holders by amount (descending)
        # Sorting is less critical now as SolanaFM provides them sorted, but doesn't hurt.
        holders.sort(key=lambda x: x.get('amount', Decimal(0)), reverse=True)
        
        logger.info(f"Processed {len(holders)} holders from SolanaFM for {token_address}, returning top {limit}")
        # Ensure we only return the amount requested by the original limit parameter
        return holders[:limit] 

    except SolanaRpcError as e:
        # Raised if the call to solana_client.get_token_holders failed
        logger.error(f"Error calling SolanaFM API via client for {token_address}: {e}")
        raise # Re-raise to be handled by caller
    except Exception as e:
        logger.error(f"Unexpected error processing holders from SolanaFM for {token_address}: {e}", exc_info=True)
        raise # Re-raise


# Renamed from get_wallet_tokens_for_owner to clarify internal use and refactored
async def _get_wallet_token_balances_internal(owner_address: str, solana_client: SolanaClient) -> List[Dict[str, Any]]:
    """(Internal) Get all token balances (including SOL) for a wallet owner.

    Args:
        owner_address: Owner address
        solana_client: Solana client

    Returns:
        List of tokens with balance info (token, mint, amount_decimal, decimals)
        Returns empty list on failure to fetch critical data.
    """
    logger.debug(f"Fetching token balances for owner: {owner_address}")
    balances = []
    try:
        # Concurrently fetch SOL balance and token accounts
        sol_task = solana_client.get_balance(owner_address)
        tokens_task = solana_client.get_token_accounts_by_owner(owner_address, encoding="jsonParsed")

        results = await asyncio.gather(sol_task, tokens_task, return_exceptions=True)

        sol_balance_lamports = results[0]
        token_accounts_result = results[1]

        # Process SOL balance
        if isinstance(sol_balance_lamports, Exception):
            logger.warning(f"Could not fetch SOL balance for {owner_address}: {sol_balance_lamports}")
        elif sol_balance_lamports is not None:
            try:
                sol_balance_decimal = Decimal(sol_balance_lamports) / Decimal(10**9)
                if sol_balance_decimal > 0:
                    balances.append({
                        'token': 'SOL',
                        'mint': 'So11111111111111111111111111111111111111112', # Native SOL mint
                        'amount_decimal': sol_balance_decimal,
                        'decimals': 9
                    })
            except Exception as e:
                 logger.warning(f"Error processing SOL balance for {owner_address}: {e}")

        # Process token accounts
        if isinstance(token_accounts_result, Exception):
            logger.warning(f"Could not fetch token accounts for {owner_address}: {token_accounts_result}")
        elif token_accounts_result and isinstance(token_accounts_result, dict) and "value" in token_accounts_result:
            token_accounts = token_accounts_result["value"]
            for account in token_accounts:
                try:
                    # Defensive parsing from the structure returned by get_token_accounts_by_owner
                    account_info = account.get("account", {})
                    if not account_info or not isinstance(account_info, dict): continue
                    data = account_info.get("data", {})
                    if not data or not isinstance(data, dict): continue
                    parsed = data.get("parsed", {})
                    if not parsed or not isinstance(parsed, dict): continue
                    info = parsed.get("info", {})
                    if not info or not isinstance(info, dict): continue

                    mint = info.get("mint")
                    token_amount_info = info.get("tokenAmount", {})
                    amount_str = token_amount_info.get("amount", "0")
                    decimals = token_amount_info.get("decimals")

                    if not mint or decimals is None: continue
                    if amount_str == "0": continue

                    # Calculate actual token amount
                    token_amount_decimal = Decimal(amount_str) / Decimal(10 ** decimals)

                    balances.append({
                        'token': mint[:6],  # Use first 6 chars as shorthand symbol
                        'mint': mint,
                        'amount_decimal': token_amount_decimal,
                        'decimals': decimals
                    })
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Error processing token account {account.get('pubkey')} for {owner_address}: {e}")
                    continue
        else:
             logger.warning(f"Unexpected format for token accounts result for {owner_address}: {token_accounts_result}")


        logger.debug(f"Found {len(balances)} token types for owner {owner_address}")
        return balances

    except Exception as e:
        logger.error(f"Unexpected error fetching balances for {owner_address}: {e}", exc_info=True)
        return [] # Return empty list on failure


if __name__ == "__main__":
    run_server() 