"""Error response models for the API.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use `solana_mcp.utils.errors` instead.
"""

import warnings
from typing import Dict, Optional, Any
from pydantic import BaseModel

from solana_mcp.utils.errors import (
    ErrorCode, ErrorResponse, SolanaMCPError, ValidationError,
    ResourceNotFoundError, DataNotFoundError, DataParsingError,
    ExternalServiceError, RpcError, RpcTimeoutError, RpcConnectionError,
    ServiceUnavailableError, TimeoutError, TaskError, TaskNotFoundError,
    TaskCompletedError, handle_api_errors, global_exception_handler
)

# Show deprecation warning
warnings.warn(
    "The `solana_mcp.models.errors` module is deprecated. "
    "Use `solana_mcp.utils.errors` instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = [
    'ErrorCode',
    'ErrorResponse',
    'SolanaMCPError',
    'ValidationError',
    'ResourceNotFoundError',
    'DataNotFoundError',
    'DataParsingError',
    'ExternalServiceError',
    'RpcError',
    'RpcTimeoutError',
    'RpcConnectionError',
    'ServiceUnavailableError',
    'TimeoutError',
    'TaskError',
    'TaskNotFoundError',
    'TaskCompletedError',
    'handle_api_errors',
    'global_exception_handler',
    'explain_solana_error'
]

# Import from solana_client for backward compatibility
from solana_mcp.solana_client import SolanaRpcError, InvalidPublicKeyError

# Legacy implementation for backward compatibility
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