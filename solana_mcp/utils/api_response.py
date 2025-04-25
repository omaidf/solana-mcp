"""
API response utilities for the Solana MCP.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use `solana_mcp.models.api_models` instead.
"""

import warnings
import time
import inspect
from typing import TypeVar, Generic, List, Dict, Any, Optional, Type, Callable, Union
from functools import wraps

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from solana_mcp.models.api_models import (
    ApiResponse, StatusCode, ErrorDetail, PaginationInfo, PaginatedResponse
)

# Show deprecation warning
warnings.warn(
    "The `solana_mcp.utils.api_response` module is deprecated. "
    "Use `solana_mcp.models.api_models` instead.",
    DeprecationWarning,
    stacklevel=2
)

# Type variable for generic response models
T = TypeVar('T')

# Re-export for backward compatibility
__all__ = [
    'ApiResponse', 
    'StatusCode', 
    'ErrorDetail', 
    'PaginationInfo', 
    'PaginatedResponse'
]

# Error classes for backward compatibility
class SolanaMCPError(Exception):
    """Base exception for Solana MCP errors (for backward compatibility)."""
    
    status_code: int = 500
    error_code: str = "internal_error"
    message: str = "An internal error occurred"
    
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the error with optional message and details."""
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation."""
        error_dict = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            error_dict["details"] = self.details
        return error_dict


class ApiResponseMeta(BaseModel):
    """Metadata for API responses."""
    
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    request_id: Optional[str] = None
    pagination: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response model with generic data type."""
    
    success: bool = True
    data: Optional[T] = None
    meta: ApiResponseMeta = Field(default_factory=ApiResponseMeta)
    
    @classmethod
    def success_response(cls, data: Optional[T] = None, 
                        pagination: Optional[Dict[str, Any]] = None, 
                        request_id: Optional[str] = None) -> 'ApiResponse[T]':
        """Create a success response."""
        meta = ApiResponseMeta(
            timestamp=int(time.time()),
            request_id=request_id,
            pagination=pagination
        )
        return cls(success=True, data=data, meta=meta)
    
    @classmethod
    def error_response(cls, errors: List[Dict[str, Any]], 
                      request_id: Optional[str] = None) -> 'ApiResponse[None]':
        """Create an error response."""
        meta = ApiResponseMeta(
            timestamp=int(time.time()),
            request_id=request_id,
            errors=errors
        )
        return cls(success=False, data=None, meta=meta)


class RateLimitError(SolanaMCPError):
    """Error raised when rate limits are exceeded."""
    
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate_limit_exceeded"
    message = "Rate limit exceeded"


class AuthenticationError(SolanaMCPError):
    """Error raised for authentication failures."""
    
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "authentication_failed"
    message = "Authentication failed"


class AuthorizationError(SolanaMCPError):
    """Error raised for authorization failures."""
    
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "permission_denied"
    message = "Permission denied"


class ValidationError(SolanaMCPError):
    """Error raised for validation failures."""
    
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "validation_error"
    message = "Validation error"


class NotFoundError(SolanaMCPError):
    """Error raised when a resource is not found."""
    
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"
    message = "Resource not found"


class DataNotFoundError(NotFoundError):
    """Error raised when data is not found."""
    
    error_code = "data_not_found"
    message = "The requested data was not found"


class RPCError(SolanaMCPError):
    """Error raised when there's an issue with the RPC call."""
    
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "rpc_error"
    message = "Error communicating with Solana RPC"


class TimeoutError(SolanaMCPError):
    """Error raised when an operation times out."""
    
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    error_code = "timeout"
    message = "Operation timed out"


def handle_api_errors(func: Callable):
    """Decorator to handle API errors and return standardized responses."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SolanaMCPError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=ApiResponse.error_response(
                    errors=[e.to_dict()]
                ).model_dump()
            )
        except Exception as e:
            # Log unexpected exceptions
            import logging
            logging.exception(f"Unexpected error in {func.__name__}: {str(e)}")
            
            # Return a generic internal error
            error = {
                "error_code": "internal_error",
                "message": "An internal server error occurred",
            }
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ApiResponse.error_response(
                    errors=[error]
                ).model_dump()
            )
    
    # Preserve signature for FastAPI
    if inspect.iscoroutinefunction(func):
        wrapper.__signature__ = inspect.signature(func)
    
    return wrapper 