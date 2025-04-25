"""
Error handling middleware for the Solana MCP API.

This module provides decorators and middleware for standardized error handling
across all API endpoints.
"""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from solana_mcp.models.api_models import ApiResponse, ErrorDetail, StatusCode
from solana_mcp.utils.error_logger import log_error

# Type variable for the return type of the decorated function
T = TypeVar('T')

# Logger for this module
logger = logging.getLogger(__name__)

class SolanaMCPError(Exception):
    """Base exception class for Solana MCP API errors."""
    
    def __init__(
        self, 
        message: str, 
        code: str = "internal_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ValidationError(SolanaMCPError):
    """Exception raised for validation errors."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="validation_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class NotFoundError(SolanaMCPError):
    """Exception raised when a requested resource is not found."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="not_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class RateLimitError(SolanaMCPError):
    """Exception raised when a rate limit is exceeded."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="rate_limit_exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


class AuthenticationError(SolanaMCPError):
    """Exception raised for authentication errors."""
    
    def __init__(
        self, 
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="authentication_error",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class SolanaRPCError(SolanaMCPError):
    """Exception raised when there's an error with the Solana RPC."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="solana_rpc_error",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details
        )


class TimeoutError(SolanaMCPError):
    """Exception raised when an operation times out."""
    
    def __init__(
        self, 
        message: str = "Operation timed out",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="timeout",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details=details
        )


def create_error_response(
    exception: Union[SolanaMCPError, Exception], 
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        exception: The exception that occurred
        status_code: HTTP status code to return
    
    Returns:
        JSONResponse with standardized error format
    """
    if isinstance(exception, SolanaMCPError):
        error_code = exception.code
        error_message = exception.message
        error_details = exception.details
        status_code = exception.status_code
    elif isinstance(exception, HTTPException):
        error_code = "http_error"
        error_message = exception.detail
        error_details = {"headers": dict(exception.headers or {})}
        status_code = exception.status_code
    elif isinstance(exception, ValidationError):
        error_code = "validation_error"
        error_message = "Validation error"
        error_details = {"errors": exception.errors() if hasattr(exception, "errors") else str(exception)}
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        error_code = "internal_server_error"
        error_message = "An unexpected error occurred"
        error_details = {"exception": str(exception)}
    
    response = ApiResponse(
        status=StatusCode.ERROR,
        error=ErrorDetail(
            code=error_code,
            message=error_message,
            details=error_details
        )
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response.dict()
    )


def handle_api_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for handling API errors and returning standardized responses.
    
    Args:
        func: The function to decorate
    
    Returns:
        Decorated function with error handling
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except (SolanaMCPError, HTTPException, ValidationError) as e:
            # Log the error
            log_error(e)
            # Return standardized error response
            return create_error_response(e)
        except Exception as e:
            # Log unexpected errors
            log_error(e, include_traceback=True)
            # Return generic internal server error
            return create_error_response(e)
    
    return wrapper


async def global_exception_handler(request: Request, exc: Exception) -> Response:
    """
    Global exception handler for FastAPI.
    
    Args:
        request: FastAPI request object
        exc: The exception that occurred
    
    Returns:
        Standardized error response
    """
    # Log the error
    log_error(exc, include_traceback=True, 
              extra={"path": request.url.path, "method": request.method})
    
    # Return standardized error response
    return create_error_response(exc) 