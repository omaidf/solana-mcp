"""
Error handling utilities for Solana MCP.

This module defines custom exception classes and error handling
decorators for the application.
"""

import functools
import logging
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

# Remove FastAPI dependencies
# from fastapi import HTTPException, Request, status
from pydantic import BaseModel

# Create minimal replacements for FastAPI dependencies
class HttpStatus:
    """Replacement for FastAPI status codes."""
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_502_BAD_GATEWAY = 502
    HTTP_503_SERVICE_UNAVAILABLE = 503
    HTTP_504_GATEWAY_TIMEOUT = 504

status = HttpStatus

class HTTPException(Exception):
    """Replacement for FastAPI HTTPException."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

class Request:
    """Minimal placeholder for FastAPI Request."""
    pass

class JSONResponse:
    """Minimal placeholder for Starlette JSONResponse."""
    def __init__(self, status_code: int, content: Dict[str, Any]):
        self.status_code = status_code
        self.content = content

# Type variable for function return type
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ErrorCode(str, Enum):
    """Error codes for the Solana MCP API."""
    
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    BAD_REQUEST = "BAD_REQUEST"
    
    # Solana RPC errors
    RPC_ERROR = "RPC_ERROR"
    RPC_TIMEOUT = "RPC_TIMEOUT"
    RPC_CONNECTION_ERROR = "RPC_CONNECTION_ERROR"
    
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    
    # Data errors
    INVALID_ACCOUNT = "INVALID_ACCOUNT"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    INVALID_TOKEN = "INVALID_TOKEN"
    DATA_PARSING_ERROR = "DATA_PARSING_ERROR"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    
    # Task errors
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_ALREADY_COMPLETED = "TASK_ALREADY_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_CANCELLED = "TASK_CANCELLED"


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SolanaMCPError(Exception):
    """Base exception for all Solana MCP errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new Solana MCP error.
        
        Args:
            message: Error message
            code: Error code
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class ValidationError(SolanaMCPError):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class ResourceNotFoundError(SolanaMCPError):
    """Exception for resource not found errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        resource_id: str
    ):
        super().__init__(
            message=message,
            code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class DataNotFoundError(SolanaMCPError):
    """Exception for data not found errors."""
    
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if data_type:
            error_details["data_type"] = data_type
            
        super().__init__(
            message=message,
            code=ErrorCode.DATA_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details=error_details
        )


class DataParsingError(SolanaMCPError):
    """Exception for data parsing errors."""
    
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if data_type:
            error_details["data_type"] = data_type
            
        super().__init__(
            message=message,
            code=ErrorCode.DATA_PARSING_ERROR,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=error_details
        )


class ExternalServiceError(SolanaMCPError):
    """Exception for errors from external services."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        error_details["service_name"] = service_name
            
        super().__init__(
            message=message,
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=error_details
        )


class RpcError(SolanaMCPError):
    """Exception for Solana RPC errors."""
    
    def __init__(
        self,
        message: str,
        rpc_error: Optional[Dict[str, Any]] = None,
        code: ErrorCode = ErrorCode.RPC_ERROR
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={"rpc_error": rpc_error or {}}
        )


class RpcTimeoutError(RpcError):
    """Exception for Solana RPC timeout errors."""
    
    def __init__(
        self,
        message: str,
        timeout: float,
        rpc_error: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            rpc_error=rpc_error,
            code=ErrorCode.RPC_TIMEOUT
        )
        self.details["timeout"] = timeout


class RpcConnectionError(RpcError):
    """Exception for Solana RPC connection errors."""
    
    def __init__(
        self,
        message: str,
        rpc_error: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            rpc_error=rpc_error,
            code=ErrorCode.RPC_CONNECTION_ERROR
        )


class ServiceUnavailableError(SolanaMCPError):
    """Exception for service unavailable errors."""
    
    def __init__(
        self,
        message: str,
        service: str,
        retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service}
        )
        if retry_after is not None:
            self.details["retry_after"] = retry_after


class TimeoutError(SolanaMCPError):
    """Exception for timeout errors."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        timeout: float
    ):
        super().__init__(
            message=message,
            code=ErrorCode.TIMEOUT,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details={"operation": operation, "timeout": timeout}
        )


class TaskError(SolanaMCPError):
    """Base exception for task-related errors."""
    
    def __init__(
        self,
        message: str,
        task_id: str,
        code: ErrorCode,
        status_code: int
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
            details={"task_id": task_id}
        )


class TaskNotFoundError(TaskError):
    """Exception for task not found errors."""
    
    def __init__(
        self,
        task_id: str
    ):
        super().__init__(
            message=f"Task with ID {task_id} not found",
            task_id=task_id,
            code=ErrorCode.TASK_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND
        )


class TaskCompletedError(TaskError):
    """Exception for trying to manipulate a completed task."""
    
    def __init__(
        self,
        task_id: str
    ):
        super().__init__(
            message=f"Task with ID {task_id} is already completed",
            task_id=task_id,
            code=ErrorCode.TASK_ALREADY_COMPLETED,
            status_code=status.HTTP_400_BAD_REQUEST
        )


def handle_api_errors(func: F) -> F:
    """
    Decorator to handle API errors.
    
    This decorator catches exceptions and converts them to appropriate
    HTTP responses.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except SolanaMCPError as e:
            # Log the error
            logging.error(f"API error: {e.code} - {e.message}")
            
            # Return a JSON response
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
        except HTTPException:
            # Re-raise FastAPI HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            logging.exception("Unexpected error in API route")
            
            # Return a generic error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": ErrorCode.UNKNOWN_ERROR,
                    "message": "An unexpected error occurred",
                    "details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
            )
    
    return cast(F, wrapper)


async def global_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Global exception handler for the FastAPI application.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        A JSON response with error details
    """
    # Get the exception traceback
    tb = traceback.format_exc()
    
    # Log the error
    logging.error(f"Unhandled exception: {str(exc)}\n{tb}")
    
    # Handle Solana MCP errors
    if isinstance(exc, SolanaMCPError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    # Handle FastAPI HTTP exceptions
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": ErrorCode.BAD_REQUEST,
                "message": exc.detail,
                "details": {}
            }
        )
    
    # Handle unknown errors
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": ErrorCode.UNKNOWN_ERROR,
            "message": "An unexpected error occurred",
            "details": {
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        }
    ) 