"""Standardized API response utilities.

DEPRECATED: This module is deprecated and will be removed in a future version.
Use `solana_mcp.models.api_models` instead.
"""

import warnings
from typing import Any, Dict, List, Optional, TypeVar, Generic

from solana_mcp.models.api_models import (
    ApiResponse, StatusCode, ErrorDetail, PaginationInfo, PaginatedResponse
)

# Show deprecation warning
warnings.warn(
    "The `solana_mcp.utils.responses` module is deprecated. "
    "Use `solana_mcp.models.api_models` instead.",
    DeprecationWarning,
    stacklevel=2
)

# Type variable for response data
T = TypeVar('T')

# Re-export for backward compatibility
__all__ = [
    'ApiResponse', 
    'StatusCode', 
    'ErrorDetail', 
    'PaginationInfo', 
    'PaginatedResponse'
]

class ApiResponse(GenericModel, Generic[T]):
    """Standardized API response model.
    
    Provides a consistent structure for all API responses.
    """
    
    data: Optional[T] = None
    error: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def success(cls, data: T, meta: Optional[Dict[str, Any]] = None) -> 'ApiResponse[T]':
        """Create a success response.
        
        Args:
            data: The response data
            meta: Optional metadata
            
        Returns:
            An ApiResponse with data
        """
        return cls(data=data, meta=meta or {})
    
    @classmethod
    def error(
        cls, 
        error_message: str, 
        error_code: str = "INTERNAL_ERROR",
        error_details: Optional[Dict[str, Any]] = None
    ) -> 'ApiResponse[T]':
        """Create an error response.
        
        Args:
            error_message: The error message
            error_code: Error code for categorization
            error_details: Optional error details
            
        Returns:
            An ApiResponse with error information
        """
        return cls(
            error={
                "message": error_message,
                "code": error_code,
                "details": error_details
            }
        )
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        error_code: str = "INTERNAL_ERROR",
        include_details: bool = True
    ) -> 'ApiResponse[T]':
        """Create an error response from an exception.
        
        Args:
            exception: The exception
            error_code: Error code for categorization
            include_details: Whether to include exception details
            
        Returns:
            An ApiResponse with error information
        """
        error_message = str(exception)
        error_details = None
        
        if include_details:
            error_details = {
                "exception_type": exception.__class__.__name__
            }
            
            # Include additional error data if available
            if hasattr(exception, "error_data"):
                error_details["error_data"] = exception.error_data
        
        return cls.error(error_message, error_code, error_details)

class PaginatedResponse(GenericModel, Generic[T]):
    """Paginated response model.
    
    Used for responses that include paginated data.
    """
    
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int
    ) -> 'PaginatedResponse[T]':
        """Create a paginated response.
        
        Args:
            items: The items for the current page
            total: Total number of items
            page: Current page number (1-based)
            page_size: Page size
            
        Returns:
            A PaginatedResponse
        """
        # Calculate total pages
        pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            pages=pages
        ) 