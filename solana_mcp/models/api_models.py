"""
API response models for the Solana MCP API.

This module defines standardized response models for the API to ensure consistent
response formats across all endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, validator

# Type variable for response data
T = TypeVar('T')

class StatusCode(str, Enum):
    """Status codes for API responses."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"

class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ApiResponse(BaseModel, Generic[T]):
    """
    Standard API response model.
    
    This model is used as the base response format for all API endpoints.
    """
    status: StatusCode = Field(
        StatusCode.SUCCESS, 
        description="Status of the response"
    )
    success: bool = Field(
        True,
        description="Whether the request was successful"
    )
    data: Optional[T] = Field(
        None, 
        description="Response data payload"
    )
    error: Optional[ErrorDetail] = Field(
        None, 
        description="Error details if status is 'error'"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the response"
    )
    
    @validator('error', always=True)
    def validate_error_for_error_status(cls, v, values):
        """Ensure error details are present when status is error."""
        if values.get('status') == StatusCode.ERROR and v is None:
            raise ValueError("Error details must be provided when status is 'error'")
        if values.get('status') != StatusCode.ERROR and v is not None:
            raise ValueError("Error details should only be provided when status is 'error'")
        return v
    
    @validator('success', always=True)
    def set_success_based_on_status(cls, v, values):
        """Set success field based on status."""
        status = values.get('status')
        if status is not None:
            return status == StatusCode.SUCCESS
        return v
    
    @classmethod
    def success_response(cls, data: Optional[T] = None, 
                       meta: Optional[Dict[str, Any]] = None) -> 'ApiResponse[T]':
        """Create a success response."""
        return cls(
            status=StatusCode.SUCCESS,
            success=True,
            data=data
        )
    
    @classmethod
    def error_response(cls, 
                      message: str,
                      code: str = "INTERNAL_ERROR",
                      details: Optional[Dict[str, Any]] = None) -> 'ApiResponse[None]':
        """Create an error response."""
        error_detail = ErrorDetail(
            code=code,
            message=message,
            details=details
        )
        return cls(
            status=StatusCode.ERROR,
            success=False,
            data=None,
            error=error_detail
        )

class PaginationInfo(BaseModel):
    """Pagination metadata for paginated responses."""
    page: int = Field(..., description="Current page number (1-indexed)")
    limit: int = Field(..., description="Number of items per page")
    total: int = Field(..., description="Total number of items available")
    next_page: Optional[int] = Field(None, description="Next page number, if available")
    prev_page: Optional[int] = Field(None, description="Previous page number, if available")
    
    @validator('next_page', 'prev_page', pre=True, always=True)
    def validate_page_numbers(cls, v, values):
        """Calculate next and previous page numbers based on current page and total."""
        page = values.get('page', 1)
        limit = values.get('limit', 10)
        total = values.get('total', 0)
        
        total_pages = (total + limit - 1) // limit if limit > 0 else 0
        
        if v == 'next_page':
            return page + 1 if page < total_pages else None
        if v == 'prev_page':
            return page - 1 if page > 1 else None
        return v

class PaginatedResponse(ApiResponse, Generic[T]):
    """
    Paginated API response model.
    
    This model extends the standard API response with pagination metadata.
    """
    data: Optional[List[T]] = Field(
        None, 
        description="List of paginated items"
    )
    pagination: Optional[PaginationInfo] = Field(
        None, 
        description="Pagination metadata"
    ) 