"""
Error handlers for the API
"""
import logging
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Setup logger
logger = structlog.get_logger("api.errors")

def register_error_handlers(app: FastAPI) -> None:
    """Register global error handlers for the application"""
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=str(exc.detail),
            path=request.url.path
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail)}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all uncaught exceptions"""
        logger.exception(
            "Uncaught exception",
            exc_info=exc,
            path=request.url.path,
            method=request.method
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if app.debug else "An unexpected error occurred"
            }
        ) 