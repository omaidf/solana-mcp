#!/usr/bin/env python
"""
Solana Model Context Protocol (MCP) Server
Main entry point for the application
"""
import os
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
from utils.logging import configure_logging
configure_logging(os.getenv("LOG_LEVEL", "INFO"))

# Get logger
logger = logging.getLogger("main")

# Initialize FastAPI app
app = FastAPI(
    title="Solana Model Context Protocol Server",
    description="A server for the Model Context Protocol on Solana blockchain",
    version="0.1.0",
    debug=os.getenv("DEBUG", "False").lower() == "true"
)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),  # Improved: Read from env vars
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API routes
from api.routes import api_router
app.include_router(api_router)

# Import analyzer routes for shutdown event
from api.analyzer_routes import close_analyzer

# Register error handlers
from api.error_handlers import register_error_handlers
register_error_handlers(app)

# Startup event to initialize resources
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    logger.info("Starting MCP Server...")
    # Any additional startup logic goes here

# Add shutdown event to cleanup resources
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources when shutting down"""
    logger.info("Shutting down MCP Server, closing resources...")
    # Make sure the analyzer gets closed
    try:
        await close_analyzer()
    except Exception as e:
        logger.error(f"Error closing analyzer: {str(e)}")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug(f"WebSocket connection established. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.debug(f"WebSocket connection closed. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for i, connection in enumerate(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to connection {i}: {str(e)}")
                disconnected.append(connection)
                
        # Clean up any failed connections
        for conn in disconnected:
            try:
                self.active_connections.remove(conn)
            except ValueError:
                pass

manager = ConnectionManager()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process WebSocket data here
            await manager.broadcast(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app", 
            host=os.getenv("HOST", "0.0.0.0"), 
            port=int(os.getenv("PORT", "8000")), 
            reload=os.getenv("RELOAD", "False").lower() == "true"
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        raise 