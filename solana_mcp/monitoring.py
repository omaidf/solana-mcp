"""Monitoring and health check functionality for Solana MCP server."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from solana_mcp.solana_client import SolanaClient, SolanaRpcError


@dataclass
class HealthStatus:
    """Health status for a component."""
    
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def is_healthy(self) -> bool:
        """Check if the component is healthy.
        
        Returns:
            Whether the component is healthy
        """
        return self.status == "healthy"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component."""
    
    component: str
    latency_ms: float = 0.0
    success_rate: float = 100.0
    request_count: int = 0
    error_count: int = 0
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def record_request(self, success: bool, latency_ms: float):
        """Record a request.
        
        Args:
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
        """
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.latency_ms = (self.latency_ms * (self.request_count - 1) + latency_ms) / self.request_count
        self.success_rate = (self.request_count - self.error_count) / self.request_count * 100.0
        self.last_update = datetime.now().isoformat()


class HealthMonitor:
    """Health monitor for the MCP server."""
    
    def __init__(self, solana_client: SolanaClient):
        """Initialize the health monitor.
        
        Args:
            solana_client: The Solana client for RPC calls
        """
        self.solana_client = solana_client
        self.health_statuses: Dict[str, HealthStatus] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.monitoring_task = None
        self.running = False
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start periodic health checks.
        
        Args:
            interval_seconds: Interval between health checks in seconds
        """
        if self.monitoring_task is not None:
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
    
    async def stop_monitoring(self):
        """Stop periodic health checks."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Run health checks periodically.
        
        Args:
            interval_seconds: Interval between health checks in seconds
        """
        while self.running:
            try:
                await self.check_all_health()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    async def check_all_health(self) -> Dict[str, HealthStatus]:
        """Check health of all components.
        
        Returns:
            Health status for all components
        """
        # Check RPC health
        await self.check_rpc_health()
        
        # Return all health statuses
        return self.health_statuses
    
    async def check_rpc_health(self) -> HealthStatus:
        """Check health of the Solana RPC connection.
        
        Returns:
            Health status for the RPC connection
        """
        start_time = time.time()
        try:
            # Check if we can get a slot
            slot = await self.solana_client.get_slot()
            
            # Successful request
            latency_ms = (time.time() - start_time) * 1000
            
            # Record performance metrics
            self._record_performance("solana_rpc", True, latency_ms)
            
            # Create health status
            status = HealthStatus(
                component="solana_rpc",
                status="healthy",
                details={
                    "slot": slot,
                    "latency_ms": latency_ms
                }
            )
        except Exception as e:
            # Failed request
            latency_ms = (time.time() - start_time) * 1000
            
            # Record performance metrics
            self._record_performance("solana_rpc", False, latency_ms)
            
            # Create health status
            status = HealthStatus(
                component="solana_rpc",
                status="unhealthy",
                details={
                    "error": str(e),
                    "latency_ms": latency_ms
                }
            )
        
        # Store and return the health status
        self.health_statuses["solana_rpc"] = status
        return status
    
    def _record_performance(self, component: str, success: bool, latency_ms: float):
        """Record performance metrics for a component.
        
        Args:
            component: The component name
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
        """
        # Get or create performance metrics for the component
        if component not in self.performance_metrics:
            self.performance_metrics[component] = PerformanceMetrics(component=component)
            
        # Record the request
        self.performance_metrics[component].record_request(success, latency_ms)
    
    def get_performance_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for all components.
        
        Returns:
            Performance metrics for all components
        """
        return self.performance_metrics


class RequestTracker:
    """Tracker for measuring request performance."""
    
    def __init__(self, health_monitor: HealthMonitor):
        """Initialize the request tracker.
        
        Args:
            health_monitor: The health monitor for recording metrics
        """
        self.health_monitor = health_monitor
    
    async def track_request(
        self, 
        component: str, 
        coroutine: asyncio.CoroutineFunctionType, 
        *args, 
        **kwargs
    ) -> Any:
        """Track a request and record performance metrics.
        
        Args:
            component: The component name
            coroutine: The coroutine to track
            *args: Arguments for the coroutine
            **kwargs: Keyword arguments for the coroutine
            
        Returns:
            The result of the coroutine
        """
        start_time = time.time()
        try:
            # Execute the coroutine
            result = await coroutine(*args, **kwargs)
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.health_monitor._record_performance(component, True, latency_ms)
            
            return result
        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self.health_monitor._record_performance(component, False, latency_ms)
            
            # Re-raise the exception
            raise e 