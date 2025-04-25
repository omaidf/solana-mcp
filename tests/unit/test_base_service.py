"""Unit tests for BaseService.

This module tests the base service functionality.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from solana_mcp.services.base_service import BaseService


class TestBaseService:
    """Test suite for BaseService."""

    @pytest.fixture
    def base_service(self):
        """Create a BaseService instance for testing."""
        return BaseService()

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, base_service):
        """Test execute_with_fallback when the coroutine succeeds."""
        # Setup
        async def success_coro():
            return "success"
        
        # Execute
        result = await base_service.execute_with_fallback(
            success_coro(),
            fallback_value="fallback"
        )
        
        # Verify
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_failure(self, base_service):
        """Test execute_with_fallback when the coroutine fails."""
        # Setup
        async def fail_coro():
            raise ValueError("Test error")
        
        # Execute
        result = await base_service.execute_with_fallback(
            fail_coro(),
            fallback_value="fallback",
            error_message="Operation failed"
        )
        
        # Verify
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self, base_service):
        """Test execute_with_timeout when the coroutine completes within the timeout."""
        # Setup
        async def fast_coro():
            return "success"
        
        # Execute
        result = await base_service.execute_with_timeout(
            fast_coro(),
            timeout=1.0,
            fallback_value="fallback"
        )
        
        # Verify
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_timeout(self, base_service):
        """Test execute_with_timeout when the coroutine times out."""
        # Setup
        async def slow_coro():
            await asyncio.sleep(0.5)
            return "success"
        
        # Execute
        result = await base_service.execute_with_timeout(
            slow_coro(),
            timeout=0.1,
            fallback_value="fallback",
            error_message="Operation timed out"
        )
        
        # Verify
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_error(self, base_service):
        """Test execute_with_timeout when the coroutine raises an error."""
        # Setup
        async def error_coro():
            raise ValueError("Test error")
        
        # Execute
        result = await base_service.execute_with_timeout(
            error_coro(),
            timeout=1.0,
            fallback_value="fallback",
            error_message="Operation failed"
        )
        
        # Verify
        assert result == "fallback"

    def test_log_with_context(self, base_service):
        """Test logging with context."""
        # Setup
        base_service.logger = MagicMock()
        base_service.logger.info = MagicMock()
        
        # Execute
        base_service.log_with_context("info", "Test message", key1="value1", key2="value2")
        
        # Verify
        base_service.logger.info.assert_called_once()
        args, kwargs = base_service.logger.info.call_args
        assert args[0] == "Test message"
        assert "extra" in kwargs
        assert "context" in kwargs["extra"]
        assert kwargs["extra"]["context"]["key1"] == "value1"
        assert kwargs["extra"]["context"]["key2"] == "value2" 