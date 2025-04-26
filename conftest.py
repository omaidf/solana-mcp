"""
Root-level conftest for pytest configuration
"""
import os
import sys
import pytest
import pytest_asyncio
import asyncio

# Configure pytest-asyncio to use the correct event loop policy
@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    try:
        # Use the new AsyncIO policy on Windows
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            # On Unix, use default policy
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        # Don't close the loop here, as it's needed for test cleanup
        loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
    except Exception as e:
        print(f"Error setting up event loop: {e}")
        raise

# Set asyncio mode to auto instead of strict
def pytest_configure(config):
    """Configure pytest"""
    # Set asyncio mode
    config.option.asyncio_mode = "auto"
    
    # Set log format for pytest
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ) 