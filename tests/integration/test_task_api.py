"""
Integration tests for the background task API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock
import uuid

from app import app
from solana_mcp.utils.background_tasks import TaskStatus, TaskResult
from solana_mcp.models.api_models import ApiResponse

client = TestClient(app)


@pytest.fixture
def mock_task_manager(monkeypatch):
    """Mock the BackgroundTaskManager for testing."""
    mock_manager = MagicMock()
    
    # Create some sample tasks
    task_id1 = str(uuid.uuid4())
    task_id2 = str(uuid.uuid4())
    
    # Set up mock responses
    mock_manager.get_all_tasks.return_value = [
        TaskResult(
            task_id=task_id1,
            status=TaskStatus.COMPLETED,
            result={"data": "task1_result"},
            error=None,
            progress=100,
            created_at=1630000000,
            started_at=1630000001,
            completed_at=1630000010
        ),
        TaskResult(
            task_id=task_id2,
            status=TaskStatus.RUNNING,
            result=None,
            error=None,
            progress=50,
            created_at=1630000000,
            started_at=1630000002,
            completed_at=None
        ),
    ]
    
    # Mock get_task_status for specific task
    def get_task_status_mock(task_id):
        if task_id == task_id1:
            return TaskResult(
                task_id=task_id1,
                status=TaskStatus.COMPLETED,
                result={"data": "task1_result"},
                error=None,
                progress=100,
                created_at=1630000000,
                started_at=1630000001,
                completed_at=1630000010
            )
        elif task_id == task_id2:
            return TaskResult(
                task_id=task_id2,
                status=TaskStatus.RUNNING,
                result=None,
                error=None,
                progress=50,
                created_at=1630000000,
                started_at=1630000002,
                completed_at=None
            )
        else:
            return None
    
    mock_manager.get_task_status.side_effect = get_task_status_mock
    
    # Mock cancel_task
    def cancel_task_mock(task_id):
        if task_id == task_id2:
            return True
        else:
            return False
    
    mock_manager.cancel_task.side_effect = cancel_task_mock
    
    # Store task IDs for tests
    mock_manager.task_id1 = task_id1
    mock_manager.task_id2 = task_id2
    
    # Patch the app state
    app.state.task_manager = mock_manager
    
    yield mock_manager


def test_list_tasks(mock_task_manager):
    """Test listing all background tasks."""
    response = client.get("/api/tasks")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert len(data["data"]) == 2
    assert data["data"][0]["task_id"] == mock_task_manager.task_id1
    assert data["data"][0]["status"] == TaskStatus.COMPLETED
    assert data["data"][1]["task_id"] == mock_task_manager.task_id2
    assert data["data"][1]["status"] == TaskStatus.RUNNING


def test_get_task_status(mock_task_manager):
    """Test getting status of a specific task."""
    # Test completed task
    response = client.get(f"/api/tasks/{mock_task_manager.task_id1}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["task_id"] == mock_task_manager.task_id1
    assert data["data"]["status"] == TaskStatus.COMPLETED
    assert data["data"]["result"] == {"data": "task1_result"}
    assert data["data"]["progress"] == 100
    
    # Test running task
    response = client.get(f"/api/tasks/{mock_task_manager.task_id2}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "data" in data
    assert data["data"]["task_id"] == mock_task_manager.task_id2
    assert data["data"]["status"] == TaskStatus.RUNNING
    assert data["data"]["result"] is None
    assert data["data"]["progress"] == 50


def test_get_nonexistent_task(mock_task_manager):
    """Test error handling for non-existent task."""
    non_existent_id = str(uuid.uuid4())
    response = client.get(f"/api/tasks/{non_existent_id}")
    
    # API should return a 404 Not Found
    assert response.status_code == 404
    data = response.json()
    
    assert "success" in data
    assert not data["success"]


def test_cancel_task(mock_task_manager):
    """Test cancelling a running task."""
    # Test cancelling a running task
    response = client.delete(f"/api/tasks/{mock_task_manager.task_id2}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert data["success"]
    
    # Test cancelling a completed task (should fail)
    response = client.delete(f"/api/tasks/{mock_task_manager.task_id1}")
    
    assert response.status_code == 400
    data = response.json()
    
    assert "success" in data
    assert not data["success"]
    
    # Test cancelling a non-existent task
    non_existent_id = str(uuid.uuid4())
    response = client.delete(f"/api/tasks/{non_existent_id}")
    
    assert response.status_code == 404
    data = response.json()
    
    assert "success" in data
    assert not data["success"] 