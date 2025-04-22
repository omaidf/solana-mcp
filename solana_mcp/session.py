"""Session management for the Solana MCP server."""

import json
import datetime
import threading
import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

from solana_mcp.config import get_app_config

# Get configuration
config = get_app_config()
SESSION_EXPIRY = config.session.expiry_minutes  # Session expiry in minutes
CLEANUP_INTERVAL = config.session.cleanup_interval_seconds  # Cleanup interval in seconds

# Global session store with thread safety
_session_store_lock = threading.Lock()
SESSION_STORE: Dict[str, 'Session'] = {}


@dataclass
class Session:
    """Session to track context across requests."""
    id: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_accessed: datetime.datetime = field(default_factory=datetime.datetime.now)
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def update_access_time(self) -> None:
        """Update the last access time."""
        self.last_accessed = datetime.datetime.now()
        
    def add_query(self, query: str, result: Any) -> None:
        """Add a query to the history.
        
        Args:
            query: The query string
            result: The query result
        """
        self.query_history.append({
            "query": query,
            "timestamp": datetime.datetime.now().isoformat(),
            "result": result
        })
        
        # Keep history to a reasonable size
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
        
    def get_context_for_entity(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get context for a specific entity.
        
        Args:
            entity_type: Type of entity (account, token, etc.)
            entity_id: ID of the entity
            
        Returns:
            Context data for the entity
        """
        key = f"{entity_type}:{entity_id}"
        return self.context.get(key, {})
    
    def update_context_for_entity(self, entity_type: str, entity_id: str, data: Dict[str, Any]) -> None:
        """Update context for a specific entity.
        
        Args:
            entity_type: Type of entity (account, token, etc.)
            entity_id: ID of the entity
            data: Context data to update
        """
        key = f"{entity_type}:{entity_id}"
        if key not in self.context:
            self.context[key] = {}
        self.context[key].update(data)
    
    def is_expired(self) -> bool:
        """Check if the session is expired.
        
        Returns:
            True if expired, False otherwise
        """
        expiry_time = self.last_accessed + datetime.timedelta(minutes=SESSION_EXPIRY)
        return datetime.datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary.
        
        Returns:
            Dictionary representation of the session
        """
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "query_history": self.query_history,
            "context": self.context
        }
    
    def to_json(self) -> str:
        """Convert session to JSON string.
        
        Returns:
            JSON string representation of the session
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session from dictionary.
        
        Args:
            data: Dictionary representation of a session
            
        Returns:
            Session object
        """
        # Convert ISO format strings back to datetime
        created_at = datetime.datetime.fromisoformat(data.get("created_at", datetime.datetime.now().isoformat()))
        last_accessed = datetime.datetime.fromisoformat(data.get("last_accessed", datetime.datetime.now().isoformat()))
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            created_at=created_at,
            last_accessed=last_accessed,
            query_history=data.get("query_history", []),
            context=data.get("context", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Session':
        """Create session from JSON string.
        
        Args:
            json_str: JSON string representation of a session
            
        Returns:
            Session object
        """
        return cls.from_dict(json.loads(json_str))


def get_or_create_session(session_id: Optional[str] = None) -> Session:
    """Get an existing session or create a new one.
    
    Args:
        session_id: Optional session ID
        
    Returns:
        The session
    """
    # Clean expired sessions
    clean_expired_sessions()
    
    with _session_store_lock:
        # Create new session if ID not provided or not found
        if not session_id or session_id not in SESSION_STORE:
            new_session = Session(id=str(uuid.uuid4()))
            SESSION_STORE[new_session.id] = new_session
            return new_session
        
        # Update access time for existing session
        session = SESSION_STORE[session_id]
        session.update_access_time()
        return session


def get_session(session_id: str) -> Optional[Session]:
    """Get an existing session.
    
    Args:
        session_id: Session ID
        
    Returns:
        The session or None if not found
    """
    with _session_store_lock:
        if session_id in SESSION_STORE:
            session = SESSION_STORE[session_id]
            session.update_access_time()
            return session
        return None


def save_session(session: Session) -> None:
    """Save a session to the store.
    
    Args:
        session: The session to save
    """
    with _session_store_lock:
        SESSION_STORE[session.id] = session


def delete_session(session_id: str) -> bool:
    """Delete a session.
    
    Args:
        session_id: The session ID to delete
        
    Returns:
        True if the session was deleted, False if not found
    """
    with _session_store_lock:
        if session_id in SESSION_STORE:
            del SESSION_STORE[session_id]
            return True
        return False


def clean_expired_sessions() -> int:
    """Remove expired sessions from the store.
    
    Returns:
        Number of sessions cleared
    """
    with _session_store_lock:
        expired_sessions = [
            session_id for session_id, session in SESSION_STORE.items() 
            if session.is_expired()
        ]
        
        for session_id in expired_sessions:
            del SESSION_STORE[session_id]
            
        return len(expired_sessions)


async def periodic_session_cleanup() -> None:
    """Periodically clean up expired sessions."""
    while True:
        try:
            cleared_count = clean_expired_sessions()
            if cleared_count > 0:
                print(f"Cleaned {cleared_count} expired sessions")
        except Exception as e:
            print(f"Error cleaning sessions: {str(e)}")
        
        # Run cleanup according to the configured interval
        await asyncio.sleep(CLEANUP_INTERVAL)


def get_session_stats() -> Dict[str, Any]:
    """Get statistics about sessions.
    
    Returns:
        Dictionary of session statistics
    """
    with _session_store_lock:
        total_sessions = len(SESSION_STORE)
        active_sessions = sum(1 for session in SESSION_STORE.values() 
                             if (datetime.datetime.now() - session.last_accessed).total_seconds() < 3600)
        oldest_session = min(SESSION_STORE.values(), 
                            key=lambda s: s.created_at).created_at if SESSION_STORE else None
        newest_session = max(SESSION_STORE.values(), 
                            key=lambda s: s.created_at).created_at if SESSION_STORE else None
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "oldest_session": oldest_session.isoformat() if oldest_session else None,
            "newest_session": newest_session.isoformat() if newest_session else None,
            "expiry_minutes": SESSION_EXPIRY,
            "cleanup_interval_seconds": CLEANUP_INTERVAL
        } 