# connectionManager.py
# Connection manager for tracking open API requests

import asyncio
import uuid
from typing import Dict, Set, Optional
from fastapi import Request
from config import logger

class ConnectionManager:
    """
    Manages open API connections and tracks requests awaiting responses.
    
    Tracks all open connections except for /models, /health, and /metrics endpoints.
    Handles both async and long-running requests with thread-safe operations.
    """
    
    def __init__(self):
        # Dictionary to store active connections with unique keys
        # Key: connection_id (str), Value: connection info (dict)
        self.connections: Dict[str, Dict] = {}

        self._lock = asyncio.Lock()

        self.excluded_endpoints = {
            "/models",
            "/health",
            "/metrics"
        }

        # Callback for when connection count changes (0 -> non-zero or non-zero -> 0)
        self._activity_callback = None
    
    async def add_connection(self, request: Request, endpoint: str, metadata: dict = None) -> str:
        """
        Add a new connection to the manager with optional metadata support.

        Args:
            request: FastAPI Request object
            endpoint: The endpoint path being requested
            metadata: Optional metadata from robai-webui containing:
                - message_id: Unique message identifier (used as connection_id)
                - chat_id: Chat session identifier
                - user_id: User identifier
                - variables: User context variables

        Returns:
            str: Connection ID (message_id if available, else UUID)
        """
        # Skip tracking for excluded endpoints
        if endpoint in self.excluded_endpoints:
            return ""

        # Use message_id as connection ID if available (unique per request)
        # Otherwise generate UUID for backward compatibility
        if metadata and metadata.get("message_id"):
            connection_id = metadata["message_id"]
            chat_id = metadata.get("chat_id")
            user_id = metadata.get("user_id")
            session_id = metadata.get("session_id")
            user_name = metadata.get("variables", {}).get("{{USER_NAME}}", "Unknown")
        else:
            connection_id = str(uuid.uuid4())
            chat_id = None
            user_id = None
            session_id = None
            user_name = "Unknown"

        # Create connection info with metadata
        connection_info = {
            "connection_id": connection_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "session_id": session_id,
            "user_name": user_name,
            "endpoint": endpoint,
            "method": request.method,
            "client_host": request.client.host if request.client else "unknown",
            "timestamp": asyncio.get_event_loop().time(),
            "status": "active",
            "has_metadata": metadata is not None
        }

        # Add connection with thread-safe operation
        async with self._lock:
            was_empty = len(self.connections) == 0
            self.connections[connection_id] = connection_info

            # Log with metadata context if available
            if metadata:
                logger.debug(
                    f"Connection added | ID: {connection_id[:8]}... | "
                    f"Chat: {chat_id[:8] if chat_id else 'none'}... | "
                    f"User: {user_name}"
                )
            else:
                logger.debug(f"Connection added: {connection_id} for {endpoint} (no metadata)")

            # Notify callback if we went from 0 to 1 connections
            if was_empty and self._activity_callback:
                try:
                    await self._activity_callback(True)  # True = activity started
                except Exception as e:
                    logger.error(f"Error in activity callback: {e}")

        return connection_id
    
    async def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a connection from the manager.
        
        Args:
            connection_id: Unique ID of the connection to remove
            
        Returns:
            bool: True if connection was found and removed, False otherwise
        """
        async with self._lock:
            if connection_id in self.connections:
                # Update status before removal
                self.connections[connection_id]["status"] = "closed"
                del self.connections[connection_id]
                logger.debug(f"Connection removed: {connection_id}")

                # Notify callback if we went from non-zero to 0 connections
                if len(self.connections) == 0 and self._activity_callback:
                    try:
                        await self._activity_callback(False)  # False = activity stopped
                    except Exception as e:
                        logger.error(f"Error in activity callback: {e}")

                return True
            return False
    
    async def _check_connection_timeout(self, connection_id: str):
        """
        Check if a connection has exceeded its timeout period.
        
        Args:
            connection_id: Unique ID of the connection to check
        """
        # Wait for the timeout period
        await asyncio.sleep(self.connection_timeout)
        
        # Check if connection still exists and is active
        async with self._lock:
            if connection_id in self.connections and self.connections[connection_id]["status"] == "active":
                # Connection has timed out - remove it
                self.connections[connection_id]["status"] = "timeout"
                del self.connections[connection_id]
                logger.warning(f"Connection timed out and removed: {connection_id}")
    
    async def get_connection(self, connection_id: str) -> Optional[Dict]:
        """
        Get connection info by ID.
        
        Args:
            connection_id: Unique ID of the connection
            
        Returns:
            Dict or None: Connection info if found, None otherwise
        """
        async with self._lock:
            return self.connections.get(connection_id)
    
    async def get_all_connections(self) -> Dict[str, Dict]:
        """
        Get all active connections.
        
        Returns:
            Dict: Dictionary of all active connections
        """
        async with self._lock:
            return self.connections.copy()
    
    async def get_connection_count(self) -> int:
        """
        Get the number of active connections.
        
        Returns:
            int: Number of active connections
        """
        async with self._lock:
            return len(self.connections)
    
    async def get_active_connections(self) -> Dict[str, Dict]:
        """
        Get all active connections (not closed).
        
        Returns:
            Dict: Dictionary of active connections
        """
        async with self._lock:
            return {
                conn_id: conn_info for conn_id, conn_info in self.connections.items()
                if conn_info["status"] == "active"
            }
    
    async def get_connection_by_endpoint(self, endpoint: str) -> Dict[str, Dict]:
        """
        Get all connections for a specific endpoint.

        Args:
            endpoint: Endpoint path to filter by

        Returns:
            Dict: Dictionary of connections for the specified endpoint
        """
        async with self._lock:
            return {
                conn_id: conn_info for conn_id, conn_info in self.connections.items()
                if conn_info["endpoint"] == endpoint and conn_info["status"] == "active"
            }

    async def get_connections_by_user(self, user_id: str) -> Dict[str, Dict]:
        """
        Get all active connections for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Dict: Dictionary of connections for the specified user
        """
        async with self._lock:
            return {
                conn_id: conn_info for conn_id, conn_info in self.connections.items()
                if conn_info.get("user_id") == user_id and conn_info["status"] == "active"
            }

    async def get_connections_by_chat(self, chat_id: str) -> Dict[str, Dict]:
        """
        Get all connections for a specific chat session.

        Args:
            chat_id: Chat session identifier

        Returns:
            Dict: Dictionary of connections for the specified chat
        """
        async with self._lock:
            return {
                conn_id: conn_info for conn_id, conn_info in self.connections.items()
                if conn_info.get("chat_id") == chat_id and conn_info["status"] == "active"
            }

    async def count_user_connections(self, user_id: str) -> int:
        """
        Count active connections for a user.

        Args:
            user_id: User identifier

        Returns:
            int: Number of active connections
        """
        async with self._lock:
            return sum(
                1 for conn_info in self.connections.values()
                if conn_info.get("user_id") == user_id and conn_info["status"] == "active"
            )
    
    async def clear_all_connections(self):
        """
        Clear all connections (for cleanup or restart).
        """
        async with self._lock:
            self.connections.clear()
            logger.info("All connections cleared")

    def set_activity_callback(self, callback):
        """
        Set a callback function to be called when connection activity changes.

        Args:
            callback: Async function that takes one bool parameter:
                     True = activity started (0 -> 1+ connections)
                     False = activity stopped (connections -> 0)
        """
        self._activity_callback = callback
    
    async def get_connection_status(self) -> Dict:
        """
        Get comprehensive connection status information.
        
        Returns:
            Dict: Status information including connection count and details
        """
        async with self._lock:
            active_connections = len(self.connections)
            return {
                "total_connections": active_connections,
                "active_connections": active_connections,
                "excluded_endpoints": list(self.excluded_endpoints),
                "connection_details": {
                    conn_id: {
                        "endpoint": conn_info["endpoint"],
                        "method": conn_info["method"],
                        "client_host": conn_info["client_host"],
                        "status": conn_info["status"],
                        "timestamp": conn_info["timestamp"]
                    }
                    for conn_id, conn_info in self.connections.items()
                }
            }

# Global instance of ConnectionManager
connection_manager = ConnectionManager()
