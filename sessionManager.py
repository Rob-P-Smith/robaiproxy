"""
Session Manager for robaiproxy with metadata integration.

Manages chat sessions using metadata from robai-webui, tracking conversation state,
token usage, and augmentation history with LRU cache for memory efficiency.
"""

import asyncio
import time
from collections import OrderedDict
from typing import Dict, Optional, Any
from config import logger

class SessionManager:
    """
    Manages chat sessions using metadata from robai-webui.

    Features:
    - LRU cache with configurable max sessions (default 1000)
    - Automatic session creation from metadata
    - Token usage tracking per session
    - RAG augmentation tracking to prevent duplicates
    - Research request counting
    - User variable storage for personalization
    - Thread-safe async operations
    """

    def __init__(self, max_sessions: int = 1000):
        """
        Initialize SessionManager with LRU cache.

        Args:
            max_sessions: Maximum number of sessions to cache (default 1000)
        """
        self.sessions = OrderedDict()  # LRU cache implementation
        self.max_sessions = max_sessions
        self._lock = asyncio.Lock()

        # Statistics tracking
        self.stats = {
            "sessions_created": 0,
            "sessions_evicted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_messages": 0,
            "total_tokens": 0
        }

    async def get_or_create_session(self, metadata: dict) -> dict:
        """
        Get existing session or create new one based on chat_id.

        Args:
            metadata: Request metadata from robai-webui containing:
                - chat_id: Unique conversation identifier
                - user_id: User identifier
                - message_id: Message identifier
                - variables: User context variables

        Returns:
            Session dictionary with tracking information:
                - user_id: User identifier
                - created_at: Session creation timestamp
                - last_activity: Last activity timestamp
                - message_count: Number of messages in session
                - total_tokens: Total tokens used in session
                - rag_augmented: Whether RAG was applied
                - research_count: Number of research requests
                - is_first_message: Whether this is the first message
                - user_variables: User context from metadata
        """
        chat_id = metadata.get("chat_id")
        user_id = metadata.get("user_id")

        if not chat_id:
            # No chat_id means direct API call (not from robai-webui)
            # Return ephemeral session that won't be cached
            logger.debug("No chat_id in metadata - creating ephemeral session")
            return {
                "chat_id": None,
                "user_id": None,
                "created_at": time.time(),
                "last_activity": time.time(),
                "message_count": 1,
                "total_tokens": 0,
                "rag_augmented": False,
                "research_count": 0,
                "is_first_message": True,
                "is_ephemeral": True,
                "user_variables": {}
            }

        async with self._lock:
            if chat_id in self.sessions:
                # Existing session - update and move to end (most recently used)
                session = self.sessions[chat_id]
                session["last_activity"] = time.time()
                session["message_count"] += 1
                session["is_first_message"] = False

                # Move to end for LRU
                self.sessions.move_to_end(chat_id)

                # Update statistics
                self.stats["cache_hits"] += 1
                self.stats["total_messages"] += 1

                logger.debug(
                    f"Session hit | Chat: {chat_id[:8]}... | "
                    f"Message #{session['message_count']} | "
                    f"Tokens: {session['total_tokens']}"
                )

                return session

            else:
                # New session - create and add to cache
                user_name = metadata.get("variables", {}).get("{{USER_NAME}}", "Unknown")

                session = {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "created_at": time.time(),
                    "last_activity": time.time(),
                    "message_count": 1,
                    "total_tokens": 0,
                    "rag_augmented": False,
                    "research_count": 0,
                    "is_first_message": True,
                    "is_ephemeral": False,
                    "user_variables": metadata.get("variables", {}),
                    "user_name": user_name,
                    "features": metadata.get("features", {}),
                    "model_info": metadata.get("model", {})
                }

                self.sessions[chat_id] = session

                # Update statistics
                self.stats["sessions_created"] += 1
                self.stats["cache_misses"] += 1
                self.stats["total_messages"] += 1

                logger.info(
                    f"New session | User: {user_name} ({user_id[:8] if user_id else 'none'}...) | "
                    f"Chat: {chat_id[:8]}..."
                )

                # LRU eviction if cache is full
                if len(self.sessions) > self.max_sessions:
                    # Remove oldest (first) item
                    evicted_id, evicted_session = self.sessions.popitem(last=False)
                    self.stats["sessions_evicted"] += 1

                    age = time.time() - evicted_session["created_at"]
                    logger.debug(
                        f"Session evicted | Chat: {evicted_id[:8]}... | "
                        f"Age: {age:.1f}s | Messages: {evicted_session['message_count']}"
                    )

                return session

    async def update_session(self, chat_id: str, updates: dict) -> bool:
        """
        Update session with new information.

        Args:
            chat_id: Chat session identifier
            updates: Dictionary of fields to update

        Returns:
            True if session was updated, False if not found
        """
        if not chat_id:
            return False

        async with self._lock:
            if chat_id in self.sessions:
                self.sessions[chat_id].update(updates)
                self.sessions[chat_id]["last_activity"] = time.time()

                # Track token updates in global stats
                if "total_tokens" in updates:
                    token_delta = updates["total_tokens"] - self.sessions[chat_id].get("total_tokens", 0)
                    self.stats["total_tokens"] += token_delta

                return True
            return False

    async def get_session(self, chat_id: str) -> Optional[dict]:
        """
        Get session without creating if not exists.

        Args:
            chat_id: Chat session identifier

        Returns:
            Session dictionary if found, None otherwise
        """
        if not chat_id:
            return None

        async with self._lock:
            if chat_id in self.sessions:
                # Move to end for LRU (accessing counts as use)
                self.sessions.move_to_end(chat_id)
                return self.sessions[chat_id].copy()
            return None

    async def mark_rag_augmented(self, chat_id: str) -> bool:
        """
        Mark session as having received RAG augmentation.

        Args:
            chat_id: Chat session identifier

        Returns:
            True if marked, False if session not found
        """
        return await self.update_session(chat_id, {
            "rag_augmented": True,
            "rag_timestamp": time.time()
        })

    async def increment_research_count(self, chat_id: str) -> bool:
        """
        Increment research request count for a session.

        Args:
            chat_id: Chat session identifier

        Returns:
            True if incremented, False if session not found
        """
        if not chat_id:
            return False

        async with self._lock:
            if chat_id in self.sessions:
                self.sessions[chat_id]["research_count"] += 1
                self.sessions[chat_id]["last_research"] = time.time()
                return True
            return False

    async def add_tokens(self, chat_id: str, token_count: int) -> bool:
        """
        Add tokens to session total.

        Args:
            chat_id: Chat session identifier
            token_count: Number of tokens to add

        Returns:
            True if added, False if session not found
        """
        if not chat_id:
            return False

        async with self._lock:
            if chat_id in self.sessions:
                current = self.sessions[chat_id].get("total_tokens", 0)
                self.sessions[chat_id]["total_tokens"] = current + token_count
                self.stats["total_tokens"] += token_count
                return True
            return False

    async def get_stats(self) -> dict:
        """
        Get session manager statistics.

        Returns:
            Dictionary containing:
                - active_sessions: Current number of cached sessions
                - max_sessions: Maximum cache size
                - sessions_created: Total sessions created
                - sessions_evicted: Total sessions evicted
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - cache_hit_rate: Percentage of cache hits
                - total_messages: Total messages processed
                - total_tokens: Total tokens processed
        """
        async with self._lock:
            total_lookups = self.stats["cache_hits"] + self.stats["cache_misses"]
            cache_hit_rate = (
                (self.stats["cache_hits"] / total_lookups * 100)
                if total_lookups > 0 else 0
            )

            return {
                **self.stats,
                "active_sessions": len(self.sessions),
                "max_sessions": self.max_sessions,
                "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "avg_tokens_per_message": (
                    self.stats["total_tokens"] / self.stats["total_messages"]
                    if self.stats["total_messages"] > 0 else 0
                )
            }

    async def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up sessions older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds (default 1 hour)

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        sessions_to_remove = []

        async with self._lock:
            for chat_id, session in self.sessions.items():
                age = current_time - session["last_activity"]
                if age > max_age_seconds:
                    sessions_to_remove.append(chat_id)

            removed_count = 0
            for chat_id in sessions_to_remove:
                del self.sessions[chat_id]
                removed_count += 1
                logger.debug(f"Cleaned up old session: {chat_id[:8]}...")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old sessions")

            return removed_count

    async def get_user_sessions(self, user_id: str) -> list:
        """
        Get all sessions for a specific user.

        Args:
            user_id: User identifier

        Returns:
            List of session dictionaries for the user
        """
        if not user_id:
            return []

        async with self._lock:
            return [
                session.copy()
                for session in self.sessions.values()
                if session.get("user_id") == user_id
            ]


# Global instance
session_manager = SessionManager()