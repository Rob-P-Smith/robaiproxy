"""
Analytics module for robaiproxy with metadata-driven tracking.

Provides comprehensive usage analytics per user, per chat, and globally,
with automatic data retention management.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Set, Optional, Any
from config import logger

class AnalyticsTracker:
    """
    Track usage analytics using metadata from robai-webui.

    Features:
    - Per-user usage statistics
    - Per-chat conversation metrics
    - Global system statistics
    - Research vs standard request tracking
    - Token usage aggregation
    - Automatic data retention management
    - Thread-safe async operations
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize AnalyticsTracker.

        Args:
            retention_hours: Hours to retain detailed analytics data (default 24)
        """
        # User statistics
        self.user_stats = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "total_chats": set(),
            "research_requests": 0,
            "rag_requests": 0,
            "multimodal_requests": 0,
            "error_count": 0,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "user_name": "Unknown"
        })

        # Chat statistics
        self.chat_stats = defaultdict(lambda: {
            "user_id": None,
            "messages": 0,
            "tokens": 0,
            "started": time.time(),
            "last_activity": time.time(),
            "research_count": 0,
            "rag_count": 0,
            "multimodal_count": 0,
            "error_count": 0
        })

        # Global statistics
        self.global_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "unique_users": set(),
            "unique_chats": set(),
            "research_requests": 0,
            "rag_requests": 0,
            "multimodal_requests": 0,
            "error_count": 0,
            "start_time": time.time()
        }

        # Hourly aggregates for time-series data
        self.hourly_stats = defaultdict(lambda: {
            "requests": 0,
            "tokens": 0,
            "unique_users": set(),
            "unique_chats": set(),
            "errors": 0
        })

        # Configuration
        self.retention_hours = retention_hours

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task = None

    async def start_cleanup_task(self):
        """Start background cleanup task for old data."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Analytics cleanup task started")

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Analytics cleanup task stopped")

    async def _cleanup_loop(self):
        """Background task to periodically clean up old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up hourly
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Analytics cleanup: {e}")

    async def log_request(
        self,
        metadata: dict,
        tokens_used: int = 0,
        request_type: str = "standard",
        error: bool = False
    ) -> None:
        """
        Log a request with metadata.

        Args:
            metadata: Request metadata from robai-webui
            tokens_used: Number of tokens consumed
            request_type: Type of request (standard, research, rag, multimodal)
            error: Whether the request resulted in an error
        """
        user_id = metadata.get("user_id", "anonymous")
        chat_id = metadata.get("chat_id", "ephemeral")
        user_name = metadata.get("variables", {}).get("{{USER_NAME}}", "Unknown")

        current_time = time.time()
        current_hour = int(current_time / 3600)

        async with self._lock:
            # Update user stats
            user_stat = self.user_stats[user_id]
            user_stat["total_requests"] += 1
            user_stat["total_tokens"] += tokens_used
            user_stat["total_chats"].add(chat_id)
            user_stat["last_seen"] = current_time
            user_stat["user_name"] = user_name

            if request_type == "research":
                user_stat["research_requests"] += 1
            elif request_type == "rag":
                user_stat["rag_requests"] += 1
            elif request_type == "multimodal":
                user_stat["multimodal_requests"] += 1

            if error:
                user_stat["error_count"] += 1

            # Update chat stats
            chat_stat = self.chat_stats[chat_id]
            if chat_stat["user_id"] is None:
                chat_stat["user_id"] = user_id
            chat_stat["messages"] += 1
            chat_stat["tokens"] += tokens_used
            chat_stat["last_activity"] = current_time

            if request_type == "research":
                chat_stat["research_count"] += 1
            elif request_type == "rag":
                chat_stat["rag_count"] += 1
            elif request_type == "multimodal":
                chat_stat["multimodal_count"] += 1

            if error:
                chat_stat["error_count"] += 1

            # Update global stats
            self.global_stats["total_requests"] += 1
            self.global_stats["total_tokens"] += tokens_used
            self.global_stats["unique_users"].add(user_id)
            self.global_stats["unique_chats"].add(chat_id)

            if request_type == "research":
                self.global_stats["research_requests"] += 1
            elif request_type == "rag":
                self.global_stats["rag_requests"] += 1
            elif request_type == "multimodal":
                self.global_stats["multimodal_requests"] += 1

            if error:
                self.global_stats["error_count"] += 1

            # Update hourly stats
            hourly = self.hourly_stats[current_hour]
            hourly["requests"] += 1
            hourly["tokens"] += tokens_used
            hourly["unique_users"].add(user_id)
            hourly["unique_chats"].add(chat_id)
            if error:
                hourly["errors"] += 1

            # Log if enabled
            if tokens_used > 0:
                logger.debug(
                    f"Analytics | User: {user_name} | "
                    f"Type: {request_type} | Tokens: {tokens_used}"
                )

    async def get_user_summary(self, user_id: str) -> dict:
        """
        Get analytics summary for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing user analytics
        """
        async with self._lock:
            if user_id not in self.user_stats:
                return {"error": "User not found"}

            stats = self.user_stats[user_id]
            current_time = time.time()

            return {
                "user_id": user_id,
                "user_name": stats["user_name"],
                "total_requests": stats["total_requests"],
                "total_tokens": stats["total_tokens"],
                "unique_chats": len(stats["total_chats"]),
                "research_requests": stats["research_requests"],
                "rag_requests": stats["rag_requests"],
                "multimodal_requests": stats["multimodal_requests"],
                "error_count": stats["error_count"],
                "avg_tokens_per_request": (
                    stats["total_tokens"] / stats["total_requests"]
                    if stats["total_requests"] > 0 else 0
                ),
                "time_active": current_time - stats["first_seen"],
                "last_seen_ago": current_time - stats["last_seen"]
            }

    async def get_chat_summary(self, chat_id: str) -> dict:
        """
        Get analytics summary for a specific chat session.

        Args:
            chat_id: Chat session identifier

        Returns:
            Dictionary containing chat analytics
        """
        async with self._lock:
            if chat_id not in self.chat_stats:
                return {"error": "Chat not found"}

            stats = self.chat_stats[chat_id]
            current_time = time.time()

            return {
                "chat_id": chat_id,
                "user_id": stats["user_id"],
                "messages": stats["messages"],
                "tokens": stats["tokens"],
                "duration": current_time - stats["started"],
                "last_activity_ago": current_time - stats["last_activity"],
                "research_count": stats["research_count"],
                "rag_count": stats["rag_count"],
                "multimodal_count": stats["multimodal_count"],
                "error_count": stats["error_count"],
                "avg_tokens_per_message": (
                    stats["tokens"] / stats["messages"]
                    if stats["messages"] > 0 else 0
                )
            }

    async def get_global_summary(self) -> dict:
        """
        Get global analytics summary.

        Returns:
            Dictionary containing global system analytics
        """
        async with self._lock:
            current_time = time.time()
            uptime = current_time - self.global_stats["start_time"]

            return {
                "uptime_seconds": uptime,
                "uptime_hours": uptime / 3600,
                "total_requests": self.global_stats["total_requests"],
                "total_tokens": self.global_stats["total_tokens"],
                "unique_users": len(self.global_stats["unique_users"]),
                "unique_chats": len(self.global_stats["unique_chats"]),
                "research_requests": self.global_stats["research_requests"],
                "rag_requests": self.global_stats["rag_requests"],
                "multimodal_requests": self.global_stats["multimodal_requests"],
                "error_count": self.global_stats["error_count"],
                "error_rate": (
                    self.global_stats["error_count"] / self.global_stats["total_requests"] * 100
                    if self.global_stats["total_requests"] > 0 else 0
                ),
                "avg_tokens_per_request": (
                    self.global_stats["total_tokens"] / self.global_stats["total_requests"]
                    if self.global_stats["total_requests"] > 0 else 0
                ),
                "requests_per_minute": (
                    self.global_stats["total_requests"] / (uptime / 60)
                    if uptime > 0 else 0
                ),
                "tokens_per_hour": (
                    self.global_stats["total_tokens"] / (uptime / 3600)
                    if uptime > 0 else 0
                )
            }

    async def get_hourly_summary(self, hours_back: int = 24) -> list:
        """
        Get hourly statistics for the last N hours.

        Args:
            hours_back: Number of hours to look back (default 24)

        Returns:
            List of hourly statistics dictionaries
        """
        current_hour = int(time.time() / 3600)
        result = []

        async with self._lock:
            for i in range(hours_back):
                hour = current_hour - i
                if hour in self.hourly_stats:
                    stats = self.hourly_stats[hour]
                    result.append({
                        "hour": hour,
                        "timestamp": hour * 3600,
                        "requests": stats["requests"],
                        "tokens": stats["tokens"],
                        "unique_users": len(stats["unique_users"]),
                        "unique_chats": len(stats["unique_chats"]),
                        "errors": stats["errors"]
                    })

        return sorted(result, key=lambda x: x["hour"])

    async def get_top_users(self, limit: int = 10) -> list:
        """
        Get top users by request count.

        Args:
            limit: Number of top users to return (default 10)

        Returns:
            List of user statistics sorted by request count
        """
        async with self._lock:
            users = []
            for user_id, stats in self.user_stats.items():
                users.append({
                    "user_id": user_id,
                    "user_name": stats["user_name"],
                    "total_requests": stats["total_requests"],
                    "total_tokens": stats["total_tokens"],
                    "unique_chats": len(stats["total_chats"])
                })

            return sorted(users, key=lambda x: x["total_requests"], reverse=True)[:limit]

    async def _cleanup_old_data(self) -> None:
        """Clean up analytics data older than retention period."""
        current_hour = int(time.time() / 3600)
        cutoff_hour = current_hour - self.retention_hours

        async with self._lock:
            # Clean up old hourly stats
            hours_to_remove = [
                hour for hour in self.hourly_stats.keys()
                if hour < cutoff_hour
            ]

            for hour in hours_to_remove:
                del self.hourly_stats[hour]

            if hours_to_remove:
                logger.info(f"Analytics: Cleaned up {len(hours_to_remove)} old hourly records")

    async def export_metrics(self) -> dict:
        """
        Export all metrics for external monitoring systems.

        Returns:
            Dictionary with all metrics suitable for Prometheus/Grafana
        """
        async with self._lock:
            current_time = time.time()
            uptime = current_time - self.global_stats["start_time"]

            return {
                "robaiproxy_uptime_seconds": uptime,
                "robaiproxy_total_requests": self.global_stats["total_requests"],
                "robaiproxy_total_tokens": self.global_stats["total_tokens"],
                "robaiproxy_unique_users": len(self.global_stats["unique_users"]),
                "robaiproxy_unique_chats": len(self.global_stats["unique_chats"]),
                "robaiproxy_research_requests": self.global_stats["research_requests"],
                "robaiproxy_rag_requests": self.global_stats["rag_requests"],
                "robaiproxy_multimodal_requests": self.global_stats["multimodal_requests"],
                "robaiproxy_error_count": self.global_stats["error_count"],
                "robaiproxy_active_user_count": len(self.user_stats),
                "robaiproxy_active_chat_count": len(self.chat_stats)
            }


# Global instance
analytics_tracker = AnalyticsTracker()