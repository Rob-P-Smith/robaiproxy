"""
Rate Limiter for robaiproxy with per-user limits.

Implements user-aware rate limiting using metadata, tracking both request frequency
and token consumption per user with automatic cleanup of old data.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Tuple, Optional
from config import logger, config

class RateLimiter:
    """
    User-aware rate limiting using metadata from robai-webui.

    Features:
    - Per-user request rate limiting (requests per minute)
    - Per-user token consumption limiting (tokens per hour)
    - Automatic cleanup of expired tracking data
    - User statistics API
    - Thread-safe async operations
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_hour: int = 100000,
        cleanup_interval: int = 300
    ):
        """
        Initialize RateLimiter with configurable limits.

        Args:
            requests_per_minute: Max requests per minute per user (default 60)
            tokens_per_hour: Max tokens per hour per user (default 100,000)
            cleanup_interval: Seconds between cleanup runs (default 300)
        """
        # Request tracking: user_id -> list of timestamps
        self.request_timestamps: Dict[str, list] = defaultdict(list)

        # Token tracking: user_id -> {hour_bucket: token_count}
        self.token_usage: Dict[str, Dict[int, int]] = defaultdict(dict)

        # Configuration
        self.requests_per_minute = requests_per_minute
        self.tokens_per_hour = tokens_per_hour
        self.cleanup_interval = cleanup_interval

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "total_requests_allowed": 0,
            "total_requests_rejected": 0,
            "total_tokens_tracked": 0,
            "unique_users": set()
        }

        # Start cleanup task
        self._cleanup_task = None

    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("RateLimiter cleanup task started")

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("RateLimiter cleanup task stopped")

    async def _cleanup_loop(self):
        """Background task to periodically clean up old tracking data."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in RateLimiter cleanup: {e}")

    async def check_and_update(
        self,
        metadata: dict,
        estimated_tokens: int = 0
    ) -> Tuple[bool, str]:
        """
        Check if request is allowed and update counters.

        Args:
            metadata: Request metadata containing user_id
            estimated_tokens: Estimated tokens for this request

        Returns:
            Tuple of (allowed: bool, reason: str)
                - allowed: Whether request should be allowed
                - reason: Rejection reason if not allowed, empty if allowed
        """
        user_id = metadata.get("user_id")

        # No user_id means direct API call - allow but don't track
        if not user_id:
            logger.debug("No user_id in metadata - allowing untracked request")
            return True, ""

        current_time = time.time()
        current_minute = int(current_time / 60)
        current_hour = int(current_time / 3600)

        async with self._lock:
            # Track unique users
            self.stats["unique_users"].add(user_id)

            # Clean old request timestamps (older than 1 minute)
            minute_ago = current_time - 60
            self.request_timestamps[user_id] = [
                ts for ts in self.request_timestamps[user_id]
                if ts > minute_ago
            ]

            # Check request rate
            request_count = len(self.request_timestamps[user_id])
            if request_count >= self.requests_per_minute:
                self.stats["total_requests_rejected"] += 1

                user_name = metadata.get("variables", {}).get("{{USER_NAME}}", "Unknown")
                reason = (
                    f"Rate limit exceeded: {request_count} requests in last minute "
                    f"(max: {self.requests_per_minute})"
                )

                logger.warning(
                    f"Rate limit exceeded | User: {user_name} ({user_id[:8]}...) | "
                    f"Requests: {request_count}/{self.requests_per_minute}"
                )

                return False, reason

            # Check token usage if tokens specified
            if estimated_tokens > 0:
                hour_tokens = self.token_usage[user_id].get(current_hour, 0)

                if hour_tokens + estimated_tokens > self.tokens_per_hour:
                    self.stats["total_requests_rejected"] += 1

                    user_name = metadata.get("variables", {}).get("{{USER_NAME}}", "Unknown")
                    reason = (
                        f"Token limit exceeded: {hour_tokens + estimated_tokens:,} tokens "
                        f"in current hour (max: {self.tokens_per_hour:,})"
                    )

                    logger.warning(
                        f"Token limit exceeded | User: {user_name} ({user_id[:8]}...) | "
                        f"Tokens: {hour_tokens + estimated_tokens:,}/{self.tokens_per_hour:,}"
                    )

                    return False, reason

                # Update token usage
                self.token_usage[user_id][current_hour] = hour_tokens + estimated_tokens
                self.stats["total_tokens_tracked"] += estimated_tokens

            # Update request timestamps
            self.request_timestamps[user_id].append(current_time)
            self.stats["total_requests_allowed"] += 1

            # Log successful check
            logger.debug(
                f"Rate limit check passed | User: {user_id[:8]}... | "
                f"Requests: {request_count + 1}/{self.requests_per_minute} | "
                f"Tokens: {estimated_tokens}"
            )

            return True, ""

    async def get_user_status(self, user_id: str) -> dict:
        """
        Get current rate limit status for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing:
                - requests_last_minute: Number of requests in last minute
                - requests_remaining: Requests remaining this minute
                - tokens_this_hour: Tokens used this hour
                - tokens_remaining: Tokens remaining this hour
                - requests_per_minute_limit: Configured limit
                - tokens_per_hour_limit: Configured limit
        """
        if not user_id:
            return {
                "error": "No user_id provided",
                "requests_per_minute_limit": self.requests_per_minute,
                "tokens_per_hour_limit": self.tokens_per_hour
            }

        current_time = time.time()
        current_hour = int(current_time / 3600)
        minute_ago = current_time - 60

        async with self._lock:
            # Count recent requests
            recent_requests = [
                ts for ts in self.request_timestamps.get(user_id, [])
                if ts > minute_ago
            ]

            # Get token usage for current hour
            hour_tokens = self.token_usage.get(user_id, {}).get(current_hour, 0)

            return {
                "user_id": user_id,
                "requests_last_minute": len(recent_requests),
                "requests_remaining": max(0, self.requests_per_minute - len(recent_requests)),
                "tokens_this_hour": hour_tokens,
                "tokens_remaining": max(0, self.tokens_per_hour - hour_tokens),
                "requests_per_minute_limit": self.requests_per_minute,
                "tokens_per_hour_limit": self.tokens_per_hour
            }

    async def add_token_usage(self, user_id: str, token_count: int) -> None:
        """
        Add token usage for a user (for post-request tracking).

        Args:
            user_id: User identifier
            token_count: Number of tokens to add
        """
        if not user_id or token_count <= 0:
            return

        current_hour = int(time.time() / 3600)

        async with self._lock:
            current = self.token_usage[user_id].get(current_hour, 0)
            self.token_usage[user_id][current_hour] = current + token_count
            self.stats["total_tokens_tracked"] += token_count

    async def _cleanup_old_data(self) -> None:
        """Clean up old tracking data to prevent memory growth."""
        current_time = time.time()
        current_hour = int(current_time / 3600)
        minute_ago = current_time - 60

        async with self._lock:
            users_to_remove = []

            # Clean up request timestamps and old token buckets
            for user_id in list(self.request_timestamps.keys()):
                # Remove old request timestamps
                self.request_timestamps[user_id] = [
                    ts for ts in self.request_timestamps[user_id]
                    if ts > minute_ago
                ]

                # If no recent requests, mark for removal
                if not self.request_timestamps[user_id]:
                    users_to_remove.append(user_id)

            # Remove users with no recent activity
            for user_id in users_to_remove:
                del self.request_timestamps[user_id]

            # Clean up old hourly token buckets (keep last 24 hours)
            for user_id in list(self.token_usage.keys()):
                old_hours = [
                    hour for hour in self.token_usage[user_id].keys()
                    if current_hour - hour > 24
                ]

                for hour in old_hours:
                    del self.token_usage[user_id][hour]

                # Remove user if no token buckets remain
                if not self.token_usage[user_id]:
                    del self.token_usage[user_id]

            if users_to_remove or old_hours:
                logger.debug(
                    f"RateLimiter cleanup: Removed {len(users_to_remove)} inactive users, "
                    f"cleaned old token buckets"
                )

    async def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary containing:
                - total_requests_allowed: Total allowed requests
                - total_requests_rejected: Total rejected requests
                - rejection_rate: Percentage of rejected requests
                - total_tokens_tracked: Total tokens tracked
                - unique_users: Number of unique users seen
                - active_users: Users with recent activity
        """
        async with self._lock:
            total_requests = (
                self.stats["total_requests_allowed"] +
                self.stats["total_requests_rejected"]
            )

            rejection_rate = (
                (self.stats["total_requests_rejected"] / total_requests * 100)
                if total_requests > 0 else 0
            )

            # Count users with recent activity
            active_users = len(self.request_timestamps)

            return {
                **self.stats,
                "unique_users": len(self.stats["unique_users"]),
                "active_users": active_users,
                "rejection_rate": f"{rejection_rate:.1f}%",
                "avg_tokens_per_request": (
                    self.stats["total_tokens_tracked"] / self.stats["total_requests_allowed"]
                    if self.stats["total_requests_allowed"] > 0 else 0
                )
            }

    async def reset_user_limits(self, user_id: str) -> bool:
        """
        Reset rate limits for a specific user (admin function).

        Args:
            user_id: User identifier

        Returns:
            True if reset, False if user not found
        """
        if not user_id:
            return False

        async with self._lock:
            had_data = False

            if user_id in self.request_timestamps:
                del self.request_timestamps[user_id]
                had_data = True

            if user_id in self.token_usage:
                del self.token_usage[user_id]
                had_data = True

            if had_data:
                logger.info(f"Reset rate limits for user: {user_id[:8]}...")

            return had_data


# Global instance with config values
rate_limiter = RateLimiter(
    requests_per_minute=config.USER_REQUESTS_PER_MINUTE,
    tokens_per_hour=config.USER_TOKENS_PER_HOUR,
    cleanup_interval=config.RATE_LIMIT_CLEANUP_INTERVAL
)