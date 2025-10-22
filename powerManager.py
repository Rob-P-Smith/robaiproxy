#!/usr/bin/env python3
"""
GPU Power Manager for AMD ROCm GPUs.

Automatically manages GPU performance levels based on API activity:
- Sets rocm-smi to 'low' after 10 minutes of idle (no active connections)
- Sets rocm-smi to 'auto' when connections arrive (if currently in 'low' mode)
- Thread-safe atomic command execution to prevent race conditions
- Integrates with connectionManager to track API activity

Performance Levels:
- auto: Automatically adjust based on workload (default for active systems)
- low: Minimum power/performance (idle state)
- high: Maximum performance (not used by this manager)
- manual: Manual control (not used by this manager)
"""

import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Literal, Optional
from enum import Enum
from config import logger

# Import connection manager for monitoring
from connectionManager import connection_manager


class PowerLevel(str, Enum):
    """Supported ROCm power levels."""
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"
    MANUAL = "manual"


class PowerManager:
    """
    Manages GPU power levels based on API activity.

    Monitors connection count and automatically switches between 'auto' and 'low'
    performance levels to save power during idle periods.
    """

    def __init__(self, idle_timeout_seconds: int = 600):
        """
        Initialize the PowerManager.

        Args:
            idle_timeout_seconds (int): Seconds of idle before switching to low power.
                                        Default: 600 seconds (10 minutes)
        """
        # Current power level state
        self._current_level: Optional[PowerLevel] = None

        # Lock for atomic rocm-smi command execution
        self._command_lock = asyncio.Lock()

        # Idle timeout configuration
        self.idle_timeout_seconds = idle_timeout_seconds

        # Timer task for switching to low power
        self._idle_timer_task: Optional[asyncio.Task] = None

        # Running state
        self._running = False

        logger.info(f"PowerManager initialized with {idle_timeout_seconds} second idle timeout")

    async def _execute_rocm_smi(self, level: PowerLevel) -> bool:
        """
        Execute rocm-smi command atomically with thread safety.

        Uses asyncio lock to ensure only one command runs at a time,
        preventing race conditions and command conflicts.

        Args:
            level (PowerLevel): Target performance level to set

        Returns:
            bool: True if command succeeded, False otherwise
        """
        async with self._command_lock:
            try:
                # Build command
                cmd = ["sudo", "rocm-smi", "--setperflevel", level.value]

                logger.info(f"Executing: {' '.join(cmd)}")

                # Execute command with timeout
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Wait for completion with 10 second timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=10.0
                )

                if process.returncode == 0:
                    logger.info(f"✓ Successfully set performance level to: {level.value}")
                    self._current_level = level
                    return True
                else:
                    error_msg = stderr.decode().strip() if stderr else "Unknown error"
                    logger.error(f"✗ rocm-smi command failed (exit {process.returncode}): {error_msg}")
                    return False

            except asyncio.TimeoutError:
                logger.error("✗ rocm-smi command timed out after 10 seconds")
                return False
            except FileNotFoundError:
                logger.error("✗ rocm-smi command not found. Is ROCm installed?")
                return False
            except Exception as e:
                logger.error(f"✗ Unexpected error executing rocm-smi: {str(e)}")
                return False

    async def set_power_level(self, level: PowerLevel, force: bool = False) -> bool:
        """
        Set GPU performance level (only if different from current state).

        Args:
            level (PowerLevel): Target performance level
            force (bool): Force execution even if already at this level

        Returns:
            bool: True if command succeeded or was skipped (already at level)
        """
        # Skip if already at this level (unless forced)
        if not force and self._current_level == level:
            logger.debug(f"Already at power level '{level.value}', skipping command")
            return True

        return await self._execute_rocm_smi(level)

    async def get_current_level(self) -> Optional[PowerLevel]:
        """
        Get the currently tracked power level.

        Returns:
            PowerLevel or None: Current level, or None if not yet set
        """
        return self._current_level

    def _cancel_idle_timer(self):
        """Cancel the idle timer if it's running."""
        if self._idle_timer_task and not self._idle_timer_task.done():
            self._idle_timer_task.cancel()
            logger.debug("Idle timer cancelled")

    async def _idle_timer(self):
        """
        Timer that waits for idle_timeout_seconds then switches to low power.
        """
        try:
            logger.info(f"Idle timer started: will switch to low power in {self.idle_timeout_seconds} seconds")
            await asyncio.sleep(self.idle_timeout_seconds)
            # Timer completed - switch to low power
            logger.info(f"Idle timeout reached ({self.idle_timeout_seconds}s), switching to low power")
            await self.set_power_level(PowerLevel.LOW)
        except asyncio.CancelledError:
            logger.debug("Idle timer was cancelled (activity detected)")
            raise

    async def on_activity_change(self, has_activity: bool):
        """
        Event handler called by connectionManager when activity state changes.

        Args:
            has_activity (bool): True if connections went from 0 to 1+,
                                False if connections went to 0
        """
        if has_activity:
            # Activity started: Cancel timer and switch to auto
            logger.info("Activity detected, switching to auto and cancelling idle timer")
            self._cancel_idle_timer()
            await self.set_power_level(PowerLevel.AUTO)
        else:
            # Activity stopped: Start idle timer
            logger.info(f"All connections closed, starting {self.idle_timeout_seconds}s idle timer")
            self._cancel_idle_timer()  # Cancel any existing timer
            self._idle_timer_task = asyncio.create_task(self._idle_timer())

    async def start(self):
        """
        Start the power manager and register with connection manager.

        Sets initial state to LOW power (fast to switch to auto when request comes).
        """
        if self._running:
            logger.warning("PowerManager already running")
            return

        self._running = True

        # Set initial state to LOW power
        await self.set_power_level(PowerLevel.LOW)

        # Register callback with connection manager
        connection_manager.set_activity_callback(self.on_activity_change)

        logger.info("PowerManager started and registered with connectionManager")

    async def stop(self):
        """
        Stop the power manager and cleanup.

        Cancels idle timer and restores auto mode.
        """
        if not self._running:
            return

        self._running = False

        # Cancel idle timer
        self._cancel_idle_timer()

        # Unregister callback
        connection_manager.set_activity_callback(None)

        # Restore auto mode on shutdown
        logger.info("PowerManager stopping, restoring auto mode")
        await self.set_power_level(PowerLevel.AUTO)

        logger.info("PowerManager stopped")

    async def get_status(self) -> dict:
        """
        Get comprehensive power manager status.

        Returns:
            dict: Status information including power level, connections, timer state
        """
        conn_count = await connection_manager.get_connection_count()
        timer_running = self._idle_timer_task is not None and not self._idle_timer_task.done()

        return {
            "running": self._running,
            "current_power_level": self._current_level.value if self._current_level else "unknown",
            "active_connections": conn_count,
            "idle_timer_running": timer_running,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "mode": "event-driven"
        }


# Global singleton instance
power_manager = PowerManager(idle_timeout_seconds=120)  # 2 minutes


if __name__ == "__main__":
    """Test the power manager standalone."""
    async def test():
        print("Testing PowerManager...")

        # Start manager
        await power_manager.start()

        # Show status
        status = await power_manager.get_status()
        print(f"\nInitial Status:")
        print(f"  Power Level: {status['current_power_level']}")
        print(f"  Active Connections: {status['active_connections']}")
        print(f"  Idle Duration: {status['idle_duration_minutes']} minutes")
        print(f"  Is Idle: {status['is_idle']}")

        # Wait a bit
        print("\nMonitoring for 60 seconds...")
        await asyncio.sleep(60)

        # Show status again
        status = await power_manager.get_status()
        print(f"\nAfter 60 seconds:")
        print(f"  Power Level: {status['current_power_level']}")
        print(f"  Idle Duration: {status['idle_duration_minutes']} minutes")

        # Stop manager
        await power_manager.stop()
        print("\nPowerManager test complete")

    asyncio.run(test())
