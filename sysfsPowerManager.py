#!/usr/bin/env python3
"""
GPU Power Manager using Direct sysfs Control for AMD ROCm GPUs.

Automatically manages GPU clock speeds and fan speeds based on API activity:
- Sets clocks to 500MHz and fans to 20% after 2 minutes of idle (no active connections)
- Sets clocks to ~3560MHz and fans to 65% when connections arrive
- Thread-safe atomic sysfs writes
- Integrates with connectionManager for event-driven updates

Performance States:
- Idle: 500MHz core clock, 20% fan speed
- Active: ~3560MHz core clock, 65% fan speed
"""

import asyncio
from typing import Optional, List
from enum import Enum
from pathlib import Path
from config import logger

# Import connection manager for monitoring
from connectionManager import connection_manager


class PerformanceState(str, Enum):
    """GPU performance states."""
    IDLE = "idle"      # 500MHz, 20% fan
    ACTIVE = "active"  # ~3560MHz, 65% fan


class GPUConfig:
    """Configuration for a single GPU."""
    def __init__(self, card_id: int, hwmon_id: int):
        """
        Initialize GPU configuration.

        Args:
            card_id (int): GPU card number (0 or 1)
            hwmon_id (int): hwmon device number for this GPU
        """
        self.card_id = card_id
        self.base_path = Path(f"/sys/class/drm/card{card_id}/device")
        self.hwmon_path = self.base_path / f"hwmon/hwmon{hwmon_id}"

        # Control file paths
        self.perf_level_file = self.base_path / "power_dpm_force_performance_level"
        self.sclk_file = self.base_path / "pp_dpm_sclk"
        self.pwm_file = self.hwmon_path / "pwm1"

    def __repr__(self):
        return f"GPU{self.card_id}"


class SysfsPowerManager:
    """
    Manages GPU power states via direct sysfs writes.

    Monitors connection activity and switches between idle (low power) and
    active (high performance) states.
    """

    def __init__(self, idle_timeout_seconds: int = 120):
        """
        Initialize the SysfsPowerManager.

        Args:
            idle_timeout_seconds (int): Seconds of idle before switching to low power.
                                        Default: 120 seconds (2 minutes)
        """
        # GPU configurations
        # GPU0 = card0, hwmon5
        # GPU1 = card1, hwmon4
        self.gpus: List[GPUConfig] = [
            GPUConfig(card_id=0, hwmon_id=5),
            GPUConfig(card_id=1, hwmon_id=4)
        ]

        # Current performance state
        self._current_state: Optional[PerformanceState] = None

        # Lock for atomic sysfs operations
        self._sysfs_lock = asyncio.Lock()

        # Idle timeout configuration
        self.idle_timeout_seconds = idle_timeout_seconds

        # Timer task for switching to idle state
        self._idle_timer_task: Optional[asyncio.Task] = None

        # Running state
        self._running = False

        logger.info(f"SysfsPowerManager initialized with {idle_timeout_seconds} second idle timeout")
        logger.info(f"Managing GPUs: {', '.join(str(gpu) for gpu in self.gpus)}")

    async def _write_sysfs(self, file_path: Path, value: str) -> bool:
        """
        Write a value to a sysfs file.

        Args:
            file_path (Path): Path to sysfs file
            value (str): Value to write

        Returns:
            bool: True if write succeeded, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                f.write(value)
            return True
        except PermissionError:
            logger.error(f"Permission denied writing to {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {str(e)}")
            return False

    async def _set_performance_state(self, state: PerformanceState) -> bool:
        """
        Set GPU performance state atomically for all GPUs.

        Args:
            state (PerformanceState): Target performance state

        Returns:
            bool: True if all operations succeeded
        """
        async with self._sysfs_lock:
            success = True

            # Define state parameters
            if state == PerformanceState.IDLE:
                # Idle: 500MHz (fan controlled by GPU automatically)
                clock_index = "0"  # 500MHz
                state_desc = "500MHz core"
            else:  # ACTIVE
                # Active: ~3560MHz (fan controlled by GPU automatically)
                clock_index = "1"  # ~3560MHz
                state_desc = "~3560MHz core"

            logger.info(f"Setting performance state to {state.value}: {state_desc}")

            # Apply to all GPUs
            for gpu in self.gpus:
                # Step 1: Set performance level to manual
                if not await self._write_sysfs(gpu.perf_level_file, "manual"):
                    logger.error(f"{gpu}: Failed to set performance level to manual")
                    success = False
                    continue

                # Step 2: Set core clock
                if not await self._write_sysfs(gpu.sclk_file, clock_index):
                    logger.error(f"{gpu}: Failed to set core clock to index {clock_index}")
                    success = False
                    continue

                logger.info(f"✓ {gpu}: Applied {state.value} state successfully (clocks only, fan auto)")

            if success:
                self._current_state = state
                logger.info(f"✓ All GPUs set to {state.value} state")
            else:
                logger.error(f"✗ Some GPUs failed to apply {state.value} state")

            return success

    async def get_current_state(self) -> Optional[PerformanceState]:
        """
        Get the currently tracked performance state.

        Returns:
            PerformanceState or None: Current state, or None if not yet set
        """
        return self._current_state

    def _cancel_idle_timer(self):
        """Cancel the idle timer if it's running."""
        if self._idle_timer_task and not self._idle_timer_task.done():
            self._idle_timer_task.cancel()
            logger.debug("Idle timer cancelled")

    async def _idle_timer(self):
        """
        Timer that waits for idle_timeout_seconds then switches to idle state.
        """
        try:
            logger.info(f"Idle timer started: will switch to idle in {self.idle_timeout_seconds} seconds")
            await asyncio.sleep(self.idle_timeout_seconds)
            # Timer completed - switch to idle state
            logger.info(f"Idle timeout reached ({self.idle_timeout_seconds}s), switching to idle state")
            await self._set_performance_state(PerformanceState.IDLE)
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
            # Activity started: Cancel timer and switch to active
            logger.info("Activity detected, switching to active state and cancelling idle timer")
            self._cancel_idle_timer()
            await self._set_performance_state(PerformanceState.ACTIVE)
        else:
            # Activity stopped: Start idle timer
            logger.info(f"All connections closed, starting {self.idle_timeout_seconds}s idle timer")
            self._cancel_idle_timer()  # Cancel any existing timer
            self._idle_timer_task = asyncio.create_task(self._idle_timer())

    async def start(self):
        """
        Start the power manager and register with connection manager.

        Sets initial state to IDLE (low power).
        """
        if self._running:
            logger.warning("SysfsPowerManager already running")
            return

        self._running = True

        # Set initial state to IDLE
        await self._set_performance_state(PerformanceState.IDLE)

        # Register callback with connection manager
        connection_manager.set_activity_callback(self.on_activity_change)

        logger.info("SysfsPowerManager started and registered with connectionManager")

    async def stop(self):
        """
        Stop the power manager and cleanup.

        Cancels idle timer and restores active state.
        """
        if not self._running:
            return

        self._running = False

        # Cancel idle timer
        self._cancel_idle_timer()

        # Unregister callback
        connection_manager.set_activity_callback(None)

        # Restore active state on shutdown
        logger.info("SysfsPowerManager stopping, restoring active state")
        await self._set_performance_state(PerformanceState.ACTIVE)

        logger.info("SysfsPowerManager stopped")

    async def get_status(self) -> dict:
        """
        Get comprehensive power manager status.

        Returns:
            dict: Status information including state, connections, timer
        """
        conn_count = await connection_manager.get_connection_count()
        timer_running = self._idle_timer_task is not None and not self._idle_timer_task.done()

        return {
            "running": self._running,
            "current_state": self._current_state.value if self._current_state else "unknown",
            "active_connections": conn_count,
            "idle_timer_running": timer_running,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "mode": "event-driven-sysfs",
            "gpus": [str(gpu) for gpu in self.gpus]
        }


# Global singleton instance
sysfs_power_manager = SysfsPowerManager(idle_timeout_seconds=120)  # 2 minutes


if __name__ == "__main__":
    """Test the sysfs power manager standalone."""
    async def test():
        print("Testing SysfsPowerManager...")

        # Start manager
        await sysfs_power_manager.start()

        # Show status
        status = await sysfs_power_manager.get_status()
        print(f"\nInitial Status:")
        print(f"  State: {status['current_state']}")
        print(f"  Active Connections: {status['active_connections']}")
        print(f"  Idle Timeout: {status['idle_timeout_seconds']} seconds")
        print(f"  GPUs: {', '.join(status['gpus'])}")

        # Simulate activity
        print("\nSimulating activity...")
        await sysfs_power_manager.on_activity_change(True)

        # Wait a bit
        print("\nWaiting 5 seconds...")
        await asyncio.sleep(5)

        # Show status
        status = await sysfs_power_manager.get_status()
        print(f"\nAfter activity:")
        print(f"  State: {status['current_state']}")

        # Stop activity
        print("\nStopping activity...")
        await sysfs_power_manager.on_activity_change(False)

        # Wait a bit
        print("\nWaiting 10 seconds...")
        await asyncio.sleep(10)

        # Stop manager
        await sysfs_power_manager.stop()
        print("\nSysfsPowerManager test complete")

    asyncio.run(test())
