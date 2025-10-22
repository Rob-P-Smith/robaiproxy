#!/usr/bin/env python3
"""
Quick utility to check power manager status.
This imports the power_manager singleton and prints current status.
"""

import asyncio
from powerManager import power_manager
from connectionManager import connection_manager

async def main():
    """Display power manager status."""
    print("\n" + "=" * 80)
    print("POWER MANAGER STATUS")
    print("=" * 80)

    # Get power manager status
    status = await power_manager.get_status()

    print(f"Running: {status['running']}")
    print(f"Current Power Level: {status['current_power_level']}")
    print(f"Active Connections: {status['active_connections']}")
    print(f"Last Activity: {status['last_activity']}")
    print(f"Idle Duration: {status['idle_duration_minutes']} minutes ({status['idle_duration_seconds']} seconds)")
    print(f"Is Idle: {status['is_idle']}")
    print(f"Idle Timeout: {status['idle_timeout_minutes']} minutes")

    if not status['is_idle']:
        time_until_low = status['time_until_low_power_seconds']
        mins = time_until_low // 60
        secs = time_until_low % 60
        print(f"Time Until Low Power: {mins}m {secs}s")
    else:
        print(f"Time Until Low Power: Already idle (should be in low power mode)")

    print("=" * 80 + "\n")

    # Also show connection details
    conn_count = await connection_manager.get_connection_count()
    if conn_count > 0:
        print("ACTIVE CONNECTIONS:")
        print("=" * 80)
        active = await connection_manager.get_active_connections()
        for conn_id, info in active.items():
            print(f"  {conn_id[:8]}... - {info['endpoint']} ({info['method']}) - {info['client_host']}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
