#!/usr/bin/env python3
"""
Quick utility to check connection manager statistics.
This imports the connection_manager singleton and prints current stats.
"""

import asyncio
from connectionManager import connection_manager

async def main():
    """Display connection manager statistics."""
    print("\n" + "=" * 80)
    print("CONNECTION MANAGER STATISTICS")
    print("=" * 80)

    # Get connection count
    count = await connection_manager.get_connection_count()
    print(f"Total Active Connections: {count}")

    # Get full status
    status = await connection_manager.get_connection_status()
    print(f"\nExcluded Endpoints: {', '.join(status['excluded_endpoints'])}")

    # Get active connections
    active = await connection_manager.get_active_connections()

    if active:
        print(f"\nActive Connections ({len(active)}):")
        print("-" * 80)
        for conn_id, info in active.items():
            print(f"ID: {conn_id[:8]}...")
            print(f"  Endpoint: {info['endpoint']}")
            print(f"  Method: {info['method']}")
            print(f"  Client: {info['client_host']}")
            print(f"  Status: {info['status']}")
            print(f"  Timestamp: {info['timestamp']}")
            print()
    else:
        print("\nNo active connections at this moment.")

    # Get all connections (includes any that might be in different states)
    all_conns = await connection_manager.get_all_connections()
    print(f"\nTotal Connections in Manager: {len(all_conns)}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
