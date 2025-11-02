#!/usr/bin/env python3
"""
Test script for transparent tool calling with guards.

Tests:
1. Simple tool call (no guards triggered)
2. Search limit guard (simple_search limit > 3)
3. Multi-search guard (multiple search types)
4. Deep crawl streaming
5. No tools needed (passthrough)
"""

import requests
import json

PROXY_URL = "http://localhost:8079/v1/chat/completions"

def test_simple_tool_call():
    """Test 1: Simple tool call with no guards triggered."""
    print("\n" + "="*80)
    print("TEST 1: Simple Tool Call (No Guards)")
    print("="*80)

    payload = {
        "model": "Qwen3-30B",
        "messages": [
            {"role": "user", "content": "What's in my knowledge base about Python async?"}
        ],
        "stream": False
    }

    response = requests.post(PROXY_URL, json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(f"Response preview: {content[:200]}...")
        print("✓ Test passed")
    else:
        print(f"✗ Test failed: {response.text}")


def test_search_limit_guard():
    """Test 2: Search limit guard (simple_search with limit > 3)."""
    print("\n" + "="*80)
    print("TEST 2: Search Limit Guard")
    print("="*80)
    print("This test requires the LLM to request simple_search(limit>3) + another search")
    print("The guard should block the secondary search.")
    print("Manual inspection required - check proxy logs for:")
    print("  - '⚠️  simple_search with limit > 3, blocking other searches'")
    print("  - Blocked tool with error message")


def test_multi_search_guard():
    """Test 3: Multi-search guard (multiple search types)."""
    print("\n" + "="*80)
    print("TEST 3: Multi-Search Guard")
    print("="*80)
    print("This test requires the LLM to request multiple search types together")
    print("Manual inspection required - check proxy logs for:")
    print("  - '⚠️  Multiple search types detected'")
    print("  - Blocked secondary searches")


def test_deep_crawl_streaming():
    """Test 4: Deep crawl with streaming progress."""
    print("\n" + "="*80)
    print("TEST 4: Deep Crawl Streaming (STREAMING MODE)")
    print("="*80)
    print("This test requires manual triggering via OpenWebUI")
    print("Expected behavior:")
    print("  1. See '🕷️ Deep Crawl in Progress' message")
    print("  2. See progress updates")
    print("  3. See final answer after crawl completes")


def test_no_tools_passthrough():
    """Test 5: No tools needed (pure passthrough)."""
    print("\n" + "="*80)
    print("TEST 5: No Tools Needed (Passthrough)")
    print("="*80)

    payload = {
        "model": "Qwen3-30B",
        "messages": [
            {"role": "user", "content": "Hello! Just testing, no tools needed."}
        ],
        "stream": False
    }

    response = requests.post(PROXY_URL, json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(f"Response preview: {content[:200]}...")
        print("✓ Test passed")
    else:
        print(f"✗ Test failed: {response.text}")


def test_health_endpoint():
    """Verify proxy is healthy."""
    print("\n" + "="*80)
    print("PRE-TEST: Health Check")
    print("="*80)

    response = requests.get("http://localhost:8079/health")

    if response.status_code == 200:
        data = response.json()
        print(f"Service: {data['service']}")
        print(f"Status: {data['status']}")
        print(f"vLLM: {data['services']['vllm-qwen3']['status']}")
        print(f"Model: {data['services']['vllm-qwen3']['model_name']}")
        print("✓ Proxy is healthy")
        return True
    else:
        print(f"✗ Proxy unhealthy: {response.status_code}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSPARENT TOOL CALLING TEST SUITE")
    print("="*80)

    # Health check first
    if not test_health_endpoint():
        print("\n✗ Proxy not healthy, aborting tests")
        exit(1)

    # Run automated tests
    test_simple_tool_call()
    test_no_tools_passthrough()

    # Manual tests
    test_search_limit_guard()
    test_multi_search_guard()
    test_deep_crawl_streaming()

    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check proxy logs: tail -f robaiproxy/proxy.log")
    print("2. Test via OpenWebUI with various queries")
    print("3. Monitor for guard triggers and tool executions")
    print("\nExpected performance improvement:")
    print("  - 4 network hops → 2 network hops")
    print("  - ~200-400ms faster tool execution")
    print("  - Transparent to OpenWebUI")
