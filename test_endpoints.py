#!/usr/bin/env python3
"""
Test all vLLM endpoints through the research proxy to identify routing issues.
"""
import httpx
import asyncio
import json
from typing import Dict, List, Tuple

PROXY_URL = "http://192.168.10.50:8079"
VLLM_URL = "http://localhost:8078"

# Test cases for each endpoint
ENDPOINTS = [
    # GET endpoints
    ("GET", "/openapi.json", None, None),
    ("GET", "/docs", None, None),
    ("GET", "/docs/oauth2-redirect", None, None),
    ("GET", "/redoc", None, None),
    ("GET", "/health", None, None),
    ("GET", "/load", None, None),
    ("GET", "/v1/models", None, None),
    ("GET", "/version", None, None),
    ("GET", "/metrics", None, None),

    # POST endpoints with minimal payloads
    ("POST", "/ping", {}, None),
    ("POST", "/tokenize", {"prompt": "Hello world"}, None),
    ("POST", "/detokenize", {"tokens": [123, 456]}, None),

    # Chat completions (handled specially by proxy)
    ("POST", "/v1/chat/completions", {
        "model": "Qwen3-30B",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }, "special"),

    # Regular completions
    ("POST", "/v1/completions", {
        "model": "Qwen3-30B",
        "prompt": "Hello",
        "max_tokens": 10
    }, None),

    # Embeddings
    ("POST", "/v1/embeddings", {
        "model": "Qwen3-30B",
        "input": "Hello world"
    }, None),

    # Test non-existent endpoints for error handling
    ("GET", "/nonexistent/endpoint", None, "error_test"),
    ("POST", "/v1/fake/route", {"test": "data"}, "error_test"),
]

async def test_endpoint(client: httpx.AsyncClient, method: str, path: str,
                       payload: dict, notes: str) -> Tuple[str, int, str, str]:
    """Test a single endpoint and return results."""
    url = f"{PROXY_URL}{path}"

    try:
        if method == "GET":
            response = await client.get(url, timeout=10.0)
        else:  # POST
            response = await client.post(url, json=payload, timeout=10.0)

        status = response.status_code

        # Try to parse response
        try:
            content = response.json()
            content_str = json.dumps(content, indent=2)[:200]
        except:
            content_str = response.text[:200]

        result = "✅ SUCCESS" if 200 <= status < 300 else f"❌ FAILED ({status})"

        return (f"{method} {path}", status, result, content_str)

    except Exception as e:
        return (f"{method} {path}", 0, f"❌ ERROR", str(e)[:200])

async def test_direct_vllm(client: httpx.AsyncClient, method: str, path: str,
                          payload: dict) -> Tuple[str, int, str]:
    """Test endpoint directly against vLLM for comparison."""
    url = f"{VLLM_URL}{path}"

    try:
        if method == "GET":
            response = await client.get(url, timeout=10.0)
        else:
            response = await client.post(url, json=payload, timeout=10.0)

        status = response.status_code
        result = "✅ OK" if 200 <= status < 300 else f"❌ {status}"
        return (path, status, result)

    except Exception as e:
        return (path, 0, f"❌ ERROR: {str(e)[:50]}")

async def main():
    print("=" * 80)
    print("TESTING RESEARCH PROXY ENDPOINTS")
    print("=" * 80)
    print(f"\nProxy URL: {PROXY_URL}")
    print(f"Backend vLLM URL: {VLLM_URL}\n")

    async with httpx.AsyncClient() as client:
        # Test proxy health first
        print("🔍 Testing proxy health endpoint...")
        try:
            response = await client.get(f"{PROXY_URL}/health", timeout=5.0)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}\n")
        except Exception as e:
            print(f"   ❌ ERROR: {e}\n")
            print("⚠️  Proxy may not be running!\n")
            return

        # Test all endpoints through proxy
        print("\n" + "=" * 80)
        print("TESTING ENDPOINTS THROUGH PROXY")
        print("=" * 80 + "\n")

        results = []
        for method, path, payload, notes in ENDPOINTS:
            result = await test_endpoint(client, method, path, payload, notes)
            results.append(result)

            endpoint, status, result_str, content = result
            print(f"{result_str:20} | {endpoint:40} | Status: {status}")
            if status >= 400:
                print(f"   Response: {content}")
            print()

            # Small delay to avoid overwhelming server
            await asyncio.sleep(0.2)

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80 + "\n")

        success_count = sum(1 for _, status, _, _ in results if 200 <= status < 300)
        total_count = len(results)

        print(f"✅ Successful: {success_count}/{total_count}")
        print(f"❌ Failed: {total_count - success_count}/{total_count}\n")

        # Show failed endpoints
        failed = [(endpoint, status, content) for endpoint, status, _, content in results
                  if not (200 <= status < 300)]

        if failed:
            print("Failed Endpoints:")
            for endpoint, status, content in failed:
                print(f"  • {endpoint} (Status: {status})")
                print(f"    {content[:100]}")

        # Test catch-all routing
        print("\n" + "=" * 80)
        print("TESTING CATCH-ALL PROXY ROUTING")
        print("=" * 80 + "\n")

        # Test a few endpoints that should be caught by catch_all_proxy
        test_paths = ["/openapi.json", "/version", "/ping"]

        for path in test_paths:
            print(f"Testing catch-all for: {path}")

            # Test direct vLLM
            vllm_result = await test_direct_vllm(client, "GET", path, None)
            print(f"  Direct vLLM: {vllm_result[2]} (Status: {vllm_result[1]})")

            # Test through proxy
            proxy_url = f"{PROXY_URL}{path}"
            try:
                response = await client.get(proxy_url, timeout=5.0)
                print(f"  Through Proxy: ✅ OK (Status: {response.status_code})")
            except Exception as e:
                print(f"  Through Proxy: ❌ ERROR - {str(e)[:50]}")

            print()

async def test_concurrent_research():
    """Test concurrent research requests to verify queue management."""
    print("\n" + "=" * 80)
    print("TESTING CONCURRENT RESEARCH QUEUE MANAGEMENT")
    print("=" * 80 + "\n")

    async with httpx.AsyncClient() as client:
        # Create 4 concurrent research requests (streaming)
        tasks = []

        # 2 standard research requests
        for i in range(2):
            tasks.append(test_streaming_research(client, f"research Python async/await", i + 1, "standard"))

        # 1 deep research request
        tasks.append(test_streaming_research(client, f"research thoroughly Kubernetes networking", 3, "deep"))

        # 1 more standard research request
        tasks.append(test_streaming_research(client, f"research Docker containers", 4, "standard"))

        # Run all 4 concurrently
        print("🚀 Launching 4 concurrent research requests...")
        print("   - Request 1: Standard research (Python async/await)")
        print("   - Request 2: Standard research (Python async/await)")
        print("   - Request 3: Deep research (Kubernetes networking)")
        print("   - Request 4: Standard research (Docker containers)\n")
        print("Expected behavior:")
        print("   - Standard research: Max 3 concurrent (all 3 standard should start)")
        print("   - Deep research: Max 1 concurrent (should queue if another running)\n")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Print results
        print("\n" + "=" * 80)
        print("CONCURRENT RESEARCH RESULTS")
        print("=" * 80 + "\n")

        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"Request {i}: ❌ ERROR - {str(result)[:100]}")
            else:
                status, queue_msg, response_preview = result
                print(f"Request {i}:")
                print(f"  Status: {status}")
                if queue_msg:
                    print(f"  Queue Message: {queue_msg}")
                print(f"  Response Preview: {response_preview[:200]}...")
                print()


async def test_streaming_research(client: httpx.AsyncClient, query: str, request_id: int, research_type: str):
    """Test a single streaming research request and capture queue messages."""
    url = f"{PROXY_URL}/v1/chat/completions"

    payload = {
        "model": "Qwen3-30B",
        "messages": [{"role": "user", "content": query}],
        "stream": True,
        "max_tokens": 500
    }

    try:
        queue_message = None
        response_chunks = []
        chunk_count = 0

        print(f"[Req {request_id}] Starting {research_type} research request...", flush=True)

        async with client.stream("POST", url, json=payload, timeout=300.0) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                return (f"ERROR {response.status_code}", None, error_text.decode()[:200])

            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices") and chunk["choices"][0].get("delta"):
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                response_chunks.append(content)
                                chunk_count += 1

                                # Check for queue message
                                if "queue is full" in content.lower() or "pending" in content.lower():
                                    queue_message = content.strip()
                                    print(f"[Req {request_id}] 📥 {queue_message}", flush=True)

                                # Print first chunk to show progress
                                if chunk_count == 1 and not queue_message:
                                    print(f"[Req {request_id}] ✅ Started processing (no queue)", flush=True)

                    except json.JSONDecodeError:
                        continue

        full_response = "".join(response_chunks)
        print(f"[Req {request_id}] ✅ Completed ({chunk_count} chunks, {len(full_response)} chars)", flush=True)

        return ("SUCCESS", queue_message, full_response)

    except Exception as e:
        print(f"[Req {request_id}] ❌ ERROR: {str(e)[:100]}", flush=True)
        return ("ERROR", None, str(e)[:200])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--concurrent":
        # Test concurrent research only
        asyncio.run(test_concurrent_research())
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Test both endpoints and concurrent research
        async def run_all():
            await main()
            await test_concurrent_research()
        asyncio.run(run_all())
    else:
        # Default: test endpoints only
        asyncio.run(main())
