# requestProxy.py
# FastAPI proxy server - routes requests between passthrough and research modes

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
import httpx
import re
import asyncio
from asyncio import Semaphore
from typing import Optional
from contextlib import asynccontextmanager

# Load configuration from environment variables
from config import config, logger

# Import research functions from researchAgent
from researchAgent import research_mode_stream, research_mode_sync

# Import connection manager
from connectionManager import connection_manager

# Import power manager
from powerManager import power_manager

# Global model state
current_model_name: Optional[str] = None
current_model_data: Optional[dict] = None  # Full model data from vLLM
model_fetch_task: Optional[asyncio.Task] = None

# Research concurrency limits (from config)
research_semaphore = Semaphore(config.MAX_STANDARD_RESEARCH)  # Max standard research requests
deep_research_semaphore = Semaphore(config.MAX_DEEP_RESEARCH)  # Max deep research requests

# vLLM endpoint paths (for routing decisions)
VLLM_ENDPOINTS = {
    "openapi.json", "docs", "docs/oauth2-redirect", "redoc",
    "load", "ping", "tokenize", "detokenize", "version",
    "v1/responses", "v1/completions", "v1/embeddings",
    "pooling", "classify", "score", "v1/score",
    "v1/audio/transcriptions", "v1/audio/translations",
    "rerank", "v1/rerank", "v2/rerank",
    "scale_elastic_ep", "is_scaling_elastic_ep",
    "invocations", "metrics"
}


def is_vllm_endpoint(path: str) -> bool:
    """
    Check if a path matches a vLLM endpoint.

    Handles both exact matches and parameterized routes.

    Args:
        path (str): The request path (without leading slash)

    Returns:
        bool: True if path is a vLLM endpoint, False otherwise

    Examples:
        is_vllm_endpoint("v1/completions") → True
        is_vllm_endpoint("v1/responses/abc123") → True
        is_vllm_endpoint("api/v1/crawl") → False
        is_vllm_endpoint("openapi.json") → True
    """
    # Exact match
    if path in VLLM_ENDPOINTS:
        return True

    # Parameterized routes: /v1/responses/{response_id}
    if path.startswith("v1/responses/"):
        return True

    return False


async def check_vllm_container_running() -> bool:
    """
    Check if a vLLM Docker container is running.

    Returns:
        bool: True if a container with name starting with 'vllm' is running
    """
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            container_names = result.stdout.strip().split('\n')
            for name in container_names:
                if name.lower().startswith('vllm'):
                    logger.debug(f"Found running vLLM container: {name}")
                    return True
        return False
    except Exception as e:
        logger.warning(f"Failed to check Docker containers: {str(e)}")
        return False


async def fetch_model_name() -> Optional[str]:
    """
    Fetch the active model name from vLLM /v1/models endpoint.

    If the 5-second timeout is hit but a vLLM container is running,
    use the previously saved model name if available.

    Returns:
        str or None: The model ID/name if successful, None if failed
    """
    global current_model_name, current_model_data

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.VLLM_BACKEND_URL}/v1/models")
            response.raise_for_status()
            data = response.json()

            # Extract model name and cache full response
            # Response format: {"object": "list", "data": [{"id": "model-name", ...}]}
            if "data" in data and len(data["data"]) > 0:
                model_name = data["data"][0].get("id")
                if model_name:
                    # Cache the full model data
                    current_model_data = data
                    logger.info(f"Fetched model name: {model_name}")
                    return model_name

            logger.warning(f"Models endpoint returned unexpected format: {data}")
            return None
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:
        # Timeout occurred - check if vLLM container is running
        logger.warning(f"Timeout fetching model name (vLLM likely busy): {str(e)}")

        # Check if vLLM container is running
        is_running = await check_vllm_container_running()

        if is_running and current_model_name:
            logger.info(f"vLLM container running & model name cached, using: {current_model_name}")
            return current_model_name
        elif is_running:
            logger.warning("vLLM container running but no cached model name, will retry")
            return None
        else:
            logger.error("No vLLM container found running")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch model name: {str(e)}")
        return None


async def model_name_manager():
    """
    Background task that manages model name state with automatic retry and recovery.

    Logic:
    - On startup: Ping /v1/models every 2 seconds until successful
    - Once fetched: Stop pinging (save resources)
    - On vLLM error: Clear model name and resume pinging
    - Resilient to Docker container restarts
    """
    global current_model_name

    logger.info("Starting model name manager...")

    while True:
        if current_model_name is None:
            # Model name not set - keep trying every 2 seconds
            logger.debug("Model name not set, attempting to fetch...")
            model_name = await fetch_model_name()

            if model_name:
                current_model_name = model_name
                logger.info(f"Model name set to: {current_model_name}")
            else:
                # Wait 2 seconds before retry
                await asyncio.sleep(2)
        else:
            await asyncio.sleep(10)


def trigger_model_refresh():
    global current_model_name
    if current_model_name:
        logger.warning(f"Clearing model name '{current_model_name}' and triggering refresh")
        current_model_name = None


def get_model_name() -> str:
    return current_model_name or "unknown-model"


def is_model_available() -> bool:
    return current_model_name is not None


async def check_research_health() -> tuple[bool, str]:
    """
    Check if vLLM is healthy for research requests.

    Verifies both model availability and Docker container status.

    Returns:
        tuple[bool, str]: (is_healthy, error_message)
            - is_healthy: True if both checks pass, False otherwise
            - error_message: Description of failure reason (empty string if healthy)
    """
    # Check 1: Model name populated
    if not is_model_available():
        return False, "vLLM model not loaded"

    # Check 2: Docker container running
    if not await check_vllm_container_running():
        return False, "vLLM container not running"

    return True, ""


def get_queue_status(is_deep: bool) -> str:
    """
    Get detailed queue status message showing slot usage.

    Args:
        is_deep (bool): True for deep research queue, False for standard

    Returns:
        str: Human-readable queue status message
    """
    if is_deep:
        # Deep research has only 1 slot
        deep_available = deep_research_semaphore._value
        deep_used = 1 - deep_available
        status = f"Deep research slot {'occupied (1/1)' if deep_used == 1 else 'available (0/1)'}"
    else:
        # Standard research has 3 slots
        available = research_semaphore._value
        used = 3 - available
        status = f"Standard research queue ({used}/3 slots used)"

    return status


async def wait_for_model(max_wait_seconds: int = 30) -> bool:
    """
    Wait for the model to become available, up to max_wait_seconds.

    Args:
        max_wait_seconds (int): Maximum seconds to wait. Default: 30

    Returns:
        bool: True if model became available, False if timeout
    """
    elapsed = 0
    while elapsed < max_wait_seconds:
        if is_model_available():
            return True
        await asyncio.sleep(1)
        elapsed += 1
    return False


async def model_loading_response_stream(model_name: str = "unknown-model"):
    """
    Generate a streaming response informing user that model is loading.

    Yields:
        bytes: SSE chunks with loading message
    """
    from researchAgent import create_sse_chunk

    message = "⏳ The model is currently loading. Please stand by...\n\n"
    message += "The vLLM backend is starting up. This usually takes 30-60 seconds.\n\n"
    message += "Please try your request again in a moment when the model is available."

    yield create_sse_chunk(message, model=model_name).encode()
    yield create_sse_chunk("", finish_reason="stop", model=model_name).encode()
    yield b"data: [DONE]\n\n"


def model_loading_response_sync(model_name: str = "unknown-model") -> dict:
    """
    Generate a sync response informing user that model is loading.

    Returns:
        dict: OpenAI-compatible completion response
    """
    message = "⏳ The model is currently loading. Please stand by...\n\n"
    message += "The vLLM backend is starting up. This usually takes 30-60 seconds.\n\n"
    message += "Please try your request again in a moment when the model is available."

    return {
        "id": "chatcmpl-loading",
        "object": "chat.completion",
        "created": 0,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": message
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager - starts background tasks on startup.
    """
    global model_fetch_task

    # Start model name manager background task
    model_fetch_task = asyncio.create_task(model_name_manager())
    logger.info("Model name manager started")

    # Start power manager
    # await power_manager.start()
    # logger.info("Power manager started")

    yield

    # Cleanup on shutdown
    # Stop power manager first
    # await power_manager.stop()
    # logger.info("Power manager stopped")

    # Stop model name manager
    if model_fetch_task:
        model_fetch_task.cancel()
        try:
            await model_fetch_task
        except asyncio.CancelledError:
            pass
    logger.info("Model name manager stopped")


app = FastAPI(lifespan=lifespan, openapi_url=None)  # Disable auto-generated OpenAPI


# Research keyword configuration
RESEARCH_KEYWORDS = {
    2: [],  # Standard (default, no modifier needed)
    4: ['thoroughly', 'carefully', 'all', 'comprehensively', 'comprehensive', 'deep', 'deeply', 'detailed', 'extensive', 'extensively']  # Deep/Comprehensive research
}


def detect_research_mode(messages: list) -> tuple[bool, int]:
    """
    Detect if research mode should be activated and determine iteration count.

    Checks the last user message for "research" keyword with optional modifiers:
    - "research" (no modifier) → 2 iterations (default)
    - "research thoroughly/carefully/all/comprehensively/deep/deeply/detailed/extensive/extensively" → 4 iterations

    Args:
        messages (list): List of message dict objects with 'role' and 'content' keys

    Returns:
        tuple[bool, int]: (is_research_mode, iteration_count)
            - is_research_mode: True if "research" keyword found
            - iteration_count: Number of iterations (2 or 4)

    Examples:
        "research python" → (True, 2)
        "research deep kubernetes" → (True, 4)
        "hello world" → (False, 2)
    """
    # Find the last user message
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")

            # Handle both string and list (multimodal) content formats
            if isinstance(content, list):
                # Extract text from multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)

            content = content.strip().lower()

            # Check if message starts with "research"
            if not content.startswith("research"):
                return (False, 2)

            # Found "research", now check for modifiers
            # Extract the word immediately after "research"
            parts = content.split(maxsplit=2)  # Split into at most 3 parts

            if len(parts) == 1:
                # Just "research" with nothing after
                return (True, 2)

            # Check the second word against our keyword lists
            modifier = parts[1]

            # Check for deep/comprehensive keywords (4 iterations)
            if modifier in RESEARCH_KEYWORDS[4]:
                return (True, 4)

            # No recognized modifier, default to 2 iterations
            return (True, 2)

    return (False, 2)


def is_multimodal(messages: list) -> bool:
    """
    Check if any message contains multimodal content (images, etc.).

    Multimodal content is represented as a list instead of a string in the content field.
    Since research mode doesn't support images, we automatically passthrough these requests.

    Args:
        messages (list): List of message dict objects with 'role' and 'content' keys

    Returns:
        bool: True if any message has list-based (multimodal) content, False otherwise

    Used by:
        - autonomous_chat(): Determines if request should bypass research mode
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            return True
    return False




async def passthrough_stream(body: dict, request: Request = None):
    """
    Pure transparent pass-through for streaming.
    Client sees raw vLLM stream as if proxy doesn't exist.

    Args:
        body: Request body to forward to vLLM
        request: FastAPI Request object (kept for signature compatibility)
    """
    # Generate unique request ID for logging
    request_id = id(request) if request else "unknown"

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{config.VLLM_BACKEND_URL}/v1/chat/completions",
                json=body,
                timeout=300.0
            ) as response:
                # Check if response is an error
                if response.status_code >= 400:
                    error_content = await response.aread()
                    logger.error(f"[ReqID: {request_id}] Backend error in passthrough_stream: {response.status_code} - {error_content.decode()}")
                    # Trigger model name refresh on backend errors
                    trigger_model_refresh()

                # Stream all chunks transparently
                async for chunk in response.aiter_bytes():
                    yield chunk
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as e:
        # Connection errors indicate vLLM might be down - trigger refresh
        logger.error(f"[ReqID: {request_id}] Connection error in passthrough_stream: {str(e)}")
        trigger_model_refresh()
        # Import SSE chunk creator from research agent for error formatting
        from researchAgent import create_sse_chunk
        error_chunk = create_sse_chunk(f"vLLM backend unavailable: {str(e)}", finish_reason="error")
        yield error_chunk.encode()
        yield b"data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"[ReqID: {request_id}] Exception in passthrough_stream: {str(e)}")
        # Import SSE chunk creator from research agent for error formatting
        from researchAgent import create_sse_chunk
        error_chunk = create_sse_chunk(f"Error: {str(e)}", finish_reason="error")
        yield error_chunk.encode()
        yield b"data: [DONE]\n\n"


async def passthrough_sync(body: dict):
    """
    Pure transparent pass-through for non-streaming.
    Client sees raw vLLM response as if proxy doesn't exist.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.VLLM_BACKEND_URL}/v1/chat/completions",
                json=body,
                timeout=300.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend HTTP error in passthrough_sync: {e.response.status_code} - {e.response.text}")
        # Trigger model name refresh on backend errors
        trigger_model_refresh()
        return {
            "error": {
                "message": f"Backend error: {e.response.text}",
                "type": "backend_error",
                "code": e.response.status_code
            }
        }
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as e:
        # Connection errors indicate vLLM might be down - trigger refresh
        logger.error(f"Connection error in passthrough_sync: {str(e)}")
        trigger_model_refresh()
        return {
            "error": {
                "message": f"vLLM backend unavailable: {str(e)}",
                "type": "connection_error",
                "code": 502
            }
        }
    except Exception as e:
        logger.error(f"Exception in passthrough_sync: {str(e)}")
        return {
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": 500
            }
        }


async def research_with_queue_management(body: dict, model: str, iterations: int, request, semaphore, is_deep: bool):
    """
    Wrapper for research mode with queue management and health checks.

    Handles queue status messages when queue is full and performs health
    checks before queueing. Once semaphore is acquired, proceeds with research.

    Args:
        body (dict): Request body to forward to research mode
        model (str): Model name for responses
        iterations (int): Number of research iterations (2 or 4)
        request: FastAPI Request object
        semaphore: Asyncio Semaphore to control concurrency
        is_deep (bool): True for deep research, False for standard

    Yields:
        bytes: SSE-formatted chunks (queue messages, research results, or errors)
    """
    from researchAgent import create_sse_chunk

    # Check if queue is full
    is_queue_full = semaphore.locked()

    if is_queue_full:
        # Health check ONLY when queue is full (before sending queue message)
        is_healthy, error_msg = await check_research_health()

        if not is_healthy:
            # vLLM unhealthy - send 503 error and abort
            yield create_sse_chunk(
                f"❌ Service unavailable: {error_msg}. Cannot queue research request.",
                finish_reason="error",
                model=model
            ).encode()
            yield b"data: [DONE]\n\n"
            return

        # Send detailed queue status message
        queue_status = get_queue_status(is_deep)
        yield create_sse_chunk(
            f"⏳ Research queue is full. {queue_status}. Your request is pending...\n\n",
            model=model
        ).encode()

    # Acquire semaphore (waits if queue is full, immediate if available)
    async with semaphore:
        # Proceed with research streaming
        async for chunk in research_mode_stream(body, model_name=model, max_iterations=iterations, request=request):
            yield chunk


@app.post("/v1/chat/completions")
async def autonomous_chat(request: Request):
    """
    Main OpenAI-compatible chat completions endpoint with research mode routing.

    This is the primary endpoint that receives chat completion requests. It analyzes
    the last user message and routes to either comprehensive research mode (if message
    starts with "research") or transparent passthrough mode (normal LLM requests).

    Model Availability Handling:
        - If model is not yet loaded, waits up to 30 seconds
        - If model doesn't load in time, returns friendly "loading" message
        - User can retry their request once model is ready

    Research Mode Flow:
        1. Detects "research" keyword in last user message
        2. Determines iteration count (2 for standard, 4 for deep research)
        3. Calls Serper API for 10 initial web search results
        4. Performs 2-4 iterations of:
           - LLM-generated search → search_memory (3 results for standard, 6 for deep)
           - URL generation → crawl 3 URLs
           - Generate 1 distinct Serper query → 1 web search (5 results)
        5. Builds massive accumulated context (potentially 100K+ tokens)
        6. Generates final answer with all research context
        7. Wraps research progress in  (^)(tag

    Passthrough Mode Flow:
        1. Forwards request directly to vLLM
        2. Streams/returns response transparently
        3. No modifications to request or response

    Args:
        request (Request): FastAPI Request object containing JSON body with:
            - messages (list): Chat messages
            - stream (bool): Whether to stream response
            - max_tokens (int): Max tokens to generate
            - temperature (float): Sampling temperature
            - stream_options (dict): Streaming options like include_usage

    Returns:
        StreamingResponse or dict: Either streaming SSE response or JSON completion

    Used by:
        - External clients as OpenAI-compatible API endpoint
    """
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Check if model is available, wait up to 30 seconds if not
    if not is_model_available():
        logger.warning("Model not yet available, waiting up to 30 seconds...")
        model_ready = await wait_for_model(max_wait_seconds=30)

        if not model_ready:
            logger.warning("Model still not available after 30 seconds, returning loading message")
            model_name = get_model_name()
            if stream:
                return StreamingResponse(
                    model_loading_response_stream(model_name),
                    media_type="text/event-stream"
                )
            else:
                return model_loading_response_sync(model_name)
        else:
            logger.info("Model became available, proceeding with request")

    # Check for multimodal content first - always passthrough if present
    if is_multimodal(messages):
        logger.debug("Multimodal content detected - passing through to vLLM")
        if stream:
            return StreamingResponse(
                passthrough_stream(body, request),
                media_type="text/event-stream"
            )
        else:
            return await passthrough_sync(body)

    # Detect research mode and iteration count
    is_research, iterations = detect_research_mode(messages)

    # Debug logging
    logger.info(f"Request received. Research mode: {is_research}, Iterations: {iterations}")

    # Extract Authorization header and log first 6 characters of bearer token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
        masked_token = token[:6] + "..."
        logger.info(f"Bearer token received: {masked_token}")

    # Add connection tracking (excluding excluded endpoints)
    path = request.url.path
    connection_id = ""
    if path not in connection_manager.excluded_endpoints:
        connection_id = await connection_manager.add_connection(request, path)
        logger.debug(f"Connection tracking started: {connection_id} for {path}")

    try:
        if is_research:
            # Get current model name for research mode
            model = get_model_name()

            # Determine which semaphore to use based on iteration count
            semaphore = deep_research_semaphore if iterations == 4 else research_semaphore
            is_deep = iterations == 4

            if stream:
                # Streaming: use queue management wrapper
                return StreamingResponse(
                    research_with_queue_management(body, model, iterations, request, semaphore, is_deep),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming: wait silently for semaphore, no queue messages
                async with semaphore:
                    return await research_mode_sync(body, model_name=model, max_iterations=iterations)
        else:
            # Pure pass-through - proxy is invisible
            if stream:
                return StreamingResponse(
                    passthrough_stream(body, request),
                    media_type="text/event-stream"
                )
            else:
                return await passthrough_sync(body)
    finally:
        # Remove connection tracking when response is complete
        if connection_id:
            await connection_manager.remove_connection(connection_id)
            logger.debug(f"Connection tracking ended: {connection_id}")

@app.get("/health")
async def health():
    """
    Comprehensive health check endpoint.

    Checks health of all related services:
    - vllm-qwen3: vLLM inference server
    - kg-service: Knowledge graph processing
    - mcprag-server: MCP RAG server
    - crawl4ai: Web crawling service
    - neo4j-kg: Neo4j graph database
    - open-webui: Open WebUI frontend
    """
    import subprocess
    from datetime import datetime

    health_status = {
        "status": "healthy",
        "service": "request-proxy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }

    # Helper function to check if Docker container is running
    def check_container(name: str) -> dict:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", name],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                return {
                    "status": "healthy" if status == "running" else "unhealthy",
                    "container_status": status,
                    "available": True
                }
            else:
                return {
                    "status": "unhealthy",
                    "container_status": "not_found",
                    "available": False
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "container_status": "error",
                "available": False,
                "error": str(e)
            }

    # Helper function to check HTTP health endpoint
    async def check_http_health(name: str, url: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "http_status": response.status_code,
                        "available": True
                    }
                else:
                    return {
                        "status": "degraded",
                        "http_status": response.status_code,
                        "available": True
                    }
        except httpx.ConnectError:
            return {
                "status": "unhealthy",
                "available": False,
                "error": "connection_refused"
            }
        except httpx.TimeoutException:
            return {
                "status": "unhealthy",
                "available": False,
                "error": "timeout"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e)
            }

    # Check vLLM (use existing model availability check + Docker)
    vllm_status = {
        "model_loaded": is_model_available(),
        "model_name": current_model_name
    }
    vllm_container = check_container("robai-qwen3")  # Updated container name
    vllm_status.update(vllm_container)
    health_status["services"]["vllm-qwen3"] = vllm_status

    # Check kg-service (Docker + HTTP health) - name unchanged
    kg_container = check_container("robaikg")  # Updated container name
    kg_http = await check_http_health("kg-service", "http://localhost:8088/health")
    health_status["services"]["kg-service"] = {**kg_container, **kg_http}

    # Check mcprag-server (Docker + HTTP health)
    mcprag_container = check_container("robaimcp")  # Updated container name
    mcprag_http = await check_http_health("mcprag-server", "http://localhost:8081/health")  # Updated port
    health_status["services"]["mcprag-server"] = {**mcprag_container, **mcprag_http}

    # Check crawl4ai (Docker only - no health endpoint)
    health_status["services"]["crawl4ai"] = check_container("robaicrawler")  # Updated container name

    # Check neo4j-kg (Docker + HTTP)
    neo4j_container = check_container("robaineo4j")  # Updated container name
    neo4j_http = await check_http_health("neo4j-kg", "http://localhost:7474")
    health_status["services"]["neo4j-kg"] = {**neo4j_container, **neo4j_http}

    # Check open-webui (Docker + HTTP)
    webui_container = check_container("open-webui")
    webui_http = await check_http_health("open-webui", "http://localhost:80/health")
    health_status["services"]["open-webui"] = {**webui_container, **webui_http}

    # Determine overall status based on critical services
    critical_services = ["vllm-qwen3", "kg-service", "neo4j-kg"]
    unhealthy_critical = [
        svc for svc in critical_services
        if health_status["services"][svc].get("status") == "unhealthy"
    ]

    if unhealthy_critical:
        health_status["status"] = "unhealthy"
        health_status["unhealthy_services"] = unhealthy_critical
    elif any(svc.get("status") == "degraded" for svc in health_status["services"].values()):
        health_status["status"] = "degraded"

    return health_status


@app.get("/v1/models")
async def list_models():
    """
    Return available models using cached model data from proxy.

    Returns cached full model response from vLLM /v1/models endpoint.
    If model not yet loaded, returns empty list.
    """
    # If model data is cached, return it directly
    if current_model_data:
        return current_model_data

    # If model is unknown/loading, return empty list
    return {
        "object": "list",
        "data": []
    }


@app.get("/openapi.json")
async def get_openapi_schema():
    """
    Proxy OpenAPI schema from robairagapi (port 8081).

    This exposes the RAG API tools to external clients like OpenWebUI.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8081/openapi.json", timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch OpenAPI schema from port 8081: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch RAG API schema: {str(e)}")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all_proxy(request: Request, path: str):
    """
    Intelligent catch-all proxy that routes requests based on endpoint type:
    - vLLM endpoints → forward to vLLM backend
    - All other endpoints → forward to robairagapi (port 8081)

    Already-handled endpoints (health, v1/chat/completions, v1/models) are skipped.
    """
    # Skip endpoints we've already defined - these are handled by their specific routes
    if path in ["health", "v1/chat/completions", "v1/models", "openapi.json"]:
        # Return 404 since these should have been caught by their specific handlers
        raise HTTPException(status_code=404, detail="Route handled by specific endpoint")

    # Route based on endpoint type
    if is_vllm_endpoint(path):
        # vLLM endpoint - forward to vLLM backend
        target_url = f"{config.VLLM_BACKEND_URL}/{path}"
        logger.debug(f"Routing vLLM endpoint {path} to {target_url}")
    else:
        # Non-vLLM endpoint - forward to robairagapi (port 8081)
        target_url = f"http://localhost:8081/{path}"
        logger.debug(f"Routing RAG API endpoint {path} to {target_url}")

    # Get query parameters
    query_params = dict(request.query_params)

    try:
        async with httpx.AsyncClient() as client:
            # Handle different request methods
            if request.method in ["GET", "HEAD", "OPTIONS"]:
                response = await client.request(
                    request.method,
                    target_url,
                    params=query_params,
                    headers=dict(request.headers),
                    timeout=30.0
                )
            else:  # POST, PUT, DELETE, PATCH
                body = await request.body()
                response = await client.request(
                    request.method,
                    target_url,
                    params=query_params,
                    headers=dict(request.headers),
                    content=body,
                    timeout=30.0
                )

            # Return the response with same status code and headers
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except httpx.RequestError as e:
        logger.error(f"Request error in catch_all_proxy for {path}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Backend connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Exception in catch_all_proxy for {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8079)
