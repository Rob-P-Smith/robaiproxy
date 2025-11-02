# requestProxy.py
# FastAPI proxy server - routes requests between passthrough and research modes

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
import httpx
import re
import json
import asyncio
from asyncio import Semaphore
from typing import Optional, Dict, Tuple
from contextlib import asynccontextmanager
import hashlib
import time

# Load configuration from environment variables
from config import config, logger

# Import research and RAG functions from robaimultiturn library
from robaimultiturn import (
    research_stream,
    research_sync,
    augment_with_rag,
    create_pondering_stream,
    wrap_stream_with_rag_indicator,
    check_if_first_user_message,
    has_research_keyword,
    create_sse_chunk
)

# Import connection manager
from connectionManager import connection_manager

# Import power manager
from powerManager import power_manager

# Import new metadata-driven modules
from sessionManager import session_manager
from rateLimiter import rate_limiter
from analytics import analytics_tracker

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
            # Clear cached data since response is invalid
            current_model_data = None
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
            # Clear cached data since vLLM is not running
            current_model_data = None
            return None
    except Exception as e:
        logger.error(f"Failed to fetch model name: {str(e)}")
        # Clear cached data on error
        current_model_data = None
        return None


async def model_name_manager():
    """
    Background task that manages model name state with automatic retry and recovery.

    Logic:
    - On startup: Ping /v1/models every 2 seconds until successful
    - Once fetched: Verify every 30 seconds that vLLM is still available
    - On vLLM error: Clear model name and resume pinging
    - Resilient to Docker container restarts
    """
    global current_model_name, current_model_data

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
            # Model is set - verify it's still available every 30 seconds
            await asyncio.sleep(30)

            # Quick health check: verify vLLM container is still running
            is_running = await check_vllm_container_running()

            if not is_running:
                logger.warning(f"vLLM container no longer running, clearing model state")
                current_model_name = None
                current_model_data = None


def trigger_model_refresh():
    global current_model_name, current_model_data
    if current_model_name:
        logger.warning(f"Clearing model name '{current_model_name}' and triggering refresh")
        current_model_name = None
        current_model_data = None


def get_model_name() -> str:
    return current_model_name or "unknown-model"


def is_model_available() -> bool:
    return current_model_name is not None


async def count_request_tokens(messages: list) -> Optional[int]:
    """
    Count tokens in a chat completion request using vLLM's tokenizer.

    Concatenates all message content and calls vLLM's /tokenize endpoint
    to get accurate token count.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        int: Token count, or None if tokenization fails
    """
    if not config.PRECOUNT_PROMPT_TOKENS:
        return None

    try:
        # Concatenate all message content
        full_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle string content
            if isinstance(content, str):
                full_text += f"{role}: {content}\n"
            # Handle multimodal content (extract text only)
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                full_text += f"{role}: {' '.join(text_parts)}\n"

        # Call vLLM tokenize endpoint
        async with httpx.AsyncClient(timeout=config.TOKENIZE_TIMEOUT) as client:
            response = await client.post(
                f"{config.VLLM_BACKEND_URL}/tokenize",
                json={"prompt": full_text}
            )
            response.raise_for_status()
            data = response.json()

            # vLLM returns: {"tokens": [123, 456, ...], "count": N}
            # Try 'count' first, fall back to len(tokens)
            token_count = data.get("count", len(data.get("tokens", [])))
            logger.debug(f"Tokenization successful: {token_count} tokens")
            return token_count

    except httpx.TimeoutException:
        logger.warning(f"Tokenization timed out after {config.TOKENIZE_TIMEOUT}s - allowing request through (fail-open)")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Tokenization failed with HTTP {e.response.status_code} - allowing request through (fail-open)")
        return None
    except Exception as e:
        logger.warning(f"Tokenization failed: {str(e)} - allowing request through (fail-open)")
        return None


async def create_token_limit_error_stream(token_count: int, model_name: str):
    """
    Generate streaming error response for token limit exceeded.

    Yields SSE-formatted chunks for streaming responses.
    """
    percentage = round((token_count / config.MAX_MODEL_CONTEXT) * 100, 1)

    error_message = (
        f"❌ **Request Rejected: Token Limit Exceeded**\n\n"
        f"Your request contains **{token_count:,} tokens**, which exceeds the configured limit.\n\n"
        f"**Limits:**\n"
        f"- Your request: {token_count:,} tokens ({percentage}%)\n"
        f"- Maximum allowed: {config.MAX_REQUEST_TOKENS:,} tokens (95%)\n"
        f"- Model capacity: {config.MAX_MODEL_CONTEXT:,} tokens\n\n"
        f"**Solution:** Please reduce the context size by:\n"
        f"- Shortening your message\n"
        f"- Removing older messages from the conversation history\n"
        f"- Splitting into multiple smaller requests"
    )

    yield create_sse_chunk(error_message, model=model_name).encode()
    yield create_sse_chunk("", finish_reason="length", model=model_name).encode()
    yield b"data: [DONE]\n\n"


def create_token_limit_error_sync(token_count: int, model_name: str) -> dict:
    """
    Generate sync error response for token limit exceeded.

    Returns dict in OpenAI chat completion format.
    """
    percentage = round((token_count / config.MAX_MODEL_CONTEXT) * 100, 1)

    error_message = (
        f"❌ **Request Rejected: Token Limit Exceeded**\n\n"
        f"Your request contains **{token_count:,} tokens**, which exceeds the configured limit.\n\n"
        f"**Limits:**\n"
        f"- Your request: {token_count:,} tokens ({percentage}%)\n"
        f"- Maximum allowed: {config.MAX_REQUEST_TOKENS:,} tokens (95%)\n"
        f"- Model capacity: {config.MAX_MODEL_CONTEXT:,} tokens\n\n"
        f"**Solution:** Please reduce the context size by:\n"
        f"- Shortening your message\n"
        f"- Removing older messages from the conversation history\n"
        f"- Splitting into multiple smaller requests"
    )

    return {
        "id": "chatcmpl-token-limit",
        "object": "chat.completion",
        "created": 0,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": error_message
            },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": token_count,
            "completion_tokens": 0,
            "total_tokens": token_count
        }
    }


def validate_token_limit(token_count: int) -> bool:
    """
    Validate that token count doesn't exceed the configured limit.

    Args:
        token_count: Number of tokens in the request

    Returns:
        bool: True if within limit, False if exceeds
    """
    # Log warning if over threshold
    if token_count >= config.WARNING_TOKEN_COUNT:
        percentage = (token_count / config.MAX_MODEL_CONTEXT) * 100
        logger.warning(
            f"⚠️  Request approaching token limit: {token_count:,} tokens "
            f"({percentage:.1f}% of {config.MAX_MODEL_CONTEXT:,})"
        )

    # Reject if over limit
    if token_count > config.MAX_REQUEST_TOKENS:
        logger.error(
            f"❌ Request EXCEEDS token limit: {token_count:,} tokens "
            f"(limit: {config.MAX_REQUEST_TOKENS:,}, max: {config.MAX_MODEL_CONTEXT:,})"
        )
        return False

    # Valid - within limit
    return True


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
    await power_manager.start()
    logger.info("Power manager started")

    # Start rate limiter cleanup task
    await rate_limiter.start_cleanup_task()
    logger.info("Rate limiter cleanup task started")

    # Start analytics cleanup task
    if config.ENABLE_ANALYTICS:
        await analytics_tracker.start_cleanup_task()
        logger.info("Analytics tracker started")

    yield

    # Cleanup on shutdown
    # Stop analytics tracker
    if config.ENABLE_ANALYTICS:
        await analytics_tracker.stop_cleanup_task()
        logger.info("Analytics tracker stopped")

    # Stop rate limiter cleanup
    await rate_limiter.stop_cleanup_task()
    logger.info("Rate limiter cleanup stopped")

    # Stop power manager
    await power_manager.stop()
    logger.info("Power manager stopped")

    # Stop model name manager
    if model_fetch_task:
        model_fetch_task.cancel()
        try:
            await model_fetch_task
        except asyncio.CancelledError:
            pass
    logger.info("Model name manager stopped")


app = FastAPI(lifespan=lifespan, openapi_url=None)  # Disable auto-generated OpenAPI


async def check_admin_authorization(request: Request) -> bool:
    """
    Check if the request is authorized for admin endpoints.

    Only "Robert P Smith" is authorized to access admin endpoints.
    Checks both custom headers and metadata in request body.

    Args:
        request: FastAPI Request object

    Returns:
        True if authorized, False otherwise
    """
    # First check custom header (easiest for GET requests)
    auth_user = request.headers.get("X-User-Name", "")
    if auth_user == "Robert P Smith":
        return True

    # Check for authorization token that maps to Robert P Smith
    auth_token = request.headers.get("X-Admin-Token", "")
    # Use a secure token for admin access (this should be in config in production)
    if auth_token == "RobertPSmith-AdminAccess-2025":
        return True

    # For requests with JSON body, check metadata
    if request.headers.get("content-type") == "application/json":
        try:
            # Read body without consuming it
            body_bytes = await request.body()
            body_text = body_bytes.decode('utf-8')
            body = json.loads(body_text) if body_text else {}

            # Check metadata for user name
            metadata = body.get("metadata", {})
            user_name = metadata.get("variables", {}).get("{{USER_NAME}}", "")
            if user_name == "Robert P Smith":
                return True
        except Exception as e:
            logger.debug(f"Error checking authorization in body: {e}")

    return False


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


def extract_text_only_messages(messages: list) -> list:
    """
    Extract text-only version of messages for token counting.

    For multimodal messages (where content is a list), extracts only the text parts.
    For regular messages (where content is a string), returns as-is.

    This is used for token validation of multimodal requests - we need to validate
    that the text portion doesn't exceed context limits, even though images are
    handled separately by vLLM.

    Args:
        messages (list): List of message dict objects with 'role' and 'content' keys

    Returns:
        list: Messages with only text content (images/audio removed)

    Example:
        Input: [{"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {...}}
        ]}]

        Output: [{"role": "user", "content": "What's in this image?"}]

    Used by:
        - autonomous_chat(): Token validation for multimodal requests
    """
    text_only_messages = []

    for msg in messages:
        content = msg.get("content")

        # If content is a list (multimodal), extract only text parts
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            # Join all text parts with space
            text_content = " ".join(text_parts)
            text_only_messages.append({
                "role": msg.get("role"),
                "content": text_content
            })
        else:
            # Regular text message, keep as-is
            text_only_messages.append(msg)

    return text_only_messages




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

    # LOG FORWARDING DESTINATION
    target_url = f"{config.VLLM_BACKEND_URL}/v1/chat/completions"
    logger.debug(f"🟢 FORWARD (stream): → {target_url}")

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                target_url,
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
        error_chunk = create_sse_chunk(f"vLLM backend unavailable: {str(e)}", finish_reason="error")
        yield error_chunk.encode()
        yield b"data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"[ReqID: {request_id}] Exception in passthrough_stream: {str(e)}")
        error_chunk = create_sse_chunk(f"Error: {str(e)}", finish_reason="error")
        yield error_chunk.encode()
        yield b"data: [DONE]\n\n"


async def passthrough_sync(body: dict):
    """
    Pure transparent pass-through for non-streaming.
    Client sees raw vLLM response as if proxy doesn't exist.
    """
    # LOG FORWARDING DESTINATION
    target_url = f"{config.VLLM_BACKEND_URL}/v1/chat/completions"
    logger.debug(f"🟢 FORWARD (sync): → {target_url}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                target_url,
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
            )
            yield "data: [DONE]\n\n"
            return

        # Send detailed queue status message
        queue_status = get_queue_status(is_deep)
        yield create_sse_chunk(
            f"⏳ Research queue is full. {queue_status}. Your request is pending...\n\n",
            model=model
        )

    # Acquire semaphore (waits if queue is full, immediate if available)
    async with semaphore:
        # Proceed with research streaming - pass proxy config to robaimultiturn
        async for chunk in research_stream(
            body,
            model_name=model,
            max_iterations=iterations,
            vllm_url=config.VLLM_BACKEND_URL + "/v1",
            mcp_url=config.REST_API_URL,
            mcp_api_key=config.REST_API_KEY,
            serper_api_key=config.SERPER_API_KEY,
            request=request
        ):
            yield chunk


@app.post("/v1/chat/completions")
async def autonomous_chat(request: Request):
    """
    Main OpenAI-compatible chat completions endpoint with metadata-driven processing.

    Leverages metadata from robai-webui for:
    - Session tracking by chat_id
    - User-aware rate limiting
    - Personalized responses
    - Accurate token validation
    - Connection tracking with message_id
    - Analytics and usage tracking

    Request Flow:
    1. Extract metadata (chat_id, user_id, message_id)
    2. Session management (get/create, check first message)
    3. Connection tracking with metadata
    4. User rate limiting check
    5. Token validation using model info from metadata
    6. Model availability check
    7. Multimodal passthrough
    8. Session-aware RAG (only once per chat)
    9. Research detection with session tracking
    10. Standard passthrough with analytics

    Args:
        request: FastAPI Request object with JSON body and metadata

    Returns:
        StreamingResponse or JSONResponse
    """
    # ========================================================================
    # STEP 1: Parse request body and extract metadata
    # ========================================================================
    body = await request.json()
    metadata = body.get("metadata", {})
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract key identifiers from metadata
    chat_id = metadata.get("chat_id")
    user_id = metadata.get("user_id")
    message_id = metadata.get("message_id")
    session_id = metadata.get("session_id")
    user_variables = metadata.get("variables", {})
    user_name = user_variables.get("{{USER_NAME}}", "Unknown")

    # Structured logging with metadata context
    logger.info(
        f"📨 Chat Request | "
        f"User: {user_name} ({user_id[:8] if user_id else 'anon'}...) | "
        f"Chat: {chat_id[:8] if chat_id else 'ephemeral'}... | "
        f"Message: {message_id[:8] if message_id else 'unknown'}..."
    )

    # Log first 20 words of last user message
    if messages:
        last_msg = messages[-1].get("content", "")
        if isinstance(last_msg, str):
            words = last_msg.split()[:20]
            preview = " ".join(words) + ("..." if len(last_msg.split()) > 20 else "")
            logger.debug(f"📝 REQUEST PREVIEW: {preview}")

    # ========================================================================
    # STEP 2-4: Run independent operations concurrently for better performance
    # ========================================================================
    # Prepare data for concurrent operations
    path = request.url.path
    estimated_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in messages)

    # Run session management, connection tracking, and rate limiting concurrently
    session_task, connection_task, rate_task = await asyncio.gather(
        session_manager.get_or_create_session(metadata),
        connection_manager.add_connection(request, path, metadata),
        rate_limiter.check_and_update(metadata, estimated_tokens)
    )

    # Unpack results
    session = session_task
    connection_id = connection_task
    rate_allowed, rate_reason = rate_task

    is_first_message = session.get("is_first_message", False)

    logger.debug(
        f"Session state | Message #{session['message_count']} | "
        f"Tokens: {session.get('total_tokens', 0)} | "
        f"RAG: {session.get('rag_augmented', False)} | "
        f"First: {is_first_message}"
    )
    logger.debug(f"Connection tracking started: {connection_id[:8] if connection_id else 'none'}...")

    # Check rate limiting result
    if not rate_allowed:
        # Clean up connection and return rate limit error
        if connection_id:
            await connection_manager.remove_connection(connection_id)

        # Log analytics
        if config.ENABLE_ANALYTICS:
            await analytics_tracker.log_request(metadata, 0, "rate_limited", error=True)

        error_response = {
            "error": {
                "message": rate_reason,
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        }

        logger.warning(f"Rate limit exceeded for {user_name} ({user_id[:8] if user_id else 'anon'}...)")

        if stream:
            async def error_stream():
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return JSONResponse(status_code=429, content=error_response)

    # ========================================================================
    # STEP 5: Check for multimodal content early (instant detection)
    # ========================================================================
    is_multimodal_request = is_multimodal(messages)
    if is_multimodal_request:
        logger.debug("🖼️  Multimodal content detected - will validate text portion and passthrough to vLLM")

    # ========================================================================
    # STEP 6: Model-aware token validation (using metadata model info)
    # ========================================================================
    # Get model limits from metadata instead of hardcoding
    model_info = metadata.get("model", {})
    max_context = model_info.get("max_model_len", config.MAX_MODEL_CONTEXT)
    model_name = model_info.get("id", get_model_name())

    if config.PRECOUNT_PROMPT_TOKENS:
        # Extract text-only messages for token counting
        messages_for_tokenization = extract_text_only_messages(messages) if is_multimodal_request else messages

        if is_multimodal_request:
            logger.debug("🔢 Validating text portion of multimodal request for token limits")

        token_count = await count_request_tokens(messages_for_tokenization)
        if token_count is not None:
            # Use 95% threshold to leave room for response
            token_limit = int(max_context * config.CONTEXT_SAFETY_MARGIN)

            if token_count > token_limit:
                # Clean up and log analytics
                if connection_id:
                    await connection_manager.remove_connection(connection_id)

                if config.ENABLE_ANALYTICS:
                    await analytics_tracker.log_request(metadata, token_count, "token_exceeded", error=True)

                error_type = "multimodal text portion" if is_multimodal_request else "request"
                error_msg = f"Token limit exceeded for {error_type}: {token_count} tokens (max: {token_limit} for model {model_name})"
                logger.info(f"❌ {error_msg}")

                if stream:
                    return StreamingResponse(
                        create_token_limit_error_stream(token_count, model_name),
                        media_type="text/event-stream"
                    )
                else:
                    return create_token_limit_error_sync(token_count, model_name)

    # ========================================================================
    # STEP 7: Check model availability (wait if necessary)
    # ========================================================================
    if not is_model_available():
        logger.warning("Model not yet available, waiting up to 30 seconds...")
        model_ready = await wait_for_model(max_wait_seconds=30)

        if not model_ready:
            logger.warning("Model still not available after 30 seconds")
            # Clean up and log analytics
            if connection_id:
                await connection_manager.remove_connection(connection_id)

            if config.ENABLE_ANALYTICS:
                await analytics_tracker.log_request(metadata, 0, "model_unavailable", error=True)

            # Personalized loading message using metadata
            loading_msg = f"Hi {user_name}, the model is still loading. Please try again in a moment."

            if stream:
                return StreamingResponse(
                    model_loading_response_stream(model_name),
                    media_type="text/event-stream"
                )
            else:
                return model_loading_response_sync(model_name)
        else:
            logger.info("Model became available, proceeding with request")

    # ========================================================================
    # STEP 8: Handle multimodal requests (early passthrough)
    # ========================================================================
    if is_multimodal_request:
        try:
            logger.debug(f"🖼️  Routing multimodal request for chat {chat_id[:8] if chat_id else 'ephemeral'}...")

            # Update session with multimodal flag
            if chat_id:
                await session_manager.update_session(chat_id, {
                    "last_multimodal": time.time()
                })

            # Log analytics
            if config.ENABLE_ANALYTICS:
                await analytics_tracker.log_request(metadata, estimated_tokens, "multimodal")

            if stream:
                return StreamingResponse(
                    passthrough_stream(body, request),
                    media_type="text/event-stream"
                )
            else:
                result = await passthrough_sync(body)
                # Update token usage from response
                if isinstance(result, dict) and "usage" in result:
                    actual_tokens = result["usage"].get("total_tokens", 0)
                    if chat_id:
                        await session_manager.add_tokens(chat_id, actual_tokens)
                return result
        finally:
            # Clean up connection tracking
            if connection_id:
                await connection_manager.remove_connection(connection_id)

    # ========================================================================
    # STEP 9: Session-aware RAG augmentation (only once per chat)
    # ========================================================================
    rag_info = {"used": False, "query": "", "succeeded": False, "url": ""}

    # Only augment first message if not already done for this chat
    should_augment_with_rag = (
        config.ENABLE_INTERNAL_TOOL_CALLING and
        is_first_message and
        not session.get("rag_augmented", False) and
        not has_research_keyword(messages)
    )

    if should_augment_with_rag:
        logger.debug(f"✅ Will augment first message with RAG for chat {chat_id[:8] if chat_id else 'ephemeral'}...")
    else:
        reasons = []
        if not config.ENABLE_INTERNAL_TOOL_CALLING:
            reasons.append("disabled")
        if not is_first_message:
            reasons.append("not first")
        if session.get("rag_augmented", False):
            reasons.append("already done")
        if has_research_keyword(messages):
            reasons.append("research mode")
        logger.debug(f"⏭️  Skipping RAG: {', '.join(reasons)}")


    if should_augment_with_rag and stream:
        # Streaming RAG with pondering indicator
        async def stream_with_rag_augmentation():
            nonlocal rag_info, body, messages
            try:
                # Send "Pondering..." while we do RAG lookup
                async for chunk in create_pondering_stream(model_name):
                    yield chunk.encode()

                # Do the RAG augmentation
                body, rag_info = await augment_with_rag(
                    body,
                    messages,
                    model_name,
                    vllm_url=config.VLLM_BACKEND_URL + "/v1",
                    ragapi_url="http://localhost:8081"
                )
                messages = body.get("messages", [])

                # Mark session as RAG augmented
                if chat_id:
                    await session_manager.mark_rag_augmented(chat_id)

                # Log analytics
                if config.ENABLE_ANALYTICS:
                    await analytics_tracker.log_request(metadata, estimated_tokens, "rag")

                # Forward to vLLM and wrap response
                async with httpx.AsyncClient(timeout=config.VLLM_TIMEOUT) as client:
                    async with client.stream(
                        "POST",
                        f"{config.VLLM_BASE_URL}/chat/completions",
                        json=body
                    ) as response:
                        response.raise_for_status()

                        # Track actual tokens from stream
                        token_count = 0
                        async for chunk in wrap_stream_with_rag_indicator(
                            response.aiter_bytes(),
                            model_name,
                            rag_info
                        ):
                            token_count += len(chunk) // 40  # Rough estimate
                            yield chunk

                        # Update session tokens
                        if chat_id:
                            await session_manager.add_tokens(chat_id, token_count)
            finally:
                # Clean up connection
                if connection_id:
                    await connection_manager.remove_connection(connection_id)

        return StreamingResponse(
            stream_with_rag_augmentation(),
            media_type="text/event-stream"
        )

    # Non-streaming RAG augmentation
    if should_augment_with_rag and not stream:
        body, rag_info = await augment_with_rag(
            body,
            messages,
            model_name,
            vllm_url=config.VLLM_BACKEND_URL + "/v1",
            ragapi_url="http://localhost:8081"
        )
        messages = body.get("messages", [])

        # Mark session and log analytics
        if chat_id:
            await session_manager.mark_rag_augmented(chat_id)
        if config.ENABLE_ANALYTICS:
            await analytics_tracker.log_request(metadata, estimated_tokens, "rag")

    # ========================================================================
    # STEP 10: Research mode detection and routing with session tracking
    # ========================================================================
    try:
        # Detect research mode and iteration count
        is_research, iterations = detect_research_mode(messages)

        logger.info(f"Research mode: {is_research}, Iterations: {iterations if is_research else 'N/A'}")

        if is_research:
            # Update session research count
            if chat_id:
                await session_manager.increment_research_count(chat_id)

            # Log analytics
            if config.ENABLE_ANALYTICS:
                await analytics_tracker.log_request(metadata, estimated_tokens, "research")

            logger.info(
                f"🔬 Research mode | User: {user_name} | "
                f"Chat: {chat_id[:8] if chat_id else 'ephemeral'}... | "
                f"Iterations: {iterations}"
            )

            # Determine which semaphore to use
            semaphore = deep_research_semaphore if iterations == 4 else research_semaphore
            is_deep = iterations == 4

            if stream:
                # Streaming research with queue management
                async def research_stream_with_tracking():
                    token_count = 0
                    async for chunk in research_with_queue_management(
                        body, model_name, iterations, request, semaphore, is_deep
                    ):
                        token_count += len(chunk) // 40  # Rough estimate
                        yield chunk

                    # Update session tokens
                    if chat_id:
                        await session_manager.add_tokens(chat_id, token_count)

                    # Clean up connection
                    if connection_id:
                        await connection_manager.remove_connection(connection_id)

                return StreamingResponse(
                    research_stream_with_tracking(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming research
                async with semaphore:
                    result = await research_sync(
                        body,
                        model_name=model_name,
                        max_iterations=iterations,
                        vllm_url=config.VLLM_BACKEND_URL + "/v1",
                        mcp_url=config.REST_API_URL,
                        mcp_api_key=config.REST_API_KEY
                    )

                    # Update session tokens and clean up
                    if isinstance(result, dict) and "usage" in result:
                        actual_tokens = result["usage"].get("total_tokens", 0)
                        if chat_id:
                            await session_manager.add_tokens(chat_id, actual_tokens)

                    if connection_id:
                        await connection_manager.remove_connection(connection_id)

                    return result

        # ========================================================================
        # STEP 11: Standard passthrough with analytics tracking
        # ========================================================================
        logger.debug(f"🟢 Standard passthrough for chat {chat_id[:8] if chat_id else 'ephemeral'}...")

        # Log analytics for standard request
        if config.ENABLE_ANALYTICS:
            await analytics_tracker.log_request(metadata, estimated_tokens, "standard")

        if stream:
            # Streaming passthrough with token tracking
            async def passthrough_stream_with_tracking():
                token_count = 0
                async for chunk in passthrough_stream(body, request):
                    token_count += len(chunk) // 40  # Rough estimate
                    yield chunk

                # Update session tokens
                if chat_id:
                    await session_manager.add_tokens(chat_id, token_count)

                # Clean up connection
                if connection_id:
                    await connection_manager.remove_connection(connection_id)

            return StreamingResponse(
                passthrough_stream_with_tracking(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming passthrough
            result = await passthrough_sync(body)

            # Update session tokens from response
            if isinstance(result, dict) and "usage" in result:
                actual_tokens = result["usage"].get("total_tokens", 0)
                if chat_id:
                    await session_manager.add_tokens(chat_id, actual_tokens)

            # Clean up connection
            if connection_id:
                await connection_manager.remove_connection(connection_id)

            return result

    except Exception as e:
        # Log error analytics
        if config.ENABLE_ANALYTICS:
            await analytics_tracker.log_request(metadata, 0, "error", error=True)

        logger.error(f"Error in autonomous_chat: {str(e)}", exc_info=True)

        # Clean up connection on error
        if connection_id:
            await connection_manager.remove_connection(connection_id)

        # Return error response
        error_msg = f"Internal server error: {str(e)}"
        if stream:
            async def error_stream():
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return JSONResponse(status_code=500, content={"error": error_msg})

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose metrics for monitoring systems (Prometheus/Grafana compatible).

    AUTHORIZATION: Only accessible by Robert P Smith.
    Use header "X-User-Name: Robert P Smith" or "X-Admin-Token: RobertPSmith-AdminAccess-2025"

    Returns comprehensive metrics from all subsystems:
    - Session manager statistics
    - Rate limiter statistics
    - Analytics data
    - Connection manager status
    """
    # Check authorization
    if not await check_admin_authorization(request):
        logger.warning(f"Unauthorized metrics access attempt from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # Authorized - return metrics
    metrics_data = {
        "session_manager": await session_manager.get_stats() if session_manager else {},
        "rate_limiter": await rate_limiter.get_stats() if rate_limiter else {},
        "connection_manager": await connection_manager.get_connection_status() if connection_manager else {},
    }

    if config.ENABLE_ANALYTICS:
        metrics_data["analytics"] = await analytics_tracker.export_metrics()
        metrics_data["global_summary"] = await analytics_tracker.get_global_summary()

    return metrics_data


@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str, request: Request):
    """
    Get session information for a specific user.

    AUTHORIZATION: Only accessible by Robert P Smith.
    Use header "X-User-Name: Robert P Smith" or "X-Admin-Token: RobertPSmith-AdminAccess-2025"

    Args:
        user_id: User identifier
        request: FastAPI Request object for authorization

    Returns:
        User sessions and rate limit status (or 404 if unauthorized)
    """
    # Check authorization
    if not await check_admin_authorization(request):
        logger.warning(f"Unauthorized session access attempt for user {user_id[:8]}... from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # Authorized - return session data
    sessions = await session_manager.get_user_sessions(user_id)
    rate_status = await rate_limiter.get_user_status(user_id)

    if config.ENABLE_ANALYTICS:
        user_summary = await analytics_tracker.get_user_summary(user_id)
    else:
        user_summary = {}

    return {
        "user_id": user_id,
        "sessions": sessions,
        "rate_limit_status": rate_status,
        "analytics": user_summary
    }


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
    vllm_container = check_container("vllm-qwen3")  # Updated container name
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
@app.get("/models")
async def list_models():
    """
    Return available models using cached model data from proxy.

    Returns cached full model response from vLLM /v1/models endpoint.
    If model not yet loaded, returns empty list.

    Available at both /v1/models and /models for compatibility.
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
    # LOG INCOMING REQUEST TO CATCH-ALL
    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"🔴 ROUTE: /{path} | FROM: {client_ip} | HANDLER: catch_all_proxy | METHOD: {request.method}")

    # Skip endpoints we've already defined - these are handled by their specific routes
    if path in ["health", "v1/chat/completions", "v1/models", "models", "openapi.json"]:
        # Return 404 since these should have been caught by their specific handlers
        logger.warning(f"⚠️  Catch-all matched reserved endpoint: {path} - this should not happen!")
        raise HTTPException(status_code=404, detail="Route handled by specific endpoint")

    # Route based on endpoint type
    if is_vllm_endpoint(path):
        # vLLM endpoint - forward to vLLM backend
        target_url = f"{config.VLLM_BACKEND_URL}/{path}"
        logger.debug(f"🟡 FORWARD (vLLM): → {target_url}")
    else:
        # Non-vLLM endpoint - forward to robairagapi (port 8081)
        target_url = f"http://localhost:8081/{path}"
        logger.debug(f"🟡 FORWARD (RAG API): → {target_url}")

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
