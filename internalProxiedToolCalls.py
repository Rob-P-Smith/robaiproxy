"""
Internal Proxied Tool Calls - Transparent RAG Augmentation

This module implements automatic RAG context injection for first messages
in new conversations. It detects first user messages, generates search queries
via forced tool calling, executes searches, and injects results before the
final LLM response - all transparently to the user.

Architecture:
1. Detection: Check if first message + not research mode
2. Phase 1: Force LLM to generate search query via tool calling
3. Phase 2: Execute simple_search against ragapi
4. Format & Inject: Add RAG context as system message
5. Continue: Normal passthrough with augmented context

Total overhead: ~150-200ms per first message
"""

import os
import json
import httpx
import logging
import asyncio
import hashlib
import random
from typing import Optional, Dict, Any, List, AsyncGenerator
from collections import OrderedDict
from datetime import datetime

# Import config for feature flag and API credentials
from config import config

logger = logging.getLogger("researchProxy")


# ============================================================================
# Pondering Messages
# ============================================================================

PONDERING_MESSAGES = [
    "Pondering...",
    "Thinking...",
    "Contemplating...",
    "Reflecting...",
    "Considering...",
    "Analyzing...",
    "Processing...",
    "Mulling over...",
    "Deliberating...",
    "Ruminating...",
    "Elucidating...",
    "Meandering...",
    "Delving...",
    "Theorizing...",
    "Daydreaming..."
]


# ============================================================================
# Conversation Tracking (LRU Cache for seen conversations)
# ============================================================================

class ConversationTracker:
    """
    Track which conversations have already been augmented with RAG.

    Uses an LRU cache to remember conversation fingerprints. When we see
    a conversation for the first time, we return True and mark it as seen.
    Subsequent requests for the same conversation return False.
    """

    def __init__(self, max_size: int = 1000):
        self.seen_conversations = OrderedDict()
        self.max_size = max_size

    def _compute_fingerprint(self, messages: List[Dict[str, Any]]) -> str:
        """
        Compute a fingerprint for the conversation.

        Uses the first user message content as the fingerprint. This identifies
        unique conversations while being resilient to system message changes.
        """
        # Find first user message
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Hash the content for privacy and fixed length
                return hashlib.sha256(content.encode()).hexdigest()[:16]
        return ""

    def is_new_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Check if this is a new conversation we haven't seen before.

        Returns True for first-time conversations, False for repeat requests.
        Marks the conversation as seen if it's new.
        """
        fingerprint = self._compute_fingerprint(messages)
        if not fingerprint:
            return False

        # Check if we've seen this conversation
        if fingerprint in self.seen_conversations:
            # Move to end (LRU)
            self.seen_conversations.move_to_end(fingerprint)
            logger.debug(f"Conversation fingerprint {fingerprint} already seen, skipping RAG")
            return False

        # New conversation - mark as seen
        self.seen_conversations[fingerprint] = datetime.now()
        logger.debug(f"New conversation fingerprint {fingerprint}, will augment with RAG")

        # Enforce max size (LRU eviction)
        if len(self.seen_conversations) > self.max_size:
            evicted = self.seen_conversations.popitem(last=False)
            logger.debug(f"Evicted old conversation fingerprint {evicted[0]} from cache")

        return True


# Global tracker instance
conversation_tracker = ConversationTracker(max_size=1000)


# ============================================================================
# Detection Functions
# ============================================================================

def is_first_user_message(messages: List[Dict[str, Any]]) -> bool:
    """
    Check if this is the first user message in a NEW conversation.

    Uses conversation tracking to detect if we've seen this conversation before.
    Only returns True for brand new conversations (based on first user message).

    Args:
        messages: List of chat messages

    Returns:
        bool: True if this is a new conversation we haven't augmented yet
    """
    # Must have exactly 1 user message for this to be a first request
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if len(user_messages) != 1:
        return False

    # Check if this is a new conversation using fingerprinting
    return conversation_tracker.is_new_conversation(messages)


def has_research_keyword(messages: List[Dict[str, Any]]) -> bool:
    """
    Check if the last user message starts with "research".

    Research mode has its own comprehensive search logic, so we skip
    internal tool calling when detected.

    Args:
        messages: List of chat messages

    Returns:
        bool: True if last user message starts with "research"
    """
    if not messages:
        return False

    last_message = messages[-1]
    if last_message.get("role") != "user":
        return False

    content = last_message.get("content", "")
    if isinstance(content, str):
        return content.lower().strip().startswith("research")

    return False


# ============================================================================
# Phase 1: Tool Call Generation
# ============================================================================

async def generate_search_query_via_tool(
    messages: List[Dict[str, Any]],
    model: str,
    vllm_url: str
) -> Optional[Dict[str, Any]]:
    """
    Force LLM to generate a search query using tool calling.

    This is Phase 1 of transparent RAG augmentation. We construct a special
    request that forces the model to call simple_search with an appropriate
    query based on the user's question.

    Args:
        messages: Original conversation messages
        model: Model name to use
        vllm_url: vLLM backend URL

    Returns:
        dict: Tool call response from vLLM, or None on failure
    """
    # System prompt to guide tool generation
    tool_prompt = {
        "role": "system",
        "content": """You have access to an enhanced_search tool that performs comprehensive knowledge base search.

Tool: enhanced_search
Parameters:
  - query (required, string): A concise search query with 2-5 keywords
  - tags (optional, string): Comma-separated tags (2-3 relevant topics/technologies)

This tool automatically returns 3 RAG results with full markdown content + 5 KG graph results.
Based on the user's question:
1. Generate ONE focused search query (2-5 keywords)
2. Generate 2-3 relevant tags (technologies, topics, or concepts)

Examples:
User: "How do I use FastAPI with async functions?"
Tool call: enhanced_search(query="FastAPI async functions", tags="python,async,web-framework")

User: "What is Redux state management?"
Tool call: enhanced_search(query="Redux state management", tags="javascript,react,state")

User: "Explain Docker containerization"
Tool call: enhanced_search(query="Docker containerization", tags="docker,containers,devops")"""
    }

    # Prepend tool instruction to original messages
    tool_messages = [tool_prompt] + messages

    # Define enhanced_search tool
    tool_definition = {
        "type": "function",
        "function": {
            "name": "enhanced_search",
            "description": "Comprehensive search using full KG pipeline. Returns 3 RAG results with full markdown + 5 KG results with referenced chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query with 2-5 keywords focused on main concepts"
                    },
                    "tags": {
                        "type": "string",
                        "description": "Optional comma-separated tags to filter results"
                    }
                },
                "required": ["query"]
            }
        }
    }

    # Build request body with forced tool calling
    tool_body = {
        "model": model,
        "messages": tool_messages,
        "tools": [tool_definition],
        "tool_choice": {"type": "function", "function": {"name": "enhanced_search"}},
        "stream": False,
        "max_tokens": 100,
        "temperature": 0.1  # Low temperature for consistent tool calls
    }

    try:
        logger.debug(f"🔧 Generating search query via tool calling (model: {model})")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{vllm_url}/v1/chat/completions",
                json=tool_body
            )
            response.raise_for_status()
            result = response.json()
            logger.debug("✅ Tool call generation successful")
            return result
    except Exception as e:
        logger.warning(f"❌ Tool generation failed: {e}")
        return None


# ============================================================================
# Phase 2: Search Execution
# ============================================================================

async def execute_enhanced_search(
    query: str,
    tags: str = None,
    ragapi_url: str = "http://localhost:8081",
    bearer_token: str = None
) -> Optional[Dict[str, Any]]:
    """
    Execute enhanced_search against ragapi.

    This is Phase 2 of transparent RAG augmentation. We take the search query
    generated by the LLM and execute it against the RAG API using the enhanced
    search endpoint which returns 3 RAG results + 5 KG results.

    Args:
        query: Search query string
        tags: Optional comma-separated tags to filter results
        ragapi_url: RAG API base URL
        bearer_token: Authentication bearer token

    Returns:
        dict: Search results from ragapi, or None on failure
    """
    if not bearer_token:
        logger.error("❌ No bearer token available for ragapi search")
        return None

    try:
        # Build request body
        search_body = {"query": query}
        if tags:
            search_body["tags"] = tags

        logger.debug(f"🔍 Executing enhanced search: '{query}'{f' (tags: {tags})' if tags else ''}")
        async with httpx.AsyncClient(timeout=10.0) as client:  # Longer timeout for enhanced search
            response = await client.post(
                f"{ragapi_url}/api/v1/search/enhanced",
                json=search_body,
                headers={"Authorization": f"Bearer {bearer_token}"}
            )
            response.raise_for_status()
            result = response.json()

            # Check if we got results
            rag_count = len(result.get("data", {}).get("rag_results", []))
            kg_count = len(result.get("data", {}).get("kg_results", []))
            logger.debug(f"✅ Enhanced search completed: {rag_count} RAG + {kg_count} KG results")
            return result
    except Exception as e:
        logger.warning(f"❌ Enhanced search execution failed: {e}")
        return None


# ============================================================================
# Context Filtering & Formatting
# ============================================================================

def filter_rag_content(content: str, max_chars: int = 9000) -> str:
    """
    Filter and truncate RAG content for first-message augmentation.

    Filters:
    1. Remove all words ≤2 letters
    2. Remove 'the', 'they', 'them'
    3. Truncate to max_chars

    Args:
        content: Raw content from RAG result
        max_chars: Maximum characters to return (default: 9000)

    Returns:
        str: Filtered and truncated content
    """
    if not content:
        return ""

    # Split into words while preserving structure
    words = content.split()

    # Filter words
    filtered_words = []
    for word in words:
        # Strip punctuation for length check but keep original word
        word_stripped = word.strip('.,!?;:"()[]{}')

        # Skip words ≤2 letters (after stripping punctuation)
        if len(word_stripped) <= 2:
            continue

        # Skip specific words (case-insensitive)
        if word_stripped.lower() in ['the', 'they', 'them']:
            continue

        filtered_words.append(word)

    # Reconstruct text
    filtered_content = ' '.join(filtered_words)

    # Truncate to max_chars
    if len(filtered_content) > max_chars:
        filtered_content = filtered_content[:max_chars] + "..."

    return filtered_content


def format_rag_context(
    search_result: Dict[str, Any],
    original_question: str
) -> Optional[str]:
    """
    Format enhanced search results for injection into conversation.

    Extracts content and URLs from RAG and KG results and formats them
    as a system message that instructs the LLM to use the information if
    relevant and provide citations.

    Args:
        search_result: Response from ragapi enhanced_search
        original_question: The user's original question

    Returns:
        str: Formatted context for injection, or None if no results
    """
    if not search_result:
        return None

    data = search_result.get("data", {})
    rag_results = data.get("rag_results", [])
    kg_results = data.get("kg_results", [])

    if not rag_results and not kg_results:
        logger.debug("No search results to inject")
        return None

    # Build context from RAG results (filtered markdown, max 9000 chars each = 27000 total)
    rag_context = []
    for i, result in enumerate(rag_results[:3], 1):  # Use all 3 RAG results
        markdown = result.get("markdown", "").strip()
        url = result.get("url", "")
        title = result.get("title", "")
        score = result.get("score", 0)

        if markdown:
            # Apply filtering: remove short words, specific words, truncate to 9000 chars
            original_len = len(markdown)
            filtered_content = filter_rag_content(markdown, max_chars=9000)
            filtered_len = len(filtered_content)

            logger.debug(f"RAG result {i} filtered: {original_len} → {filtered_len} chars ({filtered_len/original_len*100:.1f}%)")

            if filtered_content:
                rag_context.append(f"""### RAG Result {i}: {title}
Source: {url}
Score: {score:.3f}

{filtered_content}""")

    # Build context from KG results (FULL content, no filtering)
    kg_context = []
    for i, result in enumerate(kg_results[:5], 1):  # Use all 5 KG results
        # Prefer referenced_chunk (focused context) over full content
        content = result.get("referenced_chunk", "").strip() or result.get("content", "").strip()
        url = result.get("url", "")
        entity = result.get("entity", "")

        if content:
            kg_context.append(f"""### KG Result {i}: {entity}
Source: {url}

{content}""")

    # Combine all context
    all_context = []
    if rag_context:
        all_context.append("## RAG Search Results (Filtered, Max 9000 chars each)\n\n" + "\n\n".join(rag_context))
    if kg_context:
        all_context.append("## Knowledge Graph Results (Full Content)\n\n" + "\n\n".join(kg_context))

    if not all_context:
        logger.debug("No content in search results")
        return None

    # Format as instruction to LLM
    context_message = f"""Background information retrieved from the knowledge base:

{chr(10).join(all_context)}

---

The user asked: "{original_question}"

Instructions:
1. Read all the background information above (both RAG and KG results)
2. Use relevant information to enhance your response
3. Ignore information that isn't relevant to the question
4. If you use information from the sources, cite them at the end of your response

Example citation format:
Sources: [RAG Result 1 URL], [KG Result 2 URL]"""

    # Log context stats
    rag_count = len(rag_context)
    kg_count = len(kg_context)
    total_chars = len(context_message)
    logger.info(f"📦 Context prepared: {rag_count} RAG + {kg_count} KG results, {total_chars:,} chars total")

    return context_message


# ============================================================================
# Main Orchestrator
# ============================================================================

async def augment_with_rag(
    body: Dict[str, Any],
    messages: List[Dict[str, Any]],
    model_name: str
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main orchestrator for transparent RAG augmentation.

    This is the entry point called by requestProxy.py. It handles:
    1. Detection (first message + not research mode)
    2. Tool call generation to get search query
    3. Search execution against ragapi
    4. Context formatting and injection
    5. Returns modified body with augmented messages

    If anything fails at any step, returns the original body unchanged
    and logs the issue - graceful degradation.

    Args:
        body: Original request body
        messages: Chat messages array
        model_name: Active model name

    Returns:
        tuple: (Modified body with augmented messages, rag_info: Dict)
               rag_info contains: {
                   "used": bool,
                   "query": str,
                   "succeeded": bool,
                   "url": str
               }
    """
    # Default RAG info
    rag_info = {"used": False, "query": "", "succeeded": False, "url": ""}

    # Step 1: Detection
    if not is_first_user_message(messages):
        logger.debug("Not first message, skipping internal tool calling")
        return body, rag_info

    if has_research_keyword(messages):
        logger.debug("Research keyword detected, skipping internal tool calling")
        return body, rag_info

    logger.info("🎯 INTERNAL TOOL CALLING: First message detected, augmenting with RAG")

    # Step 2: Generate search query via tool calling
    tool_response = await generate_search_query_via_tool(
        messages,
        model_name,
        config.VLLM_BACKEND_URL
    )

    if not tool_response:
        logger.warning("Tool generation failed, continuing without RAG augmentation")
        return body, rag_info

    # Extract tool call from response
    choices = tool_response.get("choices", [])
    if not choices:
        logger.warning("No choices in tool response")
        return body, rag_info

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        logger.warning("No tool calls in response")
        return body, rag_info

    # Parse tool call arguments
    tool_call = tool_calls[0]
    function_name = tool_call.get("function", {}).get("name")

    if function_name != "enhanced_search":
        logger.warning(f"Unexpected tool call: {function_name}")
        return body, rag_info

    try:
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        arguments = json.loads(arguments_str)
        search_query = arguments.get("query")
        tags = arguments.get("tags")  # Optional
    except Exception as e:
        logger.warning(f"Failed to parse tool arguments: {e}")
        return body, rag_info

    if not search_query:
        logger.warning("No search query in tool call")
        return body, rag_info

    # Mark that we attempted RAG
    rag_info["used"] = True
    rag_info["query"] = search_query

    logger.info(f"🔍 Generated search query: '{search_query}'{f' (tags: {tags})' if tags else ''}")

    # Step 3: Execute search
    # Get OPENAI_API_KEY from environment (ragapi authentication)
    bearer_token = os.getenv("OPENAI_API_KEY")
    if not bearer_token:
        logger.error("❌ No OPENAI_API_KEY configured for ragapi authentication")
        return body, rag_info

    logger.debug(f"🔑 Using bearer token: ***{bearer_token[-8:] if len(bearer_token) > 8 else '***'}")

    search_result = await execute_enhanced_search(
        search_query,
        tags=tags,
        bearer_token=bearer_token
    )

    if not search_result:
        logger.warning("Search execution failed, continuing without RAG augmentation")
        rag_info["succeeded"] = False
        rag_info["url"] = "N/A"
        return body, rag_info

    # Check if search was successful and extract URL from highest scoring result
    search_success = search_result.get("success", False)
    data = search_result.get("data", {})
    rag_results = data.get("rag_results", [])
    kg_results = data.get("kg_results", [])

    if search_success and (rag_results or kg_results):
        rag_info["succeeded"] = True

        # Find the highest scoring result across both RAG and KG results
        # RAG results come first and are already sorted by score (highest first)
        # So we use the first RAG result as it has the highest score
        if rag_results:
            # First RAG result has the highest score
            rag_info["url"] = rag_results[0].get("url", "N/A")
            score = rag_results[0].get("score", 0)
            logger.debug(f"Using highest scoring result (RAG, score: {score:.3f})")
        elif kg_results:
            # Fall back to first KG result if no RAG results
            rag_info["url"] = kg_results[0].get("url", "N/A")
            logger.debug("Using first KG result (no RAG results available)")
        else:
            rag_info["url"] = "N/A"
    else:
        rag_info["succeeded"] = False
        rag_info["url"] = "N/A"

    # Step 4: Format and inject context
    original_question = messages[-1].get("content", "")
    rag_context = format_rag_context(search_result, original_question)

    if not rag_context:
        logger.warning("No relevant results to inject")
        return body, rag_info

    # Step 5: Inject as system message
    logger.info("✅ RAG context injected successfully")
    messages.append({
        "role": "system",
        "content": rag_context
    })

    # Update body with augmented messages
    body["messages"] = messages

    return body, rag_info


# ============================================================================
# Streaming Response Helpers
# ============================================================================

async def create_pondering_stream(model_name: str) -> AsyncGenerator[str, None]:
    """
    Create a streaming response that shows a random pondering message while RAG is working.

    Randomly selects from a list of pondering synonyms to add variety.

    Args:
        model_name: Model name for the stream response

    Yields:
        str: SSE-formatted streaming chunks
    """
    # Pick a random pondering message
    pondering_message = random.choice(PONDERING_MESSAGES)

    # Send initial pondering message with flashing style
    chunk = {
        "id": "rag_pondering",
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": f"⚡ *{pondering_message}*"  # Using emoji and italic for visual interest
            },
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"


async def wrap_stream_with_rag_indicator(
    upstream_stream: AsyncGenerator[bytes, None],
    model_name: str,
    rag_info: Dict[str, Any]
) -> AsyncGenerator[bytes, None]:
    """
    Wrap an existing stream to inject RAG status before the end token.

    This function:
    1. Clears "Pondering..." when real content starts
    2. Passes through all chunks from upstream
    3. Before [DONE], injects "Internal RAG Used; <query> <status> <url>"

    Args:
        upstream_stream: The original streaming response from vLLM
        model_name: Model name for generated chunks
        rag_info: Dict with keys: used, query, succeeded, url

    Yields:
        bytes: SSE-formatted streaming chunks
    """
    first_chunk_sent = False
    rag_was_used = rag_info.get("used", False)

    async for chunk in upstream_stream:
        # Check if this is the [DONE] marker
        if b"data: [DONE]" in chunk:
            # Before sending [DONE], inject RAG indicator if it was used
            if rag_was_used:
                status = "succeeded" if rag_info.get("succeeded", False) else "failed"
                query = rag_info.get("query", "unknown")
                url = rag_info.get("url", "N/A")

                rag_message = f"\n\n*Internal RAG Used; {query}; {status}; {url}*"

                rag_chunk = {
                    "id": "rag_indicator",
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": rag_message},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(rag_chunk)}\n\n".encode()

            # Now send the [DONE] marker
            yield chunk
        else:
            # For the first real chunk, we need to clear the "Pondering..." text
            if not first_chunk_sent and rag_was_used:
                # Parse the chunk to check if it has content
                try:
                    chunk_str = chunk.decode()
                    if chunk_str.startswith("data: "):
                        chunk_data = json.loads(chunk_str[6:].strip())
                        if chunk_data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            # Send a backspace/clear chunk before the first real content
                            clear_chunk = {
                                "id": "clear_pondering",
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": "\r" + " " * 20 + "\r"},  # Clear "⚡ *Pondering...*"
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(clear_chunk)}\n\n".encode()
                            first_chunk_sent = True
                except:
                    pass  # If parsing fails, just pass through

            # Pass through the original chunk
            yield chunk
