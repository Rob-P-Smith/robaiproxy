# researchAgent.py
# Core research orchestration logic - handles web search, tool calls, and research workflows

from openai import OpenAI, BadRequestError
import httpx
import json
import re

# Load configuration from environment variables
from config import config, logger

# Initialize vLLM client with config
vllm = OpenAI(base_url=config.VLLM_BASE_URL, api_key="dummy")

TOOL_SYSTEM_PROMPT = """You have access to research tools to gather comprehensive information. Available tools:

1. simple_search - Simple vector similarity search of your knowledge base (fast, good for basic queries)
2. kg_search - Knowledge Graph-enhanced search with entity extraction and expansion (powerful, best for complex queries)
3. crawl_url - Fetch fresh content from a URL (temporary, not stored)
4. crawl_and_remember - Fetch and permanently store content from a URL in knowledge base
5. crawl_temp - Fetch and store content temporarily (session only, auto-deleted)
6. deep_crawl_and_store - Deep crawl multiple pages from a starting URL (follows links, stores all)
7. list_memory - List all stored content in knowledge base
8. list_blocked_domains - List domains that are blocked from crawling
9. add_blocked_domain - Block a domain pattern from being crawled
10. remove_blocked_domain - Unblock a previously blocked domain
11. forget_url - Remove specific URL from knowledge base
12. clear_temp_memory - Clear all temporary/session-only content
13. db_stats - Get database statistics (total docs, entities, etc.)

To use tools, output them in this EXACT format:

<tool_call>
{"name": "simple_search", "arguments": {"query": "react hooks patterns", "limit": 5}}
</tool_call>

<tool_call>
{"name": "kg_search", "arguments": {"query": "machine learning optimization techniques", "limit": 10, "enable_expansion": true}}
</tool_call>

<tool_call>
{"name": "crawl_url", "arguments": {"url": "https://react.dev/reference/react"}}
</tool_call>

<tool_call>
{"name": "crawl_and_remember", "arguments": {"url": "https://docs.python.org/3/library/asyncio.html", "tags": "python,async"}}
</tool_call>

<tool_call>
{"name": "deep_crawl_and_store", "arguments": {"url": "https://example.com/docs", "max_depth": 2, "max_pages": 20, "tags": "documentation"}}
</tool_call>

IMPORTANT: You can make multiple tool calls and should make at least one simple_search tool call. After receiving <tool_response> tags with results, use that information to provide a comprehensive answer. Make tool calls FIRST, then answer."""


async def search_serper(query: str, num_results: int = 10):
    """
    Search using Serper API and return top web search results.

    This function makes an asynchronous HTTP POST request to the Serper API to perform
    a Google search and return organic search results. It's used at the beginning of
    research mode to gather current web information before starting the research loop.

    Args:
        query (str): The search query string to send to Serper API
        num_results (int, optional): Maximum number of results to return. Defaults to 10.

    Returns:
        dict: A dictionary containing:
            - success (bool): True if search succeeded, False otherwise
            - query (str): The original query string
            - results (list): List of dict objects, each containing:
                - title (str): Page title
                - link (str): URL of the page
                - snippet (str): Text snippet/description from the page
            - total_results (int): Number of results returned (only on success)
            - error (str): Error message (only on failure)

    Used by:
        - research_mode_stream(): Called at the start to get web context
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": num_results
    })

    headers = {
        'X-API-KEY': config.SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()

            # Extract organic results
            results = []
            if 'organic' in data:
                for item in data['organic'][:num_results]:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    })

            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results)
            }
    except Exception as e:
        return {
            'success': False,
            'query': query,
            'error': str(e),
            'results': []
        }


async def call_mcp_tool(tool_name: str, arguments: dict):
    """
    Execute MCP tools via REST API and return results.

    This function acts as a bridge between the research proxy and the MCP server,
    routing tool calls to appropriate REST API endpoints. Supports search_memory,
    crawl_url, and crawl_and_remember operations.

    Args:
        tool_name (str): Name of the tool to execute. Valid values:
            - "search_memory": Search the knowledge base
            - "crawl_url": Crawl a URL and return content
            - "crawl_and_remember": Crawl and store URL in knowledge base
        arguments (dict): Tool-specific arguments. Structure varies by tool:
            - search_memory: {"query": str, "limit": int}
            - crawl_url: {"url": str}
            - crawl_and_remember: {"url": str, "tags": str, "retention_policy": str}

    Returns:
        dict or str: Tool result data. Structure depends on the tool called.
            Usually contains search results, crawled content, or operation status.

    Raises:
        Exception: If tool name is unknown, API returns error, or tool execution fails

    Used by:
        - research_mode_stream(): Called for each search_memory and crawl_url operation
        - research_mode_sync(): Same usage in non-streaming mode
    """
    async with httpx.AsyncClient() as client:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.REST_API_KEY}"
        }

        # Map tool names to REST API endpoints
        if tool_name == "search_memory":
            response = await client.post(
                f"{config.REST_API_URL}/search",
                json=arguments,
                headers=headers,
                timeout=60.0
            )
        elif tool_name == "crawl_url":
            response = await client.post(
                f"{config.REST_API_URL}/crawl",
                json=arguments,
                headers=headers,
                timeout=60.0
            )
        elif tool_name == "crawl_and_remember":
            response = await client.post(
                f"{config.REST_API_URL}/crawl/store",
                json={
                    "url": arguments["url"],
                    "tags": arguments.get("tags", ""),
                    "retention_policy": "permanent"
                },
                headers=headers,
                timeout=60.0
            )
        else:
            raise Exception(f"Unknown tool: {tool_name}")

        result = response.json()

        # Handle errors
        if "detail" in result:
            raise Exception(f"API Error: {result['detail']}")

        if not result.get("success"):
            raise Exception(f"Tool failed: {result.get('error', 'Unknown error')}")

        return result.get("data", result)


def extract_tool_calls(text: str):
    """
    Parse tool calls from LLM response text.

    Extracts JSON tool call objects that are wrapped in <tool_call></tool_call> tags
    from the model's response. Used in sync mode for parsing tool invocations.

    Args:
        text (str): LLM response text that may contain tool call tags

    Returns:
        list: List of dict objects, each representing a parsed tool call with
            structure like {"name": "tool_name", "arguments": {...}}

    Used by:
        - research_mode_sync(): Parses tool calls from non-streaming LLM responses
    """
    pattern = r'<tool_call>\s*(\{[^}]+\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            tool_calls.append(json.loads(match))
        except:
            continue
    return tool_calls


def extract_urls_from_results(research_summary: list) -> list:
    """
    Extract and deduplicate URLs from research summary.

    Scans through all tool call results in the research summary and extracts
    any URLs found using regex patterns. Returns up to 5 unique URLs for
    inclusion in the final research summary sent to the user.

    Args:
        research_summary (list): List of dict objects containing tool call
            results with 'result' field

    Returns:
        list: Up to 5 unique URL strings extracted from results

    Used by:
        - research_mode_stream(): Creates "Further Reading" section in summary
    """
    urls = []
    for call in research_summary:
        # Look for URLs in the result field if it exists
        if 'result' in call:
            result_str = str(call['result'])
            # Extract URLs from the result
            url_pattern = r'https?://[^\s\'"<>]+'
            found_urls = re.findall(url_pattern, result_str)
            urls.extend(found_urls)

    # Deduplicate and return first 5
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen and len(unique_urls) < 5:
            unique_urls.append(url)
            seen.add(url)

    return unique_urls


def create_sse_chunk(content: str, finish_reason: str = None, usage: dict = None, model: str = "unknown-model"):
    """
    Create Server-Sent Event (SSE) chunk in OpenAI streaming format.

    Formats streaming response chunks to match OpenAI's chat completion streaming
    API format. Supports content streaming, finish reasons, and usage statistics.

    Args:
        content (str): Text content to stream to client
        finish_reason (str, optional): Completion finish reason ("stop", "length", etc.)
        usage (dict, optional): Token usage statistics dict with keys like
            'prompt_tokens', 'completion_tokens', 'total_tokens'
        model (str, optional): Model name to include in response. Defaults to "unknown-model"

    Returns:
        str: Formatted SSE chunk string starting with "data: " and ending with "\n\n"

    Used by:
        - research_mode_stream(): For all streaming output including research progress
        - passthrough_stream(): For transparent streaming passthrough
    """
    chunk = {
        "id": "chatcmpl-research",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason
        }]
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def get_iteration_focus(iteration: int, max_iterations: int) -> tuple[str, str]:
    """
    Get the focus and avoidance instruction for a given iteration.

    Uses VERY STRONG language to force query diversity and maximize RAG data capture.

    Args:
        iteration (int): Current iteration (0-indexed)
        max_iterations (int): Total number of iterations (2 or 4)

    Returns:
        tuple[str, str]: (iteration_focus, avoid_queries_instruction)
    """
    if max_iterations == 2:
        # Two iterations: main concepts → practical implementation
        if iteration == 0:
            return ("Focus EXCLUSIVELY on the MAIN topic and CORE foundational concepts. DO NOT cover implementation details.", "")
        else:  # iteration == 1
            avoid = "\n\n🚨 CRITICAL: You MUST use COMPLETELY DIFFERENT search terms than before. DO NOT repeat any keywords from previous queries. Focus on PRACTICAL IMPLEMENTATION terminology like 'how to', 'setup', 'usage', 'examples', 'best practices'. Your query must be TOTALLY DISTINCT from iteration 1."
            return ("Focus EXCLUSIVELY on PRACTICAL IMPLEMENTATION, usage examples, best practices, and real-world application. DO NOT repeat conceptual content.", avoid)

    elif max_iterations == 4:
        # Four iterations: main → implementation → advanced → ecosystem
        if iteration == 0:
            return ("Focus EXCLUSIVELY on the MAIN topic and CORE foundational concepts. DO NOT cover implementation or advanced topics.", "")
        elif iteration == 1:
            avoid = "\n\n🚨 CRITICAL: Use COMPLETELY DIFFERENT search keywords. Focus on 'setup', 'installation', 'configuration', 'getting started', 'integration', 'how to'. NO overlap with iteration 1."
            return ("Focus EXCLUSIVELY on IMPLEMENTATION and practical setup. Use terms like 'how to configure', 'setup guide', 'integration steps', 'usage examples'. DO NOT repeat conceptual content.", avoid)
        elif iteration == 2:
            avoid = "\n\n🚨 CRITICAL: Use ENTIRELY NEW search terms. Focus on 'best practices', 'optimization', 'performance', 'patterns', 'troubleshooting', 'pitfalls'. NO overlap with previous iterations."
            return ("Focus EXCLUSIVELY on ADVANCED topics, optimization techniques, best practices, and common pitfalls. Use terms like 'performance', 'optimization', 'debugging', 'common mistakes'. DO NOT repeat basic or setup content.", avoid)
        else:  # iteration == 3
            avoid = "\n\n🚨 FINAL ITERATION: Use COMPLETELY UNIQUE keywords. Focus on 'alternatives', 'comparison', 'ecosystem', 'related tools', 'migration', 'versus'. Cover the ecosystem and context."
            return ("Focus EXCLUSIVELY on ECOSYSTEM context, alternative solutions, comparisons, related technologies, and migration paths. Use terms like 'alternatives', 'versus', 'comparison', 'ecosystem'. This is the final chance - cover what's missing.", avoid)

    # Fallback (should not reach here)
    return ("Focus on the topic.", "")


async def _research_mode_stream_internal(body: dict, model_name: str = "unknown-model", max_iterations: int = 2, request=None):
    """
    Internal implementation of streaming research mode. Wrapped by research_mode_stream
    for automatic retry with reduced iterations on context overflow.

    See research_mode_stream() docstring for full documentation.

    Args:
        request: FastAPI Request object for disconnect detection (optional)
    """
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 20000)
    temperature = body.get("temperature", 0.7)
    stream_options = body.get("stream_options", None)

    # Generate unique request ID for logging
    request_id = id(request) if request else "unknown"

    # Helper to check if client is still connected
    async def is_client_connected():
        if request is None:
            return True
        return not await request.is_disconnected()

    # Send initial research indicator wrapped in think tags
    yield create_sse_chunk("<think>\n🔬 Researching knowledge base...\n\n", model=model_name)

    # Track all tool calls for summary
    research_summary = []

    # Get the user's original query
    user_query = ""
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

            user_query = content.strip()

            # Strip out any tool outputs that may have been appended by upstream systems
            # Tool outputs typically appear as "\n\nTool " or "Tool server:"
            tool_output_pattern = r'\n\n?Tool\s+(server:)?'
            match = re.search(tool_output_pattern, user_query)
            if match:
                # Take only the content before the tool output
                user_query = user_query[:match.start()].strip()

            # Remove "research" prefix if present
            if user_query.lower().startswith("research"):
                user_query = user_query[8:].strip()
            break

    # STEP 0: Search Serper API for web context (blocking operation)
    yield create_sse_chunk("🌐 **Searching web for current information...**\n\n", model=model_name)

    serper_results = await search_serper(user_query, num_results=10)

    if serper_results['success'] and serper_results['results']:
        yield create_sse_chunk(f"✅ Found {serper_results['total_results']} web results:\n\n", model=model_name)

        # Display Serper results in the stream (titles and URLs only)
        for idx, result in enumerate(serper_results['results'], 1):
            yield create_sse_chunk(f"{idx}. **{result['title']}**\n", model=model_name)
            yield create_sse_chunk(f"   🔗 {result['link']}\n\n", model=model_name)

        yield create_sse_chunk("\n", model=model_name)

        # Format Serper results as context for the LLM
        serper_context = "\n\n--- Web Search Results ---\n"
        for idx, result in enumerate(serper_results['results'], 1):
            serper_context += f"\n{idx}. **{result['title']}**\n"
            serper_context += f"   URL: {result['link']}\n"
            serper_context += f"   {result['snippet']}\n"
        serper_context += "\n--- End of Web Results ---\n\n"

        # Append Serper results to the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = msg["content"] + serper_context
                break
    else:
        error_msg = serper_results.get('error', 'Unknown error')
        yield create_sse_chunk(f"⚠️  Serper search failed: {error_msg}\n\n", model=model_name)

    # Build up context progressively through iterations
    accumulated_context = f"User Query: {user_query}\n\n"

    # Add Serper results to accumulated context (clamp each result at 10,000 chars)
    if serper_results['success'] and serper_results['results']:
        accumulated_context += "=== Web Search Results ===\n"
        for idx, result in enumerate(serper_results['results'], 1):
            # No character limits - use full content
            accumulated_context += f"{idx}. {result['title']}\n   URL: {result['link']}\n   {result['snippet']}\n\n"
        accumulated_context += "\n"

    # Initialize conversation thread for KV cache optimization
    # Single system message used for ALL search query generations
    conversation = [
        {"role": "system", "content": "You are a research assistant that generates diverse, non-overlapping search queries to cover different aspects of a topic. You will be given research context and asked to generate search queries across multiple iterations."}
    ]

    # Add initial context ONCE to start the conversation
    initial_context = f"""User Query: {user_query}

{accumulated_context}

I will ask you to generate search queries across {max_iterations} iteration(s). Each iteration will focus on a different aspect of the topic."""

    conversation.append({"role": "user", "content": initial_context})
    conversation.append({"role": "assistant", "content": "Understood. I'm ready to generate diverse search queries across multiple iterations to thoroughly research this topic."})

    # Do orchestrated research loop
    previous_search_queries = []  # Track queries to avoid duplicates

    for iteration in range(max_iterations):
        # Check if client is still connected before starting iteration
        if not await is_client_connected():
            logger.warning(f"[ReqID: {request_id}] Client disconnected during iteration {iteration + 1}/{max_iterations}, stopping research")
            # Properly close the SSE stream
            yield create_sse_chunk("\n\n_Research stopped (client disconnected)_\n</think>\n", model=model_name)
            yield create_sse_chunk("", finish_reason="stop", model=model_name)
            yield "data: [DONE]\n\n"
            return

        yield create_sse_chunk(f"\n**Iteration {iteration + 1}/{max_iterations}**\n\n", model=model_name)

        # STEP 1: Generate intelligent search_memory query using accumulated context
        # Get dynamic iteration focus based on current iteration and total iterations
        iteration_focus, avoid_queries_instruction = get_iteration_focus(iteration, max_iterations)

        # Add previous queries to avoidance instruction if we have any
        if previous_search_queries and avoid_queries_instruction:
            avoid_queries_instruction += f" Previous queries: {', '.join([f'`{q}`' for q in previous_search_queries])}"

        # Build user message for THIS iteration
        # The conversation already has all previous context, so we just ask for the next query
        user_turn = f"""{iteration_focus}

Generate a focused search query (3-10 words) for the knowledge base. Output ONLY the search query, no explanation.{avoid_queries_instruction}"""

        # Append user turn to conversation
        conversation.append({"role": "user", "content": user_turn})

        search_query_kwargs = {
            "model": model_name,
            "messages": conversation,  # Use full conversation for KV cache
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": False
        }

        search_query_response = vllm.chat.completions.create(**search_query_kwargs)
        current_search_term = search_query_response.choices[0].message.content.strip().replace('"', '').replace("'", "")

        # Append assistant response to conversation
        conversation.append({"role": "assistant", "content": current_search_term})

        # Track this query to avoid duplicates in next iteration
        previous_search_queries.append(current_search_term)

        # Determine search_memory limit based on research depth
        # Standard research (2 iterations): 3 results
        # Deep research (4 iterations): 6 results
        memory_limit = 6 if max_iterations >= 4 else 3

        yield create_sse_chunk(f"🔍 **Tool Call {len(research_summary) + 1}:** search_memory - Query: `{current_search_term}`\n", model=model_name)

        try:
            search_result = await call_mcp_tool("search_memory", {"query": current_search_term, "limit": memory_limit})
            research_summary.append({
                "tool": "search_memory",
                "arguments": {"query": current_search_term, "limit": memory_limit},
                "result": search_result
            })
            accumulated_context += f"=== Search Memory Results (Iteration {iteration + 1}) ===\n{str(search_result)}\n\n"

            # Add search results to conversation as a system observation
            conversation.append({"role": "user", "content": f"Search memory results for '{current_search_term}':\n{str(search_result)}"})
            conversation.append({"role": "assistant", "content": "Noted. I have this information now."})
        except Exception as e:
            yield create_sse_chunk(f"⚠️  Error searching memory: {str(e)}\n", model=model_name)
            search_result = f"Error: {str(e)}"
            research_summary.append({
                "tool": "search_memory",
                "arguments": {"query": current_search_term, "limit": memory_limit},
                "result": search_result
            })

        # STEP 2: Generate 3 URLs - ask in the same conversation thread
        url_request = "Now, based on all the research context we have so far, generate exactly 3 website URLs that would have relevant, current information to help answer the user's query. Output ONLY the URLs, one per line, with no additional text. Choose authoritative sources like official documentation, research papers, or reputable tech sites."

        conversation.append({"role": "user", "content": url_request})

        # Build kwargs for LLM call (no stream_options since stream=False)
        url_kwargs = {
            "model": model_name,
            "messages": conversation,  # Use conversation thread for KV cache
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": False
        }

        url_response = vllm.chat.completions.create(**url_kwargs)

        # Append URL response to conversation
        conversation.append({"role": "assistant", "content": url_response.choices[0].message.content})

        # Extract URLs from response
        url_text = url_response.choices[0].message.content.strip()
        urls = re.findall(r'https?://[^\s<>"]+', url_text)
        urls = urls[:3]  # Take only first 3

        if not urls:
            yield create_sse_chunk(f"⚠️  Could not generate URLs. Skipping crawl step.\n", model=model_name)

        # STEP 3: Crawl each URL and add to accumulated context
        for url in urls:
            # Check if client is still connected before expensive crawl operation
            if not await is_client_connected():
                logger.warning(f"[ReqID: {request_id}] Client disconnected during URL crawling, stopping research")
                # Properly close the SSE stream
                yield create_sse_chunk("\n\n_Research stopped (client disconnected)_\n</think>\n", model=model_name)
                yield create_sse_chunk("", finish_reason="stop", model=model_name)
                yield "data: [DONE]\n\n"
                return

            yield create_sse_chunk(f"🔍 **Tool Call {len(research_summary) + 1}:** crawl_url - URL: `{url}`\n", model=model_name)

            try:
                crawl_result = await call_mcp_tool("crawl_url", {"url": url})
                research_summary.append({
                    "tool": "crawl_url",
                    "arguments": {"url": url},
                    "result": crawl_result
                })
                # Add crawl result to accumulated context (no character limits)
                accumulated_context += f"=== Crawl Result: {url} ===\n{str(crawl_result)}\n\n"

                # Add crawl results to conversation
                conversation.append({"role": "user", "content": f"Crawled content from {url}:\n{str(crawl_result)}"})
                conversation.append({"role": "assistant", "content": "Acknowledged. I've reviewed this content."})
            except Exception as e:
                yield create_sse_chunk(f"⚠️  Error crawling {url}: {str(e)}\n", model=model_name)
                research_summary.append({
                    "tool": "crawl_url",
                    "arguments": {"url": url},
                    "result": f"Error: {str(e)}"
                })

        # STEP 4: Generate 1 distinct Serper search query and perform search
        yield create_sse_chunk(f"\n🌐 **Generating additional web search for iteration {iteration + 1}...**\n\n", model=model_name)

        # Ask LLM to generate 1 distinct search query
        serper_query_request = f"""Based on all the research context so far, generate exactly 1 distinct web search query that would uncover NEW information not yet covered. This should be COMPLETELY DIFFERENT from the initial query "{user_query}" and all previous searches.

Focus on a related but distinct angle: specific use cases, comparisons, problems, solutions, alternatives, or deeper technical aspects.

Output ONLY the search query, no additional text or numbering."""

        conversation.append({"role": "user", "content": serper_query_request})

        serper_query_kwargs = {
            "model": model_name,
            "messages": conversation,
            "max_tokens": 50,
            "temperature": 0.8,  # Higher temperature for more diversity
            "stream": False
        }

        serper_query_response = vllm.chat.completions.create(**serper_query_kwargs)
        serper_query = serper_query_response.choices[0].message.content.strip()

        # Clean up the query (remove quotes, numbering, etc.)
        serper_query = re.sub(r'^\d+[\.\)]\s*', '', serper_query)
        serper_query = serper_query.replace('"', '').replace("'", "")

        # Append response to conversation
        conversation.append({"role": "assistant", "content": serper_query})

        if not serper_query:
            yield create_sse_chunk(f"⚠️  Could not generate Serper query. Skipping additional web search.\n\n", model=model_name)
        else:
            # Check if client is still connected
            if not await is_client_connected():
                logger.warning(f"[ReqID: {request_id}] Client disconnected during Serper search, stopping research")
                yield create_sse_chunk("\n\n_Research stopped (client disconnected)_\n</think>\n", model=model_name)
                yield create_sse_chunk("", finish_reason="stop", model=model_name)
                yield "data: [DONE]\n\n"
                return

            yield create_sse_chunk(f"🔍 **Web Search:** `{serper_query}`\n", model=model_name)

            try:
                serper_result = await search_serper(serper_query, num_results=5)  # 5 results per query

                if serper_result['success'] and serper_result['results']:
                    research_summary.append({
                        "tool": "serper_search",
                        "arguments": {"query": serper_query, "num_results": 5},
                        "result": serper_result
                    })

                    # Add results to accumulated context
                    accumulated_context += f"=== Serper Search Results (Iteration {iteration + 1}): {serper_query} ===\n"
                    for res_idx, result in enumerate(serper_result['results'], 1):
                        accumulated_context += f"{res_idx}. {result['title']}\n   URL: {result['link']}\n   {result['snippet']}\n\n"
                    accumulated_context += "\n"

                    # Display results in stream
                    for res_idx, result in enumerate(serper_result['results'], 1):
                        yield create_sse_chunk(f"   {res_idx}. {result['title'][:80]}...\n", model=model_name)

                    # Add to conversation
                    conversation.append({"role": "user", "content": f"Web search results for '{serper_query}':\n{str(serper_result['results'])}"})
                    conversation.append({"role": "assistant", "content": "Noted. I have these web search results."})

                    yield create_sse_chunk("\n", model=model_name)
                else:
                    error_msg = serper_result.get('error', 'No results')
                    yield create_sse_chunk(f"   ⚠️  Search failed: {error_msg}\n\n", model=model_name)
                    research_summary.append({
                        "tool": "serper_search",
                        "arguments": {"query": serper_query, "num_results": 5},
                        "result": f"Error: {error_msg}"
                    })
            except Exception as e:
                yield create_sse_chunk(f"   ⚠️  Error: {str(e)}\n\n", model=model_name)
                research_summary.append({
                    "tool": "serper_search",
                    "arguments": {"query": serper_query, "num_results": 5},
                    "result": f"Error: {str(e)}"
                })

    # All iterations complete - close think tag and generate final answer
    logger.info(f"Research complete after {max_iterations} iteration(s)")
    yield create_sse_chunk("\n</think>\n\n", model=model_name)

    # Generate final answer with ALL accumulated context
    final_prompt = [
        {"role": "system", "content": "You are a helpful assistant. Provide a comprehensive answer based on the research data gathered. Do NOT output any tool calls - just provide your final answer directly."},
        {"role": "user", "content": f"{accumulated_context}\n\nBased on all the research above, provide a comprehensive answer to the user's original query."}
    ]

    # Build kwargs for final LLM streaming call
    final_kwargs = {
        "model": model_name,
        "messages": final_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    if stream_options:
        final_kwargs["stream_options"] = stream_options

    stream_response = vllm.chat.completions.create(**final_kwargs)

    usage_data = None
    chunk_count = 0
    for chunk in stream_response:
        chunk_count += 1

        # Debug: Log chunk structure
        logger.debug(f"Chunk {chunk_count}: choices={len(chunk.choices) if chunk.choices else 0}, has_usage={hasattr(chunk, 'usage')}, usage={chunk.usage if hasattr(chunk, 'usage') else 'N/A'}")

        # When stream_options.include_usage=true, vLLM sends a final chunk with only usage data
        # and an empty choices array. We need to handle this case.
        if not chunk.choices or len(chunk.choices) == 0:
            # This is a usage-only chunk, just capture the usage data
            if hasattr(chunk, 'usage') and chunk.usage:
                logger.debug(f"Captured usage from empty-choice chunk: {chunk.usage}")
                usage_data = chunk.usage
            continue

        if chunk.choices[0].delta.content:
            yield create_sse_chunk(chunk.choices[0].delta.content, model=model_name)

        # Capture usage data if present
        if hasattr(chunk, 'usage') and chunk.usage:
            logger.debug(f"Captured usage from regular chunk: {chunk.usage}")
            usage_data = chunk.usage

        if chunk.choices[0].finish_reason:
            # Before finishing, append research summary
            if research_summary:
                # Extract search params
                search_params = []
                for call in research_summary:
                    if 'query' in call['arguments']:
                        search_params.append(call['arguments']['query'])
                    elif 'url' in call['arguments']:
                        search_params.append(call['arguments']['url'])

                # Extract URLs from results
                urls = extract_urls_from_results(research_summary)

                summary_text = f"\n\n---\n**Research Summary:**\n"
                summary_text += f"**Tool Calls:** {len(research_summary)}\n"
                summary_text += f"**Search Params:** [{', '.join(search_params)}]\n"

                if urls:
                    summary_text += f"\n**Further Reading:**\n"
                    for idx, url in enumerate(urls, 1):
                        summary_text += f"{idx}. {url}\n"

                yield create_sse_chunk(summary_text, model=model_name)

            # Send finish_reason chunk (without usage data)
            yield create_sse_chunk("", finish_reason=chunk.choices[0].finish_reason, model=model_name)

            # Send usage data as separate chunk after finish_reason (OpenAI format)
            if stream_options and stream_options.get("include_usage"):
                if usage_data:
                    logger.warning(f"[ReqID: {request_id}] Sending usage data: {usage_data}")
                    usage_dict = usage_data.model_dump() if hasattr(usage_data, 'model_dump') else usage_data
                    yield create_sse_chunk("", usage=usage_dict, model=model_name)
                else:
                    logger.warning(f"[ReqID: {request_id}] stream_options.include_usage=true but no usage_data captured!")
            else:
                logger.warning(f"[ReqID: {request_id}] Not sending usage: stream_options={stream_options}")

            # Send final DONE marker
            yield "data: [DONE]\n\n"


async def research_mode_stream(body: dict, model_name: str = "unknown-model", max_iterations: int = 2, request=None):
    """
    Execute comprehensive streaming research mode with progressive context accumulation.

    This is the core research function that performs deep, multi-iteration research by:
    1. Calling Serper API for 10 web search results (initial)
    2. Running 2-4 research iterations (configurable), each consisting of:
       - LLM generates intelligent search_memory query (avoiding previous queries)
       - Searches knowledge base (limit: 3 for standard, 6 for deep research)
       - LLM generates 3 relevant URLs based on accumulated context
       - Crawls all 3 URLs
       - LLM generates 1 distinct Serper search query (different from initial search)
       - Performs 1 Serper search (5 results)
       - Adds all results to accumulated context (no character limits)
    3. Generates final answer using massive accumulated context (potentially 100K+ tokens)
    4. Wraps all research progress in <think></think> tags for client-side formatting
    5. Includes research summary with tool call count and URLs at the end

    Context Accumulation Strategy:
        - Standard research (2 iterations): search_memory (3 results) + 3 crawls (full) + 1 Serper search (5 results) per iteration
        - Deep research (4 iterations): search_memory (6 results) + 3 crawls (full) + 1 Serper search (5 results) per iteration
        - All results accumulate progressively (no truncation)
        - Final: All accumulated context fed to LLM for comprehensive answer

    Search Query Diversity (enforced with STRONG language):
        - 2 iterations: Main concepts → Practical implementation
        - 4 iterations: Main → Implementation → Advanced & troubleshooting → Ecosystem

    Auto-Retry on Context Overflow:
        - If context exceeds model's max (200K tokens), automatically retries with fewer iterations
        - 4 iterations → 2 iterations
        - Client receives notification about restart via streaming response

    Client Disconnect Handling:
        - Checks for client disconnection before each iteration and URL crawl
        - Gracefully stops research when client disconnects (e.g., user presses stop)
        - Prevents wasted resources on abandoned requests

    Args:
        body (dict): Request body containing:
            - messages (list): Chat messages
            - max_tokens (int, optional): Max tokens for final answer. Default: 2000
            - temperature (float, optional): Sampling temperature. Default: 0.7
            - stream_options (dict, optional): Streaming options like include_usage
        model_name (str, optional): Name of the model being used. Defaults to "unknown-model"
        max_iterations (int, optional): Number of research iterations. Can be 2 or 4. Default: 2
        request: FastAPI Request object for disconnect detection (optional)

    Yields:
        str: SSE-formatted chunks containing:
            - Research progress updates (in <think> tags)
            - Serper search results display
            - Tool call notifications
            - Final answer content (outside <think> tags)
            - Usage statistics (if stream_options.include_usage=true)

    Used by:
        - autonomous_chat(): Called when research keyword detected and stream=True
    """
    try:
        # Try executing research with requested iterations
        async for chunk in _research_mode_stream_internal(body, model_name, max_iterations, request):
            yield chunk
    except (BadRequestError, Exception) as e:
        error_msg = str(e)

        # Check if it's a context overflow error (vLLM returns BadRequestError with message about context length)
        is_context_overflow = (
            "maximum context length" in error_msg.lower() or
            ("context" in error_msg.lower() and "token" in error_msg.lower() and "exceed" in error_msg.lower()) or
            "too many tokens" in error_msg.lower()
        )

        if is_context_overflow:
            # Determine next iteration count
            if max_iterations == 4:
                # Retry with 2 iterations
                next_iterations = 2
                notification = f"\n\n⚠️ **Research data exceeded context limit ({error_msg})**\n"
                notification += f"🔄 **Restarting research with {next_iterations} iteration(s) instead of {max_iterations}**\n\n"
                yield create_sse_chunk(notification, model=model_name)

                logger.warning(f"Context overflow detected. Retrying with {next_iterations} iterations instead of {max_iterations}")

                # Retry with fewer iterations
                async for chunk in research_mode_stream(body, model_name, next_iterations, request):
                    yield chunk
            else:
                # Already at 2 iterations and still failing - can't reduce further
                error_notification = f"\n\n❌ **Fatal Error: Context overflow even with minimal research (2 iterations)**\n"
                error_notification += f"Error: {error_msg}\n\n"
                error_notification += "Please try a more specific query or shorter research topic.\n"
                yield create_sse_chunk(error_notification, model=model_name)
                yield create_sse_chunk("", finish_reason="length", model=model_name)
                yield "data: [DONE]\n\n"
        else:
            # Non-context-overflow error - propagate it
            raise


async def research_mode_sync(body: dict, model_name: str = "unknown-model", max_iterations: int = 2):
    """Non-streaming research mode"""
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2000)
    temperature = body.get("temperature", 0.7)

    # Inject tool instructions as system message
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": TOOL_SYSTEM_PROMPT})
    else:
        # Append to existing system message
        messages[0]["content"] += "\n\n" + TOOL_SYSTEM_PROMPT

    research_summary = []

    for iteration in range(max_iterations):
        response = vllm.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        assistant_message = response.choices[0].message.content
        tool_calls = extract_tool_calls(assistant_message)

        if not tool_calls:
            # Replace tool system prompt with final answer instructions
            final_messages = messages.copy()
            if final_messages and final_messages[0].get("role") == "system":
                final_messages[0]["content"] = "You are a helpful assistant. Provide a comprehensive answer based on the research data gathered. Do NOT output any tool calls - just provide your final answer directly."

            # Generate clean final answer without tool calls
            final_response = vllm.chat.completions.create(
                model=model_name,
                messages=final_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Build research summary
            summary_text = ""
            if research_summary:
                # Extract search params
                search_params = []
                for call in research_summary:
                    if 'query' in call['arguments']:
                        search_params.append(call['arguments']['query'])
                    elif 'url' in call['arguments']:
                        search_params.append(call['arguments']['url'])

                # Extract URLs from results
                urls = extract_urls_from_results(research_summary)

                summary_text = f"\n\n---\n**Research Summary:**\n"
                summary_text += f"**Tool Calls:** {len(research_summary)}\n"
                summary_text += f"**Search Params:** [{', '.join(search_params)}]\n"

                if urls:
                    summary_text += f"\n**Further Reading:**\n"
                    for idx, url in enumerate(urls, 1):
                        summary_text += f"{idx}. {url}\n"

            final_answer = final_response.choices[0].message.content + summary_text
            return {
                "id": final_response.id,
                "object": "chat.completion",
                "created": final_response.created,
                "model": final_response.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_answer
                    },
                    "finish_reason": final_response.choices[0].finish_reason
                }],
                "usage": final_response.usage.model_dump()
            }

        messages.append({"role": "assistant", "content": assistant_message})

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            try:
                result = await call_mcp_tool(tool_name, arguments)

                # Track tool call with result for summary
                research_summary.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })

                messages.append({
                    "role": "tool",
                    "content": f"<tool_response>\n{result}\n</tool_response>"
                })
            except Exception as e:
                # Track failed tool call
                research_summary.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": f"Error: {str(e)}"
                })

                messages.append({
                    "role": "tool",
                    "content": f"<tool_response>\nError: {str(e)}\n</tool_response>"
                })

    # Max iterations reached - build summary
    summary_text = ""
    if research_summary:
        # Extract search params
        search_params = []
        for call in research_summary:
            if 'query' in call['arguments']:
                search_params.append(call['arguments']['query'])
            elif 'url' in call['arguments']:
                search_params.append(call['arguments']['url'])

        # Extract URLs from results
        urls = extract_urls_from_results(research_summary)

        summary_text = f"\n\n---\n**Research Summary:**\n"
        summary_text += f"**Tool Calls:** {len(research_summary)}\n"
        summary_text += f"**Search Params:** [{', '.join(search_params)}]\n"

        if urls:
            summary_text += f"\n**Further Reading:**\n"
            for idx, url in enumerate(urls, 1):
                summary_text += f"{idx}. {url}\n"

    return {
        "id": "max_iter",
        "object": "chat.completion",
        "created": 0,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "⚠️ Research depth limit reached. Providing answer based on gathered information." + summary_text
            },
            "finish_reason": "length"
        }]
    }
