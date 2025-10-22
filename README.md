# robaiproxy

**Intelligent API Gateway and Research Orchestration Proxy**

A sophisticated FastAPI-based proxy service that sits between clients and a vLLM language model backend, providing transparent request forwarding and multi-iteration web research capabilities. Part of the robaitools ecosystem.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [System Design](#system-design)
  - [Service Dependencies](#service-dependencies)
  - [Request Flow](#request-flow)
- [Components](#components)
  - [Core Modules](#core-modules)
  - [Utilities](#utilities)
- [Operating Modes](#operating-modes)
  - [Passthrough Mode](#passthrough-mode)
  - [Research Mode](#research-mode)
- [API Endpoints](#api-endpoints)
- [Research Workflow](#research-workflow)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Queue Management](#queue-management)
- [Health Monitoring](#health-monitoring)
- [Power Management](#power-management)
- [Integration Points](#integration-points)
- [Design Patterns](#design-patterns)
- [Security](#security)
- [Limitations](#limitations)
- [Statistics](#statistics)

## Overview

**robaiproxy** is an intelligent API gateway that provides two primary modes of operation:

1. **Passthrough Mode**: Transparent forwarding of standard chat completions and API requests to vLLM
2. **Research Mode**: Multi-iteration, context-accumulating web research with integrated knowledge base searches

The service acts as a central orchestration hub coordinating between multiple backends (vLLM, MCP RAG server, Serper web search API) to provide comprehensive research capabilities while maintaining compatibility with the OpenAI chat completions API.

### What Makes This Different

Unlike traditional reverse proxies or API gateways, robaiproxy:

- **Intelligent Request Routing**: Detects research intent and routes to appropriate handler
- **Multi-Iteration Research**: Performs 2-4 research iterations with progressive context accumulation
- **Multi-Source Information Gathering**: Combines web search, knowledge base queries, and URL crawling
- **Context Accumulation**: Preserves all research context (potentially 100K+ tokens) for final answer
- **Auto-Retry Logic**: Automatically reduces iteration count on context overflow
- **Queue Management**: Limits concurrent research requests with user-friendly status messages
- **Power Management**: GPU performance level automation based on API activity

## Key Features

### Dual Mode Operation

- **Passthrough Mode**: Minimal latency, direct streaming to vLLM for regular chat
- **Research Mode**: Multi-iteration research with web search and knowledge base integration

### Research Capabilities

- **Intelligent Query Generation**: LLM generates diverse search queries to avoid repetition
- **Multi-Source Data Collection**:
  - Web search via Serper API (10 initial + 5 per iteration)
  - Knowledge base search via MCP server (3-6 results per iteration)
  - Fresh URL crawling (3 URLs per iteration)
- **Progressive Context Accumulation**: All results added without truncation
- **Auto-Retry on Context Overflow**: Automatically reduces iterations if context limit exceeded
- **Client Disconnect Detection**: Gracefully stops research on abandoned requests

### Queue Management

- **Concurrent Request Limiting**:
  - Standard research: Max 3 concurrent
  - Deep research: Max 1 concurrent
- **User-Friendly Status Messages**: Queue position and availability communicated via streaming
- **Health Checks on Queue Full**: Verifies backend availability when waiting

### Power Management

- **Activity-Based GPU Control**: Automatically adjusts GPU performance levels
- **Idle Timeout**: Reduces power after 2 minutes of inactivity
- **Automatic Recovery**: Returns to auto mode on new activity

### Health Monitoring

- **Multi-Service Health Checks**: Monitors all dependent services
- **Docker Container Status**: Verifies container presence and running state
- **HTTP Endpoint Checks**: Tests service availability
- **Critical Service Tracking**: Overall status based on critical dependencies

## Architecture

### System Design

**Intelligent Request Router with Research Orchestrator**

```
Client Request (OpenAI-compatible)
    ↓
[Model Availability Check & Wait]
    ↓
[Multimodal Content Detection]
    ↓
[Research Mode Detection]
    ├─→ "research" keyword detected
    │   ↓
    │   [Queue Management (Semaphore)]
    │   ↓
    │   [Multi-iteration Research Mode]
    │   ├─→ Initial Serper Search (10 results)
    │   ├─→ 2-4 Research Iterations, each:
    │   │   ├─→ Generate search query
    │   │   ├─→ Search knowledge base (3-6 results)
    │   │   ├─→ Generate URLs
    │   │   ├─→ Crawl URLs (3 URLs)
    │   │   └─→ Generate + execute Serper search (5 results)
    │   └─→ Generate final answer with accumulated context
    │
    └─→ No "research" keyword
        ↓
        [Direct Passthrough to vLLM]
```

### Service Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                   robaiproxy                            │
│                   Port 8079                             │
│  ┌───────────────────────────────────────────────────┐ │
│  │ requestProxy.py  │  researchAgent.py              │ │
│  │ connectionManager │ powerManager                   │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │         │         │         │         │
         │         │         │         │         │
┌────────┴─┐  ┌────┴────┐  ┌┴────────┐ ┌───────┴────┐  ┌─────────┐
│  vLLM    │  │   MCP   │  │ Serper  │ │ robairagapi│  │  rocm-  │
│  Backend │  │   RAG   │  │   API   │ │            │  │  smi    │
│          │  │  Server │  │ (HTTPS) │ │ Port 8081  │  │  (GPU)  │
│ Port 8078│  │Port 8080│  └─────────┘ └────────────┘  └─────────┘
└────┬─────┘  └────┬────┘
     │             │
     │             └─────────┬───────────┬────────────┐
┌────┴─────┐  ┌─────────────┴───┐  ┌────┴─────┐  ┌──┴────────┐
│ Qwen3    │  │ robaimodeltools │  │  Neo4j   │  │ crawl4ai  │
│  30B     │  │                 │  │ Port 7474│  │  (Docker) │
│ (Model)  │  │                 │  │          │  │           │
└──────────┘  └─────────────────┘  └──────────┘  └───────────┘
```

### Request Flow

**Passthrough Mode:**
```
Client → robaiproxy → vLLM → robaiproxy → Client
                       (direct streaming)
```

**Research Mode:**
```
Client → robaiproxy
            ↓
         Research Orchestrator
            ├→ Initial Serper Search (10 results)
            ├→ Iteration 1:
            │   ├→ vLLM: Generate search query
            │   ├→ MCP: Search knowledge base
            │   ├→ vLLM: Generate URLs
            │   ├→ MCP: Crawl URLs (3)
            │   ├→ vLLM: Generate Serper query
            │   └→ Serper: Web search (5)
            ├→ Iteration 2-4 (same pattern)
            └→ Final Answer:
                └→ vLLM: Generate with accumulated context
                   ↓
         Streaming Response → Client
```

## Components

### Core Modules

#### requestProxy.py

**Purpose**: Main FastAPI application with request routing and orchestration

**Size**: 975 lines

**Key Responsibilities**:
- HTTP request handling and routing
- Research mode detection
- Model availability monitoring
- Health check coordination
- Queue management
- Request forwarding to vLLM

**Main Functions**:

| Function | Purpose |
|----------|---------|
| `autonomous_chat()` | Main POST /v1/chat/completions endpoint |
| `detect_research_mode()` | Analyzes messages for research keywords |
| `is_multimodal()` | Detects images/audio in messages |
| `passthrough_stream()` | Transparent streaming to vLLM |
| `passthrough_sync()` | Transparent non-streaming to vLLM |
| `research_with_queue_management()` | Research with queue status messages |
| `wait_for_model()` | Waits for model availability on startup |
| `model_name_manager()` | Background task for model tracking |
| `check_research_health()` | Multi-service health verification |

**Global State**:
```python
current_model_name: Optional[str]           # Cached model name
current_model_data: Optional[dict]          # Full model response
model_fetch_task: Optional[asyncio.Task]    # Background monitor
research_semaphore: Semaphore               # Standard research (max 3)
deep_research_semaphore: Semaphore          # Deep research (max 1)
```

#### researchAgent.py

**Purpose**: Multi-iteration research workflow with context accumulation

**Size**: 1,026 lines

**Key Responsibilities**:
- Research iteration execution
- Tool call extraction and execution
- Context accumulation
- SSE (Server-Sent Events) formatting
- Auto-retry on context overflow

**Main Functions**:

| Function | Purpose |
|----------|---------|
| `search_serper()` | Query Serper API for web results |
| `call_mcp_tool()` | Execute MCP tools (search, crawl) |
| `extract_tool_calls()` | Parse `<tool_call>` XML tags |
| `extract_urls_from_results()` | Extract URLs from LLM response |
| `create_sse_chunk()` | Format OpenAI-compatible SSE chunk |
| `get_iteration_focus()` | Get research focus for iteration |
| `_research_mode_stream_internal()` | Core streaming research logic |
| `research_mode_stream()` | Wrapper with auto-retry |
| `research_mode_sync()` | Non-streaming research mode |

**Research Iterations**:
- **Standard (2 iterations)**:
  1. Main concepts and understanding
  2. Practical implementation details

- **Deep (4 iterations)**:
  1. Main concepts and understanding
  2. Practical implementation details
  3. Advanced features and troubleshooting
  4. Ecosystem, alternatives, and comparisons

#### config.py

**Purpose**: Environment-based configuration with validation

**Size**: 180 lines

**Configuration Categories**:

```python
class Config:
    # vLLM Backend
    VLLM_BASE_URL: str              # Default: http://localhost:8078/v1
    VLLM_BACKEND_URL: str           # Derived: VLLM_BASE_URL without /v1
    VLLM_TIMEOUT: int               # Default: 300 seconds

    # MCP Server
    REST_API_URL: str               # Default: http://localhost:8080/api/v1
    REST_API_KEY: str               # Required for authentication
    MCP_TIMEOUT: int                # Default: 60 seconds

    # External APIs
    SERPER_API_KEY: str             # Required for web search
    SERPER_TIMEOUT: int             # Default: 30 seconds

    # Research Limits
    MAX_STANDARD_RESEARCH: int      # Default: 3
    MAX_DEEP_RESEARCH: int          # Default: 1

    # Server Config
    HOST: str                       # Default: 0.0.0.0
    PORT: int                       # Default: 8079
    LOG_LEVEL: str                  # Default: INFO

    # Feature Flags
    AUTO_DETECT_MODEL: bool         # Default: true
    MODEL_POLL_INTERVAL: int        # Default: 2 seconds
```

**Methods**:
- `validate()`: Check for missing critical API keys
- `display()`: Print configuration with secrets masked

**Logging**:
- Dual output: Console (INFO+) + File (DEBUG+)
- File: `proxy.log` in module directory
- Format: `%(levelname)-8s | %(message)s`

#### connectionManager.py

**Purpose**: Track active API requests for monitoring and power management

**Size**: 230 lines

**Key Class**: `ConnectionManager`

**Tracked Information**:
```python
{
    "request_id": str,              # Unique identifier
    "endpoint": str,                # API endpoint path
    "method": str,                  # HTTP method
    "client_ip": str,               # Client IP address
    "started_at": datetime,         # Start timestamp
    "status": str,                  # active/completed/failed
    "model": Optional[str],         # Model name if applicable
    "is_research": bool             # Research mode flag
}
```

**Methods**:
- `add_connection()`: Register new connection
- `remove_connection()`: Deregister connection
- `get_connection_count()`: Get active count
- `get_active_connections()`: Filter by status
- `set_activity_callback()`: Register activity handler
- `get_connection_status()`: Comprehensive status

**Activity Tracking**:
- Fires callback: 0 → 1+ connections (activity started)
- Fires callback: 1+ → 0 connections (activity stopped)
- Excludes: `/models`, `/health`, `/metrics`

#### powerManager.py

**Purpose**: GPU power management based on API activity

**Size**: 286 lines

**Key Classes**:

```python
class PowerLevel(str, Enum):
    AUTO = "auto"           # Default AMD performance mode
    LOW = "low"             # Minimal power consumption
    HIGH = "high"           # Maximum performance
    MANUAL = "manual"       # User-controlled

class PowerManager:
    idle_timeout_seconds: int        # Default: 120 (2 minutes)
    _current_level: Optional[PowerLevel]
    _command_lock: asyncio.Lock
    _idle_timer_task: Optional[asyncio.Task]
```

**Power Logic**:
```
No Activity (0 connections)
    ↓
Start Idle Timer (2 minutes)
    ↓
Timer Expires → Set "low" performance
    ↓
Activity Detected (1+ connections)
    ↓
Cancel Timer + Set "auto" performance
```

**Methods**:
- `set_power_level()`: Execute `rocm-smi --setperflevel <level>`
- `on_activity_change()`: Activity callback handler
- `start()`: Initialize and register with ConnectionManager
- `stop()`: Cleanup and restore auto mode
- `get_status()`: Current power status

**Important**: Currently disabled in production (commented out in requestProxy.py)

### Utilities

#### check_connections.py

**Purpose**: Display active connection statistics

**Usage**:
```bash
python check_connections.py
```

**Output**:
```
Active Connections: 3

Connection 1:
  Request ID: abc123
  Endpoint: /v1/chat/completions
  Method: POST
  Client IP: 192.168.1.100
  Started: 2024-01-15 10:30:00
  Status: active
  Model: Qwen3-30B
  Research Mode: True
```

#### check_power.py

**Purpose**: Display GPU power manager status

**Usage**:
```bash
python check_power.py
```

**Output**:
```
Power Manager Status:
  Current Level: auto
  Idle Timeout: 120 seconds
  Idle Duration: 0 minutes
  Running: true
```

#### test_endpoints.py

**Purpose**: Comprehensive endpoint testing suite

**Size**: 313 lines

**Test Categories**:
- Model availability checks
- Passthrough mode tests
- Standard research mode tests
- Deep research mode tests
- Multimodal request tests
- Health endpoint tests
- Error handling tests

**Usage**:
```bash
python test_endpoints.py
```

## Operating Modes

### Passthrough Mode

**When Activated**:
- No "research" keyword in user message, OR
- Multimodal content detected (images, audio)

**Behavior**:
- Transparent forwarding to vLLM backend
- Preserves all request parameters
- Supports streaming and non-streaming
- Minimal latency overhead

**Use Cases**:
- Regular chat conversations
- Code generation
- Q&A without research
- Multimodal requests (images, audio)

**Example Request**:
```json
{
  "model": "Qwen3-30B",
  "messages": [
    {"role": "user", "content": "Write a Python function to sort a list"}
  ],
  "stream": true
}
```

**Response**: Direct streaming from vLLM

### Research Mode

**When Activated**:
- User message starts with "research" keyword

**Research Types**:

**Standard Research (2 iterations)**:
```json
{
  "messages": [
    {"role": "user", "content": "research kubernetes networking"}
  ]
}
```

**Deep Research (4 iterations)**:
- Triggered by modifiers: thoroughly, carefully, all, comprehensively, comprehensive, deep, deeply, detailed, extensive, extensively

```json
{
  "messages": [
    {"role": "user", "content": "research thoroughly kubernetes networking"}
  ]
}
```

**Behavior**:
- Initial Serper web search (10 results)
- 2-4 research iterations with knowledge base + web + crawling
- Progressive context accumulation (no truncation)
- Final answer generated with full context
- Streaming progress updates to client

**Use Cases**:
- Comprehensive research tasks
- Current/recent information gathering
- Multi-source data collection
- In-depth topic exploration

## API Endpoints

### Main Endpoints

#### POST /v1/chat/completions

Main chat endpoint with intelligent routing.

**Request**:
```json
{
  "model": "Qwen3-30B",
  "messages": [
    {"role": "user", "content": "research python async programming"}
  ],
  "stream": true,
  "max_tokens": 2000,
  "temperature": 0.7,
  "stream_options": {"include_usage": true}
}
```

**Response (Streaming)**:
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen3-30B","choices":[{"index":0,"delta":{"role":"assistant","content":"Starting"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen3-30B","choices":[{"index":0,"delta":{"content":" research"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen3-30B","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":150,"completion_tokens":500,"total_tokens":650}}

data: [DONE]
```

**Response (Non-Streaming)**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen3-30B",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Python async programming..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "total_tokens": 650
  }
}
```

#### GET /health

Comprehensive multi-service health check.

**Response**:
```json
{
  "status": "healthy",
  "service": "request-proxy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "vllm-qwen3": {
      "model_loaded": true,
      "model_name": "Qwen3-30B",
      "status": "healthy",
      "container_status": "running",
      "available": true
    },
    "kg-service": {
      "status": "healthy",
      "container_status": "running",
      "available": true
    },
    "mcprag-server": {
      "status": "healthy",
      "container_status": "running",
      "available": true
    },
    "crawl4ai": {
      "container_status": "running",
      "available": true
    },
    "neo4j-kg": {
      "status": "healthy",
      "container_status": "running",
      "available": true
    },
    "open-webui": {
      "status": "healthy",
      "container_status": "running",
      "available": true
    }
  }
}
```

**Status Values**:
- `healthy`: All critical services operational
- `degraded`: Some non-critical services unavailable
- `unhealthy`: Critical services unavailable

**Critical Services**: vllm-qwen3, kg-service, neo4j-kg

#### GET /v1/models

List available models (cached from vLLM).

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-30B",
      "object": "model",
      "created": 1234567890,
      "owned_by": "organization"
    }
  ]
}
```

#### GET /openapi.json

Proxy RAG API OpenAPI schema (forwarded from robairagapi:8081).

### Catch-All Routing

**Pattern**: `GET/POST /{path:path}`

**Routing Logic**:
```python
if path in vllm_endpoints:
    forward_to(VLLM_BACKEND_URL)
else:
    forward_to("http://localhost:8081")  # robairagapi
```

**vLLM Endpoints**:
- `/v1/completions`
- `/v1/embeddings`
- `/v1/chat/completions`
- `/tokenize`
- `/detokenize`
- `/v1/models`

## Research Workflow

### Detailed Research Flow

**Phase 1: Initial Search**
```
1. Detect "research" keyword
2. Check queue availability
3. Perform initial Serper search (10 results)
4. Add results to accumulated_context
```

**Phase 2: Iteration Loop** (2-4 times)
```
For each iteration:
  1. Get iteration focus instruction
  2. Generate search query (LLM)
     - Tool call: search_memory
     - Arguments: query generated by LLM
  3. Search knowledge base (MCP)
     - Returns: 3-6 results with content
  4. Add results to accumulated_context
  5. Generate URLs (LLM)
     - Tool call: suggest_urls
     - Arguments: None (uses context)
  6. Crawl URLs (MCP)
     - Tool call: crawl_url (3 URLs)
     - Returns: Crawled content
  7. Add crawled content to accumulated_context
  8. Generate Serper query (LLM)
     - Tool call: search_web
     - Arguments: query generated by LLM
  9. Execute Serper search (5 results)
  10. Add results to accumulated_context
  11. Check for client disconnect
  12. Continue to next iteration
```

**Phase 3: Final Answer**
```
1. Create final prompt with accumulated_context
2. Generate comprehensive answer (LLM)
3. Stream answer to client
4. Send [DONE] marker
```

### Iteration Focus

**Iteration 1: Main Concepts**
```
Focus on: Core concepts, fundamental understanding, key terminology
Avoid: Implementation details, advanced topics
```

**Iteration 2: Practical Implementation**
```
Focus on: Practical examples, implementation details, common patterns
Avoid: Advanced features, troubleshooting
```

**Iteration 3: Advanced Features** (Deep research only)
```
Focus on: Advanced features, edge cases, troubleshooting
Avoid: Basics, simple examples
```

**Iteration 4: Ecosystem** (Deep research only)
```
Focus on: Related tools, alternatives, ecosystem, best practices
Avoid: Basic concepts already covered
```

### Context Accumulation Example

```
Accumulated Context:
==================

[Initial Serper Search - 10 results]
1. Title: Kubernetes Networking Guide
   URL: https://kubernetes.io/docs/concepts/networking/
   Snippet: Kubernetes networking model...

2. Title: Understanding Kubernetes Services
   ...

[Iteration 1 - Knowledge Base Search - 4 results]
1. URL: https://example.com/k8s-networking
   Content: Full text of stored knowledge...

[Iteration 1 - Crawled URLs - 3 URLs]
1. URL: https://kubernetes.io/docs/concepts/networking/
   Content: Full page content (10,000+ chars)...

[Iteration 1 - Serper Search - 5 results]
1. Title: Kubernetes Network Policies
   ...

[Iteration 2 - Knowledge Base Search - 3 results]
...

[Final Context Size: 80,000+ characters]
```

### Auto-Retry on Context Overflow

```
Attempt 1: Deep research (4 iterations)
    ↓
[Accumulated context: 120,000+ chars]
    ↓
LLM Error: "maximum context length exceeded"
    ↓
Stream to client: "Context overflow detected. Restarting with 2 iterations..."
    ↓
Attempt 2: Reduced research (2 iterations)
    ↓
[Accumulated context: 60,000 chars]
    ↓
Success: Generate final answer
```

## Configuration

### Environment Variables

Create a `.env` file in the robaiproxy directory:

```bash
# vLLM Backend Configuration
VLLM_BASE_URL=http://localhost:8078/v1
VLLM_TIMEOUT=300

# MCP RAG Server Configuration
REST_API_URL=http://localhost:8080/api/v1
REST_API_KEY=your_api_key_here
MCP_TIMEOUT=60

# External APIs
SERPER_API_KEY=your_serper_api_key_here
SERPER_TIMEOUT=30

# Research Queue Limits
MAX_STANDARD_RESEARCH=3
MAX_DEEP_RESEARCH=1

# Server Configuration
HOST=0.0.0.0
PORT=8079
LOG_LEVEL=INFO

# Feature Flags
AUTO_DETECT_MODEL=true
MODEL_POLL_INTERVAL=2
```

### Configuration Validation

The config module validates critical settings on startup:

```python
# Required (will raise error if missing)
- REST_API_KEY
- SERPER_API_KEY

# Optional (uses defaults)
- All other settings
```

### Logging Configuration

**Console Logging**:
- Level: INFO or configured LOG_LEVEL
- Format: `%(levelname)-8s | %(message)s`
- Output: stdout

**File Logging**:
- Level: DEBUG
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- File: `proxy.log` in module directory
- Rotation: Manual (no auto-rotation)

## Installation

### Prerequisites

- Python 3.10+
- Docker (for dependent services)
- vLLM backend running on port 8078
- MCP RAG server running on port 8080
- Serper API key (https://serper.dev)

### Install Dependencies

```bash
cd robaiproxy

# Install Python packages
pip install -r requirements.txt
```

### Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your API keys and settings
nano .env
```

### Start the Service

```bash
# Development mode
uvicorn requestProxy:app --host 0.0.0.0 --port 8079 --reload

# Production mode
uvicorn requestProxy:app --host 0.0.0.0 --port 8079 --workers 4
```

### Verify Installation

```bash
# Check health
curl http://localhost:8079/health

# Test passthrough
curl -X POST http://localhost:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-30B",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'

# Test research mode
curl -X POST http://localhost:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-30B",
    "messages": [{"role": "user", "content": "research python async"}],
    "stream": true
  }'
```

## Usage

### Basic Chat (Passthrough)

**Python (OpenAI SDK)**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8079/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen3-30B",
    messages=[
        {"role": "user", "content": "Explain Python decorators"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Standard Research Mode

**Python**:
```python
response = client.chat.completions.create(
    model="Qwen3-30B",
    messages=[
        {"role": "user", "content": "research kubernetes networking"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**cURL**:
```bash
curl -X POST http://localhost:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-30B",
    "messages": [
      {"role": "user", "content": "research kubernetes networking"}
    ],
    "stream": true
  }'
```

### Deep Research Mode

**Python**:
```python
response = client.chat.completions.create(
    model="Qwen3-30B",
    messages=[
        {"role": "user", "content": "research thoroughly machine learning deployment on kubernetes"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Non-Streaming Mode

```python
response = client.chat.completions.create(
    model="Qwen3-30B",
    messages=[
        {"role": "user", "content": "research python async programming"}
    ],
    stream=False
)

print(response.choices[0].message.content)
```

### Health Check

```python
import requests

health = requests.get("http://localhost:8079/health").json()

if health["status"] == "healthy":
    print("All systems operational")
else:
    print(f"Status: {health['status']}")
    for service, status in health["services"].items():
        print(f"  {service}: {status}")
```

## Queue Management

### Concurrency Limits

**Standard Research**:
- Maximum concurrent: 3
- Controlled by: `research_semaphore`
- Configured via: `MAX_STANDARD_RESEARCH`

**Deep Research**:
- Maximum concurrent: 1
- Controlled by: `deep_research_semaphore`
- Configured via: `MAX_DEEP_RESEARCH`

### Queue Status Messages

When queue is full, clients receive status updates:

**Standard Research Queue Full**:
```
Research queue is full. Standard research queue (3/3 slots used).
Waiting for an available slot...

[Health check results if services are down]

Slot available. Starting research...
```

**Deep Research Queue Full**:
```
Deep research slot occupied (1/1). Waiting for the slot to become available...

[Health check results if services are down]

Deep research slot available. Starting deep research...
```

### Queue Behavior

1. **Check semaphore availability**
2. **If full**:
   - Send queue status message to client
   - Perform health check (verify backends operational)
   - Send health status if any issues
   - Wait silently for available slot (non-blocking)
3. **When slot available**:
   - Send availability message
   - Begin research
4. **On completion**:
   - Release semaphore
   - Next queued request proceeds

## Health Monitoring

### Health Check Components

**1. Model Availability**:
- Checks: `/v1/models` endpoint responds
- Verifies: Model name present in response
- Falls back to: Docker container status if endpoint slow

**2. Docker Container Status**:
- Command: `docker ps --format json`
- Checks: Container present and running
- Services monitored:
  - vllm-qwen3
  - kg-service
  - mcprag-server
  - crawl4ai
  - neo4j-kg
  - open-webui

**3. HTTP Endpoint Checks**:
- kg-service: `GET http://localhost:8088/health`
- mcprag-server: `GET http://localhost:8080/api/v1/health`
- neo4j-kg: `GET http://localhost:7474`
- open-webui: `GET http://localhost/health`

**4. Critical Services**:
Services that affect overall status:
- vllm-qwen3 (must have model loaded)
- kg-service (must be running and healthy)
- neo4j-kg (must be running and healthy)

### Health Status Logic

```python
if all_critical_services_healthy:
    status = "healthy"
elif some_critical_services_unhealthy:
    status = "unhealthy"
elif some_non_critical_services_unhealthy:
    status = "degraded"
```

### Background Model Monitoring

**Task**: `model_name_manager()`

**Behavior**:
1. Poll `/v1/models` every 2 seconds until model loads
2. Once loaded, reduce polling to every 10 seconds
3. On timeout, check Docker container status
4. Cache model name for fast access
5. Refresh on connection errors

**Benefits**:
- Decouples model availability from request processing
- Recovers from temporary vLLM slowdowns
- Provides cached model name for fast responses
- Monitors container health

## Power Management

### Overview

The PowerManager automatically adjusts GPU performance levels based on API activity to reduce power consumption during idle periods.

**Status**: Currently disabled in production (commented out in code)

### How It Works

**Activity Detection**:
```
ConnectionManager tracks active requests
    ↓
Activity callback fires on 0 ↔ 1+ transitions
    ↓
PowerManager receives activity_started/activity_stopped
    ↓
Sets GPU performance level via rocm-smi
```

**Idle Timeout**:
```
Activity stops (0 connections)
    ↓
Start 2-minute idle timer
    ↓
If timer expires without new activity:
    ↓
Set GPU to "low" performance
```

**Activity Resume**:
```
New connection arrives
    ↓
Cancel idle timer
    ↓
Set GPU to "auto" performance
```

### GPU Commands

**Set Low Performance**:
```bash
rocm-smi --setperflevel low
```

**Set Auto Performance**:
```bash
rocm-smi --setperflevel auto
```

### Configuration

```python
idle_timeout_seconds = 120  # 2 minutes
```

### Enabling Power Management

Uncomment in [requestProxy.py](requestProxy.py):

```python
# Line 343-345: Enable startup
# await power_manager.start()

# Line 351-352: Enable shutdown
# if power_manager._running:
#     await power_manager.stop()
```

## Integration Points

### Upstream Consumers

**Open WebUI (Port 80)**:
- Uses robaiproxy as OpenAI-compatible backend
- Endpoint: `http://localhost:8079/v1`
- All chat requests route through proxy

**Direct API Clients**:
- Any OpenAI SDK-compatible client
- Base URL: `http://localhost:8079/v1`

### Downstream Services

**vLLM Backend (Port 8078)**:
- **Endpoint**: `http://localhost:8078/v1`
- **Used for**:
  - Passthrough mode (all requests)
  - Research mode (LLM generations)
- **Model**: Qwen3-30B
- **APIs**: `/v1/chat/completions`, `/v1/models`

**MCP RAG Server (Port 8080)**:
- **Endpoint**: `http://localhost:8080/api/v1`
- **Used for**:
  - Knowledge base search (search_memory)
  - URL crawling (crawl_url)
- **Authentication**: Bearer token (REST_API_KEY)
- **Timeout**: 60 seconds

**Serper API**:
- **Endpoint**: `https://google.serper.dev/search`
- **Used for**: Web search in research mode
- **Authentication**: X-API-KEY header
- **Timeout**: 30 seconds
- **Rate limits**: Per Serper account tier

**robairagapi (Port 8081)**:
- **Endpoint**: `http://localhost:8081`
- **Used for**: Catch-all routing for non-vLLM endpoints
- **APIs**: OpenAPI schema, general RAG functions

### Service Communication Flow

```
Open WebUI
    ↓
robaiproxy (Port 8079)
    ├→ vLLM (8078) - LLM generations
    ├→ MCP RAG (8080) - Knowledge base & crawling
    │   └→ robaimodeltools - RAG operations
    │       ├→ crawl4ai - Web crawling
    │       └→ kg-service (8088) - Knowledge graph
    │           └→ Neo4j (7474) - Graph database
    ├→ Serper API - Web search
    └→ robairagapi (8081) - General RAG API
```

## Design Patterns

### 1. Intelligent Request Router

**Pattern**: Content-based routing with mode detection

**Implementation**:
```python
if "research" in first_user_message:
    route_to_research_mode()
elif is_multimodal(messages):
    route_to_passthrough()
else:
    route_to_passthrough()
```

**Benefits**:
- Single endpoint for multiple modes
- OpenAI API compatibility maintained
- User controls mode via natural language

### 2. Semaphore-Based Queue Management

**Pattern**: Asyncio semaphore for concurrency control

**Implementation**:
```python
research_semaphore = asyncio.Semaphore(3)
deep_research_semaphore = asyncio.Semaphore(1)

async with research_semaphore:
    # Only 3 concurrent standard research allowed
    await perform_research()
```

**Benefits**:
- Prevents backend overload
- Fair queuing (FIFO)
- User-friendly status messages

### 3. Progressive Context Accumulation

**Pattern**: Append-only context building

**Implementation**:
```python
accumulated_context = ""

for iteration in iterations:
    search_results = search_knowledge_base()
    accumulated_context += format_results(search_results)

    crawled_content = crawl_urls()
    accumulated_context += format_content(crawled_content)

    web_results = search_web()
    accumulated_context += format_results(web_results)

final_answer = llm.generate(accumulated_context)
```

**Benefits**:
- No information loss from summarization
- LLM has full context for final answer
- Better quality than truncation/chunking

### 4. Tool Call Pattern

**Pattern**: XML-tagged tool calls in LLM responses

**Implementation**:
```python
# LLM generates
<tool_call>{"name": "search_memory", "arguments": {"query": "kubernetes networking"}}</tool_call>

# Parser extracts
tool_calls = extract_tool_calls(response)

# Executor runs
result = call_mcp_tool(tool_calls[0])

# Result added back
conversation.append({"role": "user", "content": tool_result})
```

**Benefits**:
- Structured tool invocation
- Supports multiple tools per response
- LLM-controlled tool usage

### 5. Background Model Monitoring

**Pattern**: Asyncio background task with polling

**Implementation**:
```python
async def model_name_manager():
    while True:
        try:
            model = await fetch_model_name()
            if model:
                global current_model_name
                current_model_name = model
                await asyncio.sleep(10)  # Reduced polling
            else:
                await asyncio.sleep(2)  # Frequent polling until loaded
        except Exception:
            await asyncio.sleep(2)
```

**Benefits**:
- Decouples availability check from requests
- Cached model name for fast access
- Resilient to temporary failures

### 6. Auto-Retry on Context Overflow

**Pattern**: Exception-driven retry with reduced scope

**Implementation**:
```python
try:
    result = await research_internal(iterations=4)
except ContextLengthExceededError:
    yield "Context overflow. Restarting with 2 iterations..."
    result = await research_internal(iterations=2)
```

**Benefits**:
- Graceful degradation
- User informed of adjustment
- Prevents complete failure

### 7. Client Disconnect Detection

**Pattern**: Periodic request.is_disconnected() checks

**Implementation**:
```python
async for result in research_iteration():
    if await request.is_disconnected():
        logger.info("Client disconnected, stopping research")
        return

    # Expensive operation only if client still connected
    await crawl_urls()
```

**Benefits**:
- Prevents wasted computation
- Releases resources quickly
- Improves overall system efficiency

## Security

### API Key Management

**Storage**:
- Environment variables in `.env` file
- Never committed to version control
- `.gitignore` excludes `.env`

**Usage**:
- REST_API_KEY: Bearer token for MCP server
- SERPER_API_KEY: Header-based authentication for Serper

**Logging**:
- API keys masked in logs (show only last 8 chars)
- Configuration display masks secrets

### Request Authorization

**Current State**:
- No authentication on proxy endpoints
- Relies on downstream service authentication
- MCP server enforces Bearer token authentication

**Recommendation**:
- Add authentication middleware for production
- Implement rate limiting per client
- Add API key validation on proxy endpoints

### Connection Tracking

**Logged Information**:
- Client IP addresses
- Request IDs (for tracing)
- Endpoint paths
- Timestamps

**Not Exposed**:
- Connection details not available via API
- Used internally for monitoring only

### Environment Security

**Protected Files**:
```
.env           # Contains secrets
*.log          # May contain sensitive data
__pycache__/   # Bytecode cache
```

**Public Files**:
```
.env.example   # Template without secrets
```

## Limitations

### Current Known Issues

1. **Power Manager Disabled**:
   - Currently commented out in production
   - Lines 343-345, 351-352 in requestProxy.py
   - Needs testing before re-enabling

2. **Utility Scripts**:
   - `check_power.py` references status fields not in actual dict
   - `check_connections.py` has similar field mismatch issues
   - Both need updating to match current implementation

3. **Test Endpoints**:
   - Hardcoded IP address: `192.168.10.50`
   - Should be configurable via environment variable

4. **Legacy Code**:
   - `oldprototypes/` directory contains unused code
   - Should be removed or archived

5. **Model Polling**:
   - Fixed 2-second interval (no exponential backoff)
   - Could be more efficient with adaptive polling

6. **Context Overflow**:
   - Auto-retry reduces from 4 to 2 iterations
   - Could implement more granular reduction (4→3→2)

### Architectural Limitations

**Not a Traditional Reverse Proxy**:
- No connection pooling
- No response caching
- No load balancing
- No circuit breaking

**Not a Security Layer**:
- No authentication/authorization
- No rate limiting
- No request validation
- No DDoS protection

**Not a Message Bus**:
- No request persistence
- No retry queues
- No event streaming
- No message routing

### Scalability Considerations

**Single Instance Limits**:
- Max 3 concurrent standard research
- Max 1 concurrent deep research
- No horizontal scaling support
- No distributed queue

**Context Size Limits**:
- Depends on vLLM model context window
- Auto-retry helps but doesn't eliminate limit
- Very deep research may still overflow

**Backend Dependencies**:
- Single point of failure for each backend
- No failover or redundancy
- Network issues block all requests

## Statistics

### Code Metrics

- **Total Lines**: ~3,200 lines of application code
- **Main Application**: 975 lines (requestProxy.py)
- **Research Logic**: 1,026 lines (researchAgent.py)
- **Configuration**: 180 lines (config.py)
- **Connection Management**: 230 lines (connectionManager.py)
- **Power Management**: 286 lines (powerManager.py)
- **Test Suite**: 313 lines (test_endpoints.py)

### Dependencies

- **Python Packages**: 23 direct dependencies
- **Core Framework**: FastAPI 0.119.0 + Uvicorn 0.37.0
- **HTTP Client**: httpx 0.28.1
- **LLM Client**: OpenAI SDK 2.3.0

### Service Integration

- **Downstream Services**: 6 (vLLM, MCP RAG, Serper, robairagapi, crawl4ai, Neo4j)
- **Monitored Services**: 6 (vllm-qwen3, kg-service, mcprag-server, crawl4ai, neo4j-kg, open-webui)
- **Critical Services**: 3 (vllm-qwen3, kg-service, neo4j-kg)

### Research Metrics

- **Research Types**: 2 (standard, deep)
- **Standard Iterations**: 2
- **Deep Iterations**: 4
- **Max Concurrent Standard**: 3
- **Max Concurrent Deep**: 1
- **Web Search Results per Iteration**: 5-10
- **Knowledge Base Results per Iteration**: 3-6
- **URLs Crawled per Iteration**: 3

### API Endpoints

- **Main Endpoints**: 4
  - POST /v1/chat/completions
  - GET /health
  - GET /v1/models
  - GET /openapi.json
- **Catch-All**: `/{path:path}`

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [vLLM Documentation](https://docs.vllm.ai)
- [Serper API Documentation](https://serper.dev/docs)

## Contributing

When contributing to robaiproxy:

1. **Maintain OpenAI API Compatibility**: All changes to `/v1/chat/completions` must remain compatible
2. **Test Research Mode**: Verify both standard and deep research work correctly
3. **Update Health Checks**: Add health checks for new service dependencies
4. **Document Configuration**: Update `.env.example` for new environment variables
5. **Add Tests**: Update `test_endpoints.py` for new functionality

## License

[Include license information here]

## Contact

[Include contact/support information here]
