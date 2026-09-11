"""
FastAPI API Gateway with OpenAI-compatible endpoints.

Provides:
- /v1/chat/completions - Chat with RAG context
- /v1/embeddings - Generate embeddings
- /v1/files - Manage knowledge base files
- /v1/search - Semantic search
- /v1/web/fetch - Fetch web content
- /health - Health check
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .auth import TokenData, require_auth, require_local_or_auth
from .config import get_settings
from .file_service import (
    FileIngestionService,
    FileValidator,
    SecurityError as FileSecurityError,
)
from .memory import Document, get_memory
from .web_fetcher import FetchError, SecurityError, fetch_url, perform_search
from .mcp_bridge import router as mcp_router  # MCP bridge for PAI integration

logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Secure Local AI API Gateway")
    yield
    logger.info("Shutting down Secure Local AI API Gateway")


app = FastAPI(
    title="Secure Local AI Gateway",
    description="OpenAI-compatible local AI API with RAG, memory, and web tools",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include MCP bridge router for PAI integration
app.include_router(mcp_router)

# === Request/Response Models ===


class Message(BaseModel):
    """Chat message."""

    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """OpenAI-compatible chat request."""

    model: str = Field(default="gpt-oss-120b")
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)
    # RAG-specific options
    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")
    rag_k: int = Field(default=5, description="Number of RAG results to include")


class ChatChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: Message
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """OpenAI-compatible chat response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class EmbeddingRequest(BaseModel):
    """Embedding request."""

    model: str = Field(default="nomic-embed-text")
    input: str | list[str]


class EmbeddingData(BaseModel):
    """Single embedding result."""

    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


class SearchRequest(BaseModel):
    """Search request."""

    query: str
    k: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="hybrid", description="hybrid, semantic, or bm25")


class SearchResult(BaseModel):
    """Search result."""

    content: str
    metadata: dict
    score: float
    source: str


class SearchResponse(BaseModel):
    """Search response."""

    results: list[SearchResult]
    total: int
    method: str
    query: str


class WebFetchRequest(BaseModel):
    """Web fetch request."""

    url: str
    use_cache: bool = Field(default=True)
    max_tier: int = Field(default=2, ge=1, le=3)


class WebFetchResponse(BaseModel):
    """Web fetch response."""

    url: str
    content: str
    content_type: str
    status_code: int
    tier_used: str
    fetched_at: str


class WebSearchRequest(BaseModel):
    """Web search request."""
    query: str
    max_results: int = 5


@app.post(
    "/v1/web/search",
    response_model=dict,
    tags=["Web"],
)
@limiter.limit("10/minute")
async def search_web(
    request: Request,
    search_request: WebSearchRequest,
    auth: TokenData = Depends(require_auth),
):
    """Perform a web search."""
    results = perform_search(search_request.query, search_request.max_results)
    return {"results": results}


class FileInfo(BaseModel):
    """File information."""

    filename: str
    path: str
    size_bytes: int
    chunks: int
    indexed_at: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    llm_provider: str
    llm_model: str
    memory_documents: int


# === Application Setup ===


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Secure Local AI API...")

    # Initialize services
    settings = get_settings()
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Chat Model: {settings.active_chat_model}")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Secure Local AI",
    description="Private knowledge base with local file and web access",
    version="0.1.0",
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS (restrictive by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# === Helper Functions ===


async def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> dict:
    """Call the LLM (LM Studio or Ollama)."""
    settings = get_settings()

    if settings.llm_provider == "ollama":
        # Ollama API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": model or settings.ollama_chat_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        **({"num_predict": max_tokens} if max_tokens else {}),
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["message"]["content"],
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
    else:
        # LM Studio / OpenAI-compatible API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.lmstudio_host}/v1/chat/completions",
                json={
                    "model": model or settings.lmstudio_chat_model,
                    "messages": messages,
                    "temperature": temperature,
                    **({"max_tokens": max_tokens} if max_tokens else {}),
                },
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }


async def stream_llm_response(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response as SSE events.

    Yields OpenAI-compatible SSE chunks.
    """
    settings = get_settings()

    if settings.llm_provider == "ollama":
        async for chunk in _stream_ollama(messages, model, temperature):
            yield chunk
    else:
        async for chunk in _stream_openai_compatible(
            messages, model, temperature, max_tokens
        ):
            yield chunk

    # Signal completion
    yield "data: [DONE]\n\n"


async def _stream_ollama(
    messages: list[dict],
    model: str,
    temperature: float,
) -> AsyncGenerator[str, None]:
    """Stream from Ollama API."""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_host}/api/chat",
            json={
                "model": model or settings.ollama_chat_model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")

                    if content:
                        # Convert to OpenAI format
                        chunk = {
                            "id": f"chatcmpl-{datetime.utcnow().timestamp()}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.utcnow().timestamp()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    # Check if done
                    if data.get("done", False):
                        # Final chunk with finish_reason
                        final_chunk = {
                            "id": f"chatcmpl-{datetime.utcnow().timestamp()}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.utcnow().timestamp()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"

                except json.JSONDecodeError:
                    continue


async def _stream_openai_compatible(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
) -> AsyncGenerator[str, None]:
    """Stream from OpenAI-compatible API (LM Studio, etc.)."""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{settings.lmstudio_host}/v1/chat/completions",
            json={
                "model": model or settings.lmstudio_chat_model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                **({} if max_tokens is None else {"max_tokens": max_tokens}),
            },
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        ) as response:
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                # Pass through OpenAI format directly
                yield f"data: {data_str}\n\n"


# === Endpoints ===


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint (no auth required)."""
    settings = get_settings()
    memory = get_memory()

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
        llm_provider=settings.llm_provider,
        llm_model=settings.active_chat_model,
        memory_documents=memory._collection.count(),
    )


@app.post(
    "/v1/chat/completions",
    response_model=None,  # Allow streaming or JSON
    tags=["Chat"],
)
@limiter.limit("100/hour")
async def chat_completions(
    request: Request,
    chat_request: ChatRequest,
    auth: TokenData = Depends(require_auth),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=true) and non-streaming responses.
    Also supports RAG context retrieval from the knowledge base.
    """
    settings = get_settings()
    messages = [{"role": m.role, "content": m.content} for m in chat_request.messages]

    # RAG: Retrieve relevant context
    if chat_request.use_rag and messages:
        try:
            memory = get_memory()
            # Use the last user message for retrieval
            user_messages = [m for m in messages if m["role"] == "user"]
            if user_messages:
                query = user_messages[-1]["content"]
                results = memory.search(query, k=chat_request.rag_k, method="hybrid")

                if results:
                    # Build context from retrieved documents
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        source = result.document.metadata.get("source", "unknown")
                        context_parts.append(
                            f"[{i}] Source: {source}\n{result.document.content}"
                        )

                    context = "\n\n".join(context_parts)

                    # Inject context as system message
                    system_msg = {
                        "role": "system",
                        "content": f"Use the following context to answer the user's question. "
                        f"If the context doesn't contain relevant information, say so.\n\n"
                        f"Context:\n{context}",
                    }

                    # Insert after any existing system message
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] += f"\n\n{system_msg['content']}"
                    else:
                        messages.insert(0, system_msg)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            # Continue without RAG context

    # Handle streaming vs non-streaming
    if chat_request.stream:
        return StreamingResponse(
            stream_llm_response(
                messages=messages,
                model=chat_request.model,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Non-streaming: Call LLM
    try:
        llm_response = await call_llm(
            messages=messages,
            model=chat_request.model,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
        )
    except httpx.HTTPError as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {e}",
        )

    return ChatResponse(
        id=f"chatcmpl-{datetime.utcnow().timestamp()}",
        created=int(datetime.utcnow().timestamp()),
        model=chat_request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=llm_response["content"]),
            )
        ],
        usage=ChatUsage(
            prompt_tokens=llm_response["prompt_tokens"],
            completion_tokens=llm_response["completion_tokens"],
            total_tokens=llm_response["prompt_tokens"]
            + llm_response["completion_tokens"],
        ),
    )


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    tags=["Embeddings"],
)
@limiter.limit("200/hour")
async def create_embeddings(
    request: Request,
    embed_request: EmbeddingRequest,
    auth: TokenData = Depends(require_auth),
):
    """Generate embeddings for text."""
    memory = get_memory()

    inputs = embed_request.input if isinstance(embed_request.input, list) else [embed_request.input]
    embeddings = memory._embeddings.embed_batch(inputs)

    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ],
        model=embed_request.model,
        usage={"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": 0},
    )


@app.post(
    "/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
)
@limiter.limit("100/hour")
async def search_knowledge_base(
    request: Request,
    search_request: SearchRequest,
    auth: TokenData = Depends(require_auth),
):
    """Search the knowledge base using hybrid, semantic, or BM25 search."""
    memory = get_memory()
    results = memory.search(search_request.query, k=search_request.k, method=search_request.method)

    return SearchResponse(
        results=[
            SearchResult(
                content=r.document.content,
                metadata=r.document.metadata,
                score=r.score,
                source=r.source,
            )
            for r in results
        ],
        total=len(results),
        method=search_request.method,
        query=search_request.query,
    )


@app.get(
    "/v1/files",
    response_model=list[FileInfo],
    tags=["Files"],
)
async def list_files(
    auth: TokenData = Depends(require_auth),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """List indexed files in the knowledge base."""
    memory = get_memory()

    # Get unique files from metadata
    if memory._collection.count() == 0:
        return []

    results = memory._collection.get(include=["metadatas"])

    # Group by source file
    files: dict[str, FileInfo] = {}
    for metadata in results["metadatas"]:
        if not metadata:
            continue
        source = metadata.get("source", "unknown")
        if source not in files:
            files[source] = FileInfo(
                filename=metadata.get("filename", "unknown"),
                path=source,
                size_bytes=metadata.get("size_bytes", 0),
                chunks=0,
                indexed_at=metadata.get("indexed_at", ""),
            )
        files[source].chunks += 1

    return list(files.values())[:limit]


@app.post(
    "/v1/files",
    response_model=dict,
    tags=["Files"],
)
@limiter.limit("50/hour")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    auth: TokenData = Depends(require_auth),
):
    """Upload and index a file."""
    settings = get_settings()

    # Save uploaded file temporarily
    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()

        # Check size
        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max: {settings.max_file_size_mb}MB",
            )

        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        service = FileIngestionService()
        chunk_count = service.ingest_file(tmp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": chunk_count,
            "message": f"Indexed {chunk_count} chunks from {file.filename}",
        }
    except Exception as e:
        logger.error(f"Failed to index file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to index file: {e}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post(
    "/v1/web/fetch",
    response_model=WebFetchResponse,
    tags=["Web"],
)
@limiter.limit("10/minute")
async def fetch_web_content(
    request: Request,
    fetch_request: WebFetchRequest,
    auth: TokenData = Depends(require_auth),
):
    """
    Fetch content from a URL with SSRF protection.

    Uses multi-tier fallback (httpx -> curl -> playwright).
    """
    try:
        result = fetch_url(
            fetch_request.url,
            use_cache=fetch_request.use_cache,
            max_tier=fetch_request.max_tier,
        )

        return WebFetchResponse(
            url=result.url,
            content=result.content,
            content_type=result.content_type,
            status_code=result.status_code,
            tier_used=result.tier_used,
            fetched_at=result.fetched_at,
        )

    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Security error: {e}",
        )
    except FetchError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fetch failed: {e}",
        )


@app.post(
    "/v1/web/index",
    response_model=dict,
    tags=["Web"],
)
@limiter.limit("10/minute")
async def index_web_content(
    request: Request,
    fetch_request: WebFetchRequest,
    auth: TokenData = Depends(require_auth),
):
    """Fetch a URL and add its content to the knowledge base."""
    try:
        result = fetch_url(
            fetch_request.url,
            use_cache=fetch_request.use_cache,
            max_tier=fetch_request.max_tier,
        )

        # Create document and add to memory
        memory = get_memory()
        doc = Document(
            content=result.content,
            metadata={
                "source": result.url,
                "content_type": result.content_type,
                "fetched_at": result.fetched_at,
                "type": "web",
            },
        )

        doc_id = memory.add_document(doc)

        return {
            "status": "success",
            "url": result.url,
            "document_id": doc_id,
            "content_length": len(result.content),
            "message": f"Indexed content from {result.url}",
        }

    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Security error: {e}",
        )
    except FetchError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fetch failed: {e}",
        )


# === Agent Workflows ===


class AgentRequest(BaseModel):
    """Request to run an agent workflow."""
    task: str = Field(..., description="Task description for the agent")
    agent_type: str = Field(default="devops", description="Type of agent: devops or research")


class AgentResponse(BaseModel):
    """Response from agent workflow execution."""
    status: str
    task: str
    result: dict


@app.post(
    "/v1/agents/devops",
    response_model=AgentResponse,
    tags=["Agents"],
)
@limiter.limit("20/hour")
async def run_devops_agent(
    request: Request,
    agent_request: AgentRequest,
    auth: TokenData = Depends(require_auth),
):
    """
    Run DevOps agent workflow.

    The DevOps agent performs multi-step code generation:
    1. Architecture analysis (quality tier model)
    2. Code implementation (fast tier model)
    3. Code review (quality tier model)

    Example request:
    {
        "task": "Create a Python function to validate email addresses",
        "agent_type": "devops"
    }

    Returns:
    {
        "status": "completed",
        "task": "...",
        "result": {
            "context": "Architecture plan...",
            "code": "Generated code...",
            "review": "Code review..."
        }
    }
    """
    try:
        from src.agents import DevOpsAgent

        logger.info(f"Running DevOps agent for task: {agent_request.task}")

        agent = DevOpsAgent(litellm_base=get_settings().lmstudio_host)
        result = await agent.run(task=agent_request.task)

        return AgentResponse(
            status=result.get("status", "completed"),
            task=agent_request.task,
            result=result
        )

    except Exception as e:
        logger.error(f"DevOps agent error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow failed: {str(e)}"
        )


@app.post(
    "/v1/agents/research",
    response_model=AgentResponse,
    tags=["Agents"],
)
@limiter.limit("10/hour")
async def run_research_agent(
    request: Request,
    agent_request: AgentRequest,
    sources: list[str] = [],
    auth: TokenData = Depends(require_auth),
):
    """
    Run Research agent workflow.

    The Research agent synthesizes information from multiple sources
    using the ultra tier model (120B) for deep analysis.

    Example request:
    {
        "task": "What are the latest developments in quantum computing?",
        "sources": ["source1 text...", "source2 text..."]
    }

    Returns:
    {
        "status": "completed",
        "task": "...",
        "result": {
            "query": "...",
            "synthesis": "Detailed research synthesis...",
            "sources_used": 2
        }
    }
    """
    try:
        from src.agents import ResearchAgent

        logger.info(f"Running Research agent for query: {agent_request.task}")

        agent = ResearchAgent(litellm_base=get_settings().lmstudio_host)
        result = await agent.run(
            query=agent_request.task,
            sources=sources
        )

        return AgentResponse(
            status=result.get("status", "completed"),
            task=agent_request.task,
            result=result
        )

    except Exception as e:
        logger.error(f"Research agent error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow failed: {str(e)}"
        )


# === Internal File Reading API (localhost only) ===


def _allowed_read_roots(settings) -> list[Path]:
    """
    Real, existing directories that /api/read-file may serve from.

    Defaults to the container's own document/code dirs; extend via
    ALLOWED_PATH_PREFIXES (os.pathsep-separated) for deployment-specific
    mounts. Roots are resolved strictly: a configured root that does not
    exist cannot contain anything, so it is dropped rather than trusted as a
    string prefix.
    """
    configured = [
        settings.documents_path,
        settings.code_path,
        Path("/data/documents"),
        Path("/data/code"),
    ] + [
        Path(p)
        for p in os.environ.get("ALLOWED_PATH_PREFIXES", "/data").split(os.pathsep)
        if p
    ]

    roots: list[Path] = []
    for root in configured:
        try:
            resolved = Path(root).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def _is_within_allowed_roots(path: Path, roots: list[Path]) -> bool:
    """True when `path` is one of `roots` or lives under one of them.

    Uses real filesystem ancestry (Path.is_relative_to), not string prefixes,
    so `/data/allowed-evil/x` is not accepted for the root `/data/allowed`.
    """
    return any(path == root or path.is_relative_to(root) for root in roots)


class ReadFileRequest(BaseModel):
    """Request to read a local file."""
    path: str = Field(..., description="Path to the file to read")


@app.post("/api/read-file", tags=["Internal"])
async def read_local_file(
    request: Request,
    file_request: ReadFileRequest,
    auth: TokenData = Depends(require_local_or_auth),
):
    """
    Read a local file and return its contents.

    Only available from localhost for security.
    """
    from pathlib import Path as PathLib
    settings = get_settings()

    # Host -> container document-root mapping is deployment-specific, so it's
    # configurable via env rather than baked in. HOST_DOCUMENTS_ROOT is the
    # Windows/host folder your documents actually live under (e.g. a
    # per-user Documents folder on the host); CONTAINER_DOCUMENTS_ROOT is
    # where that maps to inside this service. Leave HOST_DOCUMENTS_ROOT
    # unset to disable the host-path rewrite entirely (a bare "C:..." path
    # is then left as-is and will simply fail the allowed-prefix check
    # below).
    HOST_DOCUMENTS_ROOT = os.environ.get("HOST_DOCUMENTS_ROOT", "")
    CONTAINER_DOCUMENTS_ROOT = os.environ.get("CONTAINER_DOCUMENTS_ROOT", "/data/documents")

    # Sanitize path: Handle Windows paths sent to Linux container
    raw_path = file_request.path.strip('"\'')  # Strip quotes
    if raw_path.lower().startswith("c:"):
        normalized = raw_path.replace("\\", "/")
        if HOST_DOCUMENTS_ROOT and normalized.lower().startswith(HOST_DOCUMENTS_ROOT.lower()):
            raw_path = CONTAINER_DOCUMENTS_ROOT + normalized[len(HOST_DOCUMENTS_ROOT):]
        # else: no configured host root matches — leave raw_path as the
        # normalized Windows path; it will be rejected by the allowed-prefix
        # check below unless ALLOWED_PATH_PREFIXES was configured to permit it.

    allowed_roots = _allowed_read_roots(settings)
    if not allowed_roots:
        logger.warning("No readable roots are configured; refusing file read")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Path not allowed.",
        )

    # Normalise without requiring existence first, so a path outside the roots
    # is refused with 403 whether or not it exists (no existence oracle).
    try:
        candidate = PathLib(raw_path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"Rejected unresolvable read path: {exc}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Path not allowed.",
        )

    if not _is_within_allowed_roots(candidate, allowed_roots):
        logger.warning("Access denied: resolved path is outside every allowed root")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Path not allowed.",
        )

    # Same filename policy as ingestion: extension allowlist plus blocked
    # patterns (.env, *.pem, id_rsa, ...).
    try:
        validator = FileValidator()
        validator.validate_extension(candidate)
        validator.validate_filename(candidate)
    except FileSecurityError as exc:
        logger.warning(f"Access denied by filename policy: {exc}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. {exc}",
        )

    # Strict resolution: the file must exist, and the fully symlink-resolved
    # target must still sit inside an allowed root.
    try:
        file_path = PathLib(raw_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )

    if not _is_within_allowed_roots(file_path, allowed_roots):
        logger.warning("Access denied: symlink target is outside every allowed root")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Path not allowed.",
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path is not a file",
        )

    # Check file size (max 1MB for reading)
    if file_path.stat().st_size > 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large to read (max 1MB)",
        )

    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "path": str(file_path),
            "filename": file_path.name,
            "content": content,
            "size": len(content),
        }
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a text file",
        )


# === Internal Chat API (localhost only, no auth required) ===


async def _stream_with_rag_sources(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    rag_sources: list[dict],
) -> AsyncGenerator[str, None]:
    """Stream LLM response with RAG sources in final chunk."""
    # First, send RAG sources as a special event
    if rag_sources:
        sources_event = {
            "type": "rag_sources",
            "sources": rag_sources,
        }
        yield f"data: {json.dumps(sources_event)}\n\n"

    # Stream the actual response
    async for chunk in stream_llm_response(messages, model, temperature, max_tokens):
        yield chunk


@app.post("/api/chat", tags=["Internal"])
async def internal_chat(
    request: Request,
    chat_request: ChatRequest,
    auth: TokenData = Depends(require_local_or_auth),
):
    """
    Internal chat endpoint for the UI.

    Allows localhost access without authentication.
    Supports streaming (stream=true) and returns RAG sources.
    """
    settings = get_settings()
    messages = [{"role": m.role, "content": m.content} for m in chat_request.messages]
    rag_sources = []

    # RAG: Retrieve relevant context
    if chat_request.use_rag and messages:
        try:
            memory = get_memory()
            user_messages = [m for m in messages if m["role"] == "user"]
            if user_messages:
                query = user_messages[-1]["content"]
                results = memory.search(query, k=chat_request.rag_k, method="hybrid")

                if results:
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        source = result.document.metadata.get("source", "unknown")
                        filename = result.document.metadata.get("filename", Path(source).name if source != "unknown" else "unknown")

                        # Store source info for UI
                        rag_sources.append({
                            "filename": filename,
                            "source": source,
                            "preview": result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content,
                            "score": result.score,
                        })

                        context_parts.append(
                            f"[{i}] Source: {source}\n{result.document.content}"
                        )

                    context = "\n\n".join(context_parts)
                    system_msg = {
                        "role": "system",
                        "content": f"Use the following context to answer the user's question. "
                        f"If the context doesn't contain relevant information, say so.\n\n"
                        f"Context:\n{context}",
                    }

                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] += f"\n\n{system_msg['content']}"
                    else:
                        messages.insert(0, system_msg)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            # Continue without RAG context

    # Handle streaming
    if chat_request.stream:
        return StreamingResponse(
            _stream_with_rag_sources(
                messages=messages,
                model=chat_request.model,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                rag_sources=rag_sources,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: Call LLM
    try:
        llm_response = await call_llm(
            messages=messages,
            model=chat_request.model,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
        )
    except httpx.HTTPError as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {e}",
        )

    # Return response with RAG sources
    response_data = {
        "id": f"chatcmpl-{datetime.utcnow().timestamp()}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": chat_request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": llm_response["content"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": llm_response["prompt_tokens"],
            "completion_tokens": llm_response["completion_tokens"],
            "total_tokens": llm_response["prompt_tokens"] + llm_response["completion_tokens"],
        },
    }

    # Add RAG sources if available
    if rag_sources:
        response_data["rag_sources"] = rag_sources

    return response_data


# === Chat UI ===


@app.get("/chat", tags=["UI"])
async def chat_ui():
    """Serve the chat interface."""
    static_path = Path(__file__).parent.parent / "static" / "chat.html"
    if not static_path.exists():
        raise HTTPException(status_code=404, detail="Chat UI not found")
    return FileResponse(static_path, media_type="text/html")


@app.get("/", tags=["UI"])
async def root():
    """Redirect root to chat UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat")


# === Main Entry Point ===


def main():
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api_gateway:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
