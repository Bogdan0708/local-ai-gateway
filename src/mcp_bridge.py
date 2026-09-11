"""
MCP (Model Context Protocol) bridge for Personal AI Infrastructure integration.

This module exposes Local AI's knowledge base and agent capabilities
to the Personal AI Infrastructure (PAI) framework via HTTP endpoints.

This allows PAI skills to:
1. Search the local knowledge base (ChromaDB + BM25)
2. Ingest content into the local knowledge base
3. Invoke local agents (DevOps, Research, etc.)

Usage in PAI:
    Add to PAI's .mcp.json:
    {
      "local-knowledge": {
        "type": "http",
        "url": "http://localhost:8000/mcp",
        "description": "Local LLM knowledge base and agents"
      }
    }
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp-bridge"])


class SearchRequest(BaseModel):
    """Request model for knowledge base search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    use_reranking: bool = Field(True, description="Whether to use cross-encoder reranking")


class SearchResponse(BaseModel):
    """Response model for knowledge base search."""
    query: str
    results: list[dict]
    total: int


class IngestRequest(BaseModel):
    """Request model for content ingestion."""
    content: str = Field(..., description="Content to ingest")
    metadata: dict = Field(default_factory=dict, description="Metadata for the content")
    chunk_strategy: Literal["semantic", "fixed", "adaptive"] = Field(
        "semantic",
        description="Chunking strategy to use"
    )


class IngestResponse(BaseModel):
    """Response model for content ingestion."""
    status: str
    chunks_created: int
    document_id: str


class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    agent_type: Literal["devops", "research"] = Field(..., description="Type of agent to invoke")
    task: str = Field(..., description="Task description")
    context: Optional[str] = Field(None, description="Additional context")


class AgentResponse(BaseModel):
    """Response model for agent invocation."""
    status: str
    result: dict


@router.post("/knowledge/search", response_model=SearchResponse)
async def mcp_search(request: SearchRequest):
    """
    Expose knowledge base search to PAI.

    Searches the local ChromaDB + BM25 hybrid knowledge base and returns
    relevant documents. Optionally applies cross-encoder reranking for
    improved relevance.

    Args:
        request: Search parameters (query, top_k, use_reranking)

    Returns:
        SearchResponse with matching documents

    Example PAI usage:
        POST http://localhost:8000/mcp/knowledge/search
        {
          "query": "How do I configure authentication?",
          "top_k": 5
        }
    """
    try:
        # Import here to avoid circular dependencies
        from src.memory import knowledge_base

        logger.info(f"MCP search request: {request.query}")

        # Perform hybrid search
        results = knowledge_base.search(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"Error in MCP search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.post("/knowledge/ingest", response_model=IngestResponse)
async def mcp_ingest(request: IngestRequest):
    """
    Allow PAI to ingest content into local knowledge base.

    Processes content using the specified chunking strategy and adds
    it to the ChromaDB vector store + BM25 index.

    Args:
        request: Content, metadata, and chunking strategy

    Returns:
        IngestResponse with ingestion status

    Example PAI usage:
        POST http://localhost:8000/mcp/knowledge/ingest
        {
          "content": "Local LLM setup guide: ...",
          "metadata": {"source": "PAI-skill", "skill": "research"},
          "chunk_strategy": "semantic"
        }
    """
    try:
        # Import here to avoid circular dependencies
        from src.memory import knowledge_base
        from src.chunking import chunk_document

        logger.info(f"MCP ingest request from: {request.metadata.get('source', 'unknown')}")

        # Chunk the content
        chunks = chunk_document(
            content=request.content,
            strategy=request.chunk_strategy,
            metadata=request.metadata
        )

        # Add to knowledge base
        document_id = knowledge_base.add_chunks(chunks)

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            document_id=document_id
        )

    except Exception as e:
        logger.error(f"Error in MCP ingest: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@router.post("/agents/invoke", response_model=AgentResponse)
async def mcp_invoke_agent(request: AgentRequest):
    """
    Invoke local agents from PAI.

    Allows PAI skills to leverage local LLM agents (DevOps, Research)
    for complex multi-step workflows.

    Args:
        request: Agent type and task description

    Returns:
        AgentResponse with agent execution results

    Example PAI usage:
        POST http://localhost:8000/mcp/agents/invoke
        {
          "agent_type": "devops",
          "task": "Create a function to parse JSON safely",
          "context": "Used in data processing pipeline"
        }
    """
    try:
        # Import here to avoid circular dependencies
        from src.agents import create_agent

        logger.info(f"MCP agent invocation: {request.agent_type} - {request.task}")

        # Create and run agent
        agent = create_agent(request.agent_type)
        result = await agent.run(request.task)

        return AgentResponse(
            status=result.get("status", "completed"),
            result=result
        )

    except Exception as e:
        logger.error(f"Error in MCP agent invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.get("/health")
async def mcp_health():
    """
    Health check endpoint for MCP bridge.

    Returns:
        Status information about the bridge and dependencies
    """
    try:
        # Check if knowledge base is accessible
        from src.memory import knowledge_base

        status = {
            "status": "healthy",
            "bridge": "mcp",
            "knowledge_base": "accessible",
            "agents": ["devops", "research"]
        }

        return status

    except Exception as e:
        logger.error(f"MCP health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
