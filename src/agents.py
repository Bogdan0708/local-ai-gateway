"""
Agent workflows using LangGraph for multi-step AI orchestration.

Agents call the configured chat backend (LM Studio, Ollama or any
OpenAI-compatible server) and use up to three model tiers:
- Fast tier (AGENT_FAST_MODEL) for quick tasks such as coding
- Quality tier (AGENT_QUALITY_MODEL) for architecture and review
- Ultra tier (AGENT_ULTRA_MODEL) for research synthesis
Unset tiers fall back to the active chat model.
"""

from typing import Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator
import logging

from .config import Settings, get_settings
from .reasoning import strip_think

logger = logging.getLogger(__name__)


def _chat_model(settings: Settings, tier_model: Optional[str], temperature: float) -> ChatOpenAI:
    """OpenAI-compatible client for one agent tier on the active backend."""
    if settings.llm_provider == "ollama":
        base_url = f"{settings.ollama_host.rstrip('/')}/v1"
    else:
        base_url = settings.active_chat_base_url
    return ChatOpenAI(
        model=tier_model or settings.active_chat_model,
        base_url=base_url,
        api_key=settings.openai_api_key,
        temperature=temperature,
        reasoning_effort=settings.llm_reasoning_effort,
    )


class AgentState(TypedDict):
    """State shared across all agent nodes in the workflow."""
    task: str
    context: str
    code: str
    review: str
    status: str
    iterations: Annotated[int, operator.add]
    error: str


class DevOpsAgent:
    """
    Multi-step DevOps agent that uses tiered models for different tasks:
    1. Architect (quality tier) - Analyzes task and designs approach
    2. Coder (fast tier) - Implements code quickly
    3. Reviewer (quality tier) - Reviews code for quality/security

    Usage:
        agent = DevOpsAgent()
        result = await agent.run("Create a function to validate emails")
        print(result["code"])
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize DevOps agent on the configured chat backend.

        Args:
            settings: Settings to use (default: the application settings)
        """
        settings = settings or get_settings()
        # Low temperature for deterministic code generation
        self.fast_model = _chat_model(settings, settings.agent_fast_model, temperature=0.2)
        # Slightly higher for architecture/review
        self.quality_model = _chat_model(settings, settings.agent_quality_model, temperature=0.3)

        # Build workflow graph
        self.graph = StateGraph(AgentState)
        self.graph.add_node("analyze", self.analyze_task)
        self.graph.add_node("implement", self.implement_code)
        self.graph.add_node("review", self.review_code)

        # Define workflow edges
        self.graph.set_entry_point("analyze")
        self.graph.add_edge("analyze", "implement")
        self.graph.add_edge("implement", "review")
        self.graph.add_edge("review", END)

        self.workflow = self.graph.compile()
        logger.info("DevOps agent initialized on the configured chat backend")

    async def analyze_task(self, state: AgentState) -> AgentState:
        """
        Analyze task using quality model for architecture planning.

        Uses quality tier (70B-80B) for deep understanding and planning.
        """
        try:
            logger.info(f"Analyzing task: {state['task']}")
            response = await self.quality_model.ainvoke([
                {
                    "role": "system",
                    "content": "You are a software architect. Analyze the task and provide a detailed implementation approach, considering: 1) architecture patterns, 2) edge cases, 3) security concerns, 4) testing strategy."
                },
                {
                    "role": "user",
                    "content": f"Task: {state['task']}\n\nProvide a concise implementation plan (3-5 bullet points)."
                }
            ])
            state["context"] = strip_think(response.content)
            state["iterations"] += 1
            logger.info("Task analysis complete")
        except Exception as e:
            logger.error(f"Error in analyze_task: {e}")
            state["error"] = str(e)
            state["status"] = "error"

        return state

    async def implement_code(self, state: AgentState) -> AgentState:
        """
        Implement code using fast model for quick generation.

        Uses fast tier (30B MoE) for rapid code generation based on
        the architecture plan from analyze_task.
        """
        try:
            logger.info("Implementing code based on analysis")
            response = await self.fast_model.ainvoke([
                {
                    "role": "system",
                    "content": "You are a code generator. Generate clean, well-documented code following the architecture plan. Include type hints and docstrings."
                },
                {
                    "role": "user",
                    "content": f"Architecture Plan:\n{state['context']}\n\nTask: {state['task']}\n\nGenerate the code:"
                }
            ])
            state["code"] = strip_think(response.content)
            state["iterations"] += 1
            logger.info("Code implementation complete")
        except Exception as e:
            logger.error(f"Error in implement_code: {e}")
            state["error"] = str(e)
            state["status"] = "error"

        return state

    async def review_code(self, state: AgentState) -> AgentState:
        """
        Review code using quality model for thorough analysis.

        Uses quality tier (70B-80B) for comprehensive code review
        covering security, performance, and best practices.
        """
        try:
            logger.info("Reviewing generated code")
            response = await self.quality_model.ainvoke([
                {
                    "role": "system",
                    "content": "You are a senior code reviewer. Review the code for: 1) security vulnerabilities, 2) performance issues, 3) code quality, 4) test coverage. Provide specific, actionable feedback."
                },
                {
                    "role": "user",
                    "content": f"Task: {state['task']}\n\nCode:\n{state['code']}\n\nProvide review:"
                }
            ])
            state["review"] = strip_think(response.content)
            state["iterations"] += 1
            state["status"] = "completed"
            logger.info("Code review complete")
        except Exception as e:
            logger.error(f"Error in review_code: {e}")
            state["error"] = str(e)
            state["status"] = "error"

        return state

    async def run(self, task: str) -> dict:
        """
        Execute the full DevOps agent workflow.

        Args:
            task: Description of the task to implement

        Returns:
            dict with keys: task, context, code, review, status, iterations

        Example:
            result = await agent.run("Create a function to validate email addresses")
            if result["status"] == "completed":
                print(result["code"])
                print(result["review"])
        """
        initial_state = {
            "task": task,
            "context": "",
            "code": "",
            "review": "",
            "status": "pending",
            "iterations": 0,
            "error": ""
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
            logger.info(f"Workflow completed in {result['iterations']} steps")
            return result
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {
                **initial_state,
                "status": "error",
                "error": str(e)
            }


class ResearchAgent:
    """
    Research agent that synthesizes information from multiple sources.

    Uses the ultra tier for deep research and synthesis tasks.

    Usage:
        agent = ResearchAgent()
        result = await agent.run(
            query="What are the latest developments in quantum computing?",
            sources=["..."]
        )
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize research agent with the ultra tier model."""
        settings = settings or get_settings()
        # Higher temperature for creative synthesis
        self.ultra_model = _chat_model(settings, settings.agent_ultra_model, temperature=0.4)
        logger.info("Research agent initialized with ultra tier model")

    async def run(self, query: str, sources: list[str]) -> dict:
        """
        Execute research synthesis workflow.

        Args:
            query: Research question
            sources: List of source documents/URLs

        Returns:
            dict with keys: query, synthesis, sources_used, status
        """
        try:
            logger.info(f"Starting research on: {query}")

            context = "\n\n".join([f"Source {i+1}:\n{src}" for i, src in enumerate(sources)])

            response = await self.ultra_model.ainvoke([
                {
                    "role": "system",
                    "content": "You are a research analyst. Synthesize information from multiple sources into a comprehensive, well-structured analysis. Cite sources explicitly."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nSources:\n{context}\n\nProvide detailed synthesis:"
                }
            ])

            return {
                "query": query,
                "synthesis": strip_think(response.content),
                "sources_used": len(sources),
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Research error: {e}")
            return {
                "query": query,
                "synthesis": "",
                "sources_used": 0,
                "status": "error",
                "error": str(e)
            }


# Factory function for easy agent instantiation
def create_agent(agent_type: str, settings: Optional[Settings] = None):
    """
    Create an agent instance by type.

    Args:
        agent_type: One of: "devops", "research"
        settings: Settings to use (default: the application settings)

    Returns:
        Agent instance

    Example:
        agent = create_agent("devops")
        result = await agent.run("Create a validation function")
    """
    agents = {
        "devops": DevOpsAgent,
        "research": ResearchAgent
    }

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")

    return agents[agent_type](settings=settings)
