"""Models that think inline (<think>...</think> in the content) must not leak it to clients."""

import asyncio
import json

import httpx
import pytest

from src.api_gateway import call_llm, stream_llm_response
from src.reasoning import ThinkStreamFilter, strip_think


@pytest.mark.parametrize("raw,expected", [
    ("<think>\nplan\n</think>\n\nBucharest.", "Bucharest."),
    ("plan the answer\n</think>\n\nBucharest.", "Bucharest."),  # template opened <think> in the prompt
    ("<think>a</think>One <think>b</think>two", "One two"),
    ("<think>still thinking when max_tokens hit", ""),
    ("No thinking here.", "No thinking here."),
    ("Mentions </ but no tag.", "Mentions </ but no tag."),
])
def test_strip_think(raw, expected):
    assert strip_think(raw) == expected


def stream(pieces):
    f = ThinkStreamFilter()
    return "".join(f.feed(p) for p in pieces) + f.flush()


@pytest.mark.parametrize("pieces", [
    ["<think>plan</think>\n\nBucharest."],
    ["<thi", "nk>pl", "an</thi", "nk>\n\nBuchar", "est."],      # tags split across chunks
    list("<think>plan</think>\n\nBucharest."),                  # one character per chunk
])
def test_stream_filter_drops_think_across_chunk_boundaries(pieces):
    assert stream(pieces) == "Bucharest."


def test_stream_filter_passes_plain_text_and_partial_lookalikes():
    assert stream(["a < b and ", "x <th", "ree"]) == "a < b and x <three"


def test_stream_filter_drops_unterminated_think():
    assert stream(["<think>never closed"]) == ""


@pytest.fixture
def thinking_backend(monkeypatch, settings_env):
    settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_HOST="http://studio.invalid:1234",
                 LMSTUDIO_CHAT_MODEL="thinker")
    client_class = httpx.AsyncClient
    pieces = ["<think>", "plan the", " answer</th", "ink>\n\n", "Buchar", "est."]

    def handle(request):
        if json.loads(request.content).get("stream"):
            body = "".join(
                f'data: {json.dumps({"choices": [{"index": 0, "delta": {"content": p}}]})}\n\n'
                for p in pieces
            ) + 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "".join(pieces)}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9},
        })

    monkeypatch.setattr(
        "src.api_gateway.httpx.AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs),
    )


def test_call_llm_strips_think(thinking_backend):
    result = asyncio.run(call_llm([{"role": "user", "content": "q"}], ""))
    assert result["content"] == "Bucharest."
    assert result["completion_tokens"] == 9   # usage still reports what the model generated


def test_stream_strips_think_and_keeps_finish_chunk(thinking_backend):
    async def collect():
        return [c async for c in stream_llm_response([{"role": "user", "content": "q"}], "")]

    events = asyncio.run(collect())
    payloads = [json.loads(e[6:]) for e in events if e.startswith("data: {")]
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert text == "Bucharest."
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"


def test_research_agent_strips_think(settings_env, monkeypatch):
    from types import SimpleNamespace
    from src.agents import ResearchAgent

    settings_env(LLM_PROVIDER="lmstudio")
    agent = ResearchAgent()

    async def ainvoke(messages):
        return SimpleNamespace(content="<think>weigh the sources</think>\n\nThe door is teal.")

    monkeypatch.setattr(type(agent.ultra_model), "ainvoke", lambda self, m: ainvoke(m))
    result = asyncio.run(agent.run(query="colour?", sources=["teal door"]))
    assert result["synthesis"] == "The door is teal."
