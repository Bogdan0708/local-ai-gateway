"""Minimal async client for the ComfyUI HTTP API."""

import uuid
from typing import Any, Optional

import httpx


class ComfyUIError(Exception):
    """ComfyUI rejected a request or could not be reached."""


def _describe(response: httpx.Response) -> str:
    """Pull ComfyUI's own message (and per-node validation errors) out of an error body."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    parts: list[str] = []
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            parts.append(str(error.get("message") or error.get("type") or ""))
            if error.get("details"):
                parts.append(str(error["details"]))
        elif error:
            parts.append(str(error))
        node_errors = body.get("node_errors")
        if isinstance(node_errors, dict):
            for node_id, info in node_errors.items():
                for item in (info or {}).get("errors", []) if isinstance(info, dict) else []:
                    parts.append(f"node {node_id}: {item.get('message', item)}")
    text = "; ".join(p for p in parts if p)
    return text or f"HTTP {response.status_code}"


class ComfyUIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        kwargs: dict[str, Any] = {"base_url": base_url.rstrip("/"), "timeout": timeout}
        if transport is not None:
            kwargs["transport"] = transport
        self._client = httpx.AsyncClient(**kwargs)
        self.client_id = uuid.uuid4().hex

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"ComfyUI unreachable: {exc.__class__.__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise ComfyUIError(_describe(response))
        return response

    async def submit(self, workflow: dict) -> str:
        """Queue an API-format workflow; returns the prompt id."""
        response = await self._request(
            "POST", "/prompt", json={"prompt": workflow, "client_id": self.client_id}
        )
        try:
            return str(response.json()["prompt_id"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ComfyUIError("ComfyUI returned no prompt_id") from exc

    async def history(self, prompt_id: str) -> Optional[dict]:
        """The history entry for a prompt, or None while it has not finished."""
        response = await self._request("GET", f"/history/{prompt_id}")
        try:
            entry = response.json().get(prompt_id)
        except (ValueError, AttributeError) as exc:
            raise ComfyUIError("ComfyUI returned malformed history") from exc
        return entry or None

    async def fetch_output(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        response = await self._request(
            "GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": type_}
        )
        return response.content

    async def interrupt(self) -> None:
        await self._request("POST", "/interrupt")

    async def delete_queued(self, prompt_id: str) -> None:
        await self._request("POST", "/queue", json={"delete": [prompt_id]})
