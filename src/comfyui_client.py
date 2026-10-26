"""Minimal async client for the ComfyUI HTTP API."""

import re
import uuid
from typing import Any, Optional

import httpx

DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024


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
        self._prompt_ids_supported = False

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

    async def submit(self, workflow: dict, prompt_id: Optional[str] = None) -> str:
        """Queue an API-format workflow; returns the prompt id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        if prompt_id is not None:
            # Check before POST: old servers can ignore caller IDs and leave an
            # accepted job unidentifiable if submission times out.
            if not self._prompt_ids_supported:
                stats = await self._request("GET", "/system_stats")
                try:
                    version = stats.json()["system"]["comfyui_version"]
                    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:$|[-+.])", version)
                    supported = match is not None and tuple(map(int, match.groups())) >= (0, 42, 1)
                except (ValueError, KeyError, TypeError):
                    supported = False
                if not supported:
                    raise ComfyUIError("Media jobs require ComfyUI 0.42.1 or newer for caller-assigned prompt IDs")
                self._prompt_ids_supported = True
            payload["prompt_id"] = prompt_id
        response = await self._request(
            "POST", "/prompt", json=payload
        )
        try:
            returned_id = response.json()["prompt_id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ComfyUIError("ComfyUI returned no prompt_id") from exc
        if not isinstance(returned_id, str) or not returned_id:
            raise ComfyUIError("ComfyUI returned no prompt_id")
        if prompt_id is not None and returned_id != prompt_id:
            raise ComfyUIError("ComfyUI did not honor the caller-assigned prompt ID")
        return returned_id

    async def history(self, prompt_id: str) -> Optional[dict]:
        """The history entry for a prompt, or None while it has not finished."""
        response = await self._request("GET", f"/history/{prompt_id}")
        try:
            entry = response.json().get(prompt_id)
        except (ValueError, AttributeError) as exc:
            raise ComfyUIError("ComfyUI returned malformed history") from exc
        return entry or None

    async def fetch_output(
        self, filename: str, subfolder: str = "", type_: str = "output",
        *, max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> bytes:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        try:
            async with self._client.stream(
                "GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": type_},
            ) as response:
                try:
                    declared_size = int(response.headers.get("content-length", "0"))
                except ValueError:
                    declared_size = 0
                if declared_size > max_bytes:
                    raise ComfyUIError("output image too large")
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=min(65536, max_bytes + 1)):
                    if len(data) + len(chunk) > max_bytes:
                        raise ComfyUIError("output image too large")
                    data.extend(chunk)
                if response.status_code >= 400:
                    error = httpx.Response(response.status_code, headers=response.headers, content=bytes(data))
                    raise ComfyUIError(_describe(error))
                return bytes(data)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"ComfyUI unreachable: {exc.__class__.__name__}: {exc}") from exc

    async def cancel(self, prompt_id: str) -> bool:
        """Cancel one prompt through ComfyUI's atomic, state-agnostic jobs API."""
        response = await self._request("POST", f"/api/jobs/{prompt_id}/cancel")
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIError("ComfyUI returned a malformed cancellation response") from exc
        if not isinstance(body, dict) or not isinstance(body.get("cancelled"), bool):
            raise ComfyUIError("ComfyUI returned a malformed cancellation response")
        return body["cancelled"]

    async def interrupt(self, prompt_id: str) -> None:
        """Interrupt only if `prompt_id` is the prompt ComfyUI is running right now."""
        await self._request("POST", "/interrupt", json={"prompt_id": prompt_id})

    async def queue(self) -> dict:
        response = await self._request("GET", "/queue")
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIError("ComfyUI returned a malformed queue") from exc
        return body if isinstance(body, dict) else {}

    async def delete_queued(self, prompt_id: str) -> None:
        await self._request("POST", "/queue", json={"delete": [prompt_id]})
