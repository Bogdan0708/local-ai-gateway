"""Authenticated image-generation jobs backed by saved ComfyUI workflows.

Security model:
- Callers pick a workflow by name and supply only the parameters its
  ``<name>.params.json`` whitelists. Values are typed, bounded, and written into
  existing node inputs; they never alter node types or graph structure.
- Output files go only to ``<media_output_dir>/<job_id>/<index>.<ext>`` where the
  job id is server-generated and ``ext`` comes from an allowlist. ComfyUI- and
  client-supplied names are never used as paths.
"""

import asyncio
import copy
import errno
import json
import logging
import os
import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .auth import require_auth
from .comfyui_client import ComfyUIClient, ComfyUIError, DEFAULT_MAX_OUTPUT_BYTES
from .config import Settings, get_settings

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 2.0
MAX_OUTPUT_BYTES = DEFAULT_MAX_OUTPUT_BYTES
REMOTE_CLEANUP_TIMEOUT_S = 5.0
MAX_RETAINED_JOBS = 500  # finished jobs kept in memory
PARAM_TYPES = {"integer", "string"}

WORKFLOW_NAME_RE = re.compile(r"^[a-z0-9_]+$")
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
REPO_ROOT = Path(__file__).resolve().parent.parent

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
FINISHED = {SUCCEEDED, FAILED, CANCELLED}


class UnsafeOutputPath(Exception):
    """An output location is a symlink or escapes the media output directory."""


def new_job_id() -> str:
    return uuid.uuid4().hex


def make_comfy_client(settings: Settings) -> ComfyUIClient:
    return ComfyUIClient(settings.comfyui_host)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# === Safe output storage ===


def _job_dir(root: Path, job_id: str, create: bool) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise UnsafeOutputPath("invalid job id")
    root = Path(root)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    root_real = os.path.realpath(root)
    job_dir = root / job_id
    try:
        if job_dir.is_symlink():
            raise UnsafeOutputPath("job directory is a symlink")
        if create:
            job_dir.mkdir(exist_ok=True)
        if job_dir.is_symlink() or not job_dir.is_dir():  # re-check after mkdir
            raise UnsafeOutputPath("job directory is a symlink or not a directory")
    except OSError as exc:
        raise UnsafeOutputPath(f"job directory unusable: {exc.strerror}") from exc
    real = os.path.realpath(job_dir)
    if os.path.dirname(real) != root_real:
        raise UnsafeOutputPath("job directory escapes the media output directory")
    return job_dir


def write_output(root: Path, job_id: str, index: int, ext: str, data: bytes) -> Path:
    if ext not in CONTENT_TYPES:
        raise UnsafeOutputPath(f"extension {ext!r} not allowed")
    job_dir = _job_dir(root, job_id, create=True)
    name = f"{int(index)}.{ext}"
    try:
        dir_fd = os.open(job_dir, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise UnsafeOutputPath("job directory is a symlink or unusable") from exc
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ELOOP):
                raise UnsafeOutputPath("output target already exists or is a symlink") from exc
            raise
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        except BaseException:
            # Only this newly created file is ours; never delete an existing target.
            os.unlink(name, dir_fd=dir_fd)
            raise
    finally:
        os.close(dir_fd)
    return job_dir / name


def locate_output(root: Path, job_id: str, index: int, ext: str) -> Path:
    """Re-validate a stored output before serving it."""
    job_dir = _job_dir(root, job_id, create=False)
    target = job_dir / f"{int(index)}.{ext}"
    if target.is_symlink() or not target.is_file():
        raise UnsafeOutputPath("output is missing or a symlink")
    if os.path.dirname(os.path.realpath(target)) != os.path.realpath(job_dir):
        raise UnsafeOutputPath("output escapes the job directory")
    return target


# === Workflow templates ===


class WorkflowError(Exception):
    pass


def workflows_dir(settings: Settings) -> Path:
    path = Path(settings.media_workflows_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise WorkflowError(f"{path.name} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"{path.name} unreadable: {exc}") from exc


def load_workflow(settings: Settings, name: str) -> tuple[dict, dict]:
    """Return (api graph, params spec) for a workflow, or raise WorkflowError."""
    if not isinstance(name, str) or not WORKFLOW_NAME_RE.fullmatch(name):
        raise WorkflowError("workflow name must match ^[a-z0-9_]+$")
    base = workflows_dir(settings)
    graph = _read_json(base / f"{name}.api.json")
    spec = _read_json(base / f"{name}.params.json")
    if not isinstance(graph, dict) or not isinstance(spec, dict):
        raise WorkflowError("malformed workflow files")
    for pname, pspec in spec.items():
        if not isinstance(pspec, dict) or pspec.get("type") not in PARAM_TYPES:
            raise WorkflowError(f"param {pname!r} is malformed")
        targets = pspec.get("targets")
        if not isinstance(targets, list) or not targets:
            raise WorkflowError(f"param {pname!r} needs a non-empty targets list")
        for target in targets:
            if not (
                isinstance(target, list)
                and len(target) == 2
                and all(isinstance(t, str) for t in target)
            ):
                raise WorkflowError(f"param {pname!r} has a malformed target")
            node_id, input_name = target
            node = graph.get(node_id)
            if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict) or input_name not in node["inputs"]:
                raise WorkflowError(f"param {pname!r} targets a missing node input {node_id}.{input_name}")
    return graph, spec


def list_workflows(settings: Settings) -> list[dict]:
    found = []
    base = workflows_dir(settings)
    if not base.is_dir():
        return found
    for path in sorted(base.glob("*.api.json")):
        name = path.name[: -len(".api.json")]
        if not WORKFLOW_NAME_RE.fullmatch(name):
            continue
        try:
            _, spec = load_workflow(settings, name)
        except WorkflowError as exc:
            logger.warning("Skipping workflow %s: %s", name, exc)
            continue
        found.append({"name": name, "params": {k: _public_spec(v) for k, v in spec.items()}})
    return found


def _public_spec(spec: dict) -> dict:
    """Parameter constraints without graph internals (node ids)."""
    return {k: v for k, v in spec.items() if k != "targets"}


def validate_params(spec: dict, params: dict) -> dict:
    """Check params against the spec; returns the cleaned values. Raises ValueError(list)."""
    errors: list[str] = []
    for key in params:
        if key not in spec:
            errors.append(f"unknown parameter {key!r}")
    clean: dict[str, Any] = {}
    for key, pspec in spec.items():
        if key not in params:
            if pspec.get("required"):
                errors.append(f"missing required parameter {key!r}")
            elif pspec.get("random_default"):
                clean[key] = secrets.randbelow(int(pspec.get("max", 2**32 - 1)) + 1)
            continue
        value = params[key]
        kind = pspec.get("type")
        if kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{key} must be an integer")
                continue
            if "min" in pspec and value < pspec["min"]:
                errors.append(f"{key} must be >= {pspec['min']}")
            elif "max" in pspec and value > pspec["max"]:
                errors.append(f"{key} must be <= {pspec['max']}")
            elif pspec.get("multiple_of") and value % pspec["multiple_of"]:
                errors.append(f"{key} must be a multiple of {pspec['multiple_of']}")
        elif kind == "string":
            if not isinstance(value, str):
                errors.append(f"{key} must be a string")
                continue
            if pspec.get("required") and not value.strip():
                errors.append(f"{key} must not be empty")
            elif len(value) > pspec.get("max_length", 2000):
                errors.append(f"{key} must be at most {pspec.get('max_length', 2000)} characters")
        else:
            errors.append(f"{key} has an unsupported type in the workflow definition")
            continue
        clean[key] = value
    if errors:
        raise ValueError(errors)
    return clean


def build_graph(graph: dict, spec: dict, values: dict) -> dict:
    """Place validated values into existing node inputs of a copy of the template."""
    built = copy.deepcopy(graph)
    for key, value in values.items():
        for node_id, input_name in spec[key]["targets"]:
            built[node_id]["inputs"][input_name] = value
    return built


# === Job store and worker ===


@dataclass
class Job:
    id: str
    workflow: str
    params: dict
    graph: dict
    status: str = QUEUED
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    prompt_id: Optional[str] = None
    outputs: list[dict] = field(default_factory=list)  # {index, ext, content_type, size}
    _settled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _saved_outputs: list[tuple[int, str]] = field(default_factory=list, repr=False)

    def finish(self, status: str, error: Optional[str] = None) -> None:
        if self.status in FINISHED:
            return
        self.status = status
        self.error = error
        self.finished_at = _now()

    def public(self) -> dict:
        return {
            "id": self.id,
            "workflow": self.workflow,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "outputs": [
                {"index": o["index"], "content_type": o["content_type"], "size": o["size"]}
                for o in self.outputs
            ],
        }


class MediaJobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.settings: Optional[Settings] = None
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[ComfyUIClient] = None
        self._active_job: Optional[Job] = None
        self._job_task: Optional[asyncio.Task] = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("Media worker is already started")
        self.settings = get_settings()
        self.jobs = {}
        self._queue = asyncio.Queue()
        self._client = make_comfy_client(self.settings)
        self._stopping = False
        self._task = asyncio.create_task(self._worker(), name="media-job-worker")

    async def stop(self) -> None:
        self._stopping = True
        for job in self.jobs.values():
            if job.status not in FINISHED:
                job.finish(CANCELLED, "gateway shutting down")
                if job is not self._active_job:
                    job.graph = {}
                    job._settled.set()
        active = self._active_job
        if active is not None:
            if self._job_task is not None and not self._job_task.done():
                self._job_task.cancel()
            # Do not cancel the worker while it is cleaning up remote work/files.
            await active._settled.wait()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
        self._queue = None
        self._task = None

    @property
    def ready(self) -> bool:
        return (
            not self._stopping and self._queue is not None and self.settings is not None
            and self._task is not None and not self._task.done()
        )

    def queued_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == QUEUED)

    def _prune(self) -> None:
        finished = [j for j in self.jobs.values() if j.status in FINISHED]  # insertion order
        for old in finished[: max(0, len(finished) - MAX_RETAINED_JOBS)]:
            del self.jobs[old.id]

    def enqueue(self, job: Job) -> None:
        self._prune()
        self.jobs[job.id] = job
        self._queue.put_nowait(job.id)

    async def cancel(self, job: Job) -> None:
        if job.status in FINISHED:
            return
        job.finish(CANCELLED)
        if job is self._active_job:
            if self._job_task is not None and not self._job_task.done():
                self._job_task.cancel()
            await job._settled.wait()
        else:
            job.graph = {}
            job._settled.set()

    async def _cancel_in_comfy(self, prompt_id: str) -> None:
        try:
            async with asyncio.timeout(REMOTE_CLEANUP_TIMEOUT_S):
                await self._client.cancel(prompt_id)
        except (ComfyUIError, TimeoutError) as exc:
            logger.warning("ComfyUI cancellation could not be confirmed for %s: %s", prompt_id, exc)

    async def _worker(self) -> None:
        queue = self._queue
        while True:
            job_id = await queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                queue.task_done()
                continue  # cancelled while waiting
            job.status = RUNNING
            job.started_at = _now()
            self._active_job = job
            self._job_task = asyncio.create_task(self._run(job), name=f"media-job-{job.id}")
            worker_cancelled = False
            try:
                async with asyncio.timeout(self.settings.media_job_timeout_s):
                    await self._job_task
            except TimeoutError:
                job.finish(FAILED, f"timed out after {self.settings.media_job_timeout_s} s")
            except asyncio.CancelledError:
                job.finish(CANCELLED)
                worker_cancelled = bool(asyncio.current_task().cancelling())
            except ComfyUIError as exc:
                job.finish(FAILED, str(exc))
            except UnsafeOutputPath as exc:
                logger.error("Refused unsafe output for job %s: %s", job.id, exc)
                job.finish(FAILED, f"refused to write output: {exc}")
            except Exception:
                logger.exception("Media job %s crashed", job.id)
                job.finish(FAILED, "internal error")
            else:
                if job.status == RUNNING:
                    job.finish(SUCCEEDED)
            finally:
                try:
                    if job.status != SUCCEEDED:
                        try:
                            await self._discard_remote(job)
                        except Exception:
                            logger.exception("Remote cleanup failed for job %s", job.id)
                        finally:
                            self._remove_saved_outputs(job)
                finally:
                    job.graph = {}
                    self._active_job = None
                    self._job_task = None
                    job._settled.set()
                    queue.task_done()
            if worker_cancelled:
                raise asyncio.CancelledError
            if self._stopping:
                return

    async def _discard_remote(self, job: Job) -> None:
        if job.prompt_id:
            await self._cancel_in_comfy(job.prompt_id)

    def _remove_saved_outputs(self, job: Job) -> None:
        job.outputs = []
        if not job._saved_outputs:
            return
        try:
            directory = _job_dir(self.settings.media_output_dir, job.id, create=False)
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
            try:
                for index, ext in job._saved_outputs:
                    try:
                        os.unlink(f"{index}.{ext}", dir_fd=fd)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(fd)
        except (OSError, UnsafeOutputPath):
            logger.exception("Could not remove incomplete outputs for job %s", job.id)
        finally:
            job._saved_outputs = []

    async def _write_image(self, job: Job, index: int, ext: str, data: bytes) -> None:
        writer = asyncio.create_task(asyncio.to_thread(
            write_output, self.settings.media_output_dir, job.id, index, ext, data,
        ))
        cancelled = False
        while True:
            try:
                await asyncio.shield(writer)
                break
            except asyncio.CancelledError:
                cancelled = True
                if writer.cancelled():
                    raise
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
        job._saved_outputs.append((index, ext))
        if cancelled:
            raise asyncio.CancelledError

    async def _run(self, job: Job) -> None:
        settings, client = self.settings, self._client
        job.prompt_id = str(uuid.uuid4())
        await client.submit(job.graph, prompt_id=job.prompt_id)
        job.graph = {}  # no longer needed; keeps the job store small
        if job.status == CANCELLED:  # cancelled while submitting
            return

        while True:
            entry = await client.history(job.prompt_id)
            if job.status == CANCELLED:
                return
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("completed") or status.get("status_str") in ("success", "error"):
                    break
            await asyncio.sleep(POLL_INTERVAL_S)

        if status.get("status_str") == "error":
            job.finish(FAILED, self._execution_error(status))
            return

        images = [
            img
            for node in (entry.get("outputs") or {}).values()
            for img in (node.get("images") or [])
            if isinstance(img, dict)
        ]
        if not images:
            job.finish(FAILED, "workflow produced no images")
            return
        outputs = []
        for index, img in enumerate(images):
            if job.status == CANCELLED:
                return
            ext = str(img.get("filename", "")).rsplit(".", 1)[-1].lower() if "." in str(img.get("filename", "")) else ""
            if ext not in CONTENT_TYPES:
                job.finish(FAILED, "workflow produced an unsupported output type")
                return
            data = await client.fetch_output(
                str(img["filename"]), str(img.get("subfolder", "")), str(img.get("type", "output")),
                max_bytes=MAX_OUTPUT_BYTES,
            )
            if job.status == CANCELLED:
                return
            if len(data) > MAX_OUTPUT_BYTES:
                job.finish(FAILED, "output image too large")
                return
            await self._write_image(job, index, ext, data)
            outputs.append({"index": index, "ext": ext, "content_type": CONTENT_TYPES[ext], "size": len(data)})
        if job.status == CANCELLED:
            return
        job.outputs = outputs

    @staticmethod
    def _execution_error(status: dict) -> str:
        for message in status.get("messages") or []:
            if (
                isinstance(message, list)
                and len(message) == 2
                and message[0] == "execution_error"
                and isinstance(message[1], dict)
            ):
                text = message[1].get("exception_message")
                if text:
                    return f"ComfyUI execution error: {str(text).strip()[:500]}"
        return "ComfyUI reported an execution error"


manager = MediaJobManager()


async def start_worker() -> None:
    await manager.start()


async def stop_worker() -> None:
    await manager.stop()


# === Routes ===

router = APIRouter(prefix="/v1/media", tags=["media"], dependencies=[Depends(require_auth)])


class JobRequest(BaseModel):
    workflow: str
    params: dict[str, Any] = Field(default_factory=dict)


def _get_job(job_id: str) -> Job:
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/workflows")
async def get_workflows() -> dict:
    return {"workflows": list_workflows(get_settings())}


@router.post("/jobs", status_code=202)
async def create_job(request: JobRequest) -> dict:
    if not manager.ready:
        raise HTTPException(status_code=503, detail="Media worker is not running")
    settings = manager.settings
    try:
        graph, spec = load_workflow(settings, request.workflow)
    except WorkflowError as exc:
        raise HTTPException(status_code=422, detail=f"unknown or invalid workflow: {exc}")
    try:
        values = validate_params(spec, request.params)
        built = build_graph(graph, spec, values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=exc.args[0])
    if manager.queued_count() >= settings.media_max_queued_jobs:
        raise HTTPException(status_code=429, detail="Media job queue is full")
    job = Job(id=new_job_id(), workflow=request.workflow, params=values, graph=built)
    manager.enqueue(job)
    return {"id": job.id, "status": job.status}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    return _get_job(job_id).public()


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if job.status in FINISHED:
        raise HTTPException(status_code=409, detail=f"Job already {job.status}")
    await manager.cancel(job)
    return job.public()


@router.get("/jobs/{job_id}/outputs/{index}")
async def get_output(job_id: str, index: int) -> FileResponse:
    job = _get_job(job_id)
    output = next((o for o in job.outputs if o["index"] == index), None)
    if job.status != SUCCEEDED or output is None:
        raise HTTPException(status_code=404, detail="Output not found")
    try:
        path = locate_output(manager.settings.media_output_dir, job.id, index, output["ext"])
    except UnsafeOutputPath:
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(path, media_type=output["content_type"])
