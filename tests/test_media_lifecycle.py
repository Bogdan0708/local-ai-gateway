"""Lifecycle probes using real async HTTP clients against a synthetic ComfyUI."""

import asyncio
import json
import threading
import uuid

import httpx
import pytest

from src import media_jobs
from src.comfyui_client import ComfyUIClient
from src.config import get_settings


class OutputStream(httpx.AsyncByteStream):
    def __init__(self, backend):
        self.backend = backend
        self.closed = False

    async def __aiter__(self):
        await self.backend.gate("download")
        for _ in range(self.backend.output_chunks):
            self.backend.chunks_read += 1
            yield b"image123"

    async def aclose(self):
        self.closed = True


class Backend:
    def __init__(self):
        self.delay = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.submitted = []
        self.remote = {}
        self.running = None
        self.streams = []
        self.output_chunks = 1
        self.chunks_read = 0
        self.images = 1
        self.downloads = 0
        self.hang_history = False
        self.hang_cleanup = False
        self.version = "0.42.1"
        self.output_length = None
        self.cleanup_entered = asyncio.Event()
        self.cancel_body = None
        self.cancellations = []
        self.legacy_cancellations = []
        self.cleanup_error = False

    async def gate(self, stage):
        if self.delay == stage:
            self.entered.set()
            await self.release.wait()

    async def handle(self, request):
        path = request.url.path
        if path == "/system_stats":
            await self.gate("metadata")
            return httpx.Response(200, json={"system": {"comfyui_version": self.version}})
        if path == "/prompt":
            body = json.loads(request.content)
            pid = body.get("prompt_id") or str(uuid.uuid4())
            self.submitted.append(body)
            self.remote[pid] = "pending" if self.running else "running"
            self.running = self.running or pid
            await self.gate("submit")
            return httpx.Response(200, json={"prompt_id": pid})
        if path.startswith("/history/"):
            await self.gate("history")
            pid = path.rsplit("/", 1)[1]
            if self.hang_history:
                return httpx.Response(200, json={})
            self.remote.pop(pid, None)
            if self.running == pid:
                self.running = None
            return httpx.Response(200, json={pid: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"9": {"images": [{"filename": f"{i}.png"} for i in range(self.images)]}},
            }})
        if path == "/view":
            self.downloads += 1
            if self.images > 1 and self.downloads == 2:
                self.delay = "download"
            stream = OutputStream(self)
            self.streams.append(stream)
            headers = {} if self.output_length is None else {"content-length": self.output_length}
            return httpx.Response(200, headers=headers, stream=stream)
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            pid = path.split("/")[3]
            self.cancellations.append(pid)
            if self.hang_cleanup:
                self.cleanup_entered.set()
                await asyncio.Event().wait()
            if self.cleanup_error:
                raise RuntimeError("synthetic cleanup failure")
            if self.cancel_body is not None:
                return httpx.Response(200, json=self.cancel_body)
            cancelled = pid in self.remote
            self.remote.pop(pid, None)
            if self.running == pid:
                self.running = None
            return httpx.Response(200, json={"cancelled": cancelled})
        if path in {"/queue", "/interrupt"}:
            self.legacy_cancellations.append(path)
            raise AssertionError("Legacy cancellation endpoints must not be used")
        return httpx.Response(404)


@pytest.fixture
def setup_manager(monkeypatch, settings_env, tmp_path):
    settings_env(MEDIA_OUTPUT_DIR=tmp_path / "media")

    async def setup(backend, timeout=0.1):
        # Accelerate only the internal deadline; configuration validation is unchanged.
        settings = get_settings().model_copy(update={"media_job_timeout_s": timeout})
        monkeypatch.setattr(media_jobs, "get_settings", lambda: settings)
        monkeypatch.setattr(media_jobs, "POLL_INTERVAL_S", 0.005)
        monkeypatch.setattr(media_jobs, "make_comfy_client", lambda settings: ComfyUIClient(
            "http://comfy.invalid", transport=httpx.MockTransport(backend.handle),
        ))
        manager = media_jobs.MediaJobManager()
        await manager.start()
        return manager

    return setup, tmp_path / "media"


def enqueue(manager):
    job = media_jobs.Job(id=uuid.uuid4().hex, workflow="synthetic", params={}, graph={"1": {}})
    manager.enqueue(job)
    return job


async def finished(job):
    async def wait():
        while job.status not in media_jobs.FINISHED:
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 1)


@pytest.mark.parametrize("stage", ["metadata", "submit", "history", "download"])
def test_deadline_covers_every_network_stage(setup_manager, stage):
    setup, output = setup_manager

    async def run():
        backend = Backend()
        backend.delay = stage
        manager = await setup(backend)
        job = enqueue(manager)
        try:
            await asyncio.wait_for(backend.entered.wait(), 1)
            await finished(job)
            assert job.status == media_jobs.FAILED
            assert "timed out" in job.error
        finally:
            await manager.stop()
        assert backend.remote == {}
        assert not list(output.rglob("*.png"))

    asyncio.run(run())


def test_cancel_during_download_closes_stream_without_saving(setup_manager):
    setup, output = setup_manager

    async def run():
        backend = Backend()
        backend.delay = "download"
        manager = await setup(backend, timeout=5)
        job = enqueue(manager)
        try:
            await asyncio.wait_for(backend.entered.wait(), 1)
            await manager.cancel(job)
            backend.release.set()
            await asyncio.sleep(0.03)
            assert job.status == media_jobs.CANCELLED
            assert not list(output.rglob("*.png"))
            assert backend.streams[0].closed
        finally:
            backend.release.set()
            await manager.stop()

    asyncio.run(run())


def test_shutdown_cancels_accepted_submission_and_queued_jobs(setup_manager):
    setup, output = setup_manager

    async def run():
        backend = Backend()
        backend.delay = "submit"
        manager = await setup(backend, timeout=5)
        first = enqueue(manager)
        await asyncio.wait_for(backend.entered.wait(), 1)
        second = enqueue(manager)
        await manager.stop()
        assert first.status == second.status == media_jobs.CANCELLED
        assert first.finished_at and second.finished_at
        assert backend.remote == {}
        assert not manager.ready
        assert not list(output.rglob("*.png"))

    asyncio.run(run())


def test_oversized_output_stops_reading_early(setup_manager, monkeypatch):
    setup, output = setup_manager
    monkeypatch.setattr(media_jobs, "MAX_OUTPUT_BYTES", 16)

    async def run():
        backend = Backend()
        backend.output_chunks = 64
        manager = await setup(backend, timeout=5)
        job = enqueue(manager)
        try:
            await finished(job)
            assert job.status == media_jobs.FAILED
            assert backend.chunks_read < 64
            assert backend.streams[0].closed
            assert not list(output.rglob("*.png"))
        finally:
            await manager.stop()

    asyncio.run(run())


def test_cancel_removes_already_downloaded_outputs(setup_manager):
    setup, output = setup_manager

    async def run():
        backend = Backend()
        backend.images = 2
        manager = await setup(backend, timeout=5)
        job = enqueue(manager)
        try:
            await asyncio.wait_for(backend.entered.wait(), 1)
            assert len(list(output.rglob("*.png"))) == 1
            await manager.cancel(job)
            backend.release.set()
            await asyncio.sleep(0.03)
            assert not list(output.rglob("*.png"))
            assert job.outputs == []
        finally:
            backend.release.set()
            await manager.stop()

    asyncio.run(run())


def test_cancel_waits_for_inflight_file_write_then_removes_it(setup_manager, monkeypatch):
    setup, output = setup_manager
    entered, release = threading.Event(), threading.Event()
    original = media_jobs.write_output

    def delayed_write(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)

    monkeypatch.setattr(media_jobs, "write_output", delayed_write)

    async def run():
        manager = await setup(Backend(), timeout=5)
        job = enqueue(manager)
        cancellation = None
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            cancellation = asyncio.create_task(manager.cancel(job))
            await asyncio.sleep(0.03)
            assert not cancellation.done()
            release.set()
            await asyncio.wait_for(cancellation, 1)
            assert job.status == media_jobs.CANCELLED
            assert not list(output.rglob("*.png"))
        finally:
            release.set()
            if cancellation is not None:
                await cancellation
            await manager.stop()

    asyncio.run(run())


@pytest.mark.parametrize("version", ["0.41.0", "unknown"])
def test_unsupported_backend_is_refused_before_submission(setup_manager, version):
    setup, _ = setup_manager

    async def run():
        backend = Backend()
        backend.version = version
        manager = await setup(backend, timeout=5)
        job = enqueue(manager)
        try:
            await finished(job)
            assert job.status == media_jobs.FAILED
            assert backend.submitted == []
            assert backend.remote == {}
        finally:
            await manager.stop()

    asyncio.run(run())


def test_timeout_preserves_other_clients_prompt_and_processes_next_job(setup_manager):
    setup, _ = setup_manager

    async def run():
        backend = Backend()
        backend.running = "other-client"
        backend.remote[backend.running] = "running"
        backend.delay = "submit"
        manager = await setup(backend)
        first = enqueue(manager)
        try:
            await asyncio.wait_for(backend.entered.wait(), 1)
            second = enqueue(manager)
            await asyncio.wait_for(first._settled.wait(), 1)
            backend.release.set()
            await asyncio.wait_for(second._settled.wait(), 1)
            assert first.status == media_jobs.FAILED
            assert second.status == media_jobs.SUCCEEDED
            assert backend.running == "other-client"
            assert backend.remote == {"other-client": "running"}
            assert manager.ready
        finally:
            backend.release.set()
            await manager.stop()

    asyncio.run(run())


def test_shutdown_during_remote_cleanup_waits_for_bounded_cleanup(setup_manager, monkeypatch, caplog):
    setup, _ = setup_manager
    monkeypatch.setattr(media_jobs, "REMOTE_CLEANUP_TIMEOUT_S", 0.05)

    async def run():
        backend = Backend()
        backend.delay = "history"
        backend.hang_cleanup = True
        manager = await setup(backend, timeout=0.03)
        job = enqueue(manager)
        await asyncio.wait_for(backend.cleanup_entered.wait(), 1)
        await asyncio.wait_for(manager.stop(), 0.5)
        assert job.status == media_jobs.FAILED
        assert job._settled.is_set()
        assert not manager.ready
        assert "cancellation could not be confirmed" in caplog.text

    asyncio.run(run())


@pytest.mark.parametrize("length", ["1024", "1", "invalid"])
def test_download_limit_checks_headers_and_actual_stream_size(setup_manager, monkeypatch, length):
    setup, output = setup_manager
    monkeypatch.setattr(media_jobs, "MAX_OUTPUT_BYTES", 16)

    async def run():
        backend = Backend()
        backend.output_chunks = 64
        backend.output_length = length
        manager = await setup(backend, timeout=5)
        job = enqueue(manager)
        try:
            await asyncio.wait_for(job._settled.wait(), 1)
            assert job.status == media_jobs.FAILED
            assert "too large" in job.error
            if length == "1024":
                assert backend.chunks_read == 0
            else:
                assert backend.chunks_read < 64
            assert backend.streams[0].closed
            assert not list(output.rglob("*.png"))
        finally:
            await manager.stop()

    asyncio.run(run())


def test_partial_file_write_failure_removes_only_new_output(tmp_path, monkeypatch):
    root, job_id = tmp_path / "media", uuid.uuid4().hex
    original_fdopen = media_jobs.os.fdopen

    class FailingWriter:
        def __init__(self, fd, mode):
            self.handle = original_fdopen(fd, mode)

        def __enter__(self):
            return self

        def write(self, data):
            self.handle.write(data[:2])
            raise OSError("synthetic disk failure")

        def __exit__(self, *args):
            self.handle.close()

    monkeypatch.setattr(media_jobs.os, "fdopen", FailingWriter)
    with pytest.raises(OSError, match="synthetic disk failure"):
        media_jobs.write_output(root, job_id, 0, "png", b"image123")
    assert not (root / job_id / "0.png").exists()
    existing = root / job_id / "1.png"
    existing.write_bytes(b"existing output")
    with pytest.raises(media_jobs.UnsafeOutputPath):
        media_jobs.write_output(root, job_id, 1, "png", b"replacement")
    assert existing.read_bytes() == b"existing output"


@pytest.mark.parametrize("cancel_body,unexpected_error", [
    ({"cancelled": 1}, False),
    ([], False),
    ({}, False),
    (None, True),
])
def test_remote_cleanup_failure_keeps_worker_and_local_cleanup_running(
    setup_manager, caplog, cancel_body, unexpected_error,
):
    setup, output = setup_manager

    async def run():
        backend = Backend()
        backend.images = 2
        backend.cancel_body = cancel_body
        backend.cleanup_error = unexpected_error
        manager = await setup(backend)
        first = enqueue(manager)
        try:
            await asyncio.wait_for(backend.entered.wait(), 1)
            assert list(output.rglob("*.png"))
            second = enqueue(manager)
            await asyncio.wait_for(first._settled.wait(), 1)
            backend.release.set()
            assert not list((output / first.id).glob("*.png"))
            await asyncio.wait_for(second._settled.wait(), 1)
            assert first.status == media_jobs.FAILED
            assert second.status == media_jobs.SUCCEEDED
            assert manager.ready
            assert "cleanup" in caplog.text or "cancellation could not be confirmed" in caplog.text
            assert backend.legacy_cancellations == []
        finally:
            backend.release.set()
            await manager.stop()

    asyncio.run(run())


def test_cancellation_uses_atomic_prompt_endpoint_without_legacy_interrupt(setup_manager):
    setup, _ = setup_manager

    async def run():
        backend = Backend()
        backend.delay = "history"
        manager = await setup(backend, timeout=5)
        job = enqueue(manager)
        try:
            await asyncio.wait_for(backend.entered.wait(), 1)
            await manager.cancel(job)
            assert backend.cancellations == [job.prompt_id]
            assert backend.legacy_cancellations == []
            assert backend.remote == {}
        finally:
            backend.release.set()
            await manager.stop()

    asyncio.run(run())
