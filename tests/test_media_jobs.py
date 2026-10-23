"""Image job API against a mocked ComfyUI (httpx MockTransport); no GPU, no network."""

import json
import os
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from src import media_jobs
from src.api_gateway import app
from src.comfyui_client import ComfyUIClient, ComfyUIError
from tests.conftest import TEST_API_KEY

HEADERS = {"X-API-Key": TEST_API_KEY}
PNG = b"\x89PNG\r\n\x1a\n" + b"synthetic-image-bytes" * 8


class FakeComfy:
    """Scriptable stand-in for the ComfyUI HTTP API."""

    def __init__(self):
        self.submitted = []
        self.interrupts = 0
        self.interrupt_bodies = []
        self.deleted = []
        self.mode = "success"  # success | error | hang | reject
        self.history_calls = 0
        self.running = "auto"  # prompt id ComfyUI reports as running; "auto" = ours
        self.images = [{"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}]

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/prompt":
            if self.mode == "reject":
                return httpx.Response(400, json={
                    "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                    "node_errors": {"9": {"errors": [{"message": "Value not in list: ckpt_name"}]}},
                })
            body = json.loads(request.content)
            self.submitted.append(body)
            return httpx.Response(200, json={"prompt_id": f"pid-{len(self.submitted)}"})
        if path.startswith("/history/"):
            self.history_calls += 1
            pid = path.rsplit("/", 1)[1]
            if self.mode == "boom":
                return httpx.Response(500, text="boom")
            if self.mode == "hang":
                return httpx.Response(200, json={})
            if self.mode == "error":
                return httpx.Response(200, json={pid: {
                    "status": {"status_str": "error", "completed": False, "messages": [
                        ["execution_error", {"exception_message": "CUDA out of memory"}]]},
                    "outputs": {}}})
            return httpx.Response(200, json={pid: {
                "status": {"status_str": "success", "completed": True, "messages": []},
                "outputs": {"9": {"images": self.images}}}})
        if path == "/view":
            return httpx.Response(200, content=PNG)
        if path == "/interrupt":
            self.interrupts += 1
            self.interrupt_bodies.append(json.loads(request.content) if request.content else None)
            return httpx.Response(200)
        if path == "/queue" and request.method == "GET":
            running = f"pid-{len(self.submitted)}" if self.running == "auto" else self.running
            return httpx.Response(200, json={
                "queue_running": [[0, running, {}]] if running else [],
                "queue_pending": [[1, f"pid-{len(self.submitted)}", {}]],
            })
        if path == "/queue":
            self.deleted.append(json.loads(request.content))
            return httpx.Response(200)
        return httpx.Response(404)


@pytest.fixture
def comfy(monkeypatch, settings_env, tmp_path):
    fake = FakeComfy()
    settings_env(
        MEDIA_OUTPUT_DIR=tmp_path / "media",
        MEDIA_JOB_TIMEOUT_S=30,
        MEDIA_MAX_QUEUED_JOBS=2,
        ALLOW_LOOPBACK_UNAUTHENTICATED="false",
    )
    monkeypatch.setattr(media_jobs, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(
        media_jobs,
        "make_comfy_client",
        lambda settings: ComfyUIClient("http://comfy.invalid", transport=httpx.MockTransport(fake.handle)),
    )
    fake.out = tmp_path / "media"
    return fake


@pytest.fixture
def client(comfy):
    with TestClient(app, client=("203.0.113.7", 1234)) as c:
        yield c


def submit(client, **params):
    params.setdefault("prompt", "a stone bridge")
    return client.post("/v1/media/jobs", json={"workflow": "sdxl_t2i", "params": params}, headers=HEADERS)


def wait_for(client, job_id, wanted, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/media/jobs/{job_id}", headers=HEADERS).json()
        if body["status"] in wanted:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job stuck at {body['status']}")


def test_workflows_listing(client):
    response = client.get("/v1/media/workflows", headers=HEADERS)
    assert response.status_code == 200
    by_name = {w["name"]: w for w in response.json()["workflows"]}
    assert {"qwen_image_t2i", "sdxl_t2i"} <= set(by_name)
    params = by_name["sdxl_t2i"]["params"]
    assert set(params) == {"prompt", "negative_prompt", "seed", "width", "height", "steps"}
    assert params["steps"]["max"] == 60
    assert params["prompt"]["required"] is True
    assert "targets" not in params["prompt"]  # graph internals are not exposed


@pytest.mark.parametrize("workflow,params", [
    ("../etc/passwd", {"prompt": "x"}),
    ("SDXL_T2I", {"prompt": "x"}),
    ("sdxl_t2i.api", {"prompt": "x"}),
    ("does_not_exist", {"prompt": "x"}),
    ("sdxl_t2i", {}),
    ("sdxl_t2i", {"prompt": "x", "cfg": 3}),
    ("sdxl_t2i", {"prompt": 5}),
    ("sdxl_t2i", {"prompt": "x", "steps": "20"}),
    ("sdxl_t2i", {"prompt": "x", "steps": True}),
    ("sdxl_t2i", {"prompt": "x", "steps": 61}),
    ("sdxl_t2i", {"prompt": "x", "steps": 0}),
    ("sdxl_t2i", {"prompt": "x", "width": 1000}),
    ("sdxl_t2i", {"prompt": "x", "width": 128}),
    ("sdxl_t2i", {"prompt": "x", "seed": -1}),
    ("sdxl_t2i", {"prompt": "x" * 2001}),
])
def test_invalid_submissions_are_422(client, comfy, workflow, params):
    response = client.post("/v1/media/jobs", json={"workflow": workflow, "params": params}, headers=HEADERS)
    assert response.status_code == 422, response.text
    assert comfy.submitted == []


def test_happy_path_saves_and_serves_output(client, comfy):
    response = submit(client, width=512, height=768, steps=12, seed=7, negative_prompt="blurry")
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["status"] == "queued"

    job = wait_for(client, job_id, {"succeeded", "failed"})
    assert job["status"] == "succeeded", job
    assert job["error"] is None
    assert job["outputs"] == [{"index": 0, "content_type": "image/png", "size": len(PNG)}]
    assert job["created_at"] and job["finished_at"]

    saved = comfy.out / job_id / "0.png"
    assert saved.read_bytes() == PNG

    served = client.get(f"/v1/media/jobs/{job_id}/outputs/0", headers=HEADERS)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content == PNG

    graph = comfy.submitted[0]["prompt"]
    assert graph["6"]["inputs"]["text"] == "a stone bridge"
    assert graph["7"]["inputs"]["text"] == "blurry"
    assert graph["3"]["inputs"]["seed"] == 7 and graph["3"]["inputs"]["steps"] == 12
    assert (graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]) == (512, 768)


def test_prompt_text_is_data_not_graph_structure(client, comfy):
    evil = '"}, "99": {"class_type": "LoadImage"}, "x": {"'
    job_id = submit(client, prompt=evil).json()["id"]
    wait_for(client, job_id, {"succeeded", "failed"})
    graph = comfy.submitted[0]["prompt"]
    assert "99" not in graph and "x" not in graph
    assert graph["6"]["inputs"]["text"] == evil
    assert {n["class_type"] for n in graph.values()} == {
        "CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"}


def test_seed_is_random_when_omitted(client, comfy):
    for _ in range(2):
        wait_for(client, submit(client).json()["id"], {"succeeded", "failed"})
    seeds = [s["prompt"]["3"]["inputs"]["seed"] for s in comfy.submitted]
    assert all(0 <= s <= 4294967295 for s in seeds)
    assert seeds[0] != seeds[1]


def test_unknown_job_is_404(client):
    assert client.get("/v1/media/jobs/nope", headers=HEADERS).status_code == 404
    assert client.delete("/v1/media/jobs/nope", headers=HEADERS).status_code == 404
    assert client.get("/v1/media/jobs/nope/outputs/0", headers=HEADERS).status_code == 404


def test_output_index_out_of_range_is_404(client):
    job_id = submit(client).json()["id"]
    wait_for(client, job_id, {"succeeded"})
    assert client.get(f"/v1/media/jobs/{job_id}/outputs/5", headers=HEADERS).status_code == 404


def test_execution_error_marks_failed_with_message(client, comfy):
    comfy.mode = "error"
    job = wait_for(client, submit(client).json()["id"], {"succeeded", "failed"})
    assert job["status"] == "failed"
    assert "CUDA out of memory" in job["error"]


def test_comfyui_validation_rejection_marks_failed_with_message(client, comfy):
    comfy.mode = "reject"
    job = wait_for(client, submit(client).json()["id"], {"succeeded", "failed"})
    assert job["status"] == "failed"
    assert "Prompt outputs failed validation" in job["error"]
    assert "Value not in list" in job["error"]


def test_timeout_interrupts_and_fails(comfy, settings_env, tmp_path):
    comfy.mode = "hang"
    settings_env(MEDIA_OUTPUT_DIR=tmp_path / "media", MEDIA_JOB_TIMEOUT_S=1, ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    with TestClient(app, client=("203.0.113.7", 1234)) as client:
        job = wait_for(client, submit(client).json()["id"], {"succeeded", "failed"}, timeout=10)
    assert job["status"] == "failed"
    assert "timed out" in job["error"]
    assert comfy.interrupts >= 1


def test_cancel_running_interrupts(client, comfy):
    comfy.mode = "hang"
    job_id = submit(client).json()["id"]
    wait_for(client, job_id, {"running"})
    response = client.delete(f"/v1/media/jobs/{job_id}", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert comfy.interrupts == 1
    assert comfy.interrupt_bodies == [{"prompt_id": "pid-1"}]
    time.sleep(0.2)  # the worker must not resurrect or fail a cancelled job
    assert client.get(f"/v1/media/jobs/{job_id}", headers=HEADERS).json()["status"] == "cancelled"


def test_cancel_queued_never_reaches_comfyui(client, comfy):
    comfy.mode = "hang"
    first = submit(client).json()["id"]
    wait_for(client, first, {"running"})
    second = submit(client).json()["id"]
    response = client.delete(f"/v1/media/jobs/{second}", headers=HEADERS)
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert comfy.interrupts == 0
    client.delete(f"/v1/media/jobs/{first}", headers=HEADERS)
    time.sleep(0.2)
    assert len(comfy.submitted) == 1  # the cancelled job was skipped


def test_cancel_finished_is_409(client):
    job_id = submit(client).json()["id"]
    wait_for(client, job_id, {"succeeded"})
    assert client.delete(f"/v1/media/jobs/{job_id}", headers=HEADERS).status_code == 409


def test_queue_full_is_429(client, comfy):
    comfy.mode = "hang"
    first = submit(client).json()["id"]
    wait_for(client, first, {"running"})
    assert submit(client).status_code == 202
    assert submit(client).status_code == 202
    assert submit(client).status_code == 429


def test_symlinked_job_dir_is_refused(client, comfy, monkeypatch, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    comfy.out.mkdir(parents=True, exist_ok=True)
    job_id = "a" * 32
    os.symlink(victim, comfy.out / job_id)
    monkeypatch.setattr(media_jobs, "new_job_id", lambda: job_id)
    job = wait_for(client, submit(client).json()["id"], {"succeeded", "failed"})
    assert job["status"] == "failed"
    assert "symlink" in job["error"]
    assert list(victim.iterdir()) == []


def test_symlinked_target_file_is_refused(tmp_path):
    root = tmp_path / "media"
    (root / ("b" * 32)).mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    os.symlink(victim, root / ("b" * 32) / "0.png")
    with pytest.raises(media_jobs.UnsafeOutputPath):
        media_jobs.write_output(root, "b" * 32, 0, "png", b"data")
    assert victim.read_text() == "keep"


def test_output_extension_comes_from_allowlist(client, comfy):
    comfy.images = [{"filename": "../../evil.sh", "subfolder": "../..", "type": "output"}]
    job = wait_for(client, submit(client).json()["id"], {"succeeded", "failed"})
    assert job["status"] == "failed"
    assert not any(p.suffix == ".sh" for p in comfy.out.rglob("*"))


def test_comfy_filenames_never_become_paths(client, comfy):
    comfy.images = [{"filename": "../../../x/pic.JPG", "subfolder": "a/b", "type": "output"}]
    job_id = submit(client).json()["id"]
    job = wait_for(client, job_id, {"succeeded", "failed"})
    assert job["status"] == "succeeded", job
    assert job["outputs"][0]["content_type"] == "image/jpeg"
    assert [p.relative_to(comfy.out).as_posix() for p in comfy.out.rglob("*") if p.is_file()] == [f"{job_id}/0.jpg"]


def test_media_routes_require_auth(client):
    for method, path in [
        ("get", "/v1/media/workflows"),
        ("post", "/v1/media/jobs"),
        ("get", "/v1/media/jobs/abc"),
        ("delete", "/v1/media/jobs/abc"),
        ("get", "/v1/media/jobs/abc/outputs/0"),
    ]:
        assert getattr(client, method)(path).status_code == 401, path


def test_client_surfaces_validation_errors():
    import asyncio

    async def run():
        fake = FakeComfy()
        fake.mode = "reject"
        comfy_client = ComfyUIClient("http://comfy.invalid", transport=httpx.MockTransport(fake.handle))
        try:
            await comfy_client.submit({"1": {}})
        finally:
            await comfy_client.aclose()

    with pytest.raises(ComfyUIError) as err:
        asyncio.run(run())
    assert "Prompt outputs failed validation" in str(err.value)
    assert "Value not in list" in str(err.value)


def test_timeout_does_not_interrupt_other_clients_work(comfy, settings_env, tmp_path):
    comfy.mode = "hang"
    comfy.running = "someone-elses-prompt"
    settings_env(MEDIA_OUTPUT_DIR=tmp_path / "media", MEDIA_JOB_TIMEOUT_S=1, ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    with TestClient(app, client=("203.0.113.7", 1234)) as client:
        job = wait_for(client, submit(client).json()["id"], {"succeeded", "failed"}, timeout=10)
    assert job["status"] == "failed"
    assert comfy.deleted == [{"delete": ["pid-1"]}]
    assert comfy.interrupts == 0


def test_timeout_of_running_prompt_interrupts_once(comfy, settings_env, tmp_path):
    comfy.mode = "hang"
    settings_env(MEDIA_OUTPUT_DIR=tmp_path / "media", MEDIA_JOB_TIMEOUT_S=1, ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    with TestClient(app, client=("203.0.113.7", 1234)) as client:
        wait_for(client, submit(client).json()["id"], {"failed"}, timeout=10)
    assert comfy.interrupts == 1
    assert comfy.interrupt_bodies == [{"prompt_id": "pid-1"}]


def test_cancel_does_not_interrupt_other_clients_work(client, comfy):
    comfy.mode = "hang"
    comfy.running = "someone-elses-prompt"
    job_id = submit(client).json()["id"]
    wait_for(client, job_id, {"running"})
    time.sleep(0.1)
    assert client.delete(f"/v1/media/jobs/{job_id}", headers=HEADERS).status_code == 200
    assert comfy.deleted == [{"delete": ["pid-1"]}]
    assert comfy.interrupts == 0


def test_failed_job_is_removed_from_comfyui(client, comfy):
    comfy.mode = "boom"
    job = wait_for(client, submit(client).json()["id"], {"failed", "succeeded"})
    assert job["status"] == "failed"
    assert comfy.deleted == [{"delete": ["pid-1"]}]


@pytest.mark.parametrize("name", ["sdxl_t2i\n", "sdxl_t2i\n\n"])
def test_trailing_newline_in_workflow_name_rejected(client, name):
    response = client.post("/v1/media/jobs", json={"workflow": name, "params": {"prompt": "x"}}, headers=HEADERS)
    assert response.status_code == 422


def test_malformed_params_file_is_reported_not_crashing(client, comfy, settings_env, tmp_path):
    import shutil
    wf = tmp_path / "wf"
    shutil.copytree(media_jobs.REPO_ROOT / "config/media_workflows", wf)
    (wf / "broken_t2i.api.json").write_text(json.dumps({"1": {"class_type": "X", "inputs": {"a": 1}}}))
    (wf / "broken_t2i.params.json").write_text(json.dumps({"p": {"type": "integer", "targets": "1.a"}}))
    (wf / "broken2_t2i.api.json").write_text(json.dumps({"1": {"class_type": "X", "inputs": {"a": 1}}}))
    (wf / "broken2_t2i.params.json").write_text(json.dumps({"p": "nope"}))
    settings_env(MEDIA_WORKFLOWS_DIR=wf, MEDIA_OUTPUT_DIR=tmp_path / "media", ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    with TestClient(app, client=("203.0.113.7", 1234)) as c:
        listing = c.get("/v1/media/workflows", headers=HEADERS)
        assert listing.status_code == 200
        names = {w["name"] for w in listing.json()["workflows"]}
        assert "sdxl_t2i" in names and not names & {"broken_t2i", "broken2_t2i"}
        for bad in ("broken_t2i", "broken2_t2i"):
            r = c.post("/v1/media/jobs", json={"workflow": bad, "params": {"p": 1}}, headers=HEADERS)
            assert r.status_code == 422


def test_cancel_during_download_stops_writing(client, comfy):
    comfy.images = [
        {"filename": f"img{i}.png", "subfolder": "", "type": "output"} for i in range(3)
    ]
    original = media_jobs.write_output
    holder = {}

    def cancelling_write(*args, **kwargs):
        result = original(*args, **kwargs)
        job = media_jobs.manager.jobs[args[1]]
        job.status = media_jobs.CANCELLED  # cancelled mid-download
        holder["job"] = job
        return result

    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(media_jobs, "write_output", cancelling_write)
    try:
        job_id = submit(client).json()["id"]
        wait_for(client, job_id, {"cancelled"})
        time.sleep(0.2)
    finally:
        mp.undo()
    files = [p for p in comfy.out.rglob("*") if p.is_file()]
    assert len(files) == 1


def test_finished_jobs_are_pruned(client, comfy, monkeypatch):
    monkeypatch.setattr(media_jobs, "MAX_RETAINED_JOBS", 3)
    monkeypatch.setattr(media_jobs, "POLL_INTERVAL_S", 0.01)
    comfy_ids = []
    for _ in range(6):
        job_id = submit(client).json()["id"]
        comfy_ids.append(job_id)
        wait_for(client, job_id, {"succeeded", "failed"})
    assert len(media_jobs.manager.jobs) <= 4  # cap + the job being enqueued
    assert client.get(f"/v1/media/jobs/{comfy_ids[0]}", headers=HEADERS).status_code == 404
    assert client.get(f"/v1/media/jobs/{comfy_ids[-1]}", headers=HEADERS).status_code == 200


def test_graph_is_dropped_after_submit(client, comfy):
    job_id = submit(client).json()["id"]
    wait_for(client, job_id, {"succeeded"})
    assert media_jobs.manager.jobs[job_id].graph == {}


def test_sdxl_template_prompts_are_blank():
    graph = json.loads((media_jobs.REPO_ROOT / "config/media_workflows/sdxl_t2i.api.json").read_text())
    assert graph["6"]["inputs"]["text"] == "" and graph["7"]["inputs"]["text"] == ""
