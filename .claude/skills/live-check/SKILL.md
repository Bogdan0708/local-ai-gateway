---
name: live-check
description: Use when the user requests or authorizes a live gateway smoke test, a gateway restart, or endpoint verification against the configured model backend.
---

Verify the configured gateway and backend. Never print credentials.

1. **Scope:** a live check can restart a service and load models. Run these actions only when the user has authorized them for this check. A code edit, passing tests, or a request for an offline review does not authorize restart or inference. Respect any GPU pause or bounded-test limits. Record a pending live check when it is outside the authorized scope.
2. **Resolve effective configuration** from the intended gateway checkout and the environment it will run with, using `get_settings()`, rather than parsing `.env` with grep/cut. Print only these non-secret fields:
   ```python
   from src.config import get_settings
   s = get_settings()
   host = "127.0.0.1" if s.host == "0.0.0.0" else "::1" if s.host == "::" else s.host
   authority = f"[{host}]" if ":" in host else host
   gateway_url = f"http://{authority}:{s.port}"
   models_url = (f"{s.ollama_host.rstrip('/')}/api/tags" if s.llm_provider == "ollama"
                 else f"{s.active_chat_base_url}/models")
   print({"gateway_url": gateway_url, "port": s.port,
          "provider": s.llm_provider, "models_url": models_url,
          "chat_model": s.active_chat_model})
   ```
   If an existing service uses a different checkout or process environment, resolve that mismatch before claiming it is the target. Do not read or dump its environment to logs.
3. **Backend available?** Request the resolved `models_url` with a five-second timeout. For an authenticated backend, obtain its credential through Settings in memory, never in printed shell arguments. Verify the resolved `chat_model` exists (`/api/tags` for Ollama, `/models` for compatible providers). If unavailable, report failure; starting or reconfiguring a backend requires its own authorization.
4. **Identify the gateway before stopping it:** inspect the listener on the resolved `s.port` with `ss -ltnp`. For each candidate PID inspect `/proc/<pid>/cwd` and `/proc/<pid>/cmdline`; require the resolved cwd to match the intended checkout and the command to run that checkout's `run.py` or `uvicorn src.api_gateway:app`. A matching port alone is insufficient. If either identity check fails or cannot be read, leave the process running and report the mismatch. Recheck the same process identity immediately before `kill <verified-pid>`, then confirm the configured listener is gone. Never use `pkill -f run.py` or kill an unrelated listener to free a port.
5. **Start the intended checkout** using its Python environment and the same resolved settings:
   `mkdir -p data/logs && (nohup .venv/bin/python run.py > data/logs/gateway.log 2>&1 &)`
   If the worktree has no `.venv`, use the explicitly identified existing Python environment. Wait for `Application startup complete`; fail on `Traceback`, bind failure, or the configured listener being absent. Verify the new listener's PID, cwd and command before smoke tests.
6. **Smoke tests** against the resolved `gateway_url`, with credentials loaded using Settings inside the test process:
   - `GET /health` returns healthy and the configured provider/model. This does not prove inference works.
   - `POST /v1/chat/completions` with no key and a wrong key both return 401.
   - Within the authorized inference budget, send one short prompt with `use_rag=false`, a small `max_tokens` bound and an explicit request timeout. Require HTTP 200 and a sensible answer.
   - Exercise the changed endpoint using its documented payload. RAG/embedding and media checks can load additional models; include them only in the authorized budget.
7. Inspect `data/logs/gateway.log` for errors/tracebacks. Report each check's target, HTTP code, timing and actual result; separate offline tests, service startup and successful inference. Do not claim live acceptance for checks left pending.
