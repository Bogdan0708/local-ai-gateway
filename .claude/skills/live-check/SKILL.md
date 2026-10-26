---
name: live-check
description: Restart the local gateway and verify it end-to-end against the LM Studio hub. Use after any change to gateway code, before calling the work done, or when asked to start/test the gateway.
---

Verify the gateway against the real stack. Never print the API key.

1. **Hub up?** `curl -s -m 5 localhost:1234/v1/models` must list the model in `.env` (`LMSTUDIO_CHAT_MODEL`). If LM Studio is down, stop and tell the user — don't start it blind.
2. **Stop the old gateway by PID:** `ss -ltnp | grep ':8000 '` → `kill <pid>` → confirm the port is free. Never `pkill -f run.py` (matches and kills your own shell).
3. **Start detached** from the repo root so it survives the session:
   `mkdir -p data/logs && (nohup .venv/bin/python run.py > data/logs/gateway.log 2>&1 &)` (`data/` is gitignored) then wait for `Application startup complete` in the log (fail on `Traceback`).
4. **Smoke tests** (key via `K=$(grep -E '^API_KEY=' .env | cut -d= -f2- | sed 's/\s*#.*//')`, header `X-API-Key: $K`):
   - `GET /health` → `healthy` and the expected `llm_model`.
   - `POST /v1/chat/completions` without a key and with a wrong key → both `401`.
   - `POST /v1/chat/completions` with a short prompt → `200`, sensible answer. The first call after a reboot can take ~20 s while LM Studio JIT-loads the embedding model; time a second call.
5. **Exercise what changed:** call the endpoint(s) touched by the diff using the request shape documented in the handler's docstring, and check status, body, and the effect (e.g. `sources_used`). $ARGUMENTS names extra endpoints or payloads to try.
6. `grep -iE 'error|traceback' data/logs/gateway.log`, then report a table of check → result with HTTP codes and timings. Report failures as failures.
