# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Python venv is `.venv/` at the repo root: use `.venv/bin/python`. pytest is not in `requirements.txt` (CI installs it separately).
- Tests: `.venv/bin/python -m pytest -q` (`pytest.ini` sets `pythonpath = .`). CI runs the same on Python 3.12.
- Run: `.venv/bin/python run.py` (uvicorn `src.api_gateway:app`, `HOST`/`PORT` from `.env`, default `0.0.0.0:8000`). `run.py --setup` generates `.env`; `run.py --ingest DIR` indexes a folder.
- Lint: `.venv/bin/ruff check .` (`ruff.toml`: correctness rules only — `E4,E7,E9,F`; no formatter, match the surrounding style). CI runs it; a PostToolUse hook runs it on each edited `.py` file. ruff/pytest aren't in `requirements.txt`: `.venv/bin/pip install ruff pytest`.
- Some files use CRLF line endings (e.g. `src/chunking.py`); preserve them when editing via scripts.

## Tests

- `tests/conftest.py` sets synthetic secrets/dirs before `src` is imported and sets `Settings.model_config["env_file"] = None`, so a developer `.env` never leaks into tests. Keep it that way.
- `get_settings`, `get_domain_config`, `get_file_whitelist` are `@lru_cache`d (`src/config.py`): after changing env, call `cache_clear()` — use the `settings_env` fixture.
- Bypass auth per module with `app.dependency_overrides[require_auth]` (or `require_local_or_auth` for `/api/*`, `/mcp/*`) and pop it in fixture teardown so it can't leak into other files.
- Act as a non-loopback caller with `TestClient(app, client=("203.0.113.7", 1234))`.
- The slowapi limiter is a single module-level instance keyed on client address and never reset between tests; many calls to a rate-limited route from one test client can 429.
- No test may need a live LM Studio/ComfyUI: mock HTTP with `httpx.MockTransport`, memory with `patch("src.api_gateway.get_memory")`.

## Config gotchas

- `JWT_SECRET` and `API_KEY` must be ≥32 chars and not start with `your-`, or `Settings()` fails at import.
- `LLM_PROVIDER`: `lmstudio` (default, `LMSTUDIO_HOST`), `ollama`, or `openai-compatible` (`OPENAI_BASE_URL` is the full `/v1` prefix + `OPENAI_CHAT_MODEL`). `LLM_REASONING_EFFORT=none` keeps reasoning models from burning tokens; unset means the field is omitted.
- `ALLOW_LOOPBACK_UNAUTHENTICATED` only takes effect when `HOST` is loopback.
- The domain allowlist is fail-closed (`config/allowed_domains.yaml` → `.example.yaml` → in-code defaults).

## Architecture notes

- There is exactly one FastAPI `app` in `src/api_gateway.py`. Never instantiate it again further down — that once silently dropped every route registered earlier.
- Auth (`src/auth.py`): `require_auth` accepts `X-API-Key`, `Authorization: Bearer <API_KEY>`, or a JWT; `require_scope(...)` for scoped routes; `require_local_or_auth` guards internal `/api/*` and `/mcp/*`.
- A FastAPI handler with two body parameters expects them wrapped by parameter name (`{"agent_request": {...}}`). Keep one Pydantic body model per endpoint so the documented flat JSON works.

## Workflow

- Changes land via branch (`fix/<slug>`, `feat/<slug>`) → PR to `main` → CI green → squash merge. Commits use Conventional Commits: `type(scope): lowercase summary` (scopes like `agents`, `media`, `llm`, `auth`, `fetch`).
- Validate changes with pytest and Ruff. Report live acceptance separately: use `/live-check` when the user authorizes service restart and inference against the configured backend. Code edits and offline reviews do not authorize live actions; preserve any GPU pause or test budget.
- Before an authorized restart, resolve `HOST`/`PORT` through Settings and verify the listener PID belongs to this checkout using both `/proc/<pid>/cwd` and `/proc/<pid>/cmdline`. Leave unrelated or unverifiable processes running. Follow `/live-check`; never `pkill -f run.py`.
