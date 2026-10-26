# Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Integrate reviewed hardening and repair the remaining review findings.
**Architecture:** Preserve both branch histories in an isolated integration branch.
Normalize chunker calls, remove cancelled pending jobs, and correct live-check guidance.
**Tech Stack:** Python, FastAPI, asyncio, ChromaDB, pytest, Ruff.
**Spec:** docs/superpowers/specs/2026-10-26-review-remediation-design.md

## Global Constraints

- Keep main, original worktrees, live databases, gateway and GPU workloads unchanged.
- Use synthetic credentials, temporary storage and mocked inference for tests.
- Preserve CRLF in src/chunking.py. No dependency installs or remote publication.

## Review Focus

- Explicit chunking modes preserve text and source ownership for Markdown, HTML and plain text.
- Semantic tests substitute only external model inference, including unavailable-model fallback.
- Cancelled queued jobs release physical storage while preserving FIFO and worker wakeups.
- Queue capacity still produces 429 and shutdown settles pending jobs.
- Live-check must reject unrelated listeners and derive nondefault ports and providers.

### Task 1: Integrate existing fixes

**Files:** Merge tracked security branch changes; resolve .github/workflows/ci.yml.
**Interfaces:** Produces integrated ingestion/media behavior and existing public APIs.
- [x] Merge 14274df into isolated branch based on aabc461; keep Ruff and pip check in CI.
- [x] Run full offline pytest baseline; expected 291 passed.
- [x] Verify Ruff; fix integration-only lint regressions without behavior changes.
- [x] Commit merge and design/plan.

### Task 2: Normalize chunker interfaces

**Files:** src/chunking.py; tests/test_chunking_strategies.py.
**Interfaces:** get_chunker(config).chunk(text, content_type=None, filename=None).
- [x] Add regressions exercising DocumentProcessor across recursive, semantic and structure strategies, including content, provenance, HTML and semantic fallback.
- [x] Run focused tests; expected failures for incompatible keywords or missing chunk method.
- [x] Implement a consistent chunker interface and structure dispatch; preserve algorithms and adaptive behavior.
- [x] Run focused chunking/ingestion tests; expected pass, then commit.

### Task 3: Remove cancelled pending entries

**Files:** src/media_jobs.py; tests/test_media_lifecycle.py.
**Interfaces:** MediaJobManager.enqueue/cancel/queued_count and worker; preserve API capacity semantics.
- [x] Add regressions with a blocked active job, repeated queued cancellation, surviving FIFO jobs, capacity refusal and worker wakeup.
- [x] Run focused tests; expected failure from retained cancelled entries.
- [x] Implement bounded pending storage with immediate removal and existing active cleanup semantics.
- [x] Run media tests; expected pass, then commit.

### Task 4: Correct live-check instructions

**Files:** .claude/skills/live-check/SKILL.md; CLAUDE.md.
**Interfaces:** Existing operator instructions only; no automatic service actions.
- [x] Simulate old instructions against nondefault port/provider and an unrelated listener; record wrong target.
- [x] Derive endpoints from effective Settings, verify process command and resolved checkout, and scope restarts/inference to explicit user authorization.
- [x] Simulate corrected instructions against same scenario; expected correct endpoints and refusal to stop unrelated listener.
- [ ] Commit guidance, run full suite/lint/pip check/secret scans and independent final review.
