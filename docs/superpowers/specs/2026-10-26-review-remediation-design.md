# Review remediation design

The user approved continuing the review recommendations on 2026-10-26.
Integrate security branch `14274df` with current branch `aabc461` in an isolated
local worktree, retaining the flat research request fix and Ruff checks. Repair
the three remaining findings without changing live services or data.

## Approach

Merge the reviewed branch to preserve its tested history and provenance; then
make focused commits. Reimplementing those fixes would duplicate reviewed work;
cherry-picking would obscure their integration ancestry. Execution is inline with
one independent final review (implementation choice).

## Required behavior

- Every chunker returned by get_chunker accepts text, content_type and filename;
  recursive and semantic strategies retain their algorithms, structure selects
  HTML versus Markdown and handles plain text, and helper entry points work.
- Cancelling a waiting media job removes its pending queue entry immediately;
  repeated cancellation cannot grow physical pending storage. Preserve FIFO,
  wakeups, existing capacity responses, active cancellation and cleanup.
- Live-check resolves backend and gateway endpoints from effective Settings,
  verifies listener process identity and checkout before any stop, and retains
  explicit user control of service restarts and inference. No broad pkill.

## Constraints

Keep main, original worktrees, live databases, gateway and GPU workloads unchanged.
Use synthetic credentials, temporary storage and mocked inference for tests.
Preserve CRLF in src/chunking.py. No dependency installs or remote publication.
