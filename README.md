# Local AI Gateway

> Status: implementation evidence; no live/hosted service claimed. This is a
> working FastAPI codebase with passing unit tests — it is not deployed or
> operated as a hosted service.

A private knowledge base with local file and web access. Runs entirely on your machine with no data leaving your network. FastAPI gateway + hybrid (BM25 + vector) retrieval, document/web ingestion with chunking, an MCP bridge, and LangGraph-based agents, all backed by a local LLM (LM Studio/Ollama/any OpenAI-compatible endpoint).

## Features

- **Hybrid Search**: Combines BM25 keyword search with semantic vector search, with optional cross-encoder reranking
- **Local File Access**: Index documents (PDF, Markdown, code) from your filesystem
- **Secure Web Fetching**: Fetch web content with SSRF protection and domain whitelisting
- **OpenAI-Compatible API**: Works with any client that supports the OpenAI API format
- **Multiple LLM Support**: Works with LM Studio (gpt-oss-20B/120B), Ollama, or any OpenAI-compatible endpoint
- **Remote Access**: Secure access via Tailscale VPN (zero public exposure)

## Quick Start

### 1. Setup

```bash
# Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# Generate secrets and create .env
python run.py --setup
```

### 2. Configure LLM

Edit `.env` to match your setup:

```env
# For LM Studio (default)
LLM_PROVIDER=lmstudio
LMSTUDIO_HOST=http://localhost:1234
LMSTUDIO_CHAT_MODEL=gpt-oss-120b

# Or for Ollama
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
```

### 3. Start the Server

```bash
python run.py
```

The API will be available at `http://localhost:8000`

### 4. Test It

```bash
# Health check
curl http://localhost:8000/health

# Chat (with your API key)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth) |
| `/v1/chat/completions` | POST | Chat with RAG context |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/search` | POST | Search knowledge base |
| `/v1/files` | GET | List indexed files |
| `/v1/files` | POST | Upload and index a file |
| `/v1/web/fetch` | POST | Fetch web content |
| `/v1/web/index` | POST | Fetch and index web content |

## Ingesting Documents

### Via CLI

```bash
# Ingest a directory
python run.py --ingest /path/to/documents
```

### Via API

```bash
# Upload a file
curl -X POST http://localhost:8000/v1/files \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@document.pdf"

# Index a webpage
curl -X POST http://localhost:8000/v1/web/index \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/tutorial/"}'
```

## Security

### Authentication

Every endpoint requires a credential. Send the API key from `.env` as either
header:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/chat ...
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/chat/completions ...
```

`/health` and `/mcp/health` are the only open endpoints.

The internal endpoints (`/api/chat`, `/api/read-file`, `/mcp/*`) can
additionally accept unauthenticated requests **from the loopback interface
only**, and only when you opt in:

```env
ALLOW_LOOPBACK_UNAUTHENTICATED=true
```

It defaults to `false`. Leave it off unless the server is bound to
`127.0.0.1`: the default bind is `0.0.0.0`, and a reverse proxy on the same
host makes remote traffic look like loopback traffic. With the flag off,
loopback callers must present the API key like anyone else.

### SSRF Protection

The web fetcher blocks requests to:
- localhost, 127.0.0.1, 0.0.0.0
- Private networks (10.x, 172.16-31.x, 192.168.x)
- Link-local and cloud metadata endpoints (169.254.169.254)
- IPv6 loopback, unique-local (fc00::/7) and link-local (fe80::/10)

Redirects are not followed by the HTTP transport. Each hop is returned to the
fetcher, which re-runs the full check (scheme, hostname blocklist, DNS
resolution, resolved-IP ranges, domain policy) against the new destination
before requesting it, and the chain is capped at 5 hops. The curl fallback
runs without `-L` and is subject to the same loop.

### Domain Whitelist

Configure allowed domains in `config/allowed_domains.yaml`.

### File Security

- Only whitelisted file types are indexed or read
- Blocked: `.env`, credentials, private keys (`*.pem`, `id_rsa`, ...)
- `/api/read-file` resolves the path (symlinks included) and requires it to be
  a real descendant of a configured root, so `..` traversal, symlink escapes
  and sibling directories that merely share a name prefix are all refused
- The same extension allowlist and blocked-filename patterns apply to reads
  and to ingestion

The policy lives in `config/file_whitelist.yaml`. The repository ships
`config/file_whitelist.example.yaml`; if you do not create your own file, the
example is used, and if that is missing too the equivalent defaults in
`src/config.py` apply (`.txt`, `.md`, `.pdf`, `.csv`, `.json`, `.yaml`).
Copy it to customise:

```bash
cp config/file_whitelist.example.yaml config/file_whitelist.yaml
```

## Remote Access with Tailscale

### 1. Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### 2. Get Your Tailscale IP

```bash
tailscale ip -4
# Example: 100.64.1.23
```

### 3. Bind the Server to the Tailscale Interface

There is no deployment directory in this repository; run the server directly
and let Tailscale provide the private network.

```bash
HOST=100.64.1.23 PORT=8000 python run.py
```

### 4. Access from Any Device

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://100.64.1.23:8000/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `lmstudio` | LLM provider (lmstudio, ollama) |
| `LMSTUDIO_HOST` | `http://localhost:1234` | LM Studio API URL |
| `LMSTUDIO_CHAT_MODEL` | `gpt-oss-120b` | Chat model name |
| `JWT_SECRET` | (required) | Secret for JWT tokens |
| `API_KEY` | (required) | API key for authentication (`X-API-Key` or `Authorization: Bearer`) |
| `ALLOW_LOOPBACK_UNAUTHENTICATED` | `false` | Allow unauthenticated calls to the internal endpoints from 127.0.0.1/::1 only |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated CORS origins |
| `PORT` | `8000` | Server port |
| `HOST_DOCUMENTS_ROOT` | (unset) | Windows/host folder your documents live under, for the `/api/read-file` path-rewrite (e.g. a per-user Documents folder on the host). Leave unset to disable the rewrite. |
| `CONTAINER_DOCUMENTS_ROOT` | `/data/documents` | Where `HOST_DOCUMENTS_ROOT` maps to inside this service. |
| `ALLOWED_PATH_PREFIXES` | `/data` | `os.pathsep`-separated list of extra path prefixes `/api/read-file` may read from, beyond the configured documents/code dirs. |

### File Types

The shipped defaults (`config/file_whitelist.example.yaml`) are deliberately
narrow: `.txt`, `.md`, `.pdf`, `.csv`, `.json`, `.yaml`. Add the code and
config extensions you actually want indexed to your own
`config/file_whitelist.yaml`; anything not listed is refused by both the
ingestion pipeline and `/api/read-file`.

## Architecture

```
┌─────────────────────────────────────────┐
│      TLS terminator / VPN (optional)    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           FastAPI Gateway               │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │  Chat   │ │ Search  │ │   Web    │  │
│  │   API   │ │   API   │ │ Fetcher  │  │
│  └────┬────┘ └────┬────┘ └────┬─────┘  │
│       │           │           │        │
│  ┌────▼───────────▼───────────▼────┐   │
│  │      Hybrid Memory Layer        │   │
│  │  ┌─────────┐  ┌─────────────┐   │   │
│  │  │ChromaDB │  │   BM25      │   │   │
│  │  │(vectors)│  │ (keywords)  │   │   │
│  │  └─────────┘  └─────────────┘   │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    LM Studio / Ollama (Local LLM)       │
└─────────────────────────────────────────┘
```

## Tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
