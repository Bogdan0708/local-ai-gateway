# Local AI Gateway

> Status: implementation evidence; no live/hosted service claimed. This is a
> working FastAPI codebase with passing unit tests — it is not deployed or
> operated as a hosted service.

A private knowledge base with local file and web access. Runs entirely on your machine with no data leaving your network. FastAPI gateway + hybrid (BM25 + vector) retrieval, document/web ingestion with chunking, an MCP bridge, and LangGraph-based agents, all backed by a local LLM (LM Studio/Ollama/any OpenAI-compatible endpoint).

## Features

- **Hybrid Search**: Combines BM25 keyword search with semantic vector search for 2-3x better retrieval
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

### SSRF Protection

The web fetcher blocks requests to:
- localhost, 127.0.0.1, 0.0.0.0
- Private networks (10.x, 172.16-31.x, 192.168.x)
- Cloud metadata endpoints (169.254.169.254)

### Domain Whitelist

Configure allowed domains in `config/allowed_domains.yaml`.

### File Security

- Only whitelisted file types are indexed
- Blocked: `.env`, credentials, private keys
- Read-only mounts in Docker

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

### 3. Start with Docker

```bash
cd deploy
TAILSCALE_IP=100.64.1.23 docker-compose up -d
```

### 4. Access from Any Device

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://100.64.1.23:8080/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `lmstudio` | LLM provider (lmstudio, ollama) |
| `LMSTUDIO_HOST` | `http://localhost:1234` | LM Studio API URL |
| `LMSTUDIO_CHAT_MODEL` | `gpt-oss-120b` | Chat model name |
| `JWT_SECRET` | (required) | Secret for JWT tokens |
| `API_KEY` | (required) | API key for authentication |
| `PORT` | `8000` | Server port |

### File Types

Supported formats (see `config/file_whitelist.yaml`):
- Documents: PDF, Markdown, TXT, HTML
- Code: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, etc.
- Config: JSON, YAML, TOML

## Architecture

```
┌─────────────────────────────────────────┐
│           Caddy (TLS + Auth)            │
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
