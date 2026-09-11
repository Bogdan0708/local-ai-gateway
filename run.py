#!/usr/bin/env python3
"""
Quick start script for Secure Local AI.

Usage:
    python run.py              # Start the API server
    python run.py --setup      # Generate secrets and setup
    python run.py --ingest DIR # Ingest a directory
"""

import argparse
import os
import secrets
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def setup():
    """Generate secrets and create .env file."""
    env_path = Path(__file__).parent / ".env"
    example_path = Path(__file__).parent / ".env.example"

    if env_path.exists():
        print(f"Warning: {env_path} already exists!")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Generate secrets
    jwt_secret = secrets.token_hex(32)
    api_key = secrets.token_hex(24)
    encryption_key = secrets.token_hex(32)

    # Read example and replace placeholders
    if example_path.exists():
        content = example_path.read_text()
    else:
        content = """# Secure Local AI Configuration

# === Security Secrets ===
JWT_SECRET={jwt_secret}
API_KEY={api_key}
ENCRYPTION_KEY={encryption_key}

# === LLM Configuration ===
LLM_PROVIDER=lmstudio
LMSTUDIO_HOST=http://localhost:1234
LMSTUDIO_CHAT_MODEL=gpt-oss-120b
LMSTUDIO_EMBED_MODEL=nomic-embed-text

# === Server Configuration ===
HOST=0.0.0.0
PORT=8000
DEBUG=false
"""

    # Replace placeholder values
    content = content.replace(
        "your-jwt-secret-here-generate-with-openssl", jwt_secret
    )
    content = content.replace("your-api-key-here-generate-with-openssl", api_key)
    content = content.replace("your-encryption-key-here", encryption_key)

    env_path.write_text(content)

    print(f"Created {env_path}")
    print(f"\nYour API key: {api_key}")
    print("\nSave this key! You'll need it to authenticate requests.")
    print("\nTo start the server:")
    print("  python run.py")


def ingest_directory(directory: str):
    """Ingest a directory into the knowledge base."""
    from src.file_service import ingest_directory as do_ingest

    path = Path(directory).resolve()
    if not path.exists():
        print(f"Error: Directory does not exist: {path}")
        sys.exit(1)

    print(f"Ingesting {path}...")
    results = do_ingest(path)

    total_files = len(results)
    total_chunks = sum(results.values())
    print(f"\nIngested {total_files} files ({total_chunks} chunks)")


def start_server():
    """Start the API server."""
    import uvicorn

    from src.config import get_settings

    settings = get_settings()

    print(f"Starting Secure Local AI on {settings.host}:{settings.port}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Chat Model: {settings.active_chat_model}")
    print(f"\nAPI Docs: http://localhost:{settings.port}/docs")

    uvicorn.run(
        "src.api_gateway:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


def main():
    parser = argparse.ArgumentParser(description="Secure Local AI")
    parser.add_argument("--setup", action="store_true", help="Generate secrets and setup")
    parser.add_argument("--ingest", metavar="DIR", help="Ingest a directory")

    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.ingest:
        ingest_directory(args.ingest)
    else:
        start_server()


if __name__ == "__main__":
    main()
