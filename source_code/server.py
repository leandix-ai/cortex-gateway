"""
server.py — Entry point for the Cortex Context Router.

Runs independently from the core module (leandix_cortex.py).
Configure host/port/reload here — do not touch the core logic.

Usage:
    python server.py
    python server.py --host 0.0.0.0 --port 9000
    python server.py --reload         # dev mode

Continue config.yaml (client):
    models:
      - name: Cortex
        provider: openai
        model: claude-haiku-4-5
        apiBase: http://<host>:8000/v1
        apiKey: sk-ant-...
"""

import argparse
import os

import uvicorn

# ─────────────────────────────────────────────
# Server config defaults
# ─────────────────────────────────────────────

DEFAULT_HOST   = "0.0.0.0"
DEFAULT_PORT   = 8000
DEFAULT_RELOAD = False
APP_MODULE     = "leandix_cortex:app"


# ─────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cortex Context Router — server entry point"
    )
    parser.add_argument("--host",   default=DEFAULT_HOST,   help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=DEFAULT_PORT,   type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=DEFAULT_RELOAD, help="Enable hot reload — watches leandix_cortex.py (default: False)")
    parser.add_argument("--workers", default=1,             type=int, help="Number of worker processes")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"""
  ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
 ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
 ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝
 ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗
 ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
  Context Router — Leandix
  http://{args.host}:{args.port}
    """)

    uvicorn.run(
        APP_MODULE,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[os.path.dirname(os.path.abspath(__file__))] if args.reload else None,
        reload_includes=["leandix_cortex.py"] if args.reload else None,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
