"""
cli.py — CLI entry point for Cortex Gateway.

Usage:
    cortex-gateway
    cortex-gateway --host 0.0.0.0 --port 9000
    cortex-gateway --reload

Or via module:
    python -m cortex_gateway
"""

import argparse
import os

import uvicorn
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# ─────────────────────────────────────────────
# Server config defaults (from .env)
# ─────────────────────────────────────────────

DEFAULT_HOST   = os.getenv("GATEWAY_HOST", "0.0.0.0")
DEFAULT_PORT   = int(os.getenv("GATEWAY_PORT", "8000"))
DEFAULT_RELOAD = False
APP_MODULE     = "cortex_gateway:app"

_BANNER = r"""
   ____           _
  / ___|___  _ __| |_ _____  __
 | |   / _ \| '__| __/ _ \ \/ /
 | |__| (_) | |  | ||  __/>  <
  \____\___/|_|   \__\___/_/\_\
  Cortex Gateway --- Leandix
"""


# ─────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="cortex-gateway",
        description="Cortex Gateway — OpenAI-compatible AI gateway with auto-routing",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--host",    default=DEFAULT_HOST,   help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument("--port",    default=DEFAULT_PORT,   type=int, help=f"Bind port (default: {DEFAULT_PORT})")
    parser.add_argument("--reload",  action="store_true",    default=DEFAULT_RELOAD, help="Enable hot reload (default: False)")
    parser.add_argument("--workers", default=1,              type=int, help="Number of worker processes")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"{_BANNER}  http://{args.host}:{args.port}\n")
    print(f"  UI:        http://{args.host}:{args.port}/")
    print(f"  Dashboard: http://{args.host}:{args.port}/dashboard\n")

    uvicorn.run(
        APP_MODULE,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[os.path.dirname(os.path.abspath(__file__))] if args.reload else None,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )
