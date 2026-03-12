"""
cortex_gateway — Cortex Gateway package.

An OpenAI-compatible AI gateway with automatic complexity-based
model routing, client API key management, and admin dashboard.

Public surface:
  app         — FastAPI application instance
  __version__ — Package version string
"""

__version__ = "0.2.0"

from .app import app  # noqa: F401
