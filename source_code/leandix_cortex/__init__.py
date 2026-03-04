"""
leandix_cortex — Cortex Context Router package.

Public surface:
  app  — FastAPI application instance (consumed by server.py via uvicorn)
"""

from .leandix_cortex import app  # noqa: F401
