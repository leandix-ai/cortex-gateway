"""
db.py — SQLite persistence for Cortex Gateway.

Tables:
  - admin: stores hashed admin password
  - model_config: master model settings (simple/complex)
  - client_keys: client API keys with name and status
  - request_logs: per-request tracking (client key, model, tokens)
"""

import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# ─────────────────────────────────────────────
# Database path
# ─────────────────────────────────────────────

DB_PATH = Path(os.environ.get("CORTEX_DB_PATH", "cortex_gateway.db"))

DEFAULT_ADMIN_PASSWORD = "admin123"


# ─────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist and seed default admin password."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS admin (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                password_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_config (
                tier TEXT PRIMARY KEY,
                provider TEXT NOT NULL DEFAULT 'anthropic',
                model_name TEXT NOT NULL,
                api_key TEXT NOT NULL DEFAULT '',
                base_url TEXT NOT NULL DEFAULT 'https://api.anthropic.com'
            );

            CREATE TABLE IF NOT EXISTS client_keys (
                api_key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                client_key TEXT NOT NULL,
                model TEXT NOT NULL,
                complexity TEXT NOT NULL DEFAULT '',
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (client_key) REFERENCES client_keys(api_key)
            );

            CREATE INDEX IF NOT EXISTS idx_logs_client ON request_logs(client_key);
            CREATE INDEX IF NOT EXISTS idx_logs_ts ON request_logs(timestamp);
        """)

        # Seed admin password if not exists
        row = conn.execute("SELECT COUNT(*) as c FROM admin").fetchone()
        if row["c"] == 0:
            conn.execute(
                "INSERT INTO admin (id, password_hash) VALUES (1, ?)",
                (_hash_password(DEFAULT_ADMIN_PASSWORD),),
            )

        # Seed model config from .env if not exists
        row = conn.execute("SELECT COUNT(*) as c FROM model_config").fetchone()
        if row["c"] == 0:
            conn.execute(
                "INSERT INTO model_config (tier, provider, model_name, api_key, base_url) VALUES (?, ?, ?, ?, ?)",
                ("simple",
                 os.environ.get("SIMPLE_MODEL_PROVIDER", "anthropic"),
                 os.environ.get("SIMPLE_MODEL_NAME", "claude-haiku-4-5"),
                 os.environ.get("SIMPLE_MODEL_API_KEY", ""),
                 os.environ.get("SIMPLE_MODEL_BASE_URL", "https://api.anthropic.com")),
            )
            conn.execute(
                "INSERT INTO model_config (tier, provider, model_name, api_key, base_url) VALUES (?, ?, ?, ?, ?)",
                ("complex",
                 os.environ.get("COMPLEX_MODEL_PROVIDER", "anthropic"),
                 os.environ.get("COMPLEX_MODEL_NAME", "claude-sonnet-4-5"),
                 os.environ.get("COMPLEX_MODEL_API_KEY", ""),
                 os.environ.get("COMPLEX_MODEL_BASE_URL", "https://api.anthropic.com")),
            )


# ─────────────────────────────────────────────
# Password hashing (SHA-256 + salt)
# ─────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    parts = stored.split(":", 1)
    if len(parts) != 2:
        return False
    salt, expected = parts
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return h == expected


# ─────────────────────────────────────────────
# Admin operations
# ─────────────────────────────────────────────

def verify_admin_password(password: str) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT password_hash FROM admin WHERE id = 1").fetchone()
        if not row:
            return False
        return _verify_password(password, row["password_hash"])


def change_admin_password(old_password: str, new_password: str) -> bool:
    """Change admin password. Returns True on success, False if old password is wrong."""
    if not verify_admin_password(old_password):
        return False
    with get_db() as conn:
        conn.execute(
            "UPDATE admin SET password_hash = ? WHERE id = 1",
            (_hash_password(new_password),),
        )
    return True


# ─────────────────────────────────────────────
# Client key operations
# ─────────────────────────────────────────────

def create_client_key(name: str) -> str:
    """Create a new client API key. Returns the key string."""
    api_key = f"cgw-{secrets.token_hex(24)}"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO client_keys (api_key, name, created_at) VALUES (?, ?, ?)",
            (api_key, name, time.time()),
        )
    return api_key


def list_client_keys() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT api_key, name, created_at, active FROM client_keys ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def revoke_client_key(api_key: str) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE client_keys SET active = 0 WHERE api_key = ?", (api_key,)
        )
        return cur.rowcount > 0


def validate_client_key(api_key: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT active FROM client_keys WHERE api_key = ?", (api_key,)
        ).fetchone()
        return bool(row and row["active"])


# ─────────────────────────────────────────────
# Request logging
# ─────────────────────────────────────────────

def log_request(
    client_key: str,
    model: str,
    complexity: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO request_logs (timestamp, client_key, model, complexity, input_tokens, output_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), client_key, model, complexity, input_tokens, output_tokens),
        )


def get_stats() -> dict:
    """Get aggregated stats grouped by client key and model."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                ck.name as client_name,
                rl.client_key,
                rl.model,
                COUNT(*) as request_count,
                SUM(rl.input_tokens) as total_input_tokens,
                SUM(rl.output_tokens) as total_output_tokens
            FROM request_logs rl
            JOIN client_keys ck ON rl.client_key = ck.api_key
            GROUP BY rl.client_key, rl.model
            ORDER BY request_count DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_total_requests() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM request_logs").fetchone()
        return row["c"] if row else 0


# ─────────────────────────────────────────────
# Model config operations
# ─────────────────────────────────────────────

def get_model_config() -> dict:
    """Get model config for both tiers. Returns {simple: {...}, complex: {...}}."""
    with get_db() as conn:
        rows = conn.execute("SELECT tier, provider, model_name, api_key, base_url FROM model_config").fetchall()
        result = {}
        for r in rows:
            d = dict(r)
            # Mask API key for display (show first 8 + last 4 chars)
            key = d["api_key"]
            if len(key) > 16:
                d["api_key_masked"] = key[:8] + "..." + key[-4:]
            elif key:
                d["api_key_masked"] = key[:4] + "..."
            else:
                d["api_key_masked"] = "(not set)"
            result[d["tier"]] = d
        return result


def get_model_config_raw(tier: str) -> dict | None:
    """Get raw model config for a tier (with full API key). For internal use."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT provider, model_name, api_key, base_url FROM model_config WHERE tier = ?", (tier,)
        ).fetchone()
        return dict(row) if row else None


def update_model_config(tier: str, provider: str, model_name: str, api_key: str, base_url: str) -> bool:
    """Update model config for a tier. Returns True on success."""
    if tier not in ("simple", "complex"):
        return False
    if provider not in ("anthropic", "openai"):
        return False
    with get_db() as conn:
        conn.execute(
            "UPDATE model_config SET provider = ?, model_name = ?, api_key = ?, base_url = ? WHERE tier = ?",
            (provider, model_name, api_key, base_url, tier),
        )
    return True
