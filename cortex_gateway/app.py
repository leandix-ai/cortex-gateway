"""
app.py — Cortex Gateway: HTTP layer & application entry point.

Routes:
  - /v1/chat/completions  — OpenAI-compatible chat API (client API key auth)
  - /v1/models            — List available models
  - /health               — Health check
  - /api/status            — Public gateway status
  - /api/login             — Admin login → JWT
  - /api/change-password   — Change admin password (JWT required)
  - /api/keys              — CRUD client API keys (JWT required)
  - /api/stats             — Request stats (JWT required)
  - /                      — Public home page
  - /login                 — Login page
  - /dashboard             — Dashboard page

Entry point: cortex-gateway CLI → uvicorn → cortex_gateway:app
"""

import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import jwt
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    JWT_ALGORITHM,
    JWT_EXPIRE_HOURS,
    JWT_SECRET,
    LANG_NAMES,
    MAX_TOKENS,
    PROMPT_TIER,
    MessageList,
    log,
)
from .db import (
    change_admin_password,
    create_client_key,
    get_model_config,
    get_model_config_raw,
    get_stats,
    get_total_requests,
    init_db,
    list_client_keys,
    log_request,
    revoke_client_key,
    update_model_config,
    validate_client_key,
    verify_admin_password,
)
from .router import Complexity, classify_complexity, detect_language, extract_text

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Cortex Gateway")

# Track server start time
_start_time = time.time()

# Static files directory
_STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
async def startup():
    init_db()
    log.info("[Startup] Database initialised")
    mc = get_model_config()
    log.info(f"[Startup] Simple model: {mc.get('simple', {}).get('model_name', '?')}")
    log.info(f"[Startup] Complex model: {mc.get('complex', {}).get('model_name', '?')}")


# Mount static files
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────

def _create_token() -> str:
    payload = {
        "sub": "admin",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _require_admin(request: Request):
    """Validates JWT token from Authorization header. Raises 401 on failure."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = auth[len("Bearer "):].strip()
    try:
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# ─────────────────────────────────────────────
# Client API key auth
# ─────────────────────────────────────────────

def _extract_client_key(request: Request) -> str:
    """Extracts and validates client API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        key = auth[len("Bearer "):].strip()
        if key and validate_client_key(key):
            return key
    raise HTTPException(401, "Invalid or missing client API key")


# ─────────────────────────────────────────────
# Model client factory (reads from DB, supports both providers)
# ─────────────────────────────────────────────

def _get_tier_config(complexity: Complexity) -> dict:
    """Get the raw model config for the tier matching the given complexity."""
    tier = "complex" if complexity == Complexity.COMPLEX else "simple"
    return get_model_config_raw(tier) or {"provider": "anthropic", "model_name": "unknown", "api_key": "", "base_url": "https://api.anthropic.com"}


def _make_anthropic_client(cfg: dict) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", "https://api.anthropic.com"),
    )


def _make_openai_client(cfg: dict) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
    )


def _get_model(complexity: Complexity) -> str:
    tier = "complex" if complexity == Complexity.COMPLEX else "simple"
    cfg = get_model_config_raw(tier)
    return cfg["model_name"] if cfg else "unknown"


# ─────────────────────────────────────────────
# System prompt selector
# ─────────────────────────────────────────────

def _get_system_prompt(complexity: Complexity) -> str:
    return PROMPT_TIER.get(complexity.value, PROMPT_TIER["simple"])


# ─────────────────────────────────────────────
# Language instruction block
# ─────────────────────────────────────────────

def _build_language_block(lang: str) -> dict:
    lang_name = LANG_NAMES.get(lang, "English")
    if lang == "en":
        instruction = "Respond in English."
    else:
        instruction = (
            f"The user's message is in {lang_name}. "
            f"Respond in {lang_name}. "
            "If the response contains code, keep all code, identifiers, and inline comments in English; "
            f"use {lang_name} only for prose explanations."
        )
    return {"type": "text", "text": f"<language>\n  {instruction}\n</language>"}


# ─────────────────────────────────────────────
# OpenAI → Anthropic message format converter
# ─────────────────────────────────────────────

_TOOL_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_tool_id(tool_id: str) -> str:
    if tool_id and _TOOL_ID_RE.match(tool_id):
        return tool_id
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id or "")
    return clean or f"tool_{uuid.uuid4().hex[:8]}"


def _is_anthropic_format(msg: dict) -> bool:
    content = msg.get("content", "")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") in
               ("text", "tool_use", "tool_result", "thinking") for b in content)


def _convert_messages(messages: MessageList) -> MessageList:
    """
    Converts an OpenAI-format message list to Anthropic format.
    Handles tool_calls → tool_use blocks, tool role → user/tool_result,
    and sanitises tool IDs.
    """
    result: MessageList = []
    i = 0

    while i < len(messages):
        msg  = messages[i]
        role = msg.get("role", "")

        if _is_anthropic_format(msg):
            if role == "assistant":
                content = msg["content"]
                fixed = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        b = {**b, "id": _sanitize_tool_id(b.get("id", ""))}
                    fixed.append(b)
                result.append({**msg, "content": fixed})
            else:
                content = msg.get("content", [])
                if isinstance(content, list):
                    fixed = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            b = {**b, "tool_use_id": _sanitize_tool_id(b.get("tool_use_id", ""))}
                        fixed.append(b)
                    result.append({**msg, "content": fixed})
                else:
                    result.append(msg)
            i += 1
            continue

        if role == "assistant":
            tool_calls   = msg.get("tool_calls")
            text_content = msg.get("content") or ""

            if tool_calls:
                blocks = []
                if text_content:
                    blocks.append({"type": "text", "text": text_content})
                for tc in tool_calls:
                    fn   = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        parsed_input = json.loads(args)
                    except Exception:
                        parsed_input = {"raw": args}
                    blocks.append({
                        "type":  "tool_use",
                        "id":    _sanitize_tool_id(tc.get("id", "")),
                        "name":  fn.get("name", "unknown"),
                        "input": parsed_input,
                    })
                result.append({"role": "assistant", "content": blocks})
            else:
                result.append(msg)
            i += 1
            continue

        if role == "tool":
            tool_result_blocks = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tm = messages[i]
                tool_result_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": _sanitize_tool_id(tm.get("tool_call_id", "")),
                    "content":     tm.get("content", ""),
                })
                i += 1
            result.append({"role": "user", "content": tool_result_blocks})
            continue

        result.append(msg)
        i += 1

    return result


# ─────────────────────────────────────────────
# Payload builder
# ─────────────────────────────────────────────

def build_params(
    messages: MessageList,
    tools: list,
    max_tokens: int,
    model: str,
    complexity: Complexity,
    lang: str = "en",
) -> dict:
    """
    Builds the full kwargs dict for anthropic.messages.create / .stream.
    """
    messages = _convert_messages(messages)

    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs   = [m for m in messages if m["role"] != "system"]

    params: dict = {"model": model, "max_tokens": max_tokens, "messages": conv_msgs}

    cortex_block: dict = {
        "type":          "text",
        "text":          _get_system_prompt(complexity),
        "cache_control": {"type": "ephemeral"},
    }

    lang_block: dict = _build_language_block(lang)

    client_blocks: list = []
    for sm in system_msgs:
        c = sm.get("content", "")
        if isinstance(c, str):
            client_blocks.append({"type": "text", "text": c})
        elif isinstance(c, list):
            client_blocks.extend(c)

    params["system"] = [cortex_block, lang_block] + client_blocks

    if tools:
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", {})
            anthropic_tools.append({
                "name":         fn.get("name", t.get("name", "unknown")),
                "description":  fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        params["tools"] = anthropic_tools

    return params


# ─────────────────────────────────────────────
# Response helpers
# ─────────────────────────────────────────────

def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def format_non_stream(anthropic_msg, routing: dict) -> dict:
    """Converts an Anthropic Messages response into an OpenAI chat.completion dict."""
    req_id, created = _make_id(), int(time.time())
    text_parts, tool_calls = [], []

    for block in anthropic_msg.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id, "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            })

    message: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
    finish   = stop_map.get(anthropic_msg.stop_reason or "end_turn", "stop")
    usage    = anthropic_msg.usage

    return {
        "id": req_id, "object": "chat.completion",
        "created": created, "model": routing["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens":               usage.input_tokens,
            "completion_tokens":           usage.output_tokens,
            "total_tokens":                usage.input_tokens + usage.output_tokens,
            "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        },
        "cortex": routing,
    }


# ─────────────────────────────────────────────
# Think-tag stream parser
# ─────────────────────────────────────────────

class _ThinkState:
    OPEN  = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.inside = False
        self.buf    = ""


def _route_think_chunk(text: str, state: _ThinkState) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    state.buf += text

    while state.buf:
        if not state.inside:
            idx = state.buf.find(_ThinkState.OPEN)
            if idx == -1:
                partial = _partial_match(state.buf, _ThinkState.OPEN)
                if partial:
                    safe = state.buf[:-partial]
                    if safe:
                        out.append(("content", safe))
                    state.buf = state.buf[-partial:]
                    break
                else:
                    out.append(("content", state.buf))
                    state.buf = ""
                    break
            else:
                if idx > 0:
                    out.append(("content", state.buf[:idx]))
                state.inside = True
                state.buf    = state.buf[idx + len(_ThinkState.OPEN):]
        else:
            idx = state.buf.find(_ThinkState.CLOSE)
            if idx == -1:
                partial = _partial_match(state.buf, _ThinkState.CLOSE)
                if partial:
                    safe = state.buf[:-partial]
                    if safe:
                        out.append(("reasoning_content", safe))
                    state.buf = state.buf[-partial:]
                    break
                else:
                    out.append(("reasoning_content", state.buf))
                    state.buf = ""
                    break
            else:
                if idx > 0:
                    out.append(("reasoning_content", state.buf[:idx]))
                state.inside = False
                state.buf    = state.buf[idx + len(_ThinkState.CLOSE):]

    return out


def _partial_match(buf: str, tag: str) -> int:
    for length in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:length]):
            return length
    return 0


# ─────────────────────────────────────────────
# SSE streaming generator
# ─────────────────────────────────────────────

async def stream_sse(
    messages: MessageList,
    tools: list,
    max_tokens: int,
    routing: dict,
    complexity: Complexity,
    client_key: str,
    lang: str = "en",
):
    cfg = _get_tier_config(complexity)
    provider = cfg.get("provider", "anthropic")

    if provider == "openai":
        async for chunk in _stream_sse_openai(messages, tools, max_tokens, routing, complexity, client_key, cfg, lang):
            yield chunk
    else:
        async for chunk in _stream_sse_anthropic(messages, tools, max_tokens, routing, complexity, client_key, cfg, lang):
            yield chunk


async def _stream_sse_anthropic(
    messages: MessageList,
    tools: list,
    max_tokens: int,
    routing: dict,
    complexity: Complexity,
    client_key: str,
    cfg: dict,
    lang: str = "en",
):
    req_id, created = _make_id(), int(time.time())
    model       = routing["model"]
    params      = build_params(messages, tools, max_tokens, model, complexity, lang)
    aclient     = _make_anthropic_client(cfg)
    think_state = _ThinkState()

    def _emit(field: str, text: str) -> str:
        return _sse({
            "id": req_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}],
        })

    try:
        async with aclient.messages.stream(**params) as stream:
            yield _sse({
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            })

            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield _sse({
                            "id": req_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {
                                "tool_calls": [{
                                    "index": 0, "id": block.id, "type": "function",
                                    "function": {"name": block.name, "arguments": ""},
                                }]
                            }, "finish_reason": None}],
                        })

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        for field, chunk in _route_think_chunk(delta.text, think_state):
                            yield _emit(field, chunk)
                    elif delta.type == "input_json_delta":
                        yield _sse({
                            "id": req_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": delta.partial_json}}]
                            }, "finish_reason": None}],
                        })

            final  = await stream.get_final_message()
            stop   = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
            finish = stop.get(final.stop_reason or "end_turn", "stop")
            usage  = final.usage

            log_request(
                client_key=client_key,
                model=model,
                complexity=routing["complexity"],
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

            yield _sse({
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens":               usage.input_tokens,
                    "completion_tokens":           usage.output_tokens,
                    "total_tokens":                usage.input_tokens + usage.output_tokens,
                    "cache_read_input_tokens":     getattr(usage, "cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                },
                "cortex": routing,
            })

    except anthropic.AuthenticationError:
        log.warning("[Auth] Authentication failed")
        yield _sse({"error": {"message": "Authentication failed.", "type": "authentication_error"}})
    except anthropic.APIError as e:
        log.error(f"Anthropic API error ({model}): {type(e).__name__}")
        yield _sse({"error": {"message": "An error occurred with the AI provider.", "type": "api_error", "model": model}})

    yield "data: [DONE]\n\n"


async def _stream_sse_openai(
    messages: MessageList,
    tools: list,
    max_tokens: int,
    routing: dict,
    complexity: Complexity,
    client_key: str,
    cfg: dict,
    lang: str = "en",
):
    """Stream SSE using OpenAI-compatible API."""
    req_id, created = _make_id(), int(time.time())
    model  = routing["model"]
    client = _make_openai_client(cfg)

    oai_messages = _build_openai_messages(messages, complexity, lang)
    params = {"model": model, "max_tokens": max_tokens, "messages": oai_messages, "stream": True}
    if tools:
        params["tools"] = tools

    total_input = 0
    total_output = 0

    try:
        stream = await client.chat.completions.create(**params)

        async for chunk in stream:
            c = chunk.choices[0] if chunk.choices else None
            if not c:
                if hasattr(chunk, 'usage') and chunk.usage:
                    total_input = chunk.usage.prompt_tokens or 0
                    total_output = chunk.usage.completion_tokens or 0
                continue

            delta_dict = {}
            if c.delta:
                if c.delta.role:
                    delta_dict["role"] = c.delta.role
                if c.delta.content:
                    delta_dict["content"] = c.delta.content
                if c.delta.tool_calls:
                    delta_dict["tool_calls"] = [
                        {
                            "index": tc.index,
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name or "", "arguments": tc.function.arguments or ""},
                        }
                        for tc in c.delta.tool_calls
                    ]

            yield _sse({
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta_dict, "finish_reason": c.finish_reason}],
            })

            if hasattr(chunk, 'usage') and chunk.usage:
                total_input = chunk.usage.prompt_tokens or 0
                total_output = chunk.usage.completion_tokens or 0

        log_request(
            client_key=client_key,
            model=model,
            complexity=routing["complexity"],
            input_tokens=total_input,
            output_tokens=total_output,
        )

    except openai.AuthenticationError:
        log.warning("[Auth] OpenAI auth failed")
        yield _sse({"error": {"message": "Authentication failed.", "type": "authentication_error"}})
    except openai.APIError as e:
        log.error(f"OpenAI API error ({model}): {type(e).__name__}")
        yield _sse({"error": {"message": "An error occurred with the AI provider.", "type": "api_error", "model": model}})

    yield "data: [DONE]\n\n"


def _build_openai_messages(messages: MessageList, complexity: Complexity, lang: str = "en") -> list:
    """Build OpenAI-compatible message list with system prompt and language instruction."""
    system_text = _get_system_prompt(complexity)
    lang_name = LANG_NAMES.get(lang, "English")
    if lang != "en":
        system_text += f"\n\nThe user's message is in {lang_name}. Respond in {lang_name}. If the response contains code, keep code in English; use {lang_name} only for prose."

    result = [{"role": "system", "content": system_text}]

    for msg in messages:
        if msg.get("role") == "system":
            # Merge additional system messages
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                result[0]["content"] += "\n\n" + content
        else:
            result.append(msg)

    return result


# ─────────────────────────────────────────────
# Static page routes
# ─────────────────────────────────────────────

@app.get("/")
async def home_page():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/login")
async def login_page():
    return FileResponse(str(_STATIC_DIR / "login.html"))


@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(str(_STATIC_DIR / "dashboard.html"))


# ─────────────────────────────────────────────
# Admin API routes
# ─────────────────────────────────────────────

@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    password = body.get("password", "")
    if verify_admin_password(password):
        return {"token": _create_token()}
    raise HTTPException(401, "Invalid password")


@app.post("/api/change-password")
async def api_change_password(request: Request):
    _require_admin(request)
    body = await request.json()
    old_pw = body.get("old_password", "")
    new_pw = body.get("new_password", "")
    if not new_pw or len(new_pw) < 4:
        raise HTTPException(400, "New password must be at least 4 characters")
    if change_admin_password(old_pw, new_pw):
        return {"ok": True}
    raise HTTPException(400, "Old password is incorrect")


@app.get("/api/status")
async def api_status():
    uptime = time.time() - _start_time
    mc = get_model_config()
    return {
        "status": "online",
        "uptime_seconds": round(uptime),
        "models": {
            "simple": mc.get("simple", {}).get("model_name", "—"),
            "complex": mc.get("complex", {}).get("model_name", "—"),
        },
        "total_requests": get_total_requests(),
    }


@app.post("/api/keys")
async def api_create_key(request: Request):
    _require_admin(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Name is required")
    key = create_client_key(name)
    return {"api_key": key, "name": name}


@app.get("/api/keys")
async def api_list_keys(request: Request):
    _require_admin(request)
    return {"keys": list_client_keys()}


@app.delete("/api/keys/{key}")
async def api_revoke_key(key: str, request: Request):
    _require_admin(request)
    if revoke_client_key(key):
        return {"ok": True}
    raise HTTPException(404, "Key not found")


@app.get("/api/stats")
async def api_get_stats(request: Request):
    _require_admin(request)
    return {"stats": get_stats()}


@app.get("/api/models")
async def api_get_models(request: Request):
    """Get model configuration (masked API keys for UI)."""
    _require_admin(request)
    mc = get_model_config()
    # Remove raw api_key from response, only send masked version
    for tier in mc:
        mc[tier].pop("api_key", None)
    return {"models": mc}


@app.put("/api/models")
async def api_update_models(request: Request):
    """Update model configuration for one or both tiers."""
    _require_admin(request)
    body = await request.json()
    updated = []
    for tier in ("simple", "complex"):
        cfg = body.get(tier)
        if cfg:
            provider   = cfg.get("provider", "anthropic").strip()
            model_name = cfg.get("model_name", "").strip()
            api_key    = cfg.get("api_key", "").strip()
            base_url   = cfg.get("base_url", "https://api.anthropic.com").strip()
            if not model_name:
                raise HTTPException(400, f"model_name is required for {tier}")
            if provider not in ("anthropic", "openai"):
                raise HTTPException(400, f"provider must be 'anthropic' or 'openai' for {tier}")
            if not api_key:
                existing = get_model_config_raw(tier)
                api_key = existing["api_key"] if existing else ""
            update_model_config(tier, provider, model_name, api_key, base_url)
            updated.append(tier)
    if not updated:
        raise HTTPException(400, "No valid model config provided")
    return {"ok": True, "updated": updated}


# ─────────────────────────────────────────────
# Chat completions API
# ─────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    client_key = _extract_client_key(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    messages   = body.get("messages", [])
    tools      = body.get("tools", [])
    max_tokens = body.get("max_tokens", MAX_TOKENS)
    do_stream  = body.get("stream", True)
    req_model  = body.get("model", "cortex-auto")

    if not messages:
        raise HTTPException(400, "No messages provided")

    # 1. Classify complexity
    if req_model and req_model.lower() != "cortex-auto":
        # Manual override — determine complexity from model name
        is_complex = any(kw in req_model.lower() for kw in ("sonnet", "opus", "complex"))
        complexity = Complexity.COMPLEX if is_complex else Complexity.SIMPLE
        method_str = "manual"
        final_model = req_model
    else:
        complexity, method = await classify_complexity(messages)
        method_str = method.value
        final_model = _get_model(complexity)

    routing = {
        "complexity":      complexity.value,
        "classify_method": method_str,
        "model":           final_model,
    }
    log.info(f"[Router] → {final_model} ({complexity.value} via {method_str}) | tools={len(tools)}")

    # 2. Detect language
    lang = detect_language(messages)

    # 3. Stream or wait
    if do_stream:
        return StreamingResponse(
            stream_sse(messages, tools, max_tokens, routing, complexity, client_key, lang),
            media_type="text/event-stream",
            headers={
                "Cache-Control":       "no-cache",
                "X-Accel-Buffering":   "no",
                "X-Cortex-Model":      final_model,
                "X-Cortex-Complexity": routing["complexity"],
                "X-Cortex-Language":   lang,
            },
        )

    # Non-stream
    cfg = _get_tier_config(complexity)
    provider = cfg.get("provider", "anthropic")

    if provider == "openai":
        client = _make_openai_client(cfg)
        oai_messages = _build_openai_messages(messages, complexity, lang)
        params = {"model": final_model, "max_tokens": max_tokens, "messages": oai_messages}
        if tools:
            params["tools"] = tools
        try:
            response = await client.chat.completions.create(**params)
        except openai.AuthenticationError:
            raise HTTPException(401, "Authentication failed.")
        except openai.APIError as e:
            log.error(f"OpenAI API error ({final_model}): {type(e).__name__}")
            raise HTTPException(502, "Downstream API error.")

        usage = response.usage
        log_request(
            client_key=client_key,
            model=final_model,
            complexity=routing["complexity"],
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

        # OpenAI response is already in OpenAI format, just add cortex metadata
        resp_dict = response.model_dump()
        resp_dict["cortex"] = routing
        return JSONResponse(resp_dict)

    else:
        params  = build_params(messages, tools, max_tokens, final_model, complexity, lang)
        aclient = _make_anthropic_client(cfg)
        try:
            response = await aclient.messages.create(**params)
        except anthropic.AuthenticationError:
            raise HTTPException(401, "Authentication failed.")
        except anthropic.APIError as e:
            log.error(f"Anthropic API error ({final_model}): {type(e).__name__}")
            raise HTTPException(502, "Downstream API error.")

        usage = response.usage
        log_request(
            client_key=client_key,
            model=final_model,
            complexity=routing["complexity"],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        return JSONResponse(format_non_stream(response, routing))


@app.get("/v1/models")
async def list_models():
    mc = get_model_config()
    s_name = mc.get("simple", {}).get("model_name", "unknown")
    c_name = mc.get("complex", {}).get("model_name", "unknown")
    return {
        "object": "list",
        "data": [
            {"id": "cortex-auto",  "object": "model", "created": 1700000000, "owned_by": "cortex"},
            {"id": s_name,         "object": "model", "created": 1700000000, "owned_by": "anthropic"},
            {"id": c_name,         "object": "model", "created": 1700000000, "owned_by": "anthropic"},
        ],
    }


@app.get("/health")
async def health():
    mc = get_model_config()
    return {
        "status":  "ok",
        "routing": "Auto-classification: simple/complex",
        "models": {
            "simple":  mc.get("simple", {}).get("model_name", "?"),
            "complex": mc.get("complex", {}).get("model_name", "?"),
        },
    }
