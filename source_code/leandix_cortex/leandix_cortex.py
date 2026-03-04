"""
leandix_cortex.py — Cortex Context Router: HTTP layer & application entry point.

Acts as the orchestrator. Imports from pipeline.py and router.py,
then wires everything into FastAPI routes.

Contains:
  - Anthropic client factory & API key extraction
  - OpenAI → Anthropic message format converter
  - Payload builder (build_params) with system prompt + language injection
  - SSE streaming generator (stream_sse) with think-tag parsing
  - Non-streaming response formatter (format_non_stream)
  - FastAPI app + routes: /v1/chat/completions, /v1/models, /health

Entry point: server.py → uvicorn → leandix_cortex:app
"""

import json
import re
import time
import uuid

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .pipeline import (
    DEFAULT_MODEL_HAIKU,
    DEFAULT_MODEL_SONNET,
    HEURISTIC_COMPLEX_THRESHOLD,
    HEURISTIC_SIMPLE_THRESHOLD,
    LANG_NAMES,
    MAX_TOKENS,
    MIN_CACHE_CHARS,
    PROMPT_TIER,
    SLIDING_WINDOW_TURNS,
    TOOL_CHAIN_SUMMARY_THRESHOLD,
    MessageList,
    detect_language,
    log,
    run_pipeline,
)
from .router import Complexity, classify_complexity

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Cortex — Context Router")

# ─────────────────────────────────────────────
# Auth & client
# ─────────────────────────────────────────────

def _extract_api_key(request: Request) -> str:
    """Extracts the API key from the Authorization: Bearer <key> header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        key = auth[len("Bearer "):].strip()
        if key:
            return key
    raise HTTPException(status_code=401, detail="Missing API key in Authorization header.")


def _make_client(api_key: str) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=api_key)


# ─────────────────────────────────────────────
# Config extraction
# ─────────────────────────────────────────────

def _extract_cortex_config(request: Request, body_model: str) -> tuple[str, str, bool]:
    """
    Parses the request to extract model configuration.
    Returns: (model_simple, model_complex, is_auto_route)
    """
    # Option 3: Dynamic Override — user forces a specific model
    if body_model and body_model.lower() != "cortex-auto":
        log.info(f"[Config] Manual override detected: forcing {body_model}")
        return (body_model, body_model, False)

    # Option 1: Read config from Headers for the Auto-Router
    headers   = request.headers
    m_simple  = headers.get("X-Cortex-Model-Simple",  DEFAULT_MODEL_HAIKU)
    m_complex = headers.get("X-Cortex-Model-Complex", DEFAULT_MODEL_SONNET)
    return (m_simple, m_complex, True)


# ─────────────────────────────────────────────
# System prompt selector
# ─────────────────────────────────────────────

def _get_system_prompt(model_name: str) -> str:
    """Selects the system prompt tier by model name substring, with safe fallback."""
    name_lower = model_name.lower()
    for keyword, prompt in PROMPT_TIER.items():
        if keyword in name_lower:
            return prompt
    return PROMPT_TIER["haiku"]  # Safe default


# ─────────────────────────────────────────────
# Language instruction block
# ─────────────────────────────────────────────

def _build_language_block(lang: str) -> dict:
    """
    Returns a non-cached system block instructing the model to respond
    in the detected language. Kept separate from the base prompt so that
    the base prompt's cache_control breakpoint is not invalidated on every
    language change.
    """
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
    and sanitises tool IDs. Already-converted messages pass through unchanged.
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
            # Collect consecutive tool result messages into a single user turn
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
    lang: str = "en",
) -> dict:
    """
    Builds the full kwargs dict for anthropic.messages.create / .stream.
    Converts messages, assembles the system prompt array, and normalises tools.
    """
    messages = _convert_messages(messages)

    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs   = [m for m in messages if m["role"] != "system"]

    params: dict = {"model": model, "max_tokens": max_tokens, "messages": conv_msgs}

    # Base system prompt — cached so repeated turns don't re-bill input tokens
    cortex_block: dict = {
        "type":          "text",
        "text":          _get_system_prompt(model),
        "cache_control": {"type": "ephemeral"},
    }

    # Language instruction — injected after the cached base prompt so it
    # does not bust the cache_control breakpoint on every new language.
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
        log.info(f"[build_params] Sending {len(anthropic_tools)} tools: {[t['name'] for t in anthropic_tools]}")
    else:
        log.info("[build_params] No tools — Chat mode")

    return params


# ─────────────────────────────────────────────
# Response helpers
# ─────────────────────────────────────────────

def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def format_non_stream(anthropic_msg, stats: dict, routing: dict) -> dict:
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
        "cortex": {**stats, "routing": routing},
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
    """
    Routes streaming text chunks into ('content', text) or
    ('reasoning_content', text) based on <think>...</think> tag presence.
    Handles partial tag boundaries across chunk boundaries safely.
    """
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
    stats: dict,
    routing: dict,
    api_key: str,
    lang: str = "en",
):
    """
    Async generator that streams an Anthropic response as OpenAI-compatible
    SSE chunks. Handles tool_use blocks, think-tag routing, cache token
    logging, and translates Anthropic errors to SSE error events.
    """
    req_id, created = _make_id(), int(time.time())
    model       = routing["model"]
    params      = build_params(messages, tools, max_tokens, model, lang)
    aclient     = _make_client(api_key)
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
                        log.info(f"[stream_sse] Tool called: {block.name} (id={block.id})")
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
            cr     = getattr(usage, "cache_read_input_tokens", 0)
            cc     = getattr(usage, "cache_creation_input_tokens", 0)

            if cr or cc:
                log.info(f"[Cache] read={cr} created={cc} tokens")

            yield _sse({
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens":               usage.input_tokens,
                    "completion_tokens":           usage.output_tokens,
                    "total_tokens":                usage.input_tokens + usage.output_tokens,
                    "cache_read_input_tokens":     cr,
                    "cache_creation_input_tokens": cc,
                },
                "cortex": {**stats, "routing": routing},
            })

    except anthropic.AuthenticationError:
        log.warning("[Auth] Authentication failed")
        yield _sse({"error": {"message": "Authentication failed.", "type": "authentication_error"}})
    except anthropic.APIError as e:
        log.error(f"Anthropic API error ({model}): {type(e).__name__}")
        yield _sse({"error": {"message": "An error occurred with the AI provider.", "type": "api_error", "model": model}})

    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    api_key = _extract_api_key(request)

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

    # 1. Extract model config from headers or manual override
    m_simple, m_complex, is_auto_route = _extract_cortex_config(request, req_model)

    # 2. Run compression pipeline (Pass 1–4)
    is_agent = bool(tools)
    processed, stats = run_pipeline(messages, is_agent=is_agent)

    # 3. Route to simple or complex model (Pass 5)
    if is_auto_route:
        complexity, method = await classify_complexity(processed, api_key, m_simple)
        final_model = m_complex if complexity == Complexity.COMPLEX else m_simple
        routing = {
            "complexity":      complexity.value,
            "classify_method": method.value,
            "model":           final_model,
        }
        log.info(f"[Router] Auto → {final_model} ({complexity.value} via {method.value}) | agent={is_agent} | tools={len(tools)}")
    else:
        final_model = req_model
        routing = {
            "complexity":      "override",
            "classify_method": "manual",
            "model":           final_model,
        }
        log.info(f"[Router] Bypass → {final_model} (Manual Override) | agent={is_agent} | tools={len(tools)}")

    if tools:
        log.info(f"[Tools] Received: {[t.get('function', t).get('name', '?') for t in tools]}")

    # 4. Detect user language (Pass 6)
    lang = detect_language(processed)
    log.info(f"[Lang] Detected language: '{lang}'")

    # 5. Stream or wait
    if do_stream:
        return StreamingResponse(
            stream_sse(processed, tools, max_tokens, stats, routing, api_key, lang),
            media_type="text/event-stream",
            headers={
                "Cache-Control":        "no-cache",
                "X-Accel-Buffering":    "no",
                "X-Cortex-Compression": f"{stats['compression_pct']}%",
                "X-Cortex-Model":       final_model,
                "X-Cortex-Complexity":  routing["complexity"],
                "X-Cortex-Language":    lang,
            },
        )

    params  = build_params(processed, tools, max_tokens, final_model, lang)
    aclient = _make_client(api_key)
    try:
        response = await aclient.messages.create(**params)
    except anthropic.AuthenticationError:
        log.warning("[Auth] Authentication failed")
        raise HTTPException(401, "Authentication failed.")
    except anthropic.APIError as e:
        log.error(f"Anthropic API error ({final_model}): {type(e).__name__}")
        raise HTTPException(502, "Downstream API error.")

    return JSONResponse(format_non_stream(response, stats, routing))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "cortex-auto",        "object": "model", "created": 1700000000, "owned_by": "cortex"},
            {"id": DEFAULT_MODEL_HAIKU,  "object": "model", "created": 1700000000, "owned_by": "anthropic"},
            {"id": DEFAULT_MODEL_SONNET, "object": "model", "created": 1700000000, "owned_by": "anthropic"},
        ],
    }


@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "auth":    "bearer",
        "routing": "Dynamic config via X-Cortex-Model-* headers or body model override",
        "passes":  [
            "pass_1: sliding_window",
            "pass_2: deduplicate_reads",
            "pass_3: summarize_chains",
            "pass_4: inject_cache",
            "pass_5: router",
            "pass_6: language_detection",
        ],
        "config": {
            "sliding_window_turns":         SLIDING_WINDOW_TURNS,
            "min_cache_chars":              MIN_CACHE_CHARS,
            "tool_chain_summary_threshold": TOOL_CHAIN_SUMMARY_THRESHOLD,
            "heuristic_complex_threshold":  HEURISTIC_COMPLEX_THRESHOLD,
            "heuristic_simple_threshold":   HEURISTIC_SIMPLE_THRESHOLD,
        },
    }
