"""
pipeline.py — Compression pipeline + language detection for Cortex Context Router.

Contains:
  - All constants and configuration (models, thresholds, system prompts, patterns)
  - Shared message helpers (extract_text, is_tool_result_msg, ...)
  - Pass 1 — Sliding window
  - Pass 2 — Deduplicate repeated file reads
  - Pass 3 — Summarize settled tool chains
  - Pass 4 — Inject cache_control
  - Pass 6 — Language detection
  - run_pipeline() — orchestrates Pass 1–4

Pure functions only. No I/O, no HTTP, no Anthropic calls.
Easy to unit test in isolation.
"""

import hashlib
import logging
import re

# ─────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────

MessageList = list[dict]

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cortex")

# Suppress DEBUG noise from third-party libs
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("anthropic").setLevel(logging.INFO)

# ─────────────────────────────────────────────
# Model defaults
# ─────────────────────────────────────────────

# Fallback model names — overridable via X-Cortex-Model-* headers
DEFAULT_MODEL_HAIKU  = "claude-haiku-4-5"
DEFAULT_MODEL_SONNET = "claude-sonnet-4-5"
MAX_TOKENS           = 8192

# ─────────────────────────────────────────────
# Pipeline thresholds
# ─────────────────────────────────────────────

SLIDING_WINDOW_TURNS         = 12
MIN_CACHE_CHARS              = 3000   # ~1024 tokens, Anthropic caching minimum
TOOL_CHAIN_SUMMARY_THRESHOLD = 6      # keep context within agent loop

HEURISTIC_COMPLEX_THRESHOLD  = 2
HEURISTIC_SIMPLE_THRESHOLD   = -2

# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────

_SYSTEM_HAIKU = """\
<s>
  <role>
    You are Cortex, a fast and precise coding assistant optimised for everyday tasks.
    You are powered by Claude Haiku and routed here because this task is straightforward.
  </role>

  <thinking>
    Before every response, write your reasoning inside <think>...</think> tags.
    Keep it concise: 1-3 sentences. Focus on what the task requires and any edge cases.
  </thinking>

  <behaviour>
    <item>Answer concisely. Prefer short, direct responses over verbose explanations.</item>
    <item>When writing code, produce only what was asked — no extra scaffolding.</item>
    <item>Do not create files unless explicitly asked.</item>
    <item>If a task turns out to be complex, say so briefly and let the user decide.</item>
  </behaviour>

  <tool_use>
    <item>If tools are provided in this request (Agent mode), use them freely and proactively.</item>
    <item>If no tools are available, answer from context only — do not attempt tool calls.</item>
    <item>Prefer reading actual file content over guessing when context is ambiguous.</item>
    <item>CRITICAL: To refactor, edit, or write code, you MUST use the provided editing tools.</item>
  </tool_use>

  <output_format>
    <item>Return code in fenced blocks with the correct language tag.</item>
    <item>Keep prose to a minimum — code speaks louder than commentary.</item>
  </output_format>
</s>"""

_SYSTEM_SONNET = """\
<s>
  <role>
    You are Cortex, a senior software engineer and architect.
    You are routed here because this task requires deep reasoning, design thinking, or hard debugging.
  </role>

  <thinking>
    Before every response, write your reasoning inside <think>...</think> tags.
    For complex tasks: outline your approach, key tradeoffs, and potential risks (4-8 sentences).
    For simpler tasks: keep it brief (1-3 sentences).
  </thinking>

  <behaviour>
    <item>Think step by step before answering complex questions.</item>
    <item>Explain trade-offs when multiple approaches exist.</item>
    <item>When refactoring or redesigning, justify decisions clearly.</item>
    <item>Do not create files unless explicitly asked.</item>
    <item>Prefer surgical changes over full rewrites unless a rewrite is clearly better.</item>
  </behaviour>

  <tool_use>
    <item>Use tools only when they are provided in this request (Agent mode).</item>
    <item>If no tools are available, answer from context only.</item>
    <item>When tools are available: use them proactively and chain multiple calls when needed.</item>
    <item>For architecture or debugging tasks, read relevant files before proposing changes.</item>
    <item>CRITICAL: To refactor, edit, or write code, you MUST use the provided editing tools.</item>
  </tool_use>

  <output_format>
    <item>Return code in fenced blocks with the correct language tag.</item>
    <item>For architecture questions, use concise diagrams or bullet points.</item>
    <item>Cite specific lines or functions when referring to existing code.</item>
  </output_format>
</s>"""

PROMPT_TIER: dict[str, str] = {
    "haiku":  _SYSTEM_HAIKU,
    "sonnet": _SYSTEM_SONNET,
    "opus":   _SYSTEM_SONNET,
}

# ─────────────────────────────────────────────
# Language detection data
# ─────────────────────────────────────────────

# Script/character-range patterns — fast, no external deps
_LANG_CHAR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("vi", re.compile(r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]", re.IGNORECASE)),
    ("zh", re.compile(r"[\u4e00-\u9fff]")),
    ("ja", re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")),
    ("ko", re.compile(r"[\uac00-\ud7af]")),
    ("ar", re.compile(r"[\u0600-\u06ff]")),
    ("th", re.compile(r"[\u0e00-\u0e7f]")),
    ("ru", re.compile(r"[\u0400-\u04ff]")),
]

# Common Vietnamese words — catches messages written without diacritics
_VI_KEYWORD_RE = re.compile(
    r"\b(bạn|tôi|mình|cho|hãy|làm|thế|nào|cái|này|đây|được|không|có|với|trong|của|và|là|một|các|như|khi|nếu|thì|vì|để|rằng|hỏi|giúp|viết|sửa|tạo|chạy|lỗi|code|file|hàm|biến|class|module)\b",
    re.IGNORECASE,
)

LANG_NAMES: dict[str, str] = {
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "th": "Thai",
    "ru": "Russian",
    "en": "English",
}

# ─────────────────────────────────────────────
# Shared message helpers
# ─────────────────────────────────────────────

def extract_text(content) -> str:
    """Recursively extracts plain text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    parts.append(extract_text(b.get("content", "")))
        return "\n".join(parts)
    return ""


def is_tool_result_msg(msg: dict) -> bool:
    content = msg.get("content", "")
    if isinstance(content, list):
        return any(b.get("type") == "tool_result" for b in content)
    return False


def has_tool_use(msg: dict) -> bool:
    content = msg.get("content", "")
    if isinstance(content, list):
        return any(b.get("type") == "tool_use" for b in content)
    return False


def extract_tool_names(msg: dict) -> list[str]:
    content = msg.get("content", [])
    if isinstance(content, list):
        return [b["name"] for b in content if b.get("type") == "tool_use"]
    return ["tool"]


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ─────────────────────────────────────────────
# Intent extraction (shared by pipeline & router)
# ─────────────────────────────────────────────

def get_last_user_text(messages: MessageList) -> str:
    """
    Extracts the user's latest intent message text.
    Used by both the complexity classifier and the language detector.
    Filters out tool results, code blocks, and non-intent messages.
    """
    conv = [m for m in messages if m["role"] != "system"]

    def _clean(raw: str) -> str:
        c = re.sub(r"```[\s\S]*?```", " ", raw)
        c = re.sub(r"`[^`\n]{1,80}`", " ", c)
        c = re.sub(r"\b\S+\.(py|js|ts|json|yaml|yml|md|sh)\b", " ", c)
        c = re.sub(r"\\[a-zA-Z]\S*", " ", c)
        c = re.sub(r"https?://\S+", " ", c)
        c = re.sub(r"\s+", " ", c).strip()
        return c

    def _is_intent(text: str) -> bool:
        if len(text) < 3 or len(text) > 400:
            return False
        if text.count("\n") > 5:
            return False
        if re.match(r"^(import |from |def |class |#|//|{)", text.strip()):
            return False
        return True

    candidates = []
    for msg in conv:
        if msg["role"] != "user":
            continue
        if is_tool_result_msg(msg):
            continue
        raw     = extract_text(msg.get("content", ""))
        cleaned = _clean(raw)
        log.debug(f"[Classify] candidate raw={repr(raw[:80])} cleaned={repr(cleaned[:80])} intent={_is_intent(cleaned)}")
        if _is_intent(cleaned):
            candidates.append(cleaned[:400].lower())

    if not candidates:
        log.info("[Classify] No intent candidates found — returning empty")
        return ""

    scored = [(len(c), c) for c in candidates if len(c.split()) >= 5]
    if scored:
        best = min(scored, key=lambda x: x[0])[1]
        log.info(f"[Classify] Best candidate ({len(scored)} options): {repr(best[:80])}")
        return best
    return min(candidates, key=len)


# ─────────────────────────────────────────────
# Pass 1 — Sliding window
# ─────────────────────────────────────────────

def pass_sliding_window(messages: MessageList) -> MessageList:
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs   = [m for m in messages if m["role"] != "system"]

    if len(conv_msgs) <= SLIDING_WINDOW_TURNS * 2:
        return messages

    keep_from = max(0, len(conv_msgs) - SLIDING_WINDOW_TURNS * 2)
    while keep_from < len(conv_msgs):
        msg = conv_msgs[keep_from]
        if msg["role"] == "user" and not is_tool_result_msg(msg):
            break
        keep_from += 1

    kept    = conv_msgs[keep_from:]
    dropped = len(conv_msgs) - len(kept)

    if dropped:
        kept = [{
            "role":    "user",
            "content": f"[Cortex] {dropped} earlier messages trimmed. Resume from current state.",
        }] + kept

    log.info(f"[Pass 1] Sliding window: kept {len(kept)}/{len(conv_msgs)} conv messages")
    return system_msgs + kept


# ─────────────────────────────────────────────
# Pass 2 — Deduplicate repeated file reads
# ─────────────────────────────────────────────

def pass_deduplicate_file_reads(messages: MessageList) -> MessageList:
    seen: dict[str, str] = {}
    total_saved = 0
    result = []

    for turn_idx, msg in enumerate(messages):
        if msg["role"] != "user":
            result.append(msg)
            continue

        content = msg.get("content", "")
        if not isinstance(content, list):
            result.append(msg)
            continue

        new_blocks = []
        for block in content:
            if block.get("type") != "tool_result":
                new_blocks.append(block)
                continue

            text = extract_text(block.get("content", ""))
            if len(text) < 300:
                new_blocks.append(block)
                continue

            key = content_hash(text)
            if key in seen:
                saved = len(text)
                total_saved += saved
                new_block = dict(block)
                new_block["content"] = (
                    f"[Cortex] Identical to {seen[key]} — omitted ({saved} chars)"
                )
                new_blocks.append(new_block)
            else:
                seen[key] = f"turn {turn_idx}/tool {block.get('tool_use_id', '?')}"
                new_blocks.append(block)

        result.append({**msg, "content": new_blocks})

    if total_saved:
        log.info(f"[Pass 2] Dedup: saved {total_saved} chars")
    return result


# ─────────────────────────────────────────────
# Pass 3 — Summarize settled tool chains
# ─────────────────────────────────────────────

def pass_summarize_tool_chains(messages: MessageList) -> MessageList:
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs   = [m for m in messages if m["role"] != "system"]

    if len(conv_msgs) <= TOOL_CHAIN_SUMMARY_THRESHOLD * 2 + 4:
        return messages

    cutoff = len(conv_msgs) - TOOL_CHAIN_SUMMARY_THRESHOLD * 2
    early, recent = conv_msgs[:cutoff], conv_msgs[cutoff:]
    compressed, n = _compress_chains(early)

    if n:
        log.info(f"[Pass 3] Summarized {n} tool chains")
    return system_msgs + compressed + recent


def _compress_chains(msgs: MessageList) -> tuple[MessageList, int]:
    result, n, i = [], 0, 0
    while i < len(msgs):
        msg = msgs[i]
        if msg["role"] == "assistant" and has_tool_use(msg):
            if i + 1 < len(msgs) and is_tool_result_msg(msgs[i + 1]):
                names   = extract_tool_names(msg)
                outcome = _summarize_results(msgs[i + 1])
                result.append({
                    "role":    "assistant",
                    "content": f"[Cortex] Used {', '.join(names)} → {outcome}",
                })
                n += 1
                i += 2
                continue
        result.append(msg)
        i += 1
    return result, n


def _summarize_results(msg: dict) -> str:
    content = msg.get("content", [])
    if not isinstance(content, list):
        return str(content)[:120]
    parts = []
    for block in content:
        if block.get("type") == "tool_result":
            text = extract_text(block.get("content", ""))
            parts.append(text.strip().split("\n")[0][:100] or "(empty)")
    return "; ".join(parts) or "(no result)"


# ─────────────────────────────────────────────
# Pass 4 — Inject cache_control
# ─────────────────────────────────────────────

def pass_inject_cache(messages: MessageList) -> MessageList:
    result, cached = [], 0
    for msg in messages:
        if msg["role"] == "system":
            injected = _cache_last_block(msg)
            if injected is not msg:
                cached += 1
            result.append(injected)
        else:
            result.append(msg)
    if cached:
        log.info(f"[Pass 4] Injected cache_control on {cached} system message(s)")
    return result


def _cache_last_block(msg: dict) -> dict:
    content = msg.get("content", "")
    if isinstance(content, str):
        if len(content) < MIN_CACHE_CHARS:
            return msg
        return {**msg, "content": [{
            "type": "text", "text": content,
            "cache_control": {"type": "ephemeral"},
        }]}
    if isinstance(content, list) and content:
        if len(extract_text(content)) < MIN_CACHE_CHARS:
            return msg
        last = {**content[-1], "cache_control": {"type": "ephemeral"}}
        return {**msg, "content": content[:-1] + [last]}
    return msg


# ─────────────────────────────────────────────
# Pass 6 — Language detection
# ─────────────────────────────────────────────

def detect_language(messages: MessageList) -> str:
    """
    Detects the language of the user's latest intent message.
    Uses character-range patterns for CJK/Arabic/etc., diacritic patterns
    for Vietnamese, and a keyword-ratio fallback for ambiguous Latin text.
    Returns a BCP-47 language tag (e.g. 'vi', 'en', 'zh') or 'en' as default.
    """
    text = get_last_user_text(messages)
    if not text:
        return "en"

    # Character-range fast path (covers VI diacritics, CJK, Arabic, Thai, Cyrillic)
    for lang, pattern in _LANG_CHAR_PATTERNS:
        if pattern.search(text):
            log.debug(f"[Lang] Detected '{lang}' via char-range pattern")
            return lang

    # Vietnamese keyword fallback (plain ASCII romanisation without diacritics)
    vi_hits = len(_VI_KEYWORD_RE.findall(text))
    words   = len(text.split())
    if words > 0 and vi_hits / words >= 0.25:
        log.debug(f"[Lang] Detected 'vi' via keyword ratio ({vi_hits}/{words})")
        return "vi"

    log.debug("[Lang] No match — defaulting to 'en'")
    return "en"


# ─────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────

def run_pipeline(messages: MessageList, is_agent: bool) -> tuple[MessageList, dict]:
    """
    Runs Pass 1–4 in sequence and returns (processed_messages, stats).
    Pass 3 (summarize tool chains) is skipped in agent mode to preserve
    live tool context.
    """
    original_chars = sum(len(extract_text(m.get("content", ""))) for m in messages)

    messages = pass_sliding_window(messages)
    messages = pass_deduplicate_file_reads(messages)

    if not is_agent:
        messages = pass_summarize_tool_chains(messages)

    messages = pass_inject_cache(messages)

    final_chars = sum(len(extract_text(m.get("content", ""))) for m in messages)
    saved = original_chars - final_chars
    ratio = round((1 - final_chars / max(original_chars, 1)) * 100, 1)

    stats = {
        "original_chars":  original_chars,
        "final_chars":     final_chars,
        "saved_chars":     saved,
        "compression_pct": ratio,
    }
    log.info(f"[Pipeline] {original_chars} → {final_chars} chars ({ratio}% saved)")
    return messages, stats
