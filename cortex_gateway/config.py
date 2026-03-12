"""
config.py — Centralised configuration for Cortex Gateway.

Loads settings from .env and contains:
  - Model config from environment
  - Logging setup
  - System prompts (_SYSTEM_SIMPLE, _SYSTEM_COMPLEX)
  - Routing patterns (_COMPLEX_PATTERNS, _SIMPLE_PATTERNS, _OVERRIDE_SIMPLE_PATTERNS)
  - Language detection data (_LANG_CHAR_PATTERNS, _VI_KEYWORD_RE, LANG_NAMES)
  - Classifier prompt (_CLASSIFY_SYSTEM)
  - JWT secret

No business logic lives here — only data and initialisation.
"""

import logging
import os
import re
import secrets

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load .env
# ─────────────────────────────────────────────

load_dotenv()

# ─────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────

MessageList = list[dict]

# ─────────────────────────────────────────────
# Model config from .env
# ─────────────────────────────────────────────

SIMPLE_MODEL_NAME     = os.getenv("SIMPLE_MODEL_NAME", "claude-haiku-4-5")
SIMPLE_MODEL_API_KEY  = os.getenv("SIMPLE_MODEL_API_KEY", "")
SIMPLE_MODEL_BASE_URL = os.getenv("SIMPLE_MODEL_BASE_URL", "https://api.anthropic.com")

COMPLEX_MODEL_NAME     = os.getenv("COMPLEX_MODEL_NAME", "claude-sonnet-4-5")
COMPLEX_MODEL_API_KEY  = os.getenv("COMPLEX_MODEL_API_KEY", "")
COMPLEX_MODEL_BASE_URL = os.getenv("COMPLEX_MODEL_BASE_URL", "https://api.anthropic.com")

MAX_TOKENS = 8192

# ─────────────────────────────────────────────
# Gateway config
# ─────────────────────────────────────────────

GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

# ─────────────────────────────────────────────
# JWT Secret (auto-generated per instance)
# ─────────────────────────────────────────────

JWT_SECRET   = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

# ─────────────────────────────────────────────
# Heuristic thresholds
# ─────────────────────────────────────────────

HEURISTIC_COMPLEX_THRESHOLD = 2
HEURISTIC_SIMPLE_THRESHOLD  = -2

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
# System prompts
# ─────────────────────────────────────────────

_SYSTEM_SIMPLE = """\
<s>
  <role>
    You are Cortex, a fast and precise coding assistant optimised for everyday tasks.
    You are powered by a fast model and routed here because this task is straightforward.
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

_SYSTEM_COMPLEX = """\
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
    "simple":  _SYSTEM_SIMPLE,
    "complex": _SYSTEM_COMPLEX,
}


# ─────────────────────────────────────────────
# Router — heuristic patterns
# ─────────────────────────────────────────────

_COMPLEX_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(architect|architecture|design pattern|system design|redesign|restructure)\b"), 3),
    (re.compile(r"\b(microservice|monolith|event.driven|domain.driven|ddd|hexagonal|clean arch)\b"), 3),
    (re.compile(r"\b(trade.?off|pros and cons|compare approach|which approach|best approach)\b"), 2),
    (re.compile(r"\b(scalab|maintainab|extensib|coupling|cohesion|solid principle)\b"), 2),
    (re.compile(r"\b(stacktrace|traceback|segfault|memory leak|deadlock|race condition)\b"), 3),
    (re.compile(r"\b(intermittent|flaky|heisenbug|hard to reproduce|happens randomly)\b"), 3),
    (re.compile(r"\b(concurren|thread.safe|async.*bug|event loop.*block)\b"), 2),
    (re.compile(r"\b(why (does|is|isn.t|doesn.t)|root cause|not working as expected)\b"), 1),
    (re.compile(r"\b(refactor (all|entire|whole)|across (all|multiple) file)\b"), 2),
    (re.compile(r"\b(codebase|entire project|end.to.end|full (flow|pipeline))\b"), 1),
    (re.compile(r"\b(tách|separate|extract|move|split).{0,30}(logic|business|service|handler|layer|module|component)\b"), 2),
    (re.compile(r"\b(logic|business|service).{0,20}(khỏi|out of|away from|from).{0,20}(ui|view|component|screen|template)\b"), 3),
    (re.compile(r"\b(decouple|isolate|segregate|separate concern)\b"), 2),
    (re.compile(r"\b(mvc|mvvm|mvp|clean architecture|layered|presentation layer|domain layer)\b"), 3),
    (re.compile(r"\b(god (class|object|component)|too much responsibilit|single responsibilit)\b"), 2),
]

_SIMPLE_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(fix typo|rename|format|lint|sort import|add comment|docstring)\b"), -2),
    (re.compile(r"\b(what is|what does|explain|how does|definition of)\b"), -1),
    (re.compile(r"\b(add (a |an )?(log|print|assert|test|type hint))\b"), -2),
    (re.compile(r"\b(simple|quick|small|minor|trivial|one.?liner)\b"), -1),
    (re.compile(r"\b(complete (the |this )?(function|method|class|snippet))\b"), -1),
]

_OVERRIDE_SIMPLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(title for (the |this )?chat|chat title|name (the |this )?conversation)\b"),
    re.compile(r"\breply with a (title|name|label)\b"),
    re.compile(r"\b(3|4|5)[- ]word (title|label)\b"),
    re.compile(r"\bgenerate (a )?(short |brief )?(title|name|label)\b"),
]

_CLASSIFY_SYSTEM = """\
You are a routing classifier. Given the user's latest message, decide if the task is:
- SIMPLE: autocomplete, rename, add comments, explain code, minor fix, add logging/tests, single-file edits
- COMPLEX: architecture/design decisions, debugging hard bugs (stacktrace, race condition, memory leak),
  cross-file refactoring, separating logic from UI/layers, decoupling modules, design pattern application

Respond with exactly one word: SIMPLE or COMPLEX. No explanation."""

# ─────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────

_LANG_CHAR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("vi", re.compile(r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]", re.IGNORECASE)),
    ("zh", re.compile(r"[\u4e00-\u9fff]")),
    ("ja", re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")),
    ("ko", re.compile(r"[\uac00-\ud7af]")),
    ("ar", re.compile(r"[\u0600-\u06ff]")),
    ("th", re.compile(r"[\u0e00-\u0e7f]")),
    ("ru", re.compile(r"[\u0400-\u04ff]")),
]

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
