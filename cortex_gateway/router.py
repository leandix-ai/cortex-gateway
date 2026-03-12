"""
router.py — Complexity classifier + language detection for Cortex Gateway.

Decides whether a request routes to the simple model or complex model
via a two-stage strategy:

  Stage 1 — Heuristic scoring  : fast, zero-cost regex pattern matching.
  Stage 2 — LLM classification : called only when heuristic is inconclusive.

Also contains:
  - Intent text extraction (get_last_user_text)
  - Language detection (detect_language)
"""

import re
from enum import Enum

import anthropic

from .config import (
    HEURISTIC_COMPLEX_THRESHOLD,
    HEURISTIC_SIMPLE_THRESHOLD,
    SIMPLE_MODEL_API_KEY,
    SIMPLE_MODEL_BASE_URL,
    SIMPLE_MODEL_NAME,
    LANG_NAMES,
    MessageList,
    _CLASSIFY_SYSTEM,
    _COMPLEX_PATTERNS,
    _LANG_CHAR_PATTERNS,
    _OVERRIDE_SIMPLE_PATTERNS,
    _SIMPLE_PATTERNS,
    _VI_KEYWORD_RE,
    log,
)

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class Complexity(str, Enum):
    SIMPLE  = "simple"
    COMPLEX = "complex"


class ClassifyMethod(str, Enum):
    HEURISTIC = "heuristic"
    LLM       = "llm"


# ─────────────────────────────────────────────
# Message helpers
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


# ─────────────────────────────────────────────
# Intent extraction
# ─────────────────────────────────────────────

def get_last_user_text(messages: MessageList) -> str:
    """
    Extracts the user's latest intent message text.
    Used by both the complexity classifier and the language detector.
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
        if _is_intent(cleaned):
            candidates.append(cleaned[:400].lower())

    if not candidates:
        return ""

    scored = [(len(c), c) for c in candidates if len(c.split()) >= 5]
    if scored:
        return min(scored, key=lambda x: x[0])[1]
    return min(candidates, key=len)


# ─────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────

def detect_language(messages: MessageList) -> str:
    """
    Detects the language of the user's latest intent message.
    Returns a BCP-47 language tag (e.g. 'vi', 'en', 'zh') or 'en' as default.
    """
    text = get_last_user_text(messages)
    if not text:
        return "en"

    for lang, pattern in _LANG_CHAR_PATTERNS:
        if pattern.search(text):
            return lang

    vi_hits = len(_VI_KEYWORD_RE.findall(text))
    words   = len(text.split())
    if words > 0 and vi_hits / words >= 0.25:
        return "vi"

    return "en"


# ─────────────────────────────────────────────
# Heuristic scorer
# ─────────────────────────────────────────────

def _heuristic_score(text: str) -> int:
    score = 0
    for pattern, weight in _COMPLEX_PATTERNS:
        if pattern.search(text):
            score += weight
    for pattern, weight in _SIMPLE_PATTERNS:
        if pattern.search(text):
            score += weight
    return score


# ─────────────────────────────────────────────
# LLM classifier (fallback for grey zone)
# ─────────────────────────────────────────────

async def _llm_classify(last_user_text: str) -> Complexity:
    """Uses the simple model to classify task complexity."""
    try:
        aclient = anthropic.AsyncAnthropic(
            api_key=SIMPLE_MODEL_API_KEY,
            base_url=SIMPLE_MODEL_BASE_URL,
        )
        resp = await aclient.messages.create(
            model=SIMPLE_MODEL_NAME,
            max_tokens=5,
            system=_CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": last_user_text[:2000]}],
        )
        verdict = extract_text(resp.content[0] if resp.content else "").strip().upper()
        return Complexity.COMPLEX if "COMPLEX" in verdict else Complexity.SIMPLE
    except Exception as e:
        log.warning(f"[Router] LLM classify failed ({e}), defaulting SIMPLE")
        return Complexity.SIMPLE


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

async def classify_complexity(
    messages: MessageList,
) -> tuple[Complexity, ClassifyMethod]:
    """
    Classifies the complexity of the latest user request.
    Returns (Complexity, ClassifyMethod) — the verdict and how it was reached.
    """
    last_text = get_last_user_text(messages)

    if not last_text:
        log.info("[Router] No classifiable text → SIMPLE (default)")
        return Complexity.SIMPLE, ClassifyMethod.HEURISTIC

    if len(last_text.split()) < 5:
        log.info(f"[Router] Text too short ({len(last_text.split())} words) → SIMPLE (default)")
        return Complexity.SIMPLE, ClassifyMethod.HEURISTIC

    for pat in _OVERRIDE_SIMPLE_PATTERNS:
        if pat.search(last_text):
            log.info("[Router] Override match → SIMPLE (meta request)")
            return Complexity.SIMPLE, ClassifyMethod.HEURISTIC

    score = _heuristic_score(last_text)
    log.info(f"[Router] Heuristic score={score} | '{last_text[:80]}'")

    if score >= HEURISTIC_COMPLEX_THRESHOLD:
        log.info("[Router] → COMPLEX (heuristic confident)")
        return Complexity.COMPLEX, ClassifyMethod.HEURISTIC

    if score <= HEURISTIC_SIMPLE_THRESHOLD:
        log.info("[Router] → SIMPLE (heuristic confident)")
        return Complexity.SIMPLE, ClassifyMethod.HEURISTIC

    log.info(f"[Router] Score {score} uncertain, calling LLM classifier...")
    verdict = await _llm_classify(last_text)
    log.info(f"[Router] → {verdict.upper()} (LLM)")
    return verdict, ClassifyMethod.LLM
