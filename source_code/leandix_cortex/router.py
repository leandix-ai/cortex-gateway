"""
router.py — Pass 5: complexity classifier for Cortex Context Router.

Decides whether a request routes to the simple model (Haiku) or the
complex model (Sonnet) via a two-stage strategy:

  Stage 1 — Heuristic scoring  : fast, zero-cost regex pattern matching.
                                  Returns immediately when score exceeds threshold.
  Stage 2 — LLM classification : called only when heuristic is inconclusive.

Imports only from pipeline.py — no FastAPI, no HTTP concerns.
"""

from enum import Enum

import anthropic

from .pipeline import (
    HEURISTIC_COMPLEX_THRESHOLD,
    HEURISTIC_SIMPLE_THRESHOLD,
    MessageList,
    extract_text,
    get_last_user_text,
    log,
)


# ─────────────────────────────────────────────
# Router patterns & classifier prompt
# ─────────────────────────────────────────────

import re

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
# Enums
# ─────────────────────────────────────────────

class Complexity(str, Enum):
    SIMPLE  = "simple"
    COMPLEX = "complex"


class ClassifyMethod(str, Enum):
    HEURISTIC = "heuristic"
    LLM       = "llm"


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

async def _llm_classify(last_user_text: str, api_key: str, classify_model: str) -> Complexity:
    try:
        aclient = anthropic.AsyncAnthropic(api_key=api_key)
        resp = await aclient.messages.create(
            model=classify_model,  # Use the simple model for classification
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
    messages: MessageList, api_key: str, classify_model: str
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

    # Override: meta-requests like "give this chat a title" are always SIMPLE
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

    # Grey zone — call the LLM classifier
    log.info(f"[Router] Score {score} uncertain, calling {classify_model} classifier...")
    verdict = await _llm_classify(last_text, api_key, classify_model)
    log.info(f"[Router] → {verdict.upper()} (LLM)")
    return verdict, ClassifyMethod.LLM
