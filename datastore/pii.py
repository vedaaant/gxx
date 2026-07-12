"""Best-effort, disclosed PII scrubbing (NOT production-grade — see PRD non-goals).

Runs client-side before any network call; the relay re-runs the same ruleset as a
backstop. Keep the two copies in sync (relay/pii.py mirrors this).
"""

from __future__ import annotations

import re

_RULES: list[tuple[str, re.Pattern]] = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # credit-card-like: 13-16 digits, optional separators
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # phone: +country / grouped digits (kept after card so long runs match card first)
    ("[PHONE]", re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2,4}\d{2,4}\b")),
    # common API-key / token shapes
    ("[SECRET]", re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,}\b")),
    ("[SECRET]", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),  # AWS access key id
    ("[SECRET]", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),  # Google API key
    # bearer tokens / long hex or base64 secrets after a key= or token: marker
    ("[SECRET]", re.compile(r"(?i)\b(?:token|api[_-]?key|secret|password)\b\s*[:=]\s*\S+")),
    # IPv4
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def scrub(text: str) -> str:
    """Replace detected PII/secret patterns with typed placeholders."""
    if not text:
        return text
    out = text
    for repl, pat in _RULES:
        out = pat.sub(repl, out)
    return out


def scrub_dict(obj: dict, keys: list[str] | None = None) -> dict:
    """Scrub string values of a dict (all string values, or a subset of keys)."""
    result = dict(obj)
    for k, v in obj.items():
        if isinstance(v, str) and (keys is None or k in keys):
            result[k] = scrub(v)
    return result
