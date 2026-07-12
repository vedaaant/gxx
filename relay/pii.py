"""Server-side PII backstop for the relay.

Intentionally a COPY of the client-side ruleset (datastore/pii.py) rather than an
import: the relay is deployed as its own container and must not depend on the
client package. Keep the two in sync. Best-effort, disclosed — not guaranteed.
"""

from __future__ import annotations

import re

_RULES: list[tuple[str, re.Pattern]] = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("[PHONE]", re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2,4}\d{2,4}\b")),
    ("[SECRET]", re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,}\b")),
    ("[SECRET]", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("[SECRET]", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("[SECRET]", re.compile(r"(?i)\b(?:token|api[_-]?key|secret|password)\b\s*[:=]\s*\S+")),
    ("[IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def scrub(text: str) -> str:
    if not text:
        return text
    out = text
    for repl, pat in _RULES:
        out = pat.sub(repl, out)
    return out
