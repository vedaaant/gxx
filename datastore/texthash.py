"""Stable text hashing for dedup.

- ``content_hash`` — exact 64-bit hash of normalized text; the primary dedup key.
  Uses blake2b (NOT builtin ``hash``) so values are stable across processes/runs,
  which matters because the hash is persisted in SQLite and compared later.
- ``simhash`` — 64-bit SimHash over word 3-shingles for *fuzzy* near-duplicate
  detection. Small ``hamming`` distance => near-duplicate (screenpipe uses ~10 on
  long text; on short summaries the signal is noisier, so we keep the collapse
  threshold conservative and rely on ``content_hash`` as the primary key).

Stored as signed ints in SQLite (which has no uint64); use ``to_sqlite_int`` /
``from_sqlite_int`` at the DB boundary.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_MASK64 = (1 << 64) - 1


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def content_hash(text: str) -> int:
    """Exact, stable 64-bit unsigned hash of normalized text."""
    h = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def _shingles(text: str, n: int = 3) -> list[str]:
    words = normalize(text).split()
    if len(words) < n:
        return words or [""]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def simhash(text: str, n: int = 3) -> int:
    """64-bit SimHash over word n-shingles."""
    bits = [0] * 64
    for sh in _shingles(text, n):
        hv = int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big")
        for b in range(64):
            bits[b] += 1 if (hv >> b) & 1 else -1
    out = 0
    for b in range(64):
        if bits[b] > 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & _MASK64).count("1")


def to_sqlite_int(u64: int) -> int:
    """Map an unsigned 64-bit int to the signed range SQLite stores."""
    u64 &= _MASK64
    return u64 - (1 << 64) if u64 >= (1 << 63) else u64


def from_sqlite_int(i: int) -> int:
    return i + (1 << 64) if i < 0 else i
