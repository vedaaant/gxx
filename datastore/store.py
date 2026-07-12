"""Unified local activity datastore: turbovec embeddings + SQLite metadata.

Both are keyed by the same monotonically-increasing integer id. The store is the
single source of truth the watcher writes to and the MCP server reads from.

Design notes:
- Content-hash dedup: ``add`` returns the existing id (no new row) when the text
  content_hash matches the most recent row and the caller marks the write as
  dedup-eligible (soft trigger). Hard checkpoints always insert.
- ``optimize`` collapses near-duplicate neighbours (exact hash + optional simhash)
  and applies retention as *media eviction* (keep the row, drop cached blobs),
  hard-deleting only rows past the outer retention window.
- Not thread-safe for concurrent writers; the watcher is the sole writer. Reads
  (MCP queries) open their own short-lived connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .texthash import (
    content_hash as compute_content_hash,
    from_sqlite_int,
    hamming,
    simhash as compute_simhash,
    to_sqlite_int,
)
from .vecindex import VectorIndex

log = logging.getLogger("contour.store")

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
_HARD_TRIGGERS = {"AppSwitch", "WindowFocus", "Idle", "Manual"}


@dataclass
class Observation:
    """One thing worth remembering about on-screen/audio activity."""

    summary: str
    app: str = ""
    window: str = ""
    salient_text: str = ""
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    trigger: str = "Manual"
    source: str = "manual"  # 'uia' | 'vision' | 'manual'
    is_actionable: bool = False
    ts: int | None = None  # unix seconds; defaults to now at insert

    def hash_text(self) -> str:
        """Text used for content/dedup hashing — the durable content, not chrome."""
        return "\n".join(p for p in (self.summary, self.salient_text) if p)


class ActivityStore:
    def __init__(
        self,
        data_dir: str | Path,
        dim: int = 768,
        bit_width: int = 4,
        backend: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "activity.db"
        self.index_path = self.data_dir / "activity.tvim"
        self.dim = dim

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

        self.index = VectorIndex.load(
            self.index_path, dim=dim, bit_width=bit_width, backend=backend
        )
        self.backend = self.index.backend

    # -- helpers --------------------------------------------------------------
    def _next_id(self) -> int:
        cur = self.conn.execute("SELECT value FROM meta WHERE key='next_id'")
        row = cur.fetchone()
        nid = row["value"] if row else 1
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('next_id',?) "
            "ON CONFLICT(key) DO UPDATE SET value=?",
            (nid + 1, nid + 1),
        )
        return nid

    def _last_row(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM activity ORDER BY id DESC LIMIT 1"
        ).fetchone()

    # -- write ----------------------------------------------------------------
    def add(
        self,
        obs: Observation,
        embedding: np.ndarray | None = None,
        dedup: bool | None = None,
        heartbeat_secs: int = 30,
    ) -> int:
        """Insert an observation. Returns the (possibly existing) row id.

        ``dedup`` controls whether an unchanged content_hash suppresses the write.
        When ``None`` it is inferred from the trigger class (soft => dedup on).
        A heartbeat floor forces a write if it's been > ``heartbeat_secs`` since
        the last row, so the timeline never goes fully silent.
        """
        ts = obs.ts if obs.ts is not None else int(time.time())
        chash = compute_content_hash(obs.hash_text())
        shash = compute_simhash(obs.hash_text())

        if dedup is None:
            dedup = obs.trigger not in _HARD_TRIGGERS

        if dedup:
            last = self._last_row()
            if last is not None and from_sqlite_int(last["content_hash"]) == chash:
                stale = (ts - last["ts"]) > heartbeat_secs
                if not stale:
                    log.debug("dedup skip: content_hash unchanged (id=%s)", last["id"])
                    return int(last["id"])

        nid = self._next_id()
        self.conn.execute(
            """INSERT INTO activity
               (id, ts, app, window, summary, salient_text, entities_json, tags,
                content_hash, simhash, trigger, source, is_actionable, embedded)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                nid,
                ts,
                obs.app,
                obs.window,
                obs.summary,
                obs.salient_text,
                json.dumps(obs.entities),
                ",".join(obs.tags),
                to_sqlite_int(chash),
                to_sqlite_int(shash),
                obs.trigger,
                obs.source,
                1 if obs.is_actionable else 0,
                1 if embedding is not None else 0,
            ),
        )
        if embedding is not None:
            self.index.add(nid, embedding)
        self.conn.commit()
        return nid

    # -- read -----------------------------------------------------------------
    def query(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        since_ts: int | None = None,
    ) -> list[dict]:
        """Semantic search; optional recency filter. Returns hydrated rows + score."""
        allowlist = None
        if since_ts is not None:
            rows = self.conn.execute(
                "SELECT id FROM activity WHERE ts >= ? AND embedded=1", (since_ts,)
            ).fetchall()
            allowlist = [int(r["id"]) for r in rows]
            if not allowlist:
                return []

        hits = self.index.search(query_embedding, k=limit, allowlist=allowlist)
        results: list[dict] = []
        for rid, score in hits:
            row = self.conn.execute(
                "SELECT * FROM activity WHERE id=?", (rid,)
            ).fetchone()
            if row is None:
                continue
            results.append(self._row_to_dict(row, score))
        return results

    def recent(self, limit: int = 20, since_ts: int | None = None) -> list[dict]:
        """Most-recent rows regardless of similarity (for timeline / debugging)."""
        if since_ts is not None:
            rows = self.conn.execute(
                "SELECT * FROM activity WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (since_ts, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM activity ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r, None) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, score: float | None) -> dict:
        d = dict(row)
        d["content_hash"] = from_sqlite_int(row["content_hash"])
        d["simhash"] = from_sqlite_int(row["simhash"]) if row["simhash"] is not None else None
        d["entities"] = json.loads(row["entities_json"] or "[]")
        d["tags"] = [t for t in (row["tags"] or "").split(",") if t]
        if score is not None:
            d["score"] = score
        return d

    # -- maintenance ----------------------------------------------------------
    def optimize(
        self,
        retention_days: int = 30,
        evict_after_days: int = 3,
        simhash_threshold: int = 4,
        now: int | None = None,
    ) -> dict:
        """Dedup near-duplicate neighbours + apply retention. Returns a report."""
        now = now if now is not None else int(time.time())
        report = {"deduped": 0, "evicted": 0, "hard_deleted": 0}

        # 1. Collapse consecutive near-duplicates (exact hash or close simhash).
        rows = self.conn.execute(
            "SELECT id, content_hash, simhash FROM activity ORDER BY id ASC"
        ).fetchall()
        remove_ids: list[int] = []
        prev = None
        for r in rows:
            if prev is not None:
                same = r["content_hash"] == prev["content_hash"]
                near = (
                    r["simhash"] is not None
                    and prev["simhash"] is not None
                    and hamming(
                        from_sqlite_int(r["simhash"]), from_sqlite_int(prev["simhash"])
                    )
                    <= simhash_threshold
                )
                if same or near:
                    remove_ids.append(int(r["id"]))
                    continue  # keep prev as the anchor of this run
            prev = r
        if remove_ids:
            self._delete_rows(remove_ids)
            report["deduped"] = len(remove_ids)

        # 2. Retention: media eviction (keep row, mark evicted), then hard delete.
        evict_before = now - evict_after_days * 86400
        cur = self.conn.execute(
            "UPDATE activity SET evicted_at=? WHERE ts < ? AND evicted_at IS NULL",
            (now, evict_before),
        )
        report["evicted"] = cur.rowcount

        delete_before = now - retention_days * 86400
        old = self.conn.execute(
            "SELECT id FROM activity WHERE ts < ?", (delete_before,)
        ).fetchall()
        old_ids = [int(r["id"]) for r in old]
        if old_ids:
            self._delete_rows(old_ids)
            report["hard_deleted"] = len(old_ids)

        self.conn.commit()
        self.save()
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('last_optimize',?) "
            "ON CONFLICT(key) DO UPDATE SET value=?",
            (now, now),
        )
        self.conn.commit()
        log.info("optimize: %s", report)
        return report

    def _delete_rows(self, ids: list[int]) -> None:
        self.index.remove(ids)
        self.conn.executemany("DELETE FROM activity WHERE id=?", [(i,) for i in ids])

    # -- lifecycle ------------------------------------------------------------
    def save(self) -> None:
        self.index.save(self.index_path)

    def close(self) -> None:
        self.save()
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "ActivityStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
