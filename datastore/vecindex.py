"""Vector index abstraction.

Primary backend is turbovec (`IdMapIndex`): an in-process, TurboQuant-quantized
index keyed by uint64 ids. turbovec stores *only* vectors keyed by id — all
summaries/metadata live in the SQLite sidecar (see ``store.py``), joined on the
same id.

If turbovec is unavailable (e.g. dev machines without it installed) we fall back
to a small exact-cosine numpy index so the datastore stays usable and testable.
Both backends expose the same interface and normalize vectors to unit length so
"score" is a cosine-like similarity where higher == more similar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("contour.vecindex")

try:  # pragma: no cover - depends on host
    from turbovec import IdMapIndex as _TurboIdMap

    _HAVE_TURBOVEC = True
except Exception:  # noqa: BLE001 - any import failure => fallback
    _TurboIdMap = None
    _HAVE_TURBOVEC = False


def _to_unit_rows(mat: np.ndarray) -> np.ndarray:
    """Return float32 rows normalized to unit L2 length (zero rows left as-is)."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorIndex:
    """Uniform interface over turbovec or the numpy fallback."""

    def __init__(self, dim: int, bit_width: int = 4, backend: str | None = None):
        self.dim = dim
        self.bit_width = bit_width
        want_turbo = _HAVE_TURBOVEC if backend is None else (backend == "turbovec")
        if want_turbo and not _HAVE_TURBOVEC:
            raise RuntimeError("turbovec backend requested but turbovec is not installed")
        self.backend = "turbovec" if want_turbo else "numpy"
        if self.backend == "turbovec":
            self._idx = _TurboIdMap(dim=dim, bit_width=bit_width)
        else:
            # id -> unit vector
            self._vecs: dict[int, np.ndarray] = {}
        log.debug("VectorIndex backend=%s dim=%d", self.backend, dim)

    # -- mutation -------------------------------------------------------------
    def add(self, id: int, vector: np.ndarray) -> None:
        vec = _to_unit_rows(vector)  # (1, dim)
        if self.backend == "turbovec":
            self._idx.add_with_ids(vec, np.array([id], dtype=np.uint64))
        else:
            self._vecs[int(id)] = vec[0]

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        if self.backend == "turbovec":
            arr = np.array(ids, dtype=np.uint64)
            # turbovec IdMapIndex supports O(1) delete; method name has varied
            # across releases, so probe defensively.
            for name in ("remove_ids", "remove", "delete_ids", "delete"):
                fn = getattr(self._idx, name, None)
                if fn is not None:
                    fn(arr)
                    return
            log.warning("turbovec index exposes no remove method; skipping delete")
        else:
            for i in ids:
                self._vecs.pop(int(i), None)

    # -- query ----------------------------------------------------------------
    def search(
        self, vector: np.ndarray, k: int = 10, allowlist: list[int] | None = None
    ) -> list[tuple[int, float]]:
        q = _to_unit_rows(vector)  # (1, dim)
        if self.backend == "turbovec":
            kwargs = {}
            if allowlist is not None:
                kwargs["allowlist"] = np.array(allowlist, dtype=np.uint64)
            scores, ids = self._idx.search(q, k=k, **kwargs)
            scores = np.asarray(scores).reshape(-1)
            ids = np.asarray(ids).reshape(-1)
            return [(int(i), float(s)) for i, s in zip(ids, scores) if int(i) >= 0]
        # numpy fallback: exact cosine over the (optionally filtered) set
        items = self._vecs.items()
        if allowlist is not None:
            allow = set(int(i) for i in allowlist)
            items = [(i, v) for i, v in items if i in allow]
        else:
            items = list(items)
        if not items:
            return []
        ids = np.array([i for i, _ in items])
        mat = np.stack([v for _, v in items])
        sims = mat @ q[0]
        order = np.argsort(-sims)[:k]
        return [(int(ids[j]), float(sims[j])) for j in order]

    def __len__(self) -> int:
        if self.backend == "turbovec":
            for name in ("__len__", "size", "count"):
                fn = getattr(self._idx, name, None)
                if fn is not None:
                    try:
                        return int(fn())
                    except TypeError:
                        return int(fn)
            return -1  # unknown
        return len(self._vecs)

    @staticmethod
    def _npz_path(path: str | Path) -> Path:
        """numpy backend persists to a sibling .npz (np.savez appends .npz)."""
        s = str(path)
        return Path(s if s.endswith(".npz") else s + ".npz")

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.backend == "turbovec":
            self._idx.write(str(path))
        else:
            data = {"__meta__": np.array([self.dim], dtype=np.int64)}
            for i, v in self._vecs.items():
                data[str(i)] = v
            np.savez(str(self._npz_path(path)), **data)

    @classmethod
    def load(
        cls, path: str | Path, dim: int, bit_width: int = 4, backend: str | None = None
    ) -> "VectorIndex":
        path = Path(path)
        inst = cls(dim=dim, bit_width=bit_width, backend=backend)
        if inst.backend == "turbovec":
            if path.exists():
                inst._idx = _TurboIdMap.load(str(path))
        else:
            npz = cls._npz_path(path)
            if npz.exists():
                with np.load(npz) as data:
                    for key in data.files:
                        if key == "__meta__":
                            continue
                        inst._vecs[int(key)] = data[key].astype(np.float32)
        return inst
