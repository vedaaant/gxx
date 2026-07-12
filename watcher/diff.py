"""Cheap "did the screen meaningfully change?" detector.

Recipe adapted from screenpipe's ``frame_comparison.rs``:
1. Downscale once (~4x, nearest-neighbour via strided slicing) to grayscale.
2. Hash the downscaled bytes; identical hash => return 0.0 (fast early-exit).
3. Otherwise compare grayscale intensity histograms with the Hellinger distance
   (0.0 == identical, 1.0 == maximally different).

This drives ONLY the input-less ``VisualChange`` trigger — it is not the store
dedup gate (that is text/content-hash based). Their tuning note: downscaling too
far (6x) caused hash collisions that hid scrolling dense text; 4x is the balance.
"""

from __future__ import annotations

import hashlib

import numpy as np


def to_gray_small(frame: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downscale by strided slicing and reduce to a 2-D grayscale float array."""
    arr = np.asarray(frame)
    small = arr[::factor, ::factor]
    if small.ndim == 3:
        # drop alpha if present, average the colour channels
        small = small[:, :, :3].mean(axis=2)
    return small.astype(np.float32)


def _histogram(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 255.0))
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float64)
    return hist.astype(np.float64) / total


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two normalized histograms, in [0, 1]."""
    bc = np.sum(np.sqrt(p * q))  # Bhattacharyya coefficient
    return float(np.sqrt(max(0.0, 1.0 - bc)))


class FrameComparer:
    def __init__(self, downscale_factor: int = 4, bins: int = 64):
        self.factor = downscale_factor
        self.bins = bins
        self._prev_hash: str | None = None
        self._prev_hist: np.ndarray | None = None

    def compare(self, frame: np.ndarray) -> float:
        """Return change score in [0, 1] vs. the previous frame; update state.

        First-ever frame returns 1.0 (treat as a full change so it gets captured).
        """
        gray = to_gray_small(frame, self.factor)
        digest = hashlib.blake2b(gray.tobytes(), digest_size=16).hexdigest()

        if self._prev_hash is None:
            self._prev_hash = digest
            self._prev_hist = _histogram(gray, self.bins)
            return 1.0

        if digest == self._prev_hash:
            return 0.0  # bit-identical downscaled frame

        hist = _histogram(gray, self.bins)
        score = hellinger(self._prev_hist, hist)
        self._prev_hash = digest
        self._prev_hist = hist
        return score
