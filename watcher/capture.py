"""Screen + audio capture. Windows-friendly, dependency-guarded.

- ``ScreenCapturer`` reuses a single ``mss`` instance (creating one per grab is the
  common perf mistake) and returns either a NumPy array (for cheap diffing) or PNG
  bytes (for the vision model).
- ``MicBuffer`` keeps a short rolling audio buffer via ``sounddevice``. System-audio
  loopback (``pyaudiowpatch``) is a stretch goal and intentionally omitted here.
"""

from __future__ import annotations

import io
import logging
import threading

import numpy as np

log = logging.getLogger("contour.capture")


class ScreenCapturer:
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self._sct = None

    def _ensure(self):
        if self._sct is None:
            import mss  # lazy

            self._sct = mss.mss()
        return self._sct

    def grab_array(self) -> np.ndarray:
        """Return the monitor as an HxWx4 BGRA NumPy array."""
        sct = self._ensure()
        mons = sct.monitors
        idx = self.monitor if self.monitor < len(mons) else 1
        shot = sct.grab(mons[idx])
        return np.asarray(shot)  # BGRA, no copy

    def grab_png(self, downscale_width: int | None = 1920) -> bytes:
        """Grab and encode to PNG bytes (optionally downscaled to bound tokens/latency)."""
        arr = self.grab_array()
        from PIL import Image

        # BGRA -> RGB
        img = Image.fromarray(arr[:, :, :3][:, :, ::-1])
        if downscale_width and img.width > downscale_width:
            h = round(img.height * downscale_width / img.width)
            img = img.resize((downscale_width, h))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def close(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001
                pass
            self._sct = None


class MicBuffer:
    """Rolling microphone buffer (mono). No-op if sounddevice is unavailable."""

    def __init__(self, seconds: float = 8.0, samplerate: int = 16000):
        self.seconds = seconds
        self.samplerate = samplerate
        self._buf = np.zeros(int(seconds * samplerate), dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> bool:
        try:
            import sounddevice as sd
        except Exception as e:  # noqa: BLE001
            log.info("mic capture disabled (sounddevice unavailable: %s)", e)
            return False

        def _cb(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("audio status: %s", status)
            chunk = indata[:, 0] if indata.ndim > 1 else indata
            with self._lock:
                n = len(chunk)
                if n >= len(self._buf):
                    self._buf[:] = chunk[-len(self._buf):]
                else:
                    self._buf[:-n] = self._buf[n:]
                    self._buf[-n:] = chunk

        try:
            self._stream = sd.InputStream(
                channels=1, samplerate=self.samplerate, callback=_cb
            )
            self._stream.start()
            return True
        except Exception as e:  # noqa: BLE001
            log.info("mic capture disabled (stream start failed: %s)", e)
            self._stream = None
            return False

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._buf.copy()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
