"""Proactive interjection gate.

Per the PRD risk table this is deliberately a small set of hardcoded, conservative
rules (no ML for v1): it must fire on a clear scripted scenario and stay SILENT on
a "boring" control. When it fires it hands a candidate observation to Hermes via a
one-shot prompt; Hermes phrases the final line and speaks it through its native
(free, no-key) Edge TTS — so all voice output flows through Hermes.

The decision logic is pure and injectable (``speaker``, ``clock``) for tests.
"""

from __future__ import annotations

import logging
import re
import time

from . import config

log = logging.getLogger("contour.gate")

# Stricter than the daemon's actionable scan: things genuinely worth interrupting for.
_PROACTIVE = re.compile(
    r"\b(error|exception|traceback|failed|failure|denied|deadline|due|overdue|"
    r"urgent|blocked|expired|payment failed|build failed|test failed)\b",
    re.IGNORECASE,
)
# Interjections are gated to context-defining moments, not every keystroke.
_ALLOWED_TRIGGERS = {"AppSwitch", "WindowFocus", "VisualChange", "Idle", "Manual"}


def _default_speaker(line: str) -> None:
    """Speak a proactive line in the user's ElevenLabs voice. Best-effort."""
    from .voice import speak

    try:
        speak(line)
    except Exception as e:  # noqa: BLE001 - proactivity must never break capture
        log.warning("failed to speak interjection: %s", e)


class ProactiveGate:
    def __init__(
        self,
        cooldown: float = config.INTERJECTION_COOLDOWN_SECS,
        speaker=_default_speaker,
        clock=time.monotonic,
        quiet: bool = False,
    ):
        self.cooldown = cooldown
        self.speaker = speaker
        self._clock = clock
        self.quiet = quiet
        self._last_fire = -1e18

    def should_fire(self, obs, trigger, now: float) -> bool:
        if self.quiet:
            return False
        if not obs.is_actionable:
            return False
        if trigger.kind not in _ALLOWED_TRIGGERS:
            return False
        if (now - self._last_fire) < self.cooldown:
            return False
        # extra-conservative: require an explicit actionable keyword in the content
        text = f"{obs.summary}\n{obs.salient_text}"
        return bool(_PROACTIVE.search(text))

    def evaluate(self, obs, trigger, now: float | None = None) -> str | None:
        """If the rules pass, ask Hermes to speak and return the candidate line."""
        now = now if now is not None else self._clock()
        if not self.should_fire(obs, trigger, now):
            return None
        self._last_fire = now
        line = self._build_line(obs)
        log.info("proactive interjection: %s", obs.summary[:80])
        self.speaker(line)
        return line

    @staticmethod
    def _build_line(obs) -> str:
        """A short, natural sentence to speak aloud (spoken directly, no LLM step)."""
        where = obs.app or "what's on screen"
        summary = (obs.summary or "").rstrip(".")
        return f"Heads up on {where}: {summary}. Want a hand with that?"
