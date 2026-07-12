from datastore import Observation
from watcher.gate import ProactiveGate
from watcher.triggers import Trigger


class Speaker:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)


def _clock(t):
    return lambda: t


def test_fires_on_actionable_error_appswitch():
    spk = Speaker()
    gate = ProactiveGate(cooldown=60, speaker=spk)
    obs = Observation(
        summary="build failed with a NullPointerException",
        salient_text="Exception in thread main",
        app="IntelliJ",
        is_actionable=True,
    )
    out = gate.evaluate(obs, Trigger("AppSwitch", ts=0.0), now=100.0)
    assert out is not None
    assert len(spk.calls) == 1
    assert "IntelliJ" in spk.calls[0]


def test_silent_on_boring_control():
    spk = Speaker()
    gate = ProactiveGate(cooldown=60, speaker=spk)
    obs = Observation(
        summary="reading a wikipedia article about otters",
        app="chrome",
        is_actionable=False,   # nothing actionable
    )
    assert gate.evaluate(obs, Trigger("AppSwitch", ts=0.0), now=100.0) is None
    assert spk.calls == []


def test_actionable_but_no_keyword_stays_silent():
    # is_actionable true but content lacks an explicit proactive keyword => conservative
    spk = Speaker()
    gate = ProactiveGate(cooldown=60, speaker=spk)
    obs = Observation(summary="a form is open", app="chrome", is_actionable=True)
    assert gate.evaluate(obs, Trigger("AppSwitch", ts=0.0), now=100.0) is None


def test_cooldown_suppresses_second_fire():
    spk = Speaker()
    gate = ProactiveGate(cooldown=60, speaker=spk)
    obs = Observation(summary="test failed", app="term", is_actionable=True)
    assert gate.evaluate(obs, Trigger("AppSwitch", ts=0.0), now=100.0) is not None
    assert gate.evaluate(obs, Trigger("AppSwitch", ts=0.0), now=130.0) is None  # within cooldown
    assert gate.evaluate(obs, Trigger("AppSwitch", ts=0.0), now=200.0) is not None  # past it
    assert len(spk.calls) == 2


def test_quiet_mode_never_fires():
    spk = Speaker()
    gate = ProactiveGate(cooldown=0, speaker=spk, quiet=True)
    obs = Observation(summary="deadline overdue", app="mail", is_actionable=True)
    assert gate.evaluate(obs, Trigger("Manual", ts=0.0), now=100.0) is None


def test_soft_typing_trigger_not_allowed():
    spk = Speaker()
    gate = ProactiveGate(cooldown=60, speaker=spk)
    obs = Observation(summary="error occurred", app="term", is_actionable=True)
    # TypingPause is not a context-defining trigger => no interjection mid-typing
    assert gate.evaluate(obs, Trigger("TypingPause", ts=0.0), now=100.0) is None
