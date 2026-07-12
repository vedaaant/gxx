from watcher.voice import speak


class FakeRelay:
    def __init__(self, audio=b"\x00\x01", rate=16000, err=None):
        self.audio, self.rate, self.err = audio, rate, err
        self.text = None
        self.voice_id = None

    def tts(self, text, voice_id=""):
        self.text = text
        self.voice_id = voice_id
        if self.err:
            raise self.err
        return self.audio, self.rate


def test_speak_synthesizes_and_plays():
    played = {}

    def player(data, rate):
        played["data"], played["rate"] = data, rate
        return True

    relay = FakeRelay(audio=b"ABCD", rate=16000)
    assert speak("hello there", relay=relay, player=player) is True
    assert played["data"] == b"ABCD" and played["rate"] == 16000
    assert relay.text == "hello there"


def test_speak_empty_is_noop():
    relay = FakeRelay()
    assert speak("   ", relay=relay, player=lambda d, r: True) is False
    assert relay.text is None  # relay never called


def test_speak_swallows_relay_errors():
    relay = FakeRelay(err=RuntimeError("relay down"))
    assert speak("hi", relay=relay, player=lambda d, r: True) is False


def test_speak_passes_voice_id():
    relay = FakeRelay()
    speak("hi", relay=relay, player=lambda d, r: True, voice_id="voiceX")
    assert relay.voice_id == "voiceX"
