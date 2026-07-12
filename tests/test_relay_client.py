import sys
import types

import pytest

from mcp_server.relay_client import RelayClient, RelayError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", content=b"", headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json


def install_fake_httpx(monkeypatch, capture: dict, response: FakeResponse):
    mod = types.ModuleType("httpx")

    class HTTPError(Exception):
        pass

    def post(url, json=None, headers=None, timeout=None):
        capture["url"] = url
        capture["json"] = json
        capture["headers"] = headers
        return response

    mod.HTTPError = HTTPError
    mod.post = post
    monkeypatch.setitem(sys.modules, "httpx", mod)


def test_search_scrubs_pii_and_sends_token(monkeypatch):
    cap = {}
    install_fake_httpx(monkeypatch, cap, FakeResponse(json_data={"results": []}))
    client = RelayClient(base_url="https://relay.example", token="dev-123")
    client.search("email me at alice@example.com about the plan", num_results=3)

    assert cap["url"] == "https://relay.example/search"
    assert "alice@example.com" not in cap["json"]["query"]  # scrubbed
    assert "[EMAIL]" in cap["json"]["query"]
    assert cap["json"]["num_results"] == 3
    assert cap["headers"]["Authorization"] == "Bearer dev-123"


def test_cloud_scrubs_prompt(monkeypatch):
    cap = {}
    install_fake_httpx(monkeypatch, cap, FakeResponse(json_data={"answer": "ok"}))
    client = RelayClient(base_url="https://relay.example", token="t")
    out = client.cloud("my card is 4111 1111 1111 1111")
    assert out["answer"] == "ok"
    assert "4111" not in cap["json"]["prompt"]
    assert "[CARD]" in cap["json"]["prompt"]


def test_rate_limit_and_auth_errors(monkeypatch):
    cap = {}
    install_fake_httpx(monkeypatch, cap, FakeResponse(status_code=429))
    with pytest.raises(RelayError, match="rate limit"):
        RelayClient(base_url="https://r", token="t").search("q")

    install_fake_httpx(monkeypatch, cap, FakeResponse(status_code=401))
    with pytest.raises(RelayError, match="token"):
        RelayClient(base_url="https://r", token="bad").search("q")


def test_tts_scrubs_and_returns_pcm(monkeypatch):
    cap = {}
    install_fake_httpx(
        monkeypatch, cap,
        FakeResponse(content=b"PCMDATA", headers={"X-Sample-Rate": "16000"}),
    )
    audio, rate = RelayClient(base_url="https://r", token="t").tts("email me at a@b.com", voice_id="v1")
    assert audio == b"PCMDATA" and rate == 16000
    assert "a@b.com" not in cap["json"]["text"] and "[EMAIL]" in cap["json"]["text"]
    assert cap["json"]["voice_id"] == "v1"


def test_unconfigured_relay_raises():
    with pytest.raises(RelayError, match="not configured"):
        RelayClient(base_url="", token="").search("q")
