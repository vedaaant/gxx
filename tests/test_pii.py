from datastore.pii import scrub as client_scrub
from datastore.pii import scrub_dict
from relay.pii import scrub as relay_scrub


def test_scrubs_common_pii():
    text = "email alice@example.com, call +1 415 555 0100, card 4111 1111 1111 1111"
    out = client_scrub(text)
    assert "alice@example.com" not in out
    assert "[EMAIL]" in out
    assert "4111 1111 1111 1111" not in out
    assert "[CARD]" in out


def test_scrubs_secrets_and_ip():
    text = "key sk-abcdefghijklmnop1234567 at 10.0.0.1 with password: hunter2"
    out = client_scrub(text)
    assert "sk-abcdefghijklmnop1234567" not in out
    assert "10.0.0.1" not in out
    assert "hunter2" not in out


def test_client_and_relay_rulesets_agree():
    # the two copies must stay in sync (server is a backstop for the client)
    samples = [
        "reach me at bob@work.io",
        "token=ghp_0123456789abcdef0123 and ip 192.168.1.55",
        "no pii here, just plain text about otters",
        "card 4111111111111111",
    ]
    for s in samples:
        assert client_scrub(s) == relay_scrub(s)


def test_scrub_dict_only_touches_strings():
    obj = {"summary": "mail me at x@y.com", "count": 5, "app": "notepad"}
    out = scrub_dict(obj)
    assert "[EMAIL]" in out["summary"]
    assert out["count"] == 5
    assert out["app"] == "notepad"


def test_plain_text_unchanged():
    s = "the user opened a document about distributed systems"
    assert client_scrub(s) == s
