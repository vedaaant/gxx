from datastore import texthash as th


def test_content_hash_stable_and_normalized():
    a = th.content_hash("Hello   World")
    b = th.content_hash("hello world")  # normalized: lowercase + collapse ws
    c = th.content_hash("goodbye world")
    assert a == b
    assert a != c
    # stable across calls (blake2b, not salted builtin hash)
    assert th.content_hash("some text") == th.content_hash("some text")


def test_simhash_near_for_similar_text():
    base = "the user opened vs code and edited the main python file"
    near = "the user opened vs code and edited the main python module"
    far = "playing a video on youtube in the browser fullscreen"
    d_near = th.hamming(th.simhash(base), th.simhash(near))
    d_far = th.hamming(th.simhash(base), th.simhash(far))
    # invariant: a one-word edit is markedly closer than unrelated text.
    # (Exact content_hash is the primary dedup key; simhash is a fuzzy assist,
    # noisier on short strings, so we assert the ordering, not a fixed bound.)
    assert d_near < d_far


def test_sqlite_int_roundtrip():
    for u in (0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
        assert th.from_sqlite_int(th.to_sqlite_int(u)) == u
        # stored value fits signed 64-bit
        assert -(1 << 63) <= th.to_sqlite_int(u) < (1 << 63)
