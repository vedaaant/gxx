import numpy as np

from watcher.understand import Understanding


class FakeClient:
    def __init__(self, chat_text="", embed_resp=None):
        self._chat_text = chat_text
        self._embed_resp = embed_resp
        self.last_chat = None
        self.last_embed = None

    def chat(self, model, messages, options=None):
        self.last_chat = {"model": model, "messages": messages, "options": options}
        return {"message": {"content": self._chat_text}}

    def embed(self, model, input, options=None):
        self.last_embed = {"model": model, "input": input, "options": options}
        return self._embed_resp


def test_describe_parses_clean_json():
    text = (
        '{"app_or_context":"VS Code","activity":"editing main.py",'
        '"salient_text":"def foo()","entities":["main.py"],"is_actionable":false}'
    )
    u = Understanding(client=FakeClient(chat_text=text))
    out = u.describe(image=b"PNGBYTES", uia_text="some code")
    assert out["app_or_context"] == "VS Code"
    assert out["activity"] == "editing main.py"
    assert out["entities"] == ["main.py"]
    assert out["is_actionable"] is False
    # uia text is fed into the prompt
    assert "some code" in u._client.last_chat["messages"][0]["content"]


def test_describe_extracts_json_embedded_in_prose():
    text = 'Sure! Here is the result:\n{"activity":"reading docs","is_actionable":true}\nHope that helps.'
    u = Understanding(client=FakeClient(chat_text=text))
    out = u.describe(image="frame.png")
    assert out["activity"] == "reading docs"
    assert out["is_actionable"] is True


def test_describe_falls_back_on_non_json():
    u = Understanding(client=FakeClient(chat_text="the user is scrolling twitter"))
    out = u.describe(image="frame.png")
    assert "scrolling twitter" in out["activity"]
    assert out["is_actionable"] is False


def test_embed_handles_batched_and_flat_shapes():
    # newer /api/embed shape: {"embeddings": [[...]]}
    u1 = Understanding(client=FakeClient(embed_resp={"embeddings": [[0.1, 0.2, 0.3]]}))
    v1 = u1.embed("hello", is_query=True)
    assert isinstance(v1, np.ndarray) and v1.dtype == np.float32
    assert np.allclose(v1, [0.1, 0.2, 0.3])
    assert u1._client.last_embed["input"].startswith("search_query: ")

    # older shape: {"embedding": [...]}
    u2 = Understanding(client=FakeClient(embed_resp={"embedding": [1.0, 2.0]}))
    v2 = u2.embed("doc text")
    assert np.allclose(v2, [1.0, 2.0])
    assert u2._client.last_embed["input"].startswith("search_document: ")
