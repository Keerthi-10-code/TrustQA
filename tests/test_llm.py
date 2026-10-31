import json

import pytest

from trustqa import llm as llm_mod
from trustqa.config import ConfigError, load_config
from trustqa.llm import FakeLLM, GroqLLM, LLMError, OllamaLLM, clean_json, get_llm


def test_fake_returns_scripted_responses_in_order():
    fake = FakeLLM(["one", "two"])
    assert fake.complete("a") == "one"
    assert fake.complete("b", system="sys") == "two"
    assert fake.calls[1] == {"prompt": "b", "system": "sys", "json": False}


def test_fake_raises_when_exhausted():
    fake = FakeLLM(["only"])
    fake.complete("a")
    with pytest.raises(LLMError):
        fake.complete("b")


def test_fake_callable_responses():
    fake = FakeLLM(lambda p, s, j: json.dumps({"echo": p}))
    assert json.loads(fake.complete("hi", json=True)) == {"echo": "hi"}


def test_json_mode_strips_fences_and_validates():
    fake = FakeLLM(['```json\n{"answer": "x"}\n```', "not json"])
    assert json.loads(fake.complete("q", json=True)) == {"answer": "x"}
    with pytest.raises(LLMError):
        fake.complete("q", json=True)


def test_clean_json_plain():
    assert clean_json(' {"a": 1} ') == '{"a": 1}'


def test_module_complete_uses_override():
    fake = FakeLLM(["hello"])
    llm_mod.set_llm(fake)
    try:
        assert llm_mod.complete("p") == "hello"
    finally:
        llm_mod.set_llm(None)


def _cfg(tmp_path, monkeypatch, **kw):
    for v in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "OLLAMA_HOST"):
        monkeypatch.delenv(v, raising=False)
    env = tmp_path / ".env"
    env.write_text("")
    return load_config(env, **kw)


def test_default_provider_is_ollama(tmp_path, monkeypatch):
    assert isinstance(get_llm(_cfg(tmp_path, monkeypatch)), OllamaLLM)


def test_unknown_provider_rejected(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="unsupported"):
        get_llm(_cfg(tmp_path, monkeypatch, llm_provider="nope"))


def test_missing_groq_key_is_explicit(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        GroqLLM(_cfg(tmp_path, monkeypatch, llm_provider="groq"))


def test_ollama_unreachable_is_explicit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, ollama_host="http://127.0.0.1:9", llm_timeout_s=2)
    with pytest.raises(LLMError, match="cannot reach Ollama"):
        OllamaLLM(cfg).complete("hi")


def test_ollama_request_shape(tmp_path, monkeypatch):
    """JSON mode sets format=json, temperature 0 and a fixed seed."""
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"message": {"content": '{"ok": true}'}, "done_reason": "stop"}).encode()

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return Resp()

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    out = OllamaLLM(_cfg(tmp_path, monkeypatch)).complete("q", system="sys", json=True)
    assert json.loads(out) == {"ok": True}
    assert sent["format"] == "json"
    assert sent["options"]["temperature"] == 0.0 and sent["options"]["seed"] == 0
    assert sent["messages"][0]["role"] == "system" and "JSON" in sent["messages"][0]["content"]
