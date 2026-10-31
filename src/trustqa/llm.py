"""Provider-swappable LLM wrapper.

Every module calls `complete(prompt, system=None, json=False) -> str`. With json=True the
returned string is guaranteed to parse as JSON (code fences stripped); otherwise LLMError.

Providers (LLM_PROVIDER):
- ollama    - local model via the Ollama server (free, offline). Default.
- groq      - Groq cloud API (free tier key from console.groq.com).
- anthropic - optional; needs ANTHROPIC_API_KEY and `pip install anthropic`.
Tests use FakeLLM so no network, server or key is needed.
"""

from __future__ import annotations

import json as _json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from typing import Protocol

from trustqa.config import Config, ConfigError, load_config

JSON_INSTRUCTION = "Respond with a single valid JSON object only. No prose, no code fences."


class LLMError(RuntimeError):
    """The model call failed or returned unusable output."""


class LLM(Protocol):
    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str: ...


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def clean_json(text: str) -> str:
    """Strip code fences and validate that `text` is JSON. Returns the clean string."""
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        _json.loads(s)
    except _json.JSONDecodeError as e:
        raise LLMError(f"model output is not valid JSON: {e}; output starts {s[:200]!r}") from e
    return s


def _messages(prompt: str, system: str | None, json: bool) -> list[dict]:
    sys_prompt = "\n\n".join(p for p in (system, JSON_INSTRUCTION if json else None) if p)
    msgs = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
    msgs.append({"role": "user", "content": prompt})
    return msgs


class OllamaLLM:
    """Local model served by Ollama (https://ollama.com). Temperature 0 + fixed seed."""

    def __init__(self, config: Config):
        self.host = config.ollama_host.rstrip("/")
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.timeout = config.llm_timeout_s

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": _messages(prompt, system, json),
            "stream": False,
            "options": {"temperature": self.temperature, "seed": 0,
                        "num_predict": self.max_tokens},
        }
        if json:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = _json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 404:
                raise LLMError(
                    f"Ollama model {self.model!r} not found; run: ollama pull {self.model}"
                ) from e
            raise LLMError(f"Ollama error {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(
                f"cannot reach Ollama at {self.host} ({e}). Is Ollama installed and running?"
            ) from e
        if data.get("done_reason") == "length":
            raise LLMError(f"output truncated at llm_max_tokens={self.max_tokens}")
        text = data.get("message", {}).get("content", "")
        return clean_json(text) if json else text


class GroqLLM:
    """Groq cloud API (free tier). Needs GROQ_API_KEY and `pip install groq`."""

    def __init__(self, config: Config):
        if not config.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com and add it to .env."
            )
        try:
            import groq
        except ImportError as e:
            raise ConfigError("groq package missing: pip install groq") from e
        self._groq = groq
        self._client = groq.Groq(api_key=config.groq_api_key, timeout=config.llm_timeout_s)
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        groq = self._groq
        kwargs = dict(
            model=self.model,
            messages=_messages(prompt, system, json),
            temperature=self.temperature,
            seed=0,
            max_tokens=self.max_tokens,
        )
        if json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except groq.AuthenticationError as e:
            raise LLMError("Groq API key was rejected") from e
        except groq.RateLimitError as e:
            raise LLMError("Groq free-tier rate limit hit; wait and retry") from e
        except groq.APIStatusError as e:
            raise LLMError(f"Groq API error {e.status_code}: {e.message}") from e
        except groq.APIConnectionError as e:
            raise LLMError("network error contacting Groq") from e
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"output truncated at llm_max_tokens={self.max_tokens}")
        text = choice.message.content or ""
        return clean_json(text) if json else text


class AnthropicLLM:
    """Optional: Claude via the Anthropic SDK (paid). Sends no temperature, since
    current Claude models reject sampling parameters; uses llm_effort instead."""

    def __init__(self, config: Config):
        if not config.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
        except ImportError as e:
            raise ConfigError("anthropic package missing: pip install anthropic") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.llm_model
        self.effort = config.llm_effort
        self.max_tokens = config.llm_max_tokens

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        anthropic = self._anthropic
        msgs = _messages(prompt, system, json)
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            messages=[m for m in msgs if m["role"] != "system"],
        )
        if msgs[0]["role"] == "system":
            kwargs["system"] = msgs[0]["content"]
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("network error contacting Anthropic API") from e
        if response.stop_reason == "refusal":
            raise LLMError("model declined the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"output truncated at llm_max_tokens={self.max_tokens}")
        text = "".join(b.text for b in response.content if b.type == "text")
        return clean_json(text) if json else text


class FakeLLM:
    """Scripted LLM for tests.

    responses: a list of strings returned in order, or a callable
    (prompt, system, json) -> str. Every call is recorded in `.calls`.
    """

    def __init__(self, responses: Iterable[str] | Callable[[str, str | None, bool], str]):
        if callable(responses):
            self._fn = responses
            self._queue: list[str] | None = None
        else:
            self._fn = None
            self._queue = list(responses)
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        self.calls.append({"prompt": prompt, "system": system, "json": json})
        if self._fn is not None:
            out = self._fn(prompt, system, json)
        else:
            if not self._queue:
                raise LLMError(f"FakeLLM exhausted after {len(self.calls) - 1} call(s)")
            out = self._queue.pop(0)
        return clean_json(out) if json else out


_PROVIDERS = {"ollama": OllamaLLM, "groq": GroqLLM, "anthropic": AnthropicLLM}


def get_llm(config: Config | None = None) -> LLM:
    """Build the configured provider."""
    config = config or load_config()
    try:
        return _PROVIDERS[config.llm_provider](config)
    except KeyError:
        raise ConfigError(
            f"unsupported LLM_PROVIDER {config.llm_provider!r}; use one of {sorted(_PROVIDERS)}"
        ) from None


_default: LLM | None = None


def set_llm(llm: LLM | None) -> None:
    """Override the module-level LLM (e.g. FakeLLM in tests). None resets it."""
    global _default
    _default = llm


def complete(prompt: str, system: str | None = None, json: bool = False) -> str:
    """Call the default LLM, creating it from config on first use."""
    global _default
    if _default is None:
        _default = get_llm()
    return _default.complete(prompt, system=system, json=json)
