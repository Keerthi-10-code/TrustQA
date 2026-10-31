import logging

import pytest

from trustqa.config import ConfigError, load_config, log_run_config

ENV_VARS = ["LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_EFFORT", "LLM_MAX_TOKENS",
            "LLM_TIMEOUT_S", "OLLAMA_HOST", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
            "EMBEDDING_MODEL", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP", "TESSERACT_CMD"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("")
    return empty


def test_loads_from_env_file(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("TOP_K=7\nCHUNK_SIZE=500\nCHUNK_OVERLAP=50\nLLM_PROVIDER=groq\n"
                   "LLM_MODEL=llama-3.3-70b-versatile\nGROQ_API_KEY=gsk-test\n")
    cfg = load_config(env)
    assert (cfg.top_k, cfg.chunk_size, cfg.chunk_overlap) == (7, 500, 50)
    assert (cfg.llm_provider, cfg.llm_model) == ("groq", "llama-3.3-70b-versatile")
    assert cfg.groq_api_key == "gsk-test"


def test_defaults_and_overrides(clean_env):
    cfg = load_config(clean_env, top_k=3)
    assert cfg.top_k == 3
    assert cfg.chunk_overlap < cfg.chunk_size
    assert cfg.anthropic_api_key is None


def test_blank_values_fall_back_to_defaults(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("TESSERACT_CMD=\nTOP_K=\n")
    cfg = load_config(env)
    assert cfg.tesseract_cmd is None and cfg.top_k == 5


@pytest.mark.parametrize("kw", [dict(top_k=0), dict(chunk_overlap=800, chunk_size=800),
                                dict(llm_effort="extreme")])
def test_invalid_config_rejected(clean_env, kw):
    with pytest.raises(ConfigError):
        load_config(clean_env, **kw)


def test_non_integer_env_rejected(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("TOP_K=five\n")
    with pytest.raises(ConfigError):
        load_config(env)


def test_log_run_config_masks_secret(clean_env, caplog):
    cfg = load_config(clean_env, groq_api_key="sk-secret-123")
    with caplog.at_level(logging.INFO, logger="trustqa"):
        record = log_run_config(cfg, extra={"run_id": "r1"})
    assert record["groq_api_key"] == "set"
    assert record["anthropic_api_key"] == "missing"
    assert record["run_id"] == "r1"
    assert "sk-secret-123" not in caplog.text
    assert "sk-secret-123" not in repr(cfg)
