"""Single source of run configuration (NFR-04). Loaded from environment / .env."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("trustqa")


class ConfigError(ValueError):
    """Invalid or missing configuration (reported explicitly, NFR-09)."""


@dataclass(frozen=True)
class Config:
    # LLM
    llm_provider: str = "ollama"  # ollama | groq | anthropic
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0  # ollama / groq
    llm_effort: str = "low"  # anthropic only (Claude rejects temperature)
    llm_max_tokens: int = 2048
    llm_timeout_s: int = 300
    ollama_host: str = "http://localhost:11434"
    groq_api_key: str | None = field(default=None, repr=False)
    anthropic_api_key: str | None = field(default=None, repr=False)

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5

    # Chunking (characters)
    chunk_size: int = 800
    chunk_overlap: int = 150

    # OCR
    tesseract_cmd: str | None = None

    # Paths
    raw_pdfs_dir: Path = PROJECT_ROOT / "data" / "raw_pdfs"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    indexes_dir: Path = PROJECT_ROOT / "data" / "indexes"
    results_dir: Path = PROJECT_ROOT / "eval" / "results"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigError(f"top_k must be >= 1, got {self.top_k}")
        if self.chunk_size < 1:
            raise ConfigError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError(
                f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap}"
            )
        if self.llm_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ConfigError(f"unknown llm_effort {self.llm_effort!r}")

    def to_loggable_dict(self) -> dict[str, Any]:
        """All settings with secrets masked and paths as strings."""
        d = asdict(self)
        for k in ("groq_api_key", "anthropic_api_key"):
            d[k] = "set" if getattr(self, k) else "missing"
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from e


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from e


def _env_str(name: str, default: str | None) -> str | None:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def load_config(env_file: str | Path | None = None, **overrides: Any) -> Config:
    """Build a Config from .env + environment variables, then apply overrides.

    Real environment variables win over the .env file. `overrides` (field=value)
    win over both; use them for tests and ablation runs.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    d = Config()
    values: dict[str, Any] = dict(
        llm_provider=_env_str("LLM_PROVIDER", d.llm_provider),
        llm_model=_env_str("LLM_MODEL", d.llm_model),
        llm_temperature=_env_float("LLM_TEMPERATURE", d.llm_temperature),
        llm_effort=_env_str("LLM_EFFORT", d.llm_effort),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", d.llm_max_tokens),
        llm_timeout_s=_env_int("LLM_TIMEOUT_S", d.llm_timeout_s),
        ollama_host=_env_str("OLLAMA_HOST", d.ollama_host),
        groq_api_key=_env_str("GROQ_API_KEY", None),
        anthropic_api_key=_env_str("ANTHROPIC_API_KEY", None),
        embedding_model=_env_str("EMBEDDING_MODEL", d.embedding_model),
        top_k=_env_int("TOP_K", d.top_k),
        chunk_size=_env_int("CHUNK_SIZE", d.chunk_size),
        chunk_overlap=_env_int("CHUNK_OVERLAP", d.chunk_overlap),
        tesseract_cmd=_env_str("TESSERACT_CMD", None),
    )
    values.update(overrides)
    return Config(**values)


def log_run_config(config: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Log the full run configuration (secrets masked) and return it for saving."""
    record = config.to_loggable_dict()
    if extra:
        record.update(extra)
    logger.info("run config: %s", json.dumps(record, sort_keys=True))
    return record
