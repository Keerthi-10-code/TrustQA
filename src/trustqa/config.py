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

    # OCR (Phase 2). A page goes to OCR when its text layer has fewer than ocr_min_chars
    # alphanumeric characters, or when more than ocr_max_nonalnum_ratio of its
    # non-whitespace characters are non-alphanumeric / undecodable (garbage text layer).
    tesseract_cmd: str | None = None
    ocr_enabled: bool = True
    ocr_min_chars: int = 50
    ocr_max_nonalnum_ratio: float = 0.6
    ocr_dpi: int = 300
    ocr_lang: str = "eng"
    ocr_psm: int = 3  # Tesseract page segmentation mode (3 = automatic)
    ocr_preprocess: bool = False  # grayscale + deskew + Otsu threshold before Tesseract

    # Tables (Phase 2) and verification switches. flatten_tables=True skips table
    # extraction: table content stays in the page text exactly as a plain-text RAG sees it.
    flatten_tables: bool = False
    verify: bool = True
    # Preset for baseline system B (OCR + plain RAG): forces flatten_tables=True and
    # verify=False regardless of their own settings.
    baseline_b: bool = False

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
        if self.ocr_min_chars < 0:
            raise ConfigError(f"ocr_min_chars must be >= 0, got {self.ocr_min_chars}")
        if not 0 < self.ocr_max_nonalnum_ratio <= 1:
            raise ConfigError("ocr_max_nonalnum_ratio must be in (0, 1], got "
                              f"{self.ocr_max_nonalnum_ratio}")
        if not 72 <= self.ocr_dpi <= 1200:
            raise ConfigError(f"ocr_dpi must be in [72, 1200], got {self.ocr_dpi}")
        if self.baseline_b:  # frozen dataclass: resolve the preset in place
            object.__setattr__(self, "flatten_tables", True)
            object.__setattr__(self, "verify", False)

    @property
    def system_name(self) -> str:
        """Which evaluation system this configuration reproduces (PRD / phases.md)."""
        if self.flatten_tables and not self.verify:
            return "B-ocr-plain-rag" if self.ocr_enabled else "A-baseline"
        return "C-trustqa"

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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (1/0/true/false), got {raw!r}")


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
        ocr_enabled=_env_bool("OCR_ENABLED", d.ocr_enabled),
        ocr_min_chars=_env_int("OCR_MIN_CHARS", d.ocr_min_chars),
        ocr_max_nonalnum_ratio=_env_float("OCR_MAX_NONALNUM_RATIO", d.ocr_max_nonalnum_ratio),
        ocr_dpi=_env_int("OCR_DPI", d.ocr_dpi),
        ocr_lang=_env_str("OCR_LANG", d.ocr_lang),
        ocr_psm=_env_int("OCR_PSM", d.ocr_psm),
        ocr_preprocess=_env_bool("OCR_PREPROCESS", d.ocr_preprocess),
        flatten_tables=_env_bool("FLATTEN_TABLES", d.flatten_tables),
        verify=_env_bool("VERIFY", d.verify),
        baseline_b=_env_bool("BASELINE_B", d.baseline_b),
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
