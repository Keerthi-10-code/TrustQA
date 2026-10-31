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
    # Retrieval upgrades (Phase 4), all off by default so baselines are unchanged. They only
    # reorder chunks: RetrievedChunk.score stays the dense cosine (the abstention gate's
    # threshold is calibrated on it). Measured in eval/retrieval_eval.py.
    retrieval_hybrid: bool = False  # BM25 + dense, merged by reciprocal rank fusion (FR-15)
    rrf_k: int = 60  # RRF constant: score = sum 1 / (rrf_k + rank)
    retrieval_candidates: int = 50  # per-retriever candidate pool for fusion/diversify/rerank
    retrieval_diversify: str = "none"  # none | quota (max doc_quota chunks per doc) | mmr
    doc_quota: int = 2
    mmr_lambda: float = 0.7  # MMR: weight of relevance vs. redundancy
    rerank: bool = False  # cross-encoder reranking of the candidate pool
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM splits a multi-entity question into sub-queries; each is retrieved and the lists are
    # merged by RRF with the original question's list (one extra LLM call per question).
    retrieval_decompose: bool = False
    max_subqueries: int = 3
    # Each document-identifying name in the question (trustqa/scope.py) gets a share of the
    # top-k from its documents, so a two-paper question retrieves both (failures.md F12).
    retrieval_entity_scope: bool = False

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
    # Record each text chunk's section heading path (processing/sections.py) and show it to
    # retrieval and generation next to the chunk text; the chunk text is unchanged.
    section_headings: bool = False
    verify: bool = True
    # Preset for baseline system B (OCR + plain RAG): forces flatten_tables=True and
    # verify=False regardless of their own settings.
    baseline_b: bool = False

    # Verification (Phase 3). The verify_* switches exist for ablations (Phase 6); with one
    # off, that check is skipped and the answer can no longer be Verified (except conflicts,
    # which only ever lower the status).
    verify_support: bool = True  # LLM judge + verbatim-quote check (FR-10)
    verify_numeric: bool = True  # numbers / units / periods / calculations (FR-11)
    verify_conflicts: bool = True  # cross-document conflict detection
    abstain_min_score: float = 0.30  # skip generation below this top retrieval score
    # ...unless one passage contains every content word of the question (abstention.py)
    gate_lexical_pass: bool = False
    ocr_warn_confidence: float = 90.0  # cited OCR chunks below this get a warning
    quote_min_similarity: float = 90.0  # fuzzy match for supporting quotes >= 20 chars
    # Answer-about-the-question checks (verification/answer_checks.py): citations from
    # documents the question does not name are withheld / capped (F13); quantity questions
    # need a number in the answer to be Verified (F10, deterministic).
    verify_scope: bool = False
    verify_answers_question: bool = False
    # Also ask the support judge whether the answer gives what the question asks. Measured
    # harmful with qwen2.5:3b (it flagged correct answers and the extra task changed its main
    # verdicts, e.g. a correct "0.656" withheld); kept for a stronger judge model.
    judge_answers_question: bool = False
    # A short numeric answer whose number is in the cited chunk, in a sentence sharing words
    # with the question, is supported even if the judge quoted a sentence without it (F14).
    support_number_fallback: bool = False
    # A number the model cites from the neighbouring chunk of the same page is checked against
    # the chunk that states it (document and page unchanged; verification still decides).
    citation_repair: bool = False
    # Ask the support judge once more for material claims it skipped (long answers).
    support_retry_missing: bool = False
    # grounded-v1 (frozen, baselines) | grounded-v2 (lists every item, answers directly)
    generation_prompt: str = "grounded-v1"

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
        if not 0 <= self.ocr_warn_confidence <= 100:
            raise ConfigError(f"ocr_warn_confidence must be in [0, 100], got {self.ocr_warn_confidence}")
        if not 0 < self.quote_min_similarity <= 100:
            raise ConfigError("quote_min_similarity must be in (0, 100], got "
                              f"{self.quote_min_similarity}")
        if self.retrieval_diversify not in {"none", "quota", "mmr"}:
            raise ConfigError("retrieval_diversify must be none, quota or mmr, got "
                              f"{self.retrieval_diversify!r}")
        if self.rrf_k < 1 or self.retrieval_candidates < 1 or self.doc_quota < 1:
            raise ConfigError("rrf_k, retrieval_candidates and doc_quota must be >= 1")
        if not 0 <= self.mmr_lambda <= 1:
            raise ConfigError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")
        if not 1 <= self.max_subqueries <= 5:
            raise ConfigError(f"max_subqueries must be in [1, 5], got {self.max_subqueries}")
        if self.generation_prompt not in {"grounded-v1", "grounded-v2"}:
            raise ConfigError("generation_prompt must be grounded-v1 or grounded-v2, got "
                              f"{self.generation_prompt!r}")
        if self.baseline_b:  # frozen dataclass: resolve the preset in place
            object.__setattr__(self, "flatten_tables", True)
            object.__setattr__(self, "verify", False)
        if self.is_frozen_baseline:
            # A and B are plain dense RAG (frozen at v0.1/v0.2): Phase 4 retrieval upgrades
            # set in .env cannot leak into them.
            object.__setattr__(self, "retrieval_hybrid", False)
            object.__setattr__(self, "retrieval_diversify", "none")
            object.__setattr__(self, "rerank", False)
            object.__setattr__(self, "retrieval_decompose", False)
            object.__setattr__(self, "section_headings", False)
            object.__setattr__(self, "retrieval_entity_scope", False)
            object.__setattr__(self, "generation_prompt", "grounded-v1")

    @property
    def is_frozen_baseline(self) -> bool:
        """True for baseline systems A and B (tables flattened, no verification)."""
        return self.flatten_tables and not self.verify

    @property
    def system_name(self) -> str:
        """Which evaluation system this configuration reproduces (PRD / phases.md)."""
        if self.is_frozen_baseline:
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
        retrieval_hybrid=_env_bool("RETRIEVAL_HYBRID", d.retrieval_hybrid),
        rrf_k=_env_int("RRF_K", d.rrf_k),
        retrieval_candidates=_env_int("RETRIEVAL_CANDIDATES", d.retrieval_candidates),
        retrieval_diversify=_env_str("RETRIEVAL_DIVERSIFY", d.retrieval_diversify),
        doc_quota=_env_int("DOC_QUOTA", d.doc_quota),
        mmr_lambda=_env_float("MMR_LAMBDA", d.mmr_lambda),
        rerank=_env_bool("RERANK", d.rerank),
        reranker_model=_env_str("RERANKER_MODEL", d.reranker_model),
        retrieval_decompose=_env_bool("RETRIEVAL_DECOMPOSE", d.retrieval_decompose),
        max_subqueries=_env_int("MAX_SUBQUERIES", d.max_subqueries),
        retrieval_entity_scope=_env_bool("RETRIEVAL_ENTITY_SCOPE", d.retrieval_entity_scope),
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
        section_headings=_env_bool("SECTION_HEADINGS", d.section_headings),
        verify=_env_bool("VERIFY", d.verify),
        baseline_b=_env_bool("BASELINE_B", d.baseline_b),
        verify_support=_env_bool("VERIFY_SUPPORT", d.verify_support),
        verify_numeric=_env_bool("VERIFY_NUMERIC", d.verify_numeric),
        verify_conflicts=_env_bool("VERIFY_CONFLICTS", d.verify_conflicts),
        abstain_min_score=_env_float("ABSTAIN_MIN_SCORE", d.abstain_min_score),
        gate_lexical_pass=_env_bool("GATE_LEXICAL_PASS", d.gate_lexical_pass),
        ocr_warn_confidence=_env_float("OCR_WARN_CONFIDENCE", d.ocr_warn_confidence),
        quote_min_similarity=_env_float("QUOTE_MIN_SIMILARITY", d.quote_min_similarity),
        verify_scope=_env_bool("VERIFY_SCOPE", d.verify_scope),
        verify_answers_question=_env_bool("VERIFY_ANSWERS_QUESTION", d.verify_answers_question),
        judge_answers_question=_env_bool("JUDGE_ANSWERS_QUESTION", d.judge_answers_question),
        support_number_fallback=_env_bool("SUPPORT_NUMBER_FALLBACK", d.support_number_fallback),
        citation_repair=_env_bool("CITATION_REPAIR", d.citation_repair),
        support_retry_missing=_env_bool("SUPPORT_RETRY_MISSING", d.support_retry_missing),
        generation_prompt=_env_str("GENERATION_PROMPT", d.generation_prompt),
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
