"""Evaluated systems and LLM-output caching (Phase 6, steps 2 and 4).

Systems share every LLM / embedding / chunking setting from .env (one model, temperature 0);
only the switches that define a system differ:

  A             plain-text RAG: no OCR, tables flattened, no verification, dense retrieval
  B             OCR + plain RAG: tables flattened, no verification, dense retrieval
  C             full TrustQA: OCR, table chunks, verification, retrieval flags from .env
  C-no_tables   C with tables flattened (no table-aware chunks)
  C-no_support  C without the claim-support judge (FR-10)
  C-no_numeric  C without numeric checks (FR-11)
  C-no_ocr      C without the OCR fallback

A and B force the Phase 4 retrieval upgrades off (Config.is_frozen_baseline); C and its
ablations keep the .env retrieval settings, so an ablation differs from C in one switch.

CachingLLM wraps the configured provider and stores every completion in a SQLite file,
keyed by sha256 of (provider, model, sampling settings, repeat, system prompt, prompt, json).
Re-running the evaluation then costs no model calls; `repeat` > 0 gives fresh calls for
variance runs while repeat 0 is shared with every earlier run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from trustqa import llm as llm_mod
from trustqa.config import Config
from trustqa.corpus import system_config

C_PINS = dict(verify_support=True, verify_numeric=True, verify_conflicts=True)
SYSTEMS: dict[str, tuple[str, str, dict]] = {
    # name: (base system, description, overrides)
    "A": ("A", "plain-text RAG", {}),
    "B": ("B", "OCR + plain RAG", {}),
    "C": ("C", "full TrustQA", C_PINS),
    "C-no_tables": ("C", "C without table-aware chunks", {**C_PINS, "flatten_tables": True}),
    "C-no_support": ("C", "C without the support check", {**C_PINS, "verify_support": False}),
    "C-no_numeric": ("C", "C without numeric checks", {**C_PINS, "verify_numeric": False}),
    "C-no_ocr": ("C", "C without OCR fallback", {**C_PINS, "ocr_enabled": False}),
    # TrustQA as tagged v0.4-demo, before the Phase 6 fixes (kept for the before/after table)
    "C-v0.4": ("C", "TrustQA v0.4 (before the Phase 6 fixes)",
               {**C_PINS, "section_headings": False, "gate_lexical_pass": False,
                "abstain_min_score": 0.30, "retrieval_entity_scope": False, "verify_scope": False,
                "verify_answers_question": False, "support_number_fallback": False,
                "judge_answers_question": False, "citation_repair": False,
                "support_retry_missing": False,
                "generation_prompt": "grounded-v1"}),
}
MAIN_SYSTEMS = ["A", "B", "C-v0.4", "C"]
ABLATIONS = ["C-no_tables", "C-no_support", "C-no_numeric", "C-no_ocr"]


def config_for(name: str, **overrides) -> Config:
    """Config of an evaluated system (temperature pinned to 0)."""
    if name not in SYSTEMS:
        raise ValueError(f"unknown system {name!r}; use one of {list(SYSTEMS)}")
    base, _, pins = SYSTEMS[name]
    return system_config(base, **{"llm_temperature": 0.0, **pins, **overrides})


class CachingLLM:
    """LLM wrapper caching completions by (prompt, model) hash in SQLite."""

    def __init__(self, inner: llm_mod.LLM, path: str | Path, namespace: dict,
                 read: bool = True):
        self.inner = inner
        self.read = read  # False: always call the model (fresh latency), still store outputs
        self.namespace = dict(namespace)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS completions "
                         "(key TEXT PRIMARY KEY, output TEXT NOT NULL)")
        self._db.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def key(self, prompt: str, system: str | None, json_mode: bool) -> str:
        payload = json.dumps([self.namespace, system, prompt, bool(json_mode)],
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, system: str | None = None, json: bool = False) -> str:
        k = self.key(prompt, system, json)
        row = None
        if self.read:
            with self._lock:
                row = self._db.execute("SELECT output FROM completions WHERE key = ?",
                                       (k,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        out = self.inner.complete(prompt, system=system, json=json)  # errors are not cached
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO completions VALUES (?, ?)", (k, out))
            self._db.commit()
        self.misses += 1
        return out

    def close(self) -> None:
        self._db.close()


def llm_namespace(config: Config, repeat: int = 0) -> dict:
    ns = {"provider": config.llm_provider, "model": config.llm_model,
          "temperature": config.llm_temperature, "effort": config.llm_effort,
          "max_tokens": config.llm_max_tokens}
    if repeat:
        ns["repeat"] = repeat
    return ns


def caching_llm(config: Config, cache_path: str | Path, repeat: int = 0,
                inner: llm_mod.LLM | None = None, read: bool = True) -> CachingLLM:
    return CachingLLM(inner or llm_mod.get_llm(config), cache_path, llm_namespace(config, repeat),
                      read=read)
