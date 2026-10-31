"""Shared fixtures: synthetic digital PDFs and a deterministic fake embedder."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from trustqa import llm as llm_mod

# Page texts for the synthetic "solar" report. Page 2 is intentionally blank.
SOLAR_PAGES = [
    "Annual Report 2025. This report describes the operations of Helios Energy. "
    "The company builds solar farms in Karnataka and Gujarat.",
    "",
    "Revenue grew to Rs. 1,03,450.50 crore in 2025. Operating margin was -2.5% "
    "because of panel import costs. The board approved a dividend of 12 percent.",
    "The Pavagada plant has an installed capacity of 600 MW. Maintenance is done by "
    "drones every quarter. Water usage per panel cleaning cycle fell by 40 percent.",
]

RIVER_PAGES = [
    "Hydrology survey. The Kaveri river basin receives most rainfall during the "
    "south-west monsoon between June and September.",
    "The Mettur dam was completed in 1934. Its reservoir supports irrigation across "
    "the delta region.",
]


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        if text:
            rect = pymupdf.Rect(50, 50, 545, 792)
            assert page.insert_textbox(rect, text, fontsize=11) >= 0, "text overflow"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def solar_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "solar_report.pdf", SOLAR_PAGES)


@pytest.fixture
def river_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "river_survey.pdf", RIVER_PAGES)


class FakeEmbedder:
    """Hashed bag-of-words embedder: deterministic, offline, good enough to rank
    passages that share words with the query."""

    dim = 512

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) < 3:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def fake_embedder(monkeypatch) -> FakeEmbedder:
    """Route every embedding call through FakeEmbedder (no model download)."""
    from trustqa.retrieval import index as index_mod

    emb = FakeEmbedder()
    monkeypatch.setattr(index_mod, "get_embedder", lambda model_name: emb)
    return emb


@pytest.fixture(autouse=True)
def _retrieval_flags_off(monkeypatch):
    """Tests do not depend on the developer's .env retrieval settings: real environment
    variables win over .env, so pin the Phase 4 flags to their defaults. Tests that want a
    flag pass it to load_config() or set it with monkeypatch.setenv."""
    for k, v in {"RETRIEVAL_HYBRID": "0", "RETRIEVAL_DIVERSIFY": "none", "RERANK": "0",
                 "RETRIEVAL_DECOMPOSE": "0"}.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(autouse=True)
def _reset_llm():
    yield
    llm_mod.set_llm(None)
