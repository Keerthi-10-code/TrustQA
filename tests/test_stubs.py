"""Signatures are frozen in Phase 0; this guards against accidental changes."""

import inspect

from trustqa.pipeline import answer_question
from trustqa.processing.process import process_pdfs
from trustqa.retrieval.index import build_index
from trustqa.retrieval.search import retrieve
from trustqa.verification.verify import generate_and_verify


def params(fn):
    return list(inspect.signature(fn).parameters)


def test_frozen_signatures():
    assert params(process_pdfs) == ["pdf_paths", "config"]
    assert params(build_index) == ["chunks", "config"]
    assert params(retrieve) == ["question", "top_k", "index"]
    assert inspect.signature(retrieve).parameters["top_k"].default == 5
    assert params(generate_and_verify) == ["question", "passages", "llm"]
    assert params(answer_question) == ["question", "uploaded_docs", "config"]
