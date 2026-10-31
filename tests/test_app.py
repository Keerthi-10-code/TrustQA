"""Smoke test: the Streamlit app renders its start page without errors."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_start_page_renders():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "TrustQA"
    assert any("Upload one or more PDFs" in i.value for i in at.info)
