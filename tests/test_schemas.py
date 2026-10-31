import json

import pytest

from trustqa.schemas import (
    Chunk,
    Citation,
    ContentType,
    ExtractionMethod,
    Result,
    RetrievedChunk,
    SchemaError,
    Status,
    TableData,
    Warning,
    WarningCode,
)


def text_chunk(**kw):
    base = dict(
        document_id="DOC01",
        document_name="annual_report.pdf",
        page_number=4,
        chunk_id="DOC01_P4_C1",
        text="The company reported revenue of Rs. 12 crore in 2025.",
    )
    base.update(kw)
    return Chunk(**base)


def table_chunk():
    return Chunk(
        document_id="DOC02",
        document_name="results.pdf",
        page_number=7,
        chunk_id="DOC02_P7_T1_R1",
        text="Table T1 (Revenue, Rs crore) | Year=2025 | Value=-12.50",
        content_type=ContentType.TABLE,
        table_data=TableData(
            table_id="T1",
            headers=["Year", "Value"],
            rows=[["2025", "-12.50"], ["2024", "1,03,450.00"]],
            units="Rs crore",
            caption="Revenue",
        ),
        extraction_method=ExtractionMethod.PDFPLUMBER,
    )


def test_prd_example_matches_contract():
    d = text_chunk().to_dict()
    for key in ["document_id", "document_name", "page_number", "chunk_id", "text",
                "content_type", "table_data"]:
        assert key in d
    assert d["content_type"] == "text"
    assert d["table_data"] is None


@pytest.mark.parametrize("chunk", [text_chunk(), table_chunk(),
                                   text_chunk(ocr_used=True, ocr_confidence=71.5,
                                              extraction_method="tesseract", section="Intro")])
def test_chunk_json_round_trip(chunk):
    s = chunk.to_json()
    json.loads(s)  # valid JSON
    assert Chunk.from_json(s) == chunk


def test_table_values_preserved_exactly():
    back = Chunk.from_json(table_chunk().to_json())
    assert back.table_data.rows == [["2025", "-12.50"], ["2024", "1,03,450.00"]]
    assert back.table_data.units == "Rs crore"


def test_page_number_is_one_indexed():
    with pytest.raises(SchemaError):
        text_chunk(page_number=0)


def test_table_chunk_requires_table_data():
    with pytest.raises(SchemaError):
        text_chunk(content_type="table")


def test_ragged_table_rejected():
    with pytest.raises(SchemaError):
        TableData(table_id="T1", headers=["a", "b"], rows=[["1"]])


def test_result_json_round_trip():
    r = Result(
        question="What was revenue in 2025?",
        answer="Rs. 12 crore.",
        status=Status.PARTIALLY_SUPPORTED,
        citations=[Citation("DOC01", "annual_report.pdf", 4, ["DOC01_P4_C1"],
                            "revenue of Rs. 12 crore in 2025")],
        warnings=[Warning(WarningCode.OCR_UNCERTAINTY, "low OCR confidence", "DOC01_P4_C1")],
    )
    d = json.loads(r.to_json())
    assert d["status"] == "Partially Supported"
    assert d["warnings"][0]["code"] == "ocr_uncertainty"
    assert Result.from_json(r.to_json()) == r


def test_verified_requires_citation():
    with pytest.raises(SchemaError):
        Result(question="q", answer="a", status=Status.VERIFIED)


def test_insufficient_evidence_allows_no_citations():
    r = Result(question="q", answer="", status="Insufficient Evidence")
    assert r.status is Status.INSUFFICIENT_EVIDENCE


def test_retrieved_chunk_round_trip():
    rc = RetrievedChunk(chunk=table_chunk(), score=0.83, rank=1)
    assert RetrievedChunk.from_dict(json.loads(json.dumps(rc.to_dict()))) == rc
