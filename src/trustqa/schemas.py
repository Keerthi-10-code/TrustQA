"""Frozen data contracts for TrustQA (PRD 3.2 canonical chunk, PRD 3.4 final response).

Change these only deliberately: every module and the evaluation harness depend on them.

Conventions
- page_number is the 1-indexed PDF page position (never 0-indexed).
- Table cells are kept as strings so numbers, signs, units and decimals are never
  silently altered (NFR-03). Parse them only where a calculation needs it.
- document_name / page_number in a Citation are always resolved from trusted chunk
  metadata, never copied from LLM output (FR-09).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Verification status (PRD 2.3)."""

    VERIFIED = "Verified"
    PARTIALLY_SUPPORTED = "Partially Supported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"


class ExtractionMethod(str, Enum):
    """Tool that produced a chunk's content."""

    PYMUPDF = "pymupdf"  # digital text layer
    TESSERACT = "tesseract"  # OCR fallback
    PDFPLUMBER = "pdfplumber"  # table extraction
    CAMELOT = "camelot"  # optional table fallback


class WarningCode(str, Enum):
    """Warning categories (PRD 3.4)."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    INVALID_CHUNK_ID = "invalid_chunk_id"
    PAGE_MISMATCH = "page_mismatch"
    OCR_UNCERTAINTY = "ocr_uncertainty"
    NUMERIC_MISMATCH = "numeric_mismatch"
    CONFLICT = "conflict"


class SchemaError(ValueError):
    """Raised when a record violates the data contract."""


@dataclass
class TableData:
    """Structured table preserved alongside a table chunk (FR-04)."""

    table_id: str
    headers: list[str]
    rows: list[list[str]]
    units: str | None = None
    caption: str | None = None

    def __post_init__(self) -> None:
        if not self.table_id:
            raise SchemaError("table_id must be non-empty")
        for i, row in enumerate(self.rows):
            if len(row) != len(self.headers):
                raise SchemaError(
                    f"table {self.table_id}: row {i} has {len(row)} cells, "
                    f"expected {len(self.headers)} (one per header)"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TableData:
        return cls(
            table_id=d["table_id"],
            headers=list(d["headers"]),
            rows=[list(r) for r in d["rows"]],
            units=d.get("units"),
            caption=d.get("caption"),
        )


@dataclass
class Chunk:
    """Canonical chunk record (PRD 3.2). One line of data/processed/*.jsonl."""

    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    text: str
    content_type: ContentType = ContentType.TEXT
    table_data: TableData | None = None
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    ocr_used: bool = False
    section: str | None = None
    ocr_confidence: float | None = None  # mean word confidence 0-100; None if not OCR

    def __post_init__(self) -> None:
        self.content_type = ContentType(self.content_type)
        self.extraction_method = ExtractionMethod(self.extraction_method)
        if not isinstance(self.page_number, int) or isinstance(self.page_number, bool):
            raise SchemaError("page_number must be an int")
        if self.page_number < 1:
            raise SchemaError(f"page_number must be 1-indexed, got {self.page_number}")
        if not self.document_id or not self.chunk_id:
            raise SchemaError("document_id and chunk_id must be non-empty")
        if self.content_type is ContentType.TABLE and self.table_data is None:
            raise SchemaError(f"{self.chunk_id}: table chunk requires table_data")
        if self.content_type is ContentType.TEXT and self.table_data is not None:
            raise SchemaError(f"{self.chunk_id}: text chunk must not carry table_data")
        if self.ocr_confidence is not None and not 0 <= self.ocr_confidence <= 100:
            raise SchemaError(f"{self.chunk_id}: ocr_confidence must be in [0, 100]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "content_type": self.content_type.value,
            "table_data": self.table_data.to_dict() if self.table_data else None,
            "extraction_method": self.extraction_method.value,
            "ocr_used": self.ocr_used,
            "section": self.section,
            "ocr_confidence": self.ocr_confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        td = d.get("table_data")
        return cls(
            document_id=d["document_id"],
            document_name=d["document_name"],
            page_number=d["page_number"],
            chunk_id=d["chunk_id"],
            text=d["text"],
            content_type=ContentType(d.get("content_type", "text")),
            table_data=TableData.from_dict(td) if td else None,
            extraction_method=ExtractionMethod(d.get("extraction_method", "pymupdf")),
            ocr_used=bool(d.get("ocr_used", False)),
            section=d.get("section"),
            ocr_confidence=d.get("ocr_confidence"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> Chunk:
        return cls.from_dict(json.loads(s))


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieve(): the original record plus its ranking."""

    chunk: Chunk
    score: float
    rank: int  # 1 = best

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": self.score, "rank": self.rank}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RetrievedChunk:
        return cls(chunk=Chunk.from_dict(d["chunk"]), score=float(d["score"]), rank=int(d["rank"]))


@dataclass
class Citation:
    """A resolved citation (PRD 3.4). Name/page come from chunk metadata only."""

    document_id: str
    document_name: str
    page_number: int
    chunk_ids: list[str]
    supporting_quote: str = ""

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise SchemaError(f"page_number must be 1-indexed, got {self.page_number}")
        if not self.chunk_ids:
            raise SchemaError("citation must reference at least one chunk_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Citation:
        return cls(
            document_id=d["document_id"],
            document_name=d["document_name"],
            page_number=d["page_number"],
            chunk_ids=list(d["chunk_ids"]),
            supporting_quote=d.get("supporting_quote", ""),
        )


@dataclass
class Warning:
    """A machine-readable warning with a human-readable message."""

    code: WarningCode
    message: str
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        self.code = WarningCode(self.code)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "chunk_id": self.chunk_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Warning:
        return cls(code=WarningCode(d["code"]), message=d["message"], chunk_id=d.get("chunk_id"))


@dataclass
class Result:
    """Final response contract (PRD 3.4) returned by answer_question()."""

    question: str
    answer: str
    status: Status
    citations: list[Citation] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = Status(self.status)
        if self.status is Status.VERIFIED and not self.citations:
            raise SchemaError("a Verified result must have at least one citation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status.value,
            "citations": [c.to_dict() for c in self.citations],
            "warnings": [w.to_dict() for w in self.warnings],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Result:
        return cls(
            question=d["question"],
            answer=d["answer"],
            status=Status(d["status"]),
            citations=[Citation.from_dict(c) for c in d.get("citations", [])],
            warnings=[Warning.from_dict(w) for w in d.get("warnings", [])],
        )

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> Result:
        return cls.from_dict(json.loads(s))
