"""OCR fallback for scanned / garbage pages: PyMuPDF render -> (optional preprocessing) ->
Tesseract image_to_data with per-word confidence.

The page text is assembled word by word from Tesseract's output, so every word keeps its
character span in the final text. Chunks cut from that text can then report the mean
confidence of exactly the words they contain (`span_confidence`), which Phase 3 uses for
OCR-uncertainty warnings on cited chunks.

Text assembly mirrors extract.normalize_text: words are joined by single spaces, and a
line-end hyphen between letters is joined only if the joined word occurs in the given
vocabulary (otherwise the hyphen is kept, so no character is dropped). Digits are never
touched (NFR-03).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pymupdf
from PIL import Image

from trustqa.processing.extract import _fix_ligatures

logger = logging.getLogger("trustqa")

OCR_VERSION = "tesseract-v1"

_LEFT_HYPHEN_RE = re.compile(r"[A-Za-z]+-$")
_RIGHT_WORD_RE = re.compile(r"^([a-z]+)")


class OcrUnavailableError(ValueError):
    """Tesseract is needed but cannot be run (reported explicitly, NFR-09)."""


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float  # Tesseract word confidence, 0-100
    start: int  # character span in OcrPage.text
    end: int


@dataclass
class OcrPage:
    text: str
    words: list[OcrWord] = field(default_factory=list)
    mean_confidence: float | None = None  # None if no words were recognised
    deskew_angle: float = 0.0  # degrees applied by preprocessing (0 if disabled)
    preprocessed: bool = False

    def span_confidence(self, start: int, end: int) -> float | None:
        """Mean confidence of the words overlapping text[start:end]."""
        confs = [w.confidence for w in self.words if w.start < end and w.end > start]
        return round(float(np.mean(confs)), 2) if confs else None


def configure_tesseract(tesseract_cmd: str | None) -> None:
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def tesseract_version(tesseract_cmd: str | None = None) -> str:
    """Return the Tesseract version, or raise OcrUnavailableError with a fix hint."""
    import pytesseract

    configure_tesseract(tesseract_cmd)
    try:
        return str(pytesseract.get_tesseract_version())
    except (pytesseract.TesseractNotFoundError, OSError) as e:
        raise OcrUnavailableError(
            "Tesseract OCR is required for scanned pages but was not found. Install it "
            "and set TESSERACT_CMD in .env to the tesseract executable.") from e


# ---------------------------------------------------------------- rendering / preprocessing

def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    """Render a page to an 8-bit grayscale image."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def is_blank_page(page: pymupdf.Page, dpi: int = 36, max_ink_pixels: int = 2) -> bool:
    """True if the page renders with no ink at all - nothing to OCR. Deliberately strict:
    a single short word already gives ~50 dark pixels at 36 DPI, a scan ~10% of pixels."""
    arr = np.asarray(render_page(page, dpi))
    return int((arr < 200).sum()) <= max_ink_pixels


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's threshold for a uint8 image (numpy only, no OpenCV)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 / total - m0) ** 2 / (w0 * w1 / total)
    between = np.nan_to_num(between)
    return int(np.argmax(between))


def binarize(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("L"))
    t = otsu_threshold(arr)
    return Image.fromarray(np.where(arr > t, 255, 0).astype(np.uint8))


def _projection_score(binary_dark: Image.Image, angle: float) -> float:
    rotated = binary_dark.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    rows = np.asarray(rotated, dtype=np.float32).sum(axis=1)
    return float(np.var(rows))


def estimate_skew(img: Image.Image, max_angle: float = 5.0) -> float:
    """Skew angle (degrees, PIL rotate convention) maximising the variance of the
    horizontal projection profile of dark pixels: text lines are sharpest when level."""
    small = img.convert("L")
    scale = 1000 / max(small.width, 1)
    if scale < 1:
        small = small.resize((1000, max(1, int(small.height * scale))), Image.BILINEAR)
    arr = np.asarray(small)
    dark = Image.fromarray(np.where(arr <= otsu_threshold(arr), 1, 0).astype(np.uint8))

    def best(candidates: np.ndarray) -> float:
        return float(max(candidates, key=lambda a: _projection_score(dark, float(a))))

    coarse = best(np.arange(-max_angle, max_angle + 1e-9, 0.25))
    return round(best(np.arange(coarse - 0.25, coarse + 0.25 + 1e-9, 0.05)), 2)


def preprocess(img: Image.Image) -> tuple[Image.Image, float]:
    """Grayscale -> deskew -> Otsu threshold. Returns (image, applied angle)."""
    gray = img.convert("L")
    angle = estimate_skew(gray)
    if abs(angle) >= 0.05:
        gray = gray.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    return binarize(gray), angle


# ---------------------------------------------------------------- text assembly

def assemble_words(lines: list[list[tuple[str, float]]], vocab: set[str] | None = None
                   ) -> tuple[str, list[OcrWord]]:
    """Join recognised words (grouped by line, in reading order) into page text.

    Returns the text and each output word with its character span. A word that merges a
    line-end hyphenation takes the lower of the two confidences.
    """
    tokens: list[tuple[str, float]] = []
    pending: tuple[str, float] | None = None  # line-final "xxx-" waiting for the next line
    for line in lines:
        line = [(t, c) for t, c in line if t]
        for i, (text, conf) in enumerate(line):
            if pending is not None:
                left, lconf = pending
                pending = None
                m = _RIGHT_WORD_RE.match(text)
                if m:
                    joined = left[:-1] + m.group(1)
                    if vocab is not None and joined.lower() in vocab:
                        tokens.append((left[:-1] + text, min(lconf, conf)))
                    else:
                        tokens.append((left + text, min(lconf, conf)))
                    continue
                tokens.append((left, lconf))
            if i == len(line) - 1 and _LEFT_HYPHEN_RE.search(text):
                pending = (text, conf)
            else:
                tokens.append((text, conf))
    if pending is not None:
        tokens.append(pending)

    words: list[OcrWord] = []
    parts: list[str] = []
    pos = 0
    for text, conf in tokens:
        if parts:
            pos += 1  # the joining space
        words.append(OcrWord(text, conf, pos, pos + len(text)))
        parts.append(text)
        pos += len(text)
    return " ".join(parts), words


def _lines_from_data(data: dict) -> list[list[tuple[str, float]]]:
    lines: dict[tuple[int, int, int], list[tuple[int, str, float]]] = {}
    order: list[tuple[int, int, int]] = []
    for i, raw in enumerate(data["text"]):
        text = _fix_ligatures((raw or "").strip())
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append((data["word_num"][i], text, conf))
    return [[(t, c) for _, t, c in sorted(lines[k])] for k in order]


def ocr_image(img: Image.Image, lang: str = "eng", psm: int = 3,
              vocab: set[str] | None = None) -> OcrPage:
    import pytesseract

    try:
        data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}",
                                         output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise OcrUnavailableError(
            "Tesseract OCR is required for scanned pages but was not found. Install it "
            "and set TESSERACT_CMD in .env to the tesseract executable.") from e
    lines = _lines_from_data(data)
    if vocab is None:  # fall back to the page's own words for hyphen decisions
        vocab = {re.sub(r"[^a-z]", "", t.lower()) for line in lines for t, _ in line}
    text, words = assemble_words(lines, vocab)
    mean = round(float(np.mean([w.confidence for w in words])), 2) if words else None
    return OcrPage(text=text, words=words, mean_confidence=mean)


def ocr_page(page: pymupdf.Page, dpi: int = 300, lang: str = "eng", psm: int = 3,
             use_preprocess: bool = False, vocab: set[str] | None = None) -> OcrPage:
    """OCR one PDF page. Requires Tesseract (configure_tesseract first if not on PATH)."""
    img = render_page(page, dpi)
    angle = 0.0
    if use_preprocess:
        img, angle = preprocess(img)
    result = ocr_image(img, lang=lang, psm=psm, vocab=vocab)
    result.deskew_angle = angle
    result.preprocessed = use_preprocess
    return result
