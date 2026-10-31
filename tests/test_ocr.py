import numpy as np
import pytest
from PIL import Image, ImageDraw

from trustqa.processing.ocr import (
    OcrPage,
    assemble_words,
    binarize,
    estimate_skew,
    otsu_threshold,
)


def test_words_joined_with_exact_spans():
    text, words = assemble_words([[("Revenue", 95.0), ("Rs.", 90.0), ("1,03,450.50", 80.0)],
                                  [("margin", 70.0), ("-2.5%", 60.0)]])
    assert text == "Revenue Rs. 1,03,450.50 margin -2.5%"
    for w in words:
        assert text[w.start:w.end] == w.text


def test_numbers_signs_and_decimals_untouched():
    tokens = ["-2.5%", "+11%", "1,03,450.50", "(3.2)", "0.0700", "10^19", "−7.4"]
    text, _ = assemble_words([[(t, 90.0) for t in tokens]])
    assert text.split(" ") == tokens


def test_line_end_hyphen_joined_only_for_known_words():
    lines = [[("the", 90.0), ("retrie-", 80.0)], [("val", 60.0), ("step", 90.0)]]
    text, words = assemble_words(lines, vocab={"retrieval", "the", "step"})
    assert text == "the retrieval step"
    assert words[1].confidence == 60.0  # merged word keeps the lower confidence
    text, _ = assemble_words(lines, vocab={"the", "step"})
    assert text == "the retrie-val step"  # unknown join: hyphen kept, nothing dropped


def test_numeric_line_end_hyphen_not_joined():
    text, _ = assemble_words([[("range", 90.0), ("10-", 90.0)], [("20", 90.0)]], vocab=set())
    assert text == "range 10- 20"


def test_span_confidence_is_mean_of_overlapping_words():
    text, words = assemble_words([[("alpha", 90.0), ("beta", 50.0), ("gamma", 70.0)]])
    page = OcrPage(text=text, words=words, mean_confidence=70.0)
    start = text.index("beta")
    assert page.span_confidence(start, len(text)) == 60.0
    assert page.span_confidence(0, 5) == 90.0
    assert page.span_confidence(len(text) + 5, len(text) + 9) is None


def test_otsu_separates_ink_from_paper():
    arr = np.full((50, 50), 230, dtype=np.uint8)
    arr[20:30, 5:45] = 30
    t = otsu_threshold(arr)
    assert 30 <= t < 230
    out = np.asarray(binarize(Image.fromarray(arr)))
    assert set(np.unique(out)) == {0, 255}


def test_deskew_recovers_rotation():
    img = Image.new("L", (900, 900), 255)
    d = ImageDraw.Draw(img)
    for y in range(80, 820, 30):
        d.rectangle([80, y, 820, y + 10], fill=0)  # text-like horizontal bars
    rotated = img.rotate(-2.0, resample=Image.BICUBIC, fillcolor=255)
    assert estimate_skew(rotated) == pytest.approx(2.0, abs=0.2)
    assert estimate_skew(img) == pytest.approx(0.0, abs=0.2)
