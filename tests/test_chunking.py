import re

import pytest

from trustqa.processing.chunking import chunk_page, chunk_pages, chunk_text, split_sentences
from trustqa.processing.extract import PageText, extract_pages

LONG = " ".join(
    f"Sentence number {i} talks about topic {i} in some detail with filler words." for i in range(1, 41)
)


def test_split_sentences_guards_abbreviations():
    s = split_sentences("RAG was proposed by Lewis et al. in 2020. See Fig. 1 for details. "
                        "J. Smith agreed, e.g. on page 4. Revenue was Rs. 12 crore.")
    assert s == [
        "RAG was proposed by Lewis et al. in 2020.",
        "See Fig. 1 for details.",
        "J. Smith agreed, e.g. on page 4.",
        "Revenue was Rs. 12 crore.",
    ]


def test_decimals_are_not_sentence_boundaries():
    assert split_sentences("Accuracy was 44.5 on NQ. It improved.") == [
        "Accuracy was 44.5 on NQ.", "It improved."]


@pytest.mark.parametrize("size,overlap", [(200, 0), (200, 60), (300, 150), (800, 150)])
def test_chunks_respect_size_and_are_sentence_aligned(size, overlap):
    sentences = set(split_sentences(LONG))
    chunks = chunk_text(LONG, size, overlap)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= size
        # every chunk is a run of whole sentences
        parts = re.split(r"(?<=\.) (?=Sentence)", c)
        assert all(p in sentences for p in parts)


def test_overlap_repeats_trailing_sentences_within_budget():
    chunks = chunk_text(LONG, 300, 150)
    for prev, nxt in zip(chunks, chunks[1:]):
        p, n = split_sentences(prev), split_sentences(nxt)
        shared = next(k for k in range(len(n), 0, -1) if p[-k:] == n[:k])
        assert shared >= 1  # next chunk starts with the previous chunk's trailing sentences
        assert len(" ".join(n[:shared])) <= 150  # and the repeated part fits the overlap
    no_overlap = chunk_text(LONG, 300, 0)
    assert " ".join(no_overlap) == LONG  # no overlap: chunks tile the text exactly


def test_all_text_is_covered():
    chunks = chunk_text(LONG, 250, 80)
    covered = set()
    for c in chunks:
        covered.update(split_sentences(c))
    assert covered == set(split_sentences(LONG))


def test_overlong_sentence_is_split_at_words():
    sentence = " ".join(["word"] * 100) + "."
    chunks = chunk_text(sentence, 50, 10)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_chunk_ids_and_metadata():
    page = PageText("DOC01", "a.pdf", 4, LONG)
    chunks = chunk_page(page, 300, 50)
    assert [c.chunk_id for c in chunks] == [f"DOC01_P4_C{i}" for i in range(1, len(chunks) + 1)]
    assert all(c.page_number == 4 and c.document_name == "a.pdf" for c in chunks)
    assert all(c.content_type.value == "text" and not c.ocr_used for c in chunks)


def test_chunks_never_cross_pages(solar_pdf):
    pages = extract_pages(solar_pdf, "DOC01")
    chunks = chunk_pages(pages, 120, 40)
    by_page = {p.page_number: p.text for p in pages}
    for c in chunks:
        assert c.text in by_page[c.page_number]  # each chunk is a substring of its own page
        assert c.chunk_id.startswith(f"DOC01_P{c.page_number}_C")
    assert 2 not in {c.page_number for c in chunks}  # blank page -> no chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        chunk_text("x.", 100, 100)
