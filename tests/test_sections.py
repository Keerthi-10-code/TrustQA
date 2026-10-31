"""Section headings (config.section_headings): detection, propagation into chunks across
pages, retrieval text, and baselines unaffected."""

import pymupdf
import pytest

from trustqa.config import load_config
from trustqa.generation import format_passages
from trustqa.processing.process import process_pdfs, processing_signature
from trustqa.processing.sections import Heading, find_headings, section_at, section_marks
from trustqa.retrieval.index import index_text
from trustqa.schemas import Chunk, ContentType, RetrievedChunk, TableData

BODY = ("Retrieval systems combine many components for search over large collections. " * 6).strip()
LIST = ("BERT showed contextual representations. T5 framed every task as text to text. "
        "FlashAttention made attention faster. PagedAttention improved memory management. ") * 3


def make_survey(path):
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    y = 60

    def line(text, font="tiro", size=10):
        nonlocal y
        p.insert_text((50, y), text, fontname=font, fontsize=size)
        y += size + 6

    def para(text, font="tiro", size=10):
        nonlocal y
        rect = pymupdf.Rect(50, y - 10, 545, y + 200)
        p.insert_textbox(rect, text, fontname=font, fontsize=size)
        lines = len(text) // 90 + 1
        y += lines * (size + 3) + 10

    line("I. LITERATURE SURVEY")                    # Roman + all caps, regular font
    para(BODY)
    line("A. Core Technologies", font="tiit")       # IEEE subsection: italic
    para("The Transformer uses self attention. " + LIST)
    line("9 Tasks", font="hebo", size=6)            # small bold figure label: not a heading
    line("1. This numbered list item ends with a period.", font="hebo")
    p2 = doc.new_page(width=595, height=842)
    p2.insert_textbox(pymupdf.Rect(50, 50, 545, 300), LIST, fontname="tiro", fontsize=10)
    p2.insert_text((50, 330), "2.1", fontname="hebo", fontsize=10)   # NeurIPS: number line
    p2.insert_text((50, 344), "Models", fontname="hebo", fontsize=10)  # ... then title line
    p2.insert_textbox(pymupdf.Rect(50, 352, 545, 600), BODY, fontname="tiro", fontsize=10)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def survey(tmp_path):
    return make_survey(tmp_path / "survey.pdf")


def test_find_headings_styles_and_exclusions(survey):
    doc = pymupdf.open(survey)
    p1 = [h.text for h in find_headings(doc[0])]
    assert p1 == ["I. LITERATURE SURVEY", "A. Core Technologies"]  # no figure label / list item
    p2 = find_headings(doc[1])
    assert [(h.text, h.level) for h in p2] == [("2.1 Models", 2)]


def test_section_marks_hierarchy_and_carry():
    text = "I. INTRO body. A. Core Technologies list here. B. Other more."
    hs = [Heading("I. INTRO", 1), Heading("A. Core Technologies", 2), Heading("B. Other", 2)]
    marks, stack = section_marks(hs, text, [])
    assert section_at(marks, 0) == "I. INTRO"
    assert section_at(marks, text.index("list")) == "I. INTRO > A. Core Technologies"
    assert section_at(marks, len(text) - 1) == "I. INTRO > B. Other"
    carried, _ = section_marks([], "next page text", stack)
    assert carried == ((0, "I. INTRO > B. Other"),)  # next page starts in the open section


def test_chunks_carry_sections_across_pages(survey):
    on = load_config(ocr_enabled=False, section_headings=True, chunk_size=300, chunk_overlap=0)
    chunks = process_pdfs([survey], on)
    by_page = {}
    for c in chunks:
        by_page.setdefault(c.page_number, []).append(c)
    # a chunk of the list that does not itself contain the heading still knows its section
    listed = [c for c in by_page[1] if "PagedAttention" in c.text and "Core Technologies" not in c.text]
    assert listed and all(c.section == "I. LITERATURE SURVEY > A. Core Technologies" for c in listed)
    assert by_page[2][0].section == "I. LITERATURE SURVEY > A. Core Technologies"  # carried
    # level-2 "2.1" replaces "A." under the still-open level-1 "I." (mixed styles in this fixture)
    assert by_page[2][-1].section == "I. LITERATURE SURVEY > 2.1 Models"
    # chunk text is untouched by the switch
    off = process_pdfs([survey], load_config(ocr_enabled=False, chunk_size=300, chunk_overlap=0))
    assert [c.text for c in off] == [c.text for c in chunks]
    assert all(c.section is None for c in off)


def test_index_text_and_prompt_show_section_only_for_text_chunks():
    t = Chunk("DOC01", "a.pdf", 1, "DOC01_P1_C2", "BERT showed ...", section="A. Core Technologies")
    assert index_text(t) == "A. Core Technologies\nBERT showed ..."
    plain = Chunk("DOC01", "a.pdf", 1, "DOC01_P1_C1", "plain text")
    assert index_text(plain) == "plain text"
    tab = Chunk("DOC01", "a.pdf", 1, "DOC01_P1_T1_R1", "Table 1 | x=1", content_type=ContentType.TABLE,
                table_data=TableData("DOC01_P1_T1", ["x"], [["1"]]), section="Table 1")
    assert index_text(tab) == "Table 1 | x=1"
    prompt = format_passages([RetrievedChunk(t, 0.5, 1), RetrievedChunk(plain, 0.4, 2)])
    assert "(source: a.pdf, page 1, section: A. Core Technologies)" in prompt
    assert "[DOC01_P1_C1] (source: a.pdf, page 1)\n" in prompt  # unchanged without a section


def test_switch_changes_signature_and_baselines_force_it_off():
    on = load_config(section_headings=True)
    assert "sections-v1" in processing_signature(on)
    assert "sections-v1" not in processing_signature(load_config(section_headings=False))
    assert load_config(section_headings=True, baseline_b=True).section_headings is False
    a = load_config(section_headings=True, ocr_enabled=False, flatten_tables=True, verify=False)
    assert a.section_headings is False and processing_signature(a) == "pymupdf-text-v1"


def test_gate_lexical_pass_needs_every_content_word():
    from trustqa.verification.abstention import content_terms, retrieval_gate

    sec = Chunk("DOC01", "a.pdf", 1, "DOC01_P1_C4", "BERT showed contextual representations.",
                section="I. LITERATURE SURVEY > A. Core Technologies")
    other = Chunk("DOC01", "a.pdf", 2, "DOC01_P2_C1", "Salary tables are not in this survey.")
    low = [RetrievedChunk(sec, 0.21, 1), RetrievedChunk(other, 0.18, 2)]
    q = "list the core technologies"
    assert content_terms(q) == {"core", "technologies"}
    assert retrieval_gate(low, 0.30, q) is not None                      # off by default
    assert retrieval_gate(low, 0.30, q, lexical_pass=True) is None        # section has both words
    assert retrieval_gate(low, 0.30, "What are the core technolgies?", lexical_pass=True) is None  # typo
    assert retrieval_gate(low, 0.30, "What was the CEO salary in 2023?", lexical_pass=True)
    assert retrieval_gate(low, 0.30, "technologies?", lexical_pass=True)  # one word is not enough
    assert retrieval_gate([], 0.30, q, lexical_pass=True) == "no passages were retrieved"
