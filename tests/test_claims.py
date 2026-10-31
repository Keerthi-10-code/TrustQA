"""Phase 3 item 2: claim decomposition."""

import pytest

from trustqa.verification.claims import decompose_claims, is_material, split_sentences

IDS = ["DOC01_P3_C1", "DOC01_P3_C2", "DOC02_P2_C1"]


def test_each_sentence_maps_to_its_own_markers():
    cl = decompose_claims("Revenue was Rs. 1,03,450.50 crore [DOC01_P3_C1]. "
                          "The dam was completed in 1934 [DOC02_P2_C1].", IDS)
    assert [(c.claim_id, c.text, c.chunk_ids, c.explicit) for c in cl] == [
        (1, "Revenue was Rs. 1,03,450.50 crore.", ("DOC01_P3_C1",), True),
        (2, "The dam was completed in 1934.", ("DOC02_P2_C1",), True)]


def test_marker_after_period_belongs_to_previous_sentence():
    cl = decompose_claims("Margin was -2.5%. [DOC01_P3_C2] The dam is old. [DOC02_P2_C1]", IDS)
    assert [c.chunk_ids for c in cl] == [("DOC01_P3_C2",), ("DOC02_P2_C1",)]
    assert cl[0].text == "Margin was -2.5%."


def test_multiple_ids_in_one_marker():
    (c,) = decompose_claims("Revenue rose and margin fell [DOC01_P3_C1, DOC01_P3_C2].", IDS)
    assert c.chunk_ids == ("DOC01_P3_C1", "DOC01_P3_C2")


def test_uncited_sentence_is_checked_against_all_accepted_chunks():
    cl = decompose_claims("Revenue grew strongly. It reached Rs. 5 crore [DOC01_P3_C1].",
                          ["DOC01_P3_C1", "DOC02_P2_C1"])
    assert cl[0].explicit is False and cl[0].chunk_ids == ("DOC01_P3_C1", "DOC02_P2_C1")
    assert cl[1].explicit is True


def test_rejected_ids_do_not_count():
    (c,) = decompose_claims("Revenue grew [DOC09_P1_C1].", ["DOC01_P3_C1"])
    assert c.explicit is False and c.chunk_ids == ("DOC01_P3_C1",)


@pytest.mark.parametrize("text, n", [
    ("Revenue was Rs. 1,03,450.50 crore in 2025.", 1),     # "Rs." and decimals
    ("Lewis et al. 2020 proposed RAG.", 1),                 # "et al."
    ("See Fig. 2 for details. It shows growth.", 2),        # "Fig."
    ("The value is 0.347. Another fact follows.", 2),
    ("e.g. retrievers like DPR. They work.", 2),
])
def test_sentence_split_respects_abbreviations_and_decimals(text, n):
    assert len(split_sentences(text)) == n


def test_boilerplate_is_not_material():
    assert not is_material("Based on the provided passages:")
    assert not is_material("- ")
    assert is_material("Revenue was Rs. 5 crore.")
    cl = decompose_claims("Based on the passages: Revenue grew [DOC01_P3_C1].", IDS)
    assert all(c.material for c in cl)  # one sentence with real content


def test_empty_answer_has_no_claims():
    assert decompose_claims("", IDS) == []
