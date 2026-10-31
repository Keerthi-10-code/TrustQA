"""LLM judge for answer correctness (long and short answers), with strict JSON output.

The judge is a proposal like every other LLM output: anything but a valid verdict is
recorded as None (no label), never guessed. Its agreement with a human is measured on a
spot-check sample (error_analysis.py writes it; report.py reads the filled-in labels), so
judge-based numbers are reported together with that agreement.
"""

from __future__ import annotations

import json

from trustqa.llm import LLM, LLMError

JUDGE_VERSION = "judge-v2"
VERDICTS = ("correct", "partially_correct", "incorrect")

JUDGE_SYSTEM_PROMPT = (
    "You grade answers to questions about documents. Compare the SYSTEM ANSWER with the "
    "REFERENCE ANSWER only; do not use outside knowledge. "
    "Judge meaning, not wording: synonyms and paraphrases of the reference count as the same "
    "fact. "
    "correct: states the reference's key facts (numbers must match exactly, extra detail is "
    "fine if it does not contradict). partially_correct: some key facts right, others missing. "
    "incorrect: wrong, contradicting, or not an answer. "
    'Return JSON: {"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one sentence>"}'
)


def judge_prompt(question: str, reference: str, answer: str) -> str:
    return (f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\n"
            f"SYSTEM ANSWER: {answer}\n\nGrade the system answer.")


def judge_answer(question: str, reference: str, answer: str, llm: LLM) -> tuple[str | None, str]:
    """(verdict or None, reason). None when the call fails or the output is not valid."""
    try:
        raw = llm.complete(judge_prompt(question, reference, answer),
                           system=JUDGE_SYSTEM_PROMPT, json=True)
        data = json.loads(raw)
    except (LLMError, ValueError) as e:
        return None, f"judge failed: {e}"
    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    if verdict not in VERDICTS:
        return None, f"invalid judge output: {raw[:200]}"
    return verdict, str(data.get("reason", ""))[:500]
