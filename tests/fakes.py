"""Routed FakeLLM for verification tests: one scripted reply per pipeline step."""

from __future__ import annotations

import json

from trustqa.generation import SYSTEM_PROMPT as GEN_SYSTEM
from trustqa.generation import SYSTEM_PROMPT_V2 as GEN_SYSTEM_V2
from trustqa.llm import FakeLLM, LLMError
from trustqa.verification.conflicts import CONFLICT_SYSTEM_PROMPT
from trustqa.verification.numeric import CALC_SYSTEM_PROMPT
from trustqa.verification.support import SUPPORT_SYSTEM_PROMPT, SUPPORT_SYSTEM_PROMPT_V2

STEPS = {GEN_SYSTEM: "generation", GEN_SYSTEM_V2: "generation", SUPPORT_SYSTEM_PROMPT: "support",
         SUPPORT_SYSTEM_PROMPT_V2: "support",
         CALC_SYSTEM_PROMPT: "calc", CONFLICT_SYSTEM_PROMPT: "conflict"}


def gen(answer="", ids=(), abstain=False):
    return {"answer": answer, "chunk_ids": list(ids), "insufficient_evidence": abstain}


def verdict(claim_id, v="supported", chunk_id="", quote=""):
    return {"claim_id": claim_id, "verdict": v, "chunk_id": chunk_id, "supporting_quote": quote}


def routed(**replies) -> FakeLLM:
    """routed(generation={...}, support={"verdicts": [...]}, calc=..., conflict=...).

    A step without a reply raises LLMError (as a failing provider would). Values may be dicts
    (sent as JSON) or raw strings. `fake.steps` records the order of steps called.
    """
    steps: list[str] = []

    def fn(prompt, system, json_mode):
        step = STEPS.get(system, "unknown")
        steps.append(step)
        if step not in replies:
            raise LLMError(f"no scripted reply for step {step}")
        r = replies[step]
        return r if isinstance(r, str) else json.dumps(r)

    fake = FakeLLM(fn)
    fake.steps = steps
    return fake
