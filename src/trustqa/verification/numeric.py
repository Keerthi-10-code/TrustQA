"""Phase 3 item 4 - numeric checks (FR-11). Matching and arithmetic are plain Python.

Every number, unit and period in a claim must appear in the claim's cited evidence:
- normalisation: thousands separators (Western 1,234,567 and Indian 1,03,450.50), unicode
  minus, "+ 11%" with a space, %, "percent", scale words (thousand, lakh, crore, million,
  billion, mn, bn, k), exponents written 10^19 / 2.3·10^19 / 2.3e19 (Phase 2 table cells use ^);
- signs: an explicitly negative claim value needs a negative evidence value; an unsigned
  claim value may restate a negative one in words ("fell by 2.5%" for -2.5%);
- rounding: a claim may round an evidence value to fewer decimals only if it keeps at least
  two significant digits (0.35 for 0.347 passes, 0.3 does not);
- units: a % claim needs a % evidence value or a chunk that states percent; a claim with a
  scale word matched only through the bare number needs that scale word in the chunk
  (e.g. the table caption "Rs crore");
- periods: FY tokens (FY2024 = FY24 = FY 2023-24) and quarters (Q3) must appear in the
  evidence; plain years are checked as numbers.
Numbers that are part of identifiers (BM25, nDCG@10, T5, FY2024) and reference labels
(Table 2, page 3, Section 4) are not values and are skipped.

Calculations: for a claim number that is not in the evidence, the LLM may propose an
arithmetic expression (strict JSON). It is evaluated here with exact fractions; it passes
only if every operand is a number from the claim's evidence and the result rounds to the
claim value.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction

from trustqa.llm import LLM, LLMError
from trustqa.schemas import Chunk, Warning, WarningCode
from trustqa.verification.claims import Claim

CALC_PROMPT_VERSION = "calc-v1"

_SCALES = {
    "thousand": 10**3, "k": 10**3, "lakh": 10**5, "lakhs": 10**5, "lac": 10**5,
    "crore": 10**7, "crores": 10**7, "cr": 10**7, "million": 10**6, "mn": 10**6,
    "billion": 10**9, "bn": 10**9, "trillion": 10**12,
}
_SCALE_ALT = "|".join(sorted(_SCALES, key=len, reverse=True))
_NUM_RE = re.compile(
    rf"""
    (?P<sign>[+\-−–]\s?)?
    (?P<int>\d{{1,3}}(?:,\d{{2,3}})+(?!\d)|\d+)
    (?P<dec>\.\d+)?
    (?:\s?(?:[x×·*]|⋅)\s?10\^(?P<mexp>[+\-−]?\d+))?
    (?:\^(?P<pow>[+\-−]?\d+))?
    (?:[eE](?P<sci>[+\-]?\d+)\b)?
    (?P<pct>\s?%|\s?(?:percent|per\scent)\b)?
    (?:\s?(?P<scale1>{_SCALE_ALT})\b)?
    (?:\s?(?P<scale2>crores?|cr)\b)?
    """, re.VERBOSE | re.IGNORECASE)
_REF_WORDS = {"table", "tables", "figure", "fig", "figs", "section", "sec", "page", "pages", "p",
              "pp", "eq", "equation", "chapter", "appendix", "step", "rank", "row", "column",
              "chunk", "claim", "reference", "ref", "line", "no", "item"}
_PERIOD_RE = re.compile(
    r"\b(?:(?P<q>[QH][1-4])\s?)?FY\s?'?(?P<y1>\d{2}|\d{4})(?:\s?[-–/]\s?(?P<y2>\d{2}|\d{4}))?\b"
    r"|\b(?P<qa>Q[1-4])\b", re.IGNORECASE)
_MARKER_RE = re.compile(r"\[[^\[\]]*\]")


@dataclass(frozen=True)
class Num:
    raw: str
    base: Decimal  # value before scale words
    scaled: Decimal  # value after scale words
    mult: int  # product of scale words
    decimals: int  # digits after the decimal point as written
    sig_digits: int
    negative: bool
    explicit_sign: bool
    percent: bool
    scale_words: tuple[str, ...]


def extract_numbers(text: str) -> list[Num]:
    """Numeric values in `text` (identifiers, reference labels, periods, markers skipped)."""
    text = _MARKER_RE.sub(lambda m: " " * len(m.group(0)), text)
    text = _PERIOD_RE.sub(lambda m: " " * len(m.group(0)), text)  # FY2023-24 is a period
    out: list[Num] = []
    for m in _NUM_RE.finditer(text):
        start = m.start("int")
        prev = text[start - 1] if start > 0 else " "
        if prev.isalpha() or prev in "@_^":
            continue  # BM25, nDCG@10, x^2
        sign_raw = m.group("sign") or ""
        before = text[m.start() - 1] if m.start() > 0 else " "
        if sign_raw and before.isalpha():
            continue  # COVID-19, FiQA-2018: identifier
        if prev == "." and start >= 2 and text[start - 2].isdigit():
            continue  # "1.2.3": tail of a version-like token
        word = re.search(r"([A-Za-z]+)\.?\s*$", text[max(0, m.start() - 14):m.start()])
        if word and word.group(1).lower() in _REF_WORDS:
            continue
        sign = sign_raw.strip()
        if sign and before.isdigit():
            sign = ""  # "3-5", "2023-24": a range, not a sign
        digits = m.group("int").replace(",", "")
        dec = m.group("dec") or ""
        try:
            base = Decimal(digits + dec)
            if m.group("pow"):
                base = base ** int(m.group("pow").replace("−", "-"))
            if m.group("mexp"):
                base = base * (Decimal(10) ** int(m.group("mexp").replace("−", "-")))
            if m.group("sci"):
                base = base * (Decimal(10) ** int(m.group("sci")))
        except (InvalidOperation, OverflowError):
            continue
        negative = sign in {"-", "−", "–"}
        if negative:
            base = -base
        scales = tuple(x.lower() for x in (m.group("scale1"), m.group("scale2")) if x)
        mult = 1
        for x in scales:
            mult *= _SCALES[x]
        sig = len((digits + dec.lstrip(".")).lstrip("0")) or 1
        out.append(Num(raw=m.group(0).strip(), base=base, scaled=base * mult, mult=mult,
                       decimals=len(dec) - 1 if dec else 0, sig_digits=sig, negative=negative,
                       explicit_sign=bool(sign), percent=bool(m.group("pct")),
                       scale_words=scales))
    return out


def extract_periods(text: str) -> set[str]:
    """Normalised period tokens: 'FY24', 'Q3FY24', 'Q3'."""
    out = set()
    for m in _PERIOD_RE.finditer(text):
        if m.group("qa"):
            out.add(m.group("qa").upper())
            continue
        year = m.group("y2") or m.group("y1")
        tok = "FY" + year[-2:]
        out.add((m.group("q").upper() + tok) if m.group("q") else tok)
        if m.group("q"):
            out.add(m.group("q").upper())
    return out


def _round_eq(evidence: Decimal, claim: Num, value: Decimal) -> bool:
    """evidence == value, or evidence rounds to value at the claim's written precision."""
    if evidence == value:
        return True
    if claim.sig_digits < 2:
        return False
    q = Decimal(1).scaleb(-claim.decimals)
    try:
        return abs(evidence).quantize(q, rounding=ROUND_HALF_UP) == abs(value)
    except InvalidOperation:
        return False


def _chunk_mentions_percent(chunk: Chunk) -> bool:
    blob = chunk.text.lower()
    if chunk.table_data is not None:
        td = chunk.table_data
        blob += " " + " ".join([td.units or "", td.caption or "", *td.headers]).lower()
    return "%" in blob or "percent" in blob or "per cent" in blob


_CTX_STOP = {
    "what", "which", "who", "whom", "how", "much", "many", "is", "are", "was", "were", "does", "do",
    "did", "the", "a", "an", "of", "on", "in", "for", "to", "by", "and", "or", "with", "than",
    "better", "worse", "value", "score", "scores", "reported", "report", "reports", "according",
    "table", "from", "at", "as", "its", "it", "be", "has", "have", "that", "this", "there",
}
_SYNONYMS = {"average": "avg", "averaged": "avg", "mean": "avg"}


def context_tokens(text: str) -> set[str]:
    """Content words of a question/claim/header for table-cell alignment ("FY2024" also
    yields "fy24" and "2024", "FiQA-2018" yields "fiqa" and "2018")."""
    toks: set[str] = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        parts = [t, *(p for p in re.findall(r"[a-z]+|\d+", t) if p != t and (p.isdigit() or len(p) >= 3))]
        toks |= {_SYNONYMS.get(p, p) for p in parts if len(p) >= 2 and p not in _CTX_STOP}
    return toks | {p.lower() for p in extract_periods(text)}


def _aligned_cell_numbers(chunk: Chunk, context: set[str] | None) -> list[Num] | None:
    """For a table-row chunk: the numbers in cells whose column header AND row label share a
    word with the question/claim. None = not a table row (or no context): match on text.

    This catches a right number from the wrong cell (failures.md F5: "0.347" is BM25+CE on
    FiQA-2018, but the question asked for BM25+CE's average gain over BM25, "+ 11%").
    Columns: only the best-matching header(s) count, scored by the share of the header's words
    found in the context ("BM25" in the question matches "Lexical / BM25" (1/2) better than
    "Re-ranking / BM25+CE" (1/4)). If no header shares a word, columns do not constrain the
    match; a column without a header never does. Rows: a row whose text label shares no word
    with the context contributes nothing; a row without a text label does not constrain.
    """
    td = chunk.table_data
    if td is None or context is None or not td.rows:
        return None
    row = td.rows[0]
    label = context_tokens(" ".join(cell for cell in row if cell.strip() and not extract_numbers(cell)))
    if label and not label & context:
        return []
    cols = [(context_tokens(h), extract_numbers(cell)) for h, cell in zip(td.headers, row)]
    cols = [(head, nums) for head, nums in cols if nums]
    scores = [len(head & context) / len(head) for head, _ in cols if head]
    best = max(scores, default=0.0)
    out: list[Num] = []
    for head, nums in cols:
        if not head or best == 0 or len(head & context) / len(head) == best:
            out.extend(nums)
    return out


def number_in_evidence(c: Num, chunks: Sequence[Chunk], context: set[str] | None = None) -> bool:
    """True if `c` occurs in one of `chunks`. With `context` (question + claim words), values in
    table-row chunks count only from cells aligned with that context."""
    for chunk in chunks:
        has_pct = None
        lower = chunk.text.lower()
        cells = _aligned_cell_numbers(chunk, context)
        for e in (cells if cells is not None else extract_numbers(chunk.text)):
            if c.negative and not e.negative:
                continue
            if c.explicit_sign and not c.negative and e.negative:
                continue
            if c.percent and not e.percent:
                if has_pct is None:
                    has_pct = _chunk_mentions_percent(chunk)
                if not has_pct:
                    continue
            if e.percent and c.scale_words:
                continue  # "12 crore" is not "12%"
            if c.scale_words:
                # compare in the claim's units: 1.03 lakh crore vs 1,03,450.50 crore
                if _round_eq(abs(e.scaled) / c.mult, c, abs(c.base)):
                    return True
                if _round_eq(abs(e.base), c, abs(c.base)) and not e.scale_words and all(
                        w.rstrip("s") in lower for w in c.scale_words):
                    return True  # bare table cell; the caption states the scale ("Rs crore")
            elif _round_eq(abs(e.base), c, abs(c.base)) or _round_eq(abs(e.scaled), c, abs(c.base)):
                return True
    return False


@dataclass
class NumericResult:
    claim_id: int
    checked: list[str] = field(default_factory=list)  # raw claim numbers/periods checked
    unmatched: list[str] = field(default_factory=list)  # not in evidence, not derived
    derived: dict[str, str] = field(default_factory=dict)  # raw number -> verified expression
    missing_periods: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unmatched and not self.missing_periods

    @property
    def numeric(self) -> bool:
        return bool(self.checked)


def check_claim_numbers(claim: Claim, chunks: Sequence[Chunk], question: str | None = None
                        ) -> tuple[NumericResult, list[Num]]:
    """Deterministic part: returns the result and the claim numbers not found in evidence.
    With `question`, table-row values must come from cells aligned with question + claim."""
    res = NumericResult(claim_id=claim.claim_id)
    missing: list[Num] = []
    context = context_tokens(f"{question} {claim.text}") if question is not None else None
    for n in extract_numbers(claim.text):
        res.checked.append(n.raw)
        if not number_in_evidence(n, chunks, context):
            missing.append(n)
    periods = extract_periods(claim.text)
    if periods:
        have = set().union(*(extract_periods(c.text) for c in chunks)) if chunks else set()
        for c in chunks:
            if c.table_data is not None:
                have |= extract_periods(" ".join(c.table_data.headers + [c.table_data.caption or ""]))
        res.checked.extend(sorted(periods))
        res.missing_periods = sorted(periods - have)
    return res, missing


# --- calculations ---------------------------------------------------------------------------

# Constants an expression may use without them being in the evidence: percentage conversion
# ("* 100") and growth written as "(a / b - 1) * 100". Every other operand must be evidence.
_CONVERSION_CONSTANTS = {Decimal(1), Decimal(100)}
_ALLOWED_EXPR = re.compile(r"^[\d.\s+\-*/()]+$")


class ExpressionError(ValueError):
    pass


def evaluate_expression(expr: str) -> Fraction:
    """Exactly evaluate + - * / and parentheses over decimal literals (no names, no power)."""
    if not isinstance(expr, str) or not expr.strip() or not _ALLOWED_EXPR.match(expr):
        raise ExpressionError(f"expression has disallowed characters: {expr!r}")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse {expr!r}") from e

    def ev(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return Fraction(ast.get_source_segment(expr, node) or str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if b == 0:
                raise ExpressionError("division by zero")
            return a / b
        raise ExpressionError(f"unsupported syntax in {expr!r}")

    return ev(tree)


def expression_operands(expr: str) -> list[Decimal]:
    return [Decimal(t) for t in re.findall(r"\d+(?:\.\d+)?", expr)]


def check_calculation(claim_num: Num, expr: str, chunks: Sequence[Chunk]) -> str | None:
    """None if `expr` uses only evidence numbers and reproduces `claim_num`; else the reason."""
    try:
        value = evaluate_expression(expr)
    except ExpressionError as e:
        return str(e)
    evidence = [abs(e.base) for c in chunks for e in extract_numbers(c.text)]
    evidence += [abs(e.scaled) for c in chunks for e in extract_numbers(c.text)]
    for op in expression_operands(expr):
        if op not in evidence and op not in _CONVERSION_CONSTANTS:
            return f"operand {op} is not a number in the cited evidence"
    if value in {Fraction(str(op)) for op in expression_operands(expr)}:
        # "0.347", "0.347 * 1": restating a passage number is not a calculation, and would
        # bypass the table-cell alignment rule (seen live on failures.md F5)
        return f"{expr} only restates a number from the passages"
    result = Decimal(value.numerator) / Decimal(value.denominator)
    target = abs(claim_num.base)
    candidates = [abs(result)]
    if claim_num.percent:
        candidates.append(abs(result) * 100)
    q = Decimal(1).scaleb(-claim_num.decimals)
    for r in candidates:
        if r.quantize(q, rounding=ROUND_HALF_UP) == target:
            return None
    return f"{expr} = {result.normalize():f}, not {claim_num.raw}"


CALC_SYSTEM_PROMPT = """Some numbers in an answer do not appear word for word in its source passages. For each listed number, decide whether it can be calculated from numbers that DO appear in the passages (for example a sum, difference, ratio or percentage change).

If it can, give an arithmetic expression that uses only + - * / and parentheses, and only numbers copied exactly from the passages (plus 1 and 100 for percentages) (digits and decimal point only: no commas, units, % signs or words). If it cannot, use null.

Return a JSON object: {"calculations": [{"id": integer, "expression": string or null}]}"""


def _calc_prompt(items: list[tuple[int, Claim, Num]], chunks: dict[str, Chunk]) -> str:
    used: list[str] = list(dict.fromkeys(cid for _, cl, _ in items for cid in cl.chunk_ids))
    passages = "\n\n".join(f"[{cid}]\n{chunks[cid].text}" for cid in used if cid in chunks)
    lines = "\n".join(f"{i}: {n.raw}  (in: \"{cl.text}\")" for i, cl, n in items)
    return f"Passages:\n\n{passages}\n\nNumbers to explain:\n{lines}\n\nAnswer in JSON as instructed."


def check_numbers(claims: Sequence[Claim], chunks: dict[str, Chunk], llm: LLM | None = None,
                  question: str | None = None
                  ) -> tuple[dict[int, NumericResult], list[Warning]]:
    """Run the numeric check on every claim. `llm` (optional) is asked only for numbers that
    are not in the evidence; its expressions are verified here, never trusted. `question`
    enables the table-cell alignment rule (see _aligned_cell_numbers)."""
    results: dict[int, NumericResult] = {}
    pending: list[tuple[int, Claim, Num]] = []
    for cl in claims:
        scope = [chunks[c] for c in cl.chunk_ids if c in chunks]
        res, missing = check_claim_numbers(cl, scope, question)
        results[cl.claim_id] = res
        for n in missing:
            pending.append((len(pending) + 1, cl, n))

    exprs: dict[int, str] = {}
    calc_error = None
    if pending and llm is not None:
        try:
            raw = llm.complete(_calc_prompt(pending, chunks), system=CALC_SYSTEM_PROMPT, json=True)
            data = json.loads(raw)
            for item in data.get("calculations", []) if isinstance(data, dict) else []:
                if isinstance(item, dict) and isinstance(item.get("expression"), str):
                    try:
                        exprs[int(item.get("id"))] = item["expression"]
                    except (TypeError, ValueError):
                        continue
        except (LLMError, json.JSONDecodeError) as e:
            calc_error = str(e)

    warnings: list[Warning] = []
    for i, cl, n in pending:
        res = results[cl.claim_id]
        scope = [chunks[c] for c in cl.chunk_ids if c in chunks]
        expr = exprs.get(i)
        reason = check_calculation(n, expr, scope) if expr else None
        if expr and reason is None:
            res.derived[n.raw] = expr
            continue
        res.unmatched.append(n.raw)
        detail = f"; proposed calculation rejected: {reason}" if expr else (
            f"; calculation check unavailable: {calc_error}" if calc_error else "")
        warnings.append(Warning(
            WarningCode.NUMERIC_MISMATCH,
            f"claim {cl.claim_id}: value '{n.raw}' does not appear in the cited evidence "
            f"({', '.join(cl.chunk_ids)}){detail}", chunk_id=cl.chunk_ids[0] if cl.chunk_ids else None))
    for res in results.values():
        for p in res.missing_periods:
            cl = next(c for c in claims if c.claim_id == res.claim_id)
            warnings.append(Warning(
                WarningCode.NUMERIC_MISMATCH,
                f"claim {cl.claim_id}: period '{p}' does not appear in the cited evidence "
                f"({', '.join(cl.chunk_ids)})", chunk_id=cl.chunk_ids[0] if cl.chunk_ids else None))
    return results, warnings
