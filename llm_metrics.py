"""
PhotonX RAG - per-answer RAGAS-style scoring via a single judge-LLM prompt
--------------------------------------------------------------------------
Replaces the previous `ragas` + `datasets` + `langchain-groq` stack. That
stack was the reason the deployed app printed "ragas unavailable
(ModuleNotFoundError)": those packages pull ~2GB of transitive dependencies
into a Streamlit Community Cloud container that cannot hold them, so the pip
install failed while the app itself still booted.

Everything here runs on `groq`, which the app already depends on for answer
generation. Zero new packages, nothing to install, nothing that can go
missing at import time.

WHAT IS MEASURED
----------------
One judge call scores the metrics RAGAS reports, from the same
(question, retrieved_contexts, answer) triple the user just produced:

  Faithfulness          higher better - claims in the answer traceable to context
  Answer Relevancy      higher better - answer actually addresses the question
  Context Precision     higher better - retrieved contexts that were worth retrieving
  Context Recall        higher better - needed information the context actually contains
  Context Entity Recall higher better - entities in play that appear in the context
  Answer Correctness    higher better - agreement with a known-correct answer

TWO MODES, AND THE DIFFERENCE MATTERS
-------------------------------------
`score_answer(...)` without a `reference` is the LIVE mode used for a real
user's question in the chat. Faithfulness, Answer Relevancy and Context
Precision are reference-free by definition and are measured as specified.
Context Recall and Context Entity Recall are NOT reference-free in real
RAGAS - they are defined against a human-written reference answer, which a
live question does not have. In live mode the judge estimates them against
"what a complete answer to this question would need to contain". Directional
indicators, not the textbook metric.

Answer Correctness genuinely cannot be estimated the same way: it exists to
check the answer against ground truth, so scoring it against a reference
drafted in the same breath as reading the actual answer just measures
self-agreement, not correctness - the reference silently anchors to whatever
answer it already saw. To make it meaningful in live mode too, score_answer
makes ONE EXTRA judge call first (_synthesize_reference) that drafts a
reference answer from ONLY the question and contexts - it is never shown the
system's actual answer, so it can't be contaminated by it. That blind
reference is then used exactly like a human-written one for scoring. This
costs one extra Groq round-trip per live question; every other metric's
definition and behavior is unchanged from before this existed.

`score_answer(..., reference="...")` is the BENCHMARK mode used by
evaluate.py over eval_dataset.json. A human-written reference already exists
there, so the blind-synthesis call is skipped entirely and the reference is
used directly - same scoring logic either way once a reference exists.

The prompt forces the judge to enumerate and count (claims supported / claims
total) rather than emit a gut-feel number. That is what keeps a single LLM
call behaving like a metric instead of a mood ring, and it is why every score
comes back with the counts that produced it.

Fail-safe by construction: any failure - missing key, rate limit, malformed
JSON - returns a JudgeScores carrying an `error` string. It never raises into
the chat flow, because a broken score must not cost the user their answer.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field

from groq import Groq

# Same model family that writes the answers. Overridable so the judge can be
# pointed at a different (e.g. cheaper, or deliberately independent) model
# without touching the answer path.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")

# Long contexts make the judge call slow and push it toward the token ceiling.
# Faithfulness only needs enough context to verify the answer's claims.
MAX_CONTEXTS = 6
MAX_CONTEXT_CHARS = 4000

# (json key, UI label, "higher" | "lower", needs_reference)
# `needs_reference` metrics need an actual REFERENCE to be scored - human
# (benchmark mode) or blind-synthesized (live mode, see _synthesize_reference
# below). Either way the metric is only meaningful once a real reference
# exists to compare against; it is never scored against thin air.
METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("faithfulness", "Faithfulness", "higher", False),
    ("answer_relevancy", "Answer Relevancy", "higher", False),
    ("context_precision", "Context Precision", "higher", False),
    ("context_recall", "Context Recall", "higher", False),
    ("context_entity_recall", "Context Entity Recall", "higher", False),
    ("answer_correctness", "Answer Correctness", "higher", True),
)

LIVE_METRICS = tuple(m for m in METRICS if not m[3])


def metrics_for(has_reference: bool) -> tuple[tuple[str, str, str, bool], ...]:
    """The metric table applicable to one scoring mode. `has_reference` is
    True whenever a REFERENCE will actually be given to the judge - whether
    it's human-written (benchmark mode) or blind-synthesized (live mode)."""
    return METRICS if has_reference else LIVE_METRICS


_PROMPT_HEAD = """\
You are a strict RAG evaluation judge. You score one retrieval-augmented answer \
on {n} metrics and reply with JSON only.

Score every metric by COUNTING, never by impression. For each metric, first \
identify the discrete items it is defined over, then count how many qualify, \
then divide. Report both counts alongside the score so the arithmetic is \
checkable. If a metric has zero items to count, set its score to null and say \
why in its reason.

METRIC DEFINITIONS - follow these exactly.

1. faithfulness (higher is better)
   Break the ANSWER into atomic factual claims. A claim is supported if it can \
be inferred from the CONTEXTS alone. Do not use outside knowledge - a claim \
that is true in the real world but absent from the contexts is UNSUPPORTED.
   score = supported_claims / total_claims

2. answer_relevancy (higher is better)
   Break the ANSWER into statements. A statement is relevant if it addresses \
the QUESTION as asked. Padding, hedging, restating the question, and correct \
but off-topic detail are all irrelevant. If the answer leaves a directly asked \
sub-question unaddressed, cap the score at 0.7.
   score = relevant_statements / total_statements

3. context_precision (higher is better)
   The CONTEXTS are given best-first, in the order the retriever ranked them. \
A context is useful if it contributed information a correct answer to the \
QUESTION needs. Weight earlier ranks more heavily: a useless context at rank 1 \
costs more than a useless one at the last rank.
   score = rank-weighted useful_contexts / total_contexts
"""

# Live mode: no human reference exists, so recall is judged against the
# judge's own reconstruction of what a complete answer would need. Unchanged
# from the original design - answer_correctness is deliberately NOT defined
# here, because it is only ever scored once a real reference exists (see
# _synthesize_reference + _DEFS_REFERENCE_BASED below).
_DEFS_REFERENCE_FREE = """\
4. context_recall (higher is better)
   There is no reference answer. Infer what a complete, correct answer to the \
QUESTION would have to state, as a list of required points. A point is covered \
if the CONTEXTS contain it.
   score = covered_points / required_points

5. context_entity_recall (higher is better)
   List the specific entities the QUESTION and ANSWER turn on - proper nouns, \
product and service names, organisations, figures, dates. Generic words are not \
entities. An entity is present if it appears in the CONTEXTS.
   score = entities_present / entities_total
"""

# Benchmark mode: a REFERENCE is supplied, so recall is measured against it as
# RAGAS defines it, and correctness becomes measurable.
_DEFS_REFERENCE_BASED = """\
4. context_recall (higher is better)
   Break the REFERENCE into atomic claims. A claim is attributable if the \
CONTEXTS contain it. This measures retrieval, not the answer - ignore the \
ANSWER entirely for this metric.
   score = attributable_reference_claims / total_reference_claims

5. context_entity_recall (higher is better)
   List the specific entities in the REFERENCE - proper nouns, product and \
service names, organisations, figures, dates. Generic words are not entities. \
An entity is present if it appears in the CONTEXTS.
   score = reference_entities_present_in_contexts / reference_entities_total

6. answer_correctness (higher is better)
   Compare the ANSWER against the REFERENCE. Classify every claim as TP (in \
both), FP (in the answer, absent from or contradicting the reference), or FN \
(required by the reference, missing from the answer). Wording may differ freely \
- judge meaning, not phrasing.
   score = TP / (TP + 0.5 * (FP + FN))
   If the REFERENCE states the documents cannot answer the question, then a \
correct ANSWER is one that declines to answer; an answer that confidently \
states facts anyway scores 0.0.
"""

_PROMPT_TAIL = """\
Reply with this JSON object and nothing else. Every "score" is a number from \
0.0 to 1.0, or null. Every "reason" is one sentence, under 25 words, and must \
cite the counts.

{{
{lines}
}}
"""


def _build_system_prompt(has_reference: bool) -> str:
    table = metrics_for(has_reference)
    width = max(len(k) for k, _, _, _ in table) + 3
    lines = ",\n".join(
        f'  {(chr(34) + k + chr(34) + ":"):<{width}} {{"score": 0.0, "reason": ""}}'
        for k, _, _, _ in table
    )
    return (
        _PROMPT_HEAD.format(n=len(table))
        + (_DEFS_REFERENCE_BASED if has_reference else _DEFS_REFERENCE_FREE)
        + "\n"
        + _PROMPT_TAIL.format(lines=lines)
    )


@dataclass
class JudgeScores:
    """`error` is set instead of the scores when anything goes wrong; callers
    render whichever is populated and never have to handle an exception."""

    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.scores)

    def as_dict(self) -> dict:
        """Flat, JSON-serialisable shape for st.session_state, so Streamlit
        replays stored scores on rerun instead of paying for them again."""
        return {"scores": self.scores, "reasons": self.reasons, "error": self.error}


_client_lock = threading.Lock()
_client: Groq | None = None


def _get_client() -> Groq:
    """One Groq client for the process. Streamlit reruns the script on every
    interaction, so building this per call would churn HTTP clients."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                try:
                    import streamlit as st

                    api_key = st.secrets["GROQ_API_KEY"]
                except Exception:
                    raise RuntimeError("GROQ_API_KEY is not configured")
            _client = Groq(api_key=api_key)
    return _client


def _trim(contexts: list[str]) -> list[str]:
    out = []
    for c in contexts[:MAX_CONTEXTS]:
        c = (c or "").strip()
        if c:
            out.append(c[:MAX_CONTEXT_CHARS])
    return out


def _coerce_score(raw) -> float | None:
    """The judge is told to emit numbers, but models occasionally return
    "0.85", "85%", or a bare "null". Accept what is unambiguous, clamp to the
    0-1 range every metric is defined on, reject the rest."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not m:
            return None
        val = float(m.group(0))
        if "%" in raw and val > 1:
            val /= 100.0
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
    if val != val:  # NaN
        return None
    return max(0.0, min(1.0, val))


def _parse_judge_json(content: str) -> dict:
    """JSON mode makes a clean object overwhelmingly likely, but a model that
    wraps it in prose or a ```json fence would otherwise throw away a paid
    call, so fall back to the outermost braces."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


_SYNTHESIS_SYSTEM_PROMPT = """\
You answer questions using ONLY the provided context. Write the single best, \
complete, accurate answer to the QUESTION using only the CONTEXTS - as if you \
were about to answer the user directly. If the CONTEXTS do not contain enough \
to answer, say so plainly instead of guessing. Plain text only, no preamble, \
no JSON, 2-5 sentences."""


def _synthesize_reference(question: str, contexts: list[str]) -> str | None:
    """Drafts a reference answer from ONLY the question and contexts, in a
    call that never sees the system's actual answer. This blind separation is
    the whole point: a reference drafted in the same call as scoring the real
    answer silently anchors to whatever answer it already read, which is why
    an earlier version of this scored answer_correctness near 1.0 regardless
    of the question - it was measuring self-agreement, not correctness.
    Returns None (never raises) on any failure, so a bad synthesis call costs
    that one metric, not the rest of the scoring."""
    context_block = "\n\n".join(f"[Context {i}]\n{c}" for i, c in enumerate(contexts, 1))
    try:
        response = _get_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"QUESTION\n{question.strip()}\n\nCONTEXTS\n{context_block}",
                },
            ],
            temperature=0,
            max_tokens=300,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


def score_answer(
    question: str,
    contexts: list[str],
    answer: str,
    reference: str | None = None,
) -> JudgeScores:
    """Score one answer with a single judge-LLM call. Never raises.

    Pass `reference` (a known-correct answer, as eval_dataset.json supplies) to
    use it directly for benchmark mode. Omit it for a live chat question and a
    reference is drafted automatically via a separate, answer-blind judge call
    (see _synthesize_reference) so Answer Correctness is still meaningful - if
    that call fails for any reason, scoring falls back to the original 5
    reference-free metrics rather than failing the whole score.

    Deliberately takes no embedder and no retrieval objects: the judge reads
    text, so scoring is decoupled from the engine that produced the answer.
    """
    contexts = _trim(contexts)
    if not question.strip() or not answer.strip() or not contexts:
        # No retrieved context means the app already declined to answer -
        # there is nothing meaningful to score.
        return JudgeScores(error="nothing to score")

    reference = (reference or "").strip()
    if not reference:
        reference = _synthesize_reference(question, contexts) or ""
    has_reference = bool(reference)
    table = metrics_for(has_reference)

    context_block = "\n\n".join(
        f"[Context {i}, retrieval rank {i}]\n{c}" for i, c in enumerate(contexts, 1)
    )
    user_prompt = (
        f"QUESTION\n{question.strip()}\n\n"
        f"CONTEXTS ({len(contexts)} retrieved, best-first)\n{context_block}\n\n"
        f"ANSWER\n{answer.strip()}\n\n"
        + (f"REFERENCE (known-correct answer)\n{reference}\n\n" if has_reference else "")
        + f"Score the {len(table)} metrics and reply with the JSON object."
    )

    try:
        response = _get_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _build_system_prompt(has_reference)},
                {"role": "user", "content": user_prompt},
            ],
            # Judging must be reproducible: the same triple should not score
            # differently on a rerun.
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        payload = _parse_judge_json(response.choices[0].message.content or "{}")
    except Exception as e:
        # Includes the message, not just the exception type - the previous
        # implementation reported only `ModuleNotFoundError` with no module
        # name, which made the failure impossible to diagnose from the UI.
        return JudgeScores(error=f"{type(e).__name__}: {e}"[:200])

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for key, _label, _direction, _needs_ref in table:
        entry = payload.get(key)
        if isinstance(entry, dict):
            val = _coerce_score(entry.get("score"))
            reason = str(entry.get("reason") or "").strip()
        else:
            # Tolerate a judge that flattened {"score": x} to a bare number.
            val = _coerce_score(entry)
            reason = ""
        if val is not None:
            scores[key] = val
            if reason:
                reasons[key] = reason

    if not scores:
        return JudgeScores(error="judge returned no usable scores")

    return JudgeScores(scores=scores, reasons=reasons)
