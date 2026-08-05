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
One judge call scores the six metrics RAGAS reports, from the same
(question, retrieved_contexts, answer) triple the user just produced:

  Faithfulness          higher better - claims in the answer traceable to context
  Answer Relevancy      higher better - answer actually addresses the question
  Context Precision     higher better - retrieved contexts that were worth retrieving
  Context Recall        higher better - needed information the context actually contains
  Context Entity Recall higher better - entities in play that appear in the context
  Noise Sensitivity     LOWER  better - claims the answer got wrong or invented

HONEST CAVEAT, WORTH KNOWING BEFORE YOU TRUST THE NUMBERS
---------------------------------------------------------
Real RAGAS computes Context Recall and Context Entity Recall against a
human-written `reference` answer. A live user's question has no reference, so
those two are judged here against "what a complete answer to this question
would need to contain" as inferred by the judge model. They are directional
indicators, not the textbook metric. Faithfulness, Answer Relevancy, Context
Precision and Noise Sensitivity are reference-free by definition and are
measured as specified.

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

# (json key, UI label, "higher" | "lower")
METRICS: tuple[tuple[str, str, str], ...] = (
    ("faithfulness", "Faithfulness", "higher"),
    ("answer_relevancy", "Answer Relevancy", "higher"),
    ("context_precision", "Context Precision", "higher"),
    ("context_recall", "Context Recall", "higher"),
    ("context_entity_recall", "Context Entity Recall", "higher"),
    ("noise_sensitivity", "Noise Sensitivity", "lower"),
)

_JUDGE_SYSTEM_PROMPT = """\
You are a strict RAG evaluation judge. You score one (question, retrieved \
contexts, answer) triple on six metrics and reply with JSON only.

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

4. context_recall (higher is better)
   There is no human reference answer. Infer what a complete, correct answer to \
the QUESTION would have to state, as a list of required points. A point is \
covered if the CONTEXTS contain it.
   score = covered_points / required_points

5. context_entity_recall (higher is better)
   List the specific entities the QUESTION and ANSWER turn on - proper nouns, \
product and service names, organisations, figures, dates. Generic words are not \
entities. An entity is present if it appears in the CONTEXTS.
   score = entities_present / entities_total

6. noise_sensitivity (LOWER is better)
   Count claims in the ANSWER that are factually wrong given the CONTEXTS, or \
that were plausibly picked up from a context that is irrelevant to the \
QUESTION. This is the answer being led astray by noise in retrieval.
   score = incorrect_or_noise_induced_claims / total_claims

Reply with this JSON object and nothing else. Every "score" is a number from \
0.0 to 1.0, or null. Every "reason" is one sentence, under 25 words, and must \
cite the counts.

{
  "faithfulness":          {"score": 0.0, "reason": ""},
  "answer_relevancy":      {"score": 0.0, "reason": ""},
  "context_precision":     {"score": 0.0, "reason": ""},
  "context_recall":        {"score": 0.0, "reason": ""},
  "context_entity_recall": {"score": 0.0, "reason": ""},
  "noise_sensitivity":     {"score": 0.0, "reason": ""}
}
"""


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


def score_answer(question: str, contexts: list[str], answer: str) -> JudgeScores:
    """Score one answer with a single judge-LLM call. Never raises.

    Deliberately takes no embedder and no retrieval objects: the judge reads
    text, so scoring is decoupled from the engine that produced the answer.
    """
    contexts = _trim(contexts)
    if not question.strip() or not answer.strip() or not contexts:
        # No retrieved context means the app already declined to answer -
        # there is nothing meaningful to score.
        return JudgeScores(error="nothing to score")

    context_block = "\n\n".join(
        f"[Context {i}, retrieval rank {i}]\n{c}" for i, c in enumerate(contexts, 1)
    )
    user_prompt = (
        f"QUESTION\n{question.strip()}\n\n"
        f"CONTEXTS ({len(contexts)} retrieved, best-first)\n{context_block}\n\n"
        f"ANSWER\n{answer.strip()}\n\n"
        "Score the six metrics and reply with the JSON object."
    )

    try:
        response = _get_client().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # Judging must be reproducible: the same triple should not score
            # differently on a rerun.
            temperature=0,
            max_tokens=900,
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
    for key, _label, _direction in METRICS:
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
