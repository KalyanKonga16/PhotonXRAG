"""
PhotonX RAG - per-answer RAGAS scoring
----------------------------------------
Scores a single (question, retrieved_contexts, answer) triple with the two
RAGAS metrics that need no ground-truth reference, so they can run on a real
user's question the moment the answer finishes streaming:

  - Faithfulness       - every claim in the answer traceable to the retrieved
                         context? (this is the anti-hallucination number)
  - Response Relevancy - does the answer actually address what was asked?

The other four metrics in ragas_eval_live.py (context precision, context
recall, context entity recall, noise sensitivity) all compare against a
`reference` answer. A live user's question doesn't have one, so they stay in
the CI report card over eval_dataset.json and cannot be computed here.

Two deliberate choices keep this affordable inside a ~1GB Streamlit container:

  1. The embedding model is NOT re-loaded. Response Relevancy needs embeddings,
     and the app already holds BAAI/bge-base-en-v1.5 in memory for retrieval -
     `_SentenceTransformerEmbeddings` adapts that same live object to the
     interface ragas expects, so nothing extra is paged in.
  2. Nothing is imported until the first score is requested. A deploy that
     never gets a question never pays the ragas import cost.

Everything here is fail-safe by construction: any failure (missing key, rate
limit, ragas version drift, import error) returns a LiveScores carrying an
`error` string. It never raises into the chat flow - a broken score must not
cost the user their answer.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass

# Groq's per-minute limits are the real ceiling here: scoring one answer costs
# roughly 3-4 extra LLM calls on top of the answer itself.
JUDGE_MODEL = os.environ.get("LIVE_METRICS_MODEL", "llama-3.3-70b-versatile")

# Long contexts make the judge calls slow and expensive. Faithfulness only
# needs enough context to verify the answer's claims.
MAX_CONTEXTS = 6
MAX_CONTEXT_CHARS = 4000


@dataclass
class LiveScores:
    """`error` is set instead of the scores when anything goes wrong; callers
    render whichever is populated and never have to handle an exception."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (
            self.faithfulness is not None or self.answer_relevancy is not None
        )


class _SentenceTransformerEmbeddings:
    """LangChain-shaped embeddings interface over the SentenceTransformer the
    app already loaded for retrieval.

    ragas calls whichever of the sync/async pairs its version prefers, so all
    four are provided. The async ones just delegate - encoding is CPU-bound,
    so there is nothing to await."""

    def __init__(self, model, query_prefix: str = ""):
        self._model = model
        self._query_prefix = query_prefix

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [self._query_prefix + text], normalize_embeddings=True
        )
        return vec[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


_judge_lock = threading.Lock()
_judge_llm = None


def _get_judge_llm():
    """One ChatGroq client for the process. Streamlit reruns the script on
    every interaction, so building this per call would churn HTTP clients."""
    global _judge_llm
    with _judge_lock:
        if _judge_llm is None:
            from langchain_groq import ChatGroq

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                try:
                    import streamlit as st

                    api_key = st.secrets["GROQ_API_KEY"]
                except Exception:
                    raise RuntimeError("GROQ_API_KEY is not configured")

            _judge_llm = ChatGroq(model=JUDGE_MODEL, api_key=api_key, temperature=0)
    return _judge_llm


def _trim(contexts: list[str]) -> list[str]:
    out = []
    for c in contexts[:MAX_CONTEXTS]:
        c = (c or "").strip()
        if c:
            out.append(c[:MAX_CONTEXT_CHARS])
    return out


def score_answer(question: str, contexts: list[str], answer: str, embedder) -> LiveScores:
    """Score one answer. `embedder` is the app's live SentenceTransformer.

    Returns a LiveScores; never raises."""
    contexts = _trim(contexts)
    if not question.strip() or not answer.strip() or not contexts:
        # No retrieved context means the app already declined to answer -
        # there is nothing meaningful to score.
        return LiveScores(error="nothing to score")

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except Exception as e:
        return LiveScores(error=f"ragas unavailable ({type(e).__name__})")

    try:
        from rag_engine import BGE_QUERY_PREFIX
    except Exception:
        BGE_QUERY_PREFIX = ""

    try:
        llm = LangchainLLMWrapper(_get_judge_llm())
        embeddings = LangchainEmbeddingsWrapper(
            _SentenceTransformerEmbeddings(embedder, BGE_QUERY_PREFIX)
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        faith = Faithfulness(llm=llm)
        relev = ResponseRelevancy(llm=llm, embeddings=embeddings)

        async def _run():
            # return_exceptions so one failing metric still lets the other
            # through - a rate limit on the claim-extraction call shouldn't
            # blank out relevancy too.
            return await asyncio.gather(
                faith.single_turn_ascore(sample),
                relev.single_turn_ascore(sample),
                return_exceptions=True,
            )

        results = asyncio.run(_run())
    except Exception as e:
        return LiveScores(error=f"{type(e).__name__}: {e}"[:160])

    def _clean(v):
        # ragas returns NaN rather than raising when it can't score a sample
        # (e.g. an answer it couldn't decompose into claims).
        if isinstance(v, BaseException) or v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if v != v else max(0.0, min(1.0, v))

    scores = LiveScores(
        faithfulness=_clean(results[0]),
        answer_relevancy=_clean(results[1]),
    )
    if scores.faithfulness is None and scores.answer_relevancy is None:
        first_exc = next((r for r in results if isinstance(r, BaseException)), None)
        scores.error = (
            f"{type(first_exc).__name__}: {first_exc}"[:160]
            if first_exc
            else "metrics returned no score"
        )
    return scores
