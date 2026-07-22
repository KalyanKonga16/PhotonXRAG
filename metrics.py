"""
PhotonX RAG - Per-answer RAGAS scoring.

Scores every answer the moment it's produced, using only that single
turn's (query, answer, retrieved_contexts) -- deliberately NOT chat
history, for two reasons:
  1. Faithfulness/relevancy are properties of "did THIS answer stay
     grounded in THESE chunks and address THIS question" -- multi-turn
     history doesn't change what's being measured.
  2. RAGAS's single-turn, reference-free metrics (no ground-truth
     answer required) are exactly Faithfulness and ResponseRelevancy,
     which take user_input + response + retrieved_contexts. That's a
     clean match for a live app with no labelled dataset.

Metrics used (both reference-free -- no ground-truth answer needed,
so this works live on real user questions, not just a curated eval set):
  - Faithfulness      : does the answer's content actually follow from
                         the retrieved chunks (i.e. is the model making
                         things up)?
  - Answer Relevancy   : does the answer actually address the question
                         asked (penalizes vague/off-target answers)?

The judge LLM is a small, fast Groq model (separate from the main
answer model) -- good enough for a 1-5 scale grounding/relevance
judgment, and cheap/fast enough to run on every turn without the user
noticing. If it fails or is slow, we fail OPEN: the chat answer already
rendered and streamed before this runs, so a scoring hiccup never
blocks or breaks the actual conversation -- the metrics line just
doesn't appear for that turn (logged, not surfaced as an error).
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from logging_setup import get_logger

logger = get_logger(__name__)

# Separate, smaller/faster model than the main answer LLM (LLM_MODEL_NAME in
# rag_engine.py) -- this is a judgment call ("is this grounded?"), not the
# user-facing answer, so we don't need the bigger 70B model for it.
JUDGE_MODEL_NAME = "llama-3.1-8b-instant"

# Soft ceiling so a slow/unavailable judge call can never hang the app --
# worst case the user just doesn't see a metrics line on that turn.
SCORE_TIMEOUT_SECONDS = 8.0

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragas-score")


@dataclass
class TurnScores:
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    duration_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and (
            self.faithfulness is not None or self.answer_relevancy is not None
        )


class _SentenceTransformerRagasEmbeddings:
    """Thin adapter so RAGAS's ResponseRelevancy metric can use the SAME
    already-loaded SentenceTransformer (BAAI/bge-base-en-v1.5) that
    rag_engine.py uses for retrieval, instead of pulling in a second
    embeddings dependency/model download just for scoring."""

    def __init__(self, sentence_transformer):
        self._model = sentence_transformer

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    def set_run_config(self, run_config):
        pass


@lru_cache(maxsize=1)
def _get_judge_and_metrics(_embedder_id: int, embedder):
    """Builds the RAGAS judge LLM + metric instances once and reuses them
    across every query (mirrors how rag_engine.load_resources() caches
    the embedder/reranker) -- constructing these per-turn would be
    wasteful and slower for no benefit.

    _embedder_id is part of the cache key purely so this rebuilds if a
    different embedder object is ever passed in; in practice it's the
    same singleton for the app's whole lifetime.
    """
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set -- cannot build RAGAS judge LLM.")

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=JUDGE_MODEL_NAME, api_key=api_key, temperature=0.0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        _SentenceTransformerRagasEmbeddings(embedder)
    )

    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings)

    logger.info("RAGAS judge initialized (model=%s)", JUDGE_MODEL_NAME)
    return faithfulness, answer_relevancy


def _score_sync(query: str, answer: str, contexts: list[str], embedder) -> TurnScores:
    from ragas.dataset_schema import SingleTurnSample

    start = time.perf_counter()
    try:
        faithfulness, answer_relevancy = _get_judge_and_metrics(id(embedder), embedder)

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
        )

        f_score = faithfulness.single_turn_score(sample)
        r_score = answer_relevancy.single_turn_score(sample)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "RAGAS scored turn | faithfulness=%.2f relevancy=%.2f (%.0fms) | query=%r",
            f_score, r_score, duration_ms, query[:80],
        )
        return TurnScores(faithfulness=f_score, answer_relevancy=r_score, duration_ms=duration_ms)

    except Exception as e:  # fail OPEN -- never break the chat over a scoring error
        duration_ms = (time.perf_counter() - start) * 1000
        logger.warning("RAGAS scoring failed after %.0fms: %s", duration_ms, e)
        return TurnScores(faithfulness=None, answer_relevancy=None, duration_ms=duration_ms, error=str(e))


def score_turn(query: str, answer: str, contexts: list[str], embedder) -> Optional[TurnScores]:
    """Public entry point, called from app.py right after an answer has
    finished streaming. Runs the (network-bound) scoring call on a
    worker thread with a hard timeout, so a slow/hung judge call degrades
    to "no metrics line this turn" rather than freezing the UI.
    """
    if not contexts:
        return None  # nothing retrieved -- faithfulness/relevancy aren't meaningful
    try:
        future = _executor.submit(_score_sync, query, answer, contexts, embedder)
        return future.result(timeout=SCORE_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.warning("RAGAS scoring timed out after %.0fs | query=%r", SCORE_TIMEOUT_SECONDS, query[:80])
        return TurnScores(faithfulness=None, answer_relevancy=None, duration_ms=SCORE_TIMEOUT_SECONDS * 1000, error="timeout")
    except Exception as e:
        logger.warning("RAGAS scoring dispatch failed: %s", e)
        return None
