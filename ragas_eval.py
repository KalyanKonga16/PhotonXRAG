"""
PhotonX RAG - RAGAS Evaluation Harness
---------------------------------------
Evaluates rag_engine.py against the 6 text-based RAG metrics from
docs.ragas.io/en/stable/concepts/metrics/available_metrics/:

  1. Faithfulness              - is the answer grounded in retrieved context?
  2. Response Relevancy        - does the answer address the question?
  3. Context Precision         - are relevant chunks ranked near the top?
  4. Context Recall            - did retrieval surface what the reference needs?
  5. Context Entities Recall   - are key reference entities present in context?
  6. Noise Sensitivity         - does irrelevant/noisy context cause wrong claims?

(Multimodal Faithfulness / Multimodal Relevance are intentionally excluded -
rag_engine.py's ingestion is text-only, so there's no image context to score.)

RAGAS defaults to OpenAI. This project has no OpenAI dependency, so:
  - Judge LLM  -> Groq's openai/gpt-oss-120b (Groq retired
    llama-3.3-70b-versatile on 2026-06-17; gpt-oss-120b is its replacement)
  - Embeddings -> the same BAAI/bge-base-en-v1.5 model rag_engine.py already
    uses for retrieval

Usage:
    export GROQ_API_KEY=your_key_here
    python ragas_eval.py                      # uses eval_dataset.json
    python ragas_eval.py my_other_set.json     # custom eval file
"""

import os
import sys
import json

import pandas as pd
from dotenv import load_dotenv

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
)

from rag_engine import load_resources, retrieve, generate_answer_stream, EMBED_MODEL_NAME

load_dotenv()

DEFAULT_EVAL_DATASET_PATH = "eval_dataset.json"
DEFAULT_RESULTS_CSV_PATH = "ragas_results.csv"

# Groq deprecated llama-3.3-70b-versatile on 2026-06-17. gpt-oss-120b is the
# recommended replacement; override with RAGAS_LLM_MODEL if you use something else.
RAGAS_LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", "llama-3.3-70b-versatile")

# All 6 metrics. Add more later (e.g. if you bring in reference-free variants)
# by just extending this list - run_evaluation() doesn't care how many there are.
METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    ContextEntityRecall(),
    NoiseSensitivity(),
]


def _get_full_answer(query: str, chunks: list[dict], chat_history: list[dict] | None = None) -> str:
    """rag_engine.generate_answer_stream() yields tokens for the Streamlit
    UI; RAGAS needs the complete string, so just join the stream here."""
    chat_history = chat_history or []
    return "".join(generate_answer_stream(query, chunks, chat_history))


def load_eval_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, row in enumerate(data):
        if "question" not in row or "ground_truth" not in row:
            raise ValueError(
                f"Row {i} in {path} is missing 'question' or 'ground_truth'. "
                "Both are needed for context_recall / context_precision / "
                "context_entity_recall / noise_sensitivity."
            )
    return data


def build_ragas_dataset(res, questions: list[dict]) -> Dataset:
    """Runs the real pipeline (retrieve -> generate) for every eval question
    and assembles the {question, contexts, answer, ground_truth} table that
    ragas.evaluate() expects."""
    rows = {"question": [], "contexts": [], "answer": [], "ground_truth": []}

    for item in questions:
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  -> retrieving + answering: {q!r}")

        chunks = retrieve(res, q)
        contexts = [c["text"] for c in chunks] if chunks else [""]
        answer = (
            _get_full_answer(q, chunks)
            if chunks
            else "I couldn't find anything relevant to that in the indexed PhotonX documents."
        )

        rows["question"].append(q)
        rows["contexts"].append(contexts)
        rows["answer"].append(answer)
        rows["ground_truth"].append(gt)

    return Dataset.from_dict(rows)


def get_ragas_llm() -> LangchainLLMWrapper:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY isn't set. All 6 metrics (faithfulness, "
            "response_relevancy, context_precision, context_recall, "
            "context_entity_recall, noise_sensitivity) need a judge LLM."
        )
    chat = ChatGroq(model=RAGAS_LLM_MODEL, api_key=api_key, temperature=0)
    return LangchainLLMWrapper(chat)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    # Reuses the exact embedding model rag_engine.py indexes/retrieves with,
    # so context_precision / context_recall stay consistent with how the
    # app itself judges "closeness" of text.
    hf = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return LangchainEmbeddingsWrapper(hf)


def run_evaluation(
    questions_path: str = DEFAULT_EVAL_DATASET_PATH,
    metrics: list | None = None,
    save_csv: str | None = DEFAULT_RESULTS_CSV_PATH,
):
    metrics = metrics or METRICS

    print("Loading RAG resources (embedder, reranker, chroma collection)...")
    res = load_resources()

    print(f"Loading eval questions from {questions_path} ...")
    questions = load_eval_questions(questions_path)

    print(f"Running pipeline over {len(questions)} eval question(s)...")
    dataset = build_ragas_dataset(res, questions)

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    # Some ragas versions read llm/embeddings off each metric instance
    # instead of (or in addition to) the evaluate() kwargs below - setting
    # both keeps this working across ragas releases.
    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    print(f"Running RAGAS evaluate() over {len(metrics)} metrics...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    df = result.to_pandas()
    pd.set_option("display.max_colwidth", 60)

    print("\n=== Per-question scores ===")
    print(df)

    print("\n=== Aggregate scores ===")
    for metric in metrics:
        name = getattr(metric, "name", str(metric))
        if name in df.columns:
            print(f"{name:>40}: {df[name].mean():.4f}")

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\nSaved detailed per-question results to {save_csv}")

    return result, df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_DATASET_PATH
    run_evaluation(questions_path=path)
