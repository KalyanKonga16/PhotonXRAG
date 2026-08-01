"""
PhotonX RAG - RAGAS Evaluation Against the LIVE Deployment
------------------------------------------------------------
Unlike ragas_eval.py (which calls rag_engine.retrieve()/generate_answer_stream()
directly), this script treats the deployed app as a black box: it drives the
real Streamlit UI at PHOTONX_LIVE_URL with Playwright, scrapes the streamed
answer and the "Sources" expander app.py renders, and feeds THAT into the same
6 RAGAS metrics.

Why this exists: Streamlit apps aren't REST APIs, there's no JSON endpoint to
call. This is the only way to test the actually-hosted environment end-to-end
(catches things a local run can't: stale chroma_db on the host, missing env
vars, cold-start failures, etc.).

Caveats vs. the local ragas_eval.py:
  - app.py truncates each source excerpt to 280 chars before rendering
    (see render_sources() in app.py), so context_precision/context_recall
    here see less text than the local run does. Scores from the two scripts
    are directionally comparable, not numerically identical.
  - Streamlit Community Cloud apps sleep after inactivity. This script
    detects the "this app has gone to sleep" screen and clicks through it,
    but a cold start can take 60-120s.
  - CSS selectors below match app.py's current DOM (data-testid attributes
    Streamlit itself renders, plus app.py's own "source-excerpt" class).
    If app.py's structure changes, update SOURCE_EXCERPT_SELECTOR etc. below.

Install (one-time, in addition to requirements.txt):
    pip install playwright
    playwright install chromium

Usage:
    export GROQ_API_KEY=your_key_here
    export PHOTONX_LIVE_URL=https://photonxrag.streamlit.app/   # optional, this is the default
    python ragas_eval_live.py                     # uses eval_dataset.json
    python ragas_eval_live.py my_other_set.json     # custom eval file
"""

import os
import sys
import json
import time
import re

import pandas as pd
from dotenv import load_dotenv

from datasets import Dataset
from playwright.sync_api import sync_playwright, Page

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

load_dotenv()

LIVE_URL = os.environ.get("PHOTONX_LIVE_URL", "https://photonxrag.streamlit.app/")
DEFAULT_EVAL_DATASET_PATH = "eval_dataset.json"
DEFAULT_RESULTS_CSV_PATH = "ragas_live_results.csv"

RAGAS_LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", "openai/gpt-oss-120b")
# Kept in sync manually with rag_engine.py's EMBED_MODEL_NAME - this script is
# intentionally decoupled from rag_engine.py since the point is testing the
# deployment without depending on local pipeline code.
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

CHAT_INPUT_SELECTOR = '[data-testid="stChatInput"] textarea'
MESSAGE_SELECTOR = '[data-testid="stChatMessage"]'
MARKDOWN_SELECTOR = '[data-testid="stMarkdownContainer"]'
EXPANDER_SELECTOR = '[data-testid="stExpander"]'
SOURCE_EXCERPT_SELECTOR = ".source-excerpt"

WAKE_UP_TEXT_PATTERN = re.compile(r"(gone to sleep|get this app back up|wake.?up)", re.IGNORECASE)
MAX_WAKE_WAIT_S = 180
MAX_ANSWER_WAIT_S = 90
STREAM_POLL_INTERVAL_S = 0.6
STABLE_READS_REQUIRED = 3  # consecutive identical reads before we call the stream "done"

METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    ContextEntityRecall(),
    NoiseSensitivity(),
]


def load_eval_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, row in enumerate(data):
        if "question" not in row or "ground_truth" not in row:
            raise ValueError(f"Row {i} in {path} is missing 'question' or 'ground_truth'.")
    return data


def _wake_app_if_sleeping(page: Page):
    """Streamlit Community Cloud puts idle apps to sleep. Detects the wake
    screen and clicks through it, then waits (generously) for the real UI."""
    try:
        wake_text = page.get_by_text(WAKE_UP_TEXT_PATTERN)
        if wake_text.count() > 0:
            print("  (deployment is asleep - waking it up, this can take a couple minutes...)")
            # The wake button's exact label has varied across Streamlit
            # versions ("Yes, get this app back up!" is the common one).
            wake_button = page.get_by_role("button", name=re.compile("get this app back up", re.IGNORECASE))
            if wake_button.count() > 0:
                wake_button.first.click()
    except Exception:
        pass  # not asleep, or already past that screen - fine either way

    page.wait_for_selector(CHAT_INPUT_SELECTOR, timeout=MAX_WAKE_WAIT_S * 1000)


def _wait_for_stream_to_finish(page: Page, message_locator) -> str:
    """st.write_stream() renders tokens as they arrive. Polls the answer's
    text content until it stops changing for a few consecutive reads."""
    last_text = None
    stable_count = 0
    deadline = time.time() + MAX_ANSWER_WAIT_S

    while time.time() < deadline:
        try:
            current_text = message_locator.locator(MARKDOWN_SELECTOR).first.inner_text()
        except Exception:
            current_text = ""

        if current_text and current_text == last_text:
            stable_count += 1
            if stable_count >= STABLE_READS_REQUIRED:
                return current_text
        else:
            stable_count = 0

        last_text = current_text
        time.sleep(STREAM_POLL_INTERVAL_S)

    print(f"  !! answer didn't stabilize within {MAX_ANSWER_WAIT_S}s, using last read")
    return last_text or ""


def _scrape_sources(message_locator) -> list[str]:
    """Returns the (280-char-truncated) excerpt text for each source chip
    app.py's render_sources() rendered, opening the expander first if needed."""
    expander = message_locator.locator(EXPANDER_SELECTOR)
    if expander.count() == 0:
        return []

    try:
        # Open it (harmless if it's already open) so excerpt text is in the DOM.
        expander.first.get_by_text(re.compile(r"^Sources?\s*\(")).first.click(timeout=3000)
    except Exception:
        pass

    excerpts = expander.first.locator(SOURCE_EXCERPT_SELECTOR).all_inner_texts()
    return [e.strip() for e in excerpts if e.strip()]


def ask_live_app(page: Page, question: str) -> tuple[str, list[str]]:
    """Submits one question to the live UI and returns (answer, contexts)."""
    baseline_count = page.locator(MESSAGE_SELECTOR).count()

    chat_input = page.locator(CHAT_INPUT_SELECTOR)
    chat_input.click()
    chat_input.fill(question)
    chat_input.press("Enter")

    # Wait for both the new user bubble and the new assistant bubble to appear.
    page.wait_for_function(
        f"""() => document.querySelectorAll('{MESSAGE_SELECTOR}').length >= {baseline_count + 2}""",
        timeout=30000,
    )

    messages = page.locator(MESSAGE_SELECTOR)
    assistant_message = messages.nth(messages.count() - 1)

    answer = _wait_for_stream_to_finish(page, assistant_message)
    contexts = _scrape_sources(assistant_message)
    return answer, contexts


def build_dataset_from_live_app(questions: list[dict]) -> Dataset:
    rows = {"question": [], "contexts": [], "answer": [], "ground_truth": []}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for item in questions:
            q = item["question"]
            gt = item["ground_truth"]
            print(f"  -> asking live app: {q!r}")

            # Fresh context per question -> fresh Streamlit session, so one
            # question's chat history never leaks into the next question's
            # answer (app.py passes prior messages to the LLM as follow-up context).
            context = browser.new_context()
            page = context.new_page()
            try:
                page.goto(LIVE_URL, wait_until="domcontentloaded", timeout=60000)
                _wake_app_if_sleeping(page)
                answer, contexts = ask_live_app(page, q)
            except Exception as e:
                print(f"     !! failed on this question, skipping it: {e}")
                context.close()
                continue
            context.close()

            rows["question"].append(q)
            rows["contexts"].append(contexts if contexts else [""])
            rows["answer"].append(answer)
            rows["ground_truth"].append(gt)

        browser.close()

    return Dataset.from_dict(rows)


def get_ragas_llm() -> LangchainLLMWrapper:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY isn't set - needed for all 6 judge-LLM-based metrics.")
    chat = ChatGroq(model=RAGAS_LLM_MODEL, api_key=api_key, temperature=0)
    return LangchainLLMWrapper(chat)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    hf = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return LangchainEmbeddingsWrapper(hf)


def run_live_evaluation(questions_path: str = DEFAULT_EVAL_DATASET_PATH, save_csv: str | None = DEFAULT_RESULTS_CSV_PATH):
    print(f"Target deployment: {LIVE_URL}")
    print(f"Loading eval questions from {questions_path} ...")
    questions = load_eval_questions(questions_path)

    print(f"Driving the live app for {len(questions)} question(s) (this is real browser automation, be patient)...")
    dataset = build_dataset_from_live_app(questions)

    if len(dataset) == 0:
        print("No questions succeeded against the live app - nothing to score. Check the errors above.")
        return None

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()
    for m in METRICS:
        if hasattr(m, "llm"):
            m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    print(f"\nRunning RAGAS evaluate() over {len(METRICS)} metrics on {len(dataset)} live sample(s)...")
    result = evaluate(dataset=dataset, metrics=METRICS, llm=llm, embeddings=embeddings)
    df = result.to_pandas()

    pd.set_option("display.max_colwidth", 60)
    print("\n=== Per-question scores (live deployment) ===")
    print(df)

    print("\n=== Aggregate scores (live deployment) ===")
    for metric in METRICS:
        name = getattr(metric, "name", str(metric))
        if name in df.columns:
            print(f"{name:>40}: {df[name].mean():.4f}")

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\nSaved detailed results to {save_csv}")

    return df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_DATASET_PATH
    run_live_evaluation(questions_path=path)
