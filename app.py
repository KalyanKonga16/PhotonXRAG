"""
PhotonX Copilot - Streamlit Interface
A polished, chat-first landing experience over the hybrid RAG engine in rag_engine.py.
"""

import base64
import html
from pathlib import Path

import streamlit as st
from rag_engine import load_resources, ask
from llm_metrics import JUDGE_MODEL, METRICS, score_answer

LOGO_PATH = Path(__file__).parent / "assets" / "photonx-logo.png"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")

st.set_page_config(
    page_title="PhotonX Copilot",
    page_icon=str(LOGO_PATH),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-deep: #0B0F19;
        --bg-panel: #131826;
        --bg-panel-hover: #171E30;
        --border: #232A3D;
        --accent-amber: #F2A93B;
        --accent-cyan: #4DD8E8;
        --text-primary: #EDEFF5;
        --text-muted: #8A93A6;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stApp {
        background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(242,169,59,0.10), transparent),
                    radial-gradient(ellipse 60% 40% at 85% 15%, rgba(77,216,232,0.08), transparent),
                    var(--bg-deep);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 3rem; max-width: 760px; }

    h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }

    /* Hero */
    .hero-wrap { text-align: center; margin-bottom: 2.2rem; }
    .hero-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 52px; height: 52px; border-radius: 14px;
        background: var(--bg-panel);
        box-shadow: 0 0 32px rgba(242,169,59,0.35);
        margin-bottom: 14px;
        overflow: hidden;
        animation: pulse-glow 3.5s ease-in-out infinite;
    }
    .hero-mark img { width: 100%; height: 100%; object-fit: cover; display: block; }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 24px rgba(242,169,59,0.30); }
        50% { box-shadow: 0 0 40px rgba(77,216,232,0.35); }
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 700; margin: 0 0 6px 0;
        background: linear-gradient(90deg, #fff 40%, var(--accent-amber) 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero-sub { color: var(--text-muted); font-size: 0.98rem; margin: 0; }

    /* Suggestion buttons */
    div[data-testid="stButton"] > button {
        width: 100%; text-align: left; white-space: normal;
        background: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        padding: 14px 16px !important;
        min-height: 76px;
        transition: all 0.18s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent-amber) !important;
        background: var(--bg-panel-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(242,169,59,0.12);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 4px 6px;
        margin-bottom: 10px;
        animation: rise-in 0.35s ease;
    }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sources -- a single collapsed expander instead of a row of dead-end
       chips. Opening it shows the actual excerpt each answer was pulled
       from, which is the practical version of "click through to that part
       of the document" given the source is a local .docx with no hosted
       page to deep-link to. */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: var(--bg-panel) !important;
        margin-top: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.76rem !important;
        color: var(--text-muted) !important;
    }
    .source-entry { margin-bottom: 10px; }
    .source-entry:last-child { margin-bottom: 0; }
    .source-heading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem; color: var(--accent-cyan);
        display: block; margin-bottom: 3px;
    }
    .source-excerpt {
        font-size: 0.85rem; color: var(--text-muted);
        line-height: 1.5; margin: 0;
    }

    div[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; }

    /* Per-answer RAGAS scores, computed by the judge-LLM call in
       llm_metrics.py and rendered under the reply they belong to.
       Deliberately quiet -- a metadata strip, not a second headline
       competing with the answer. */
    .score-strip {
        display: flex; flex-wrap: wrap; gap: 7px; align-items: center;
        margin-top: 9px;
    }
    .score-chip {
        display: inline-flex; align-items: baseline; gap: 6px;
        border: 1px solid var(--border); border-radius: 999px;
        background: var(--bg-panel); padding: 3px 11px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        color: var(--text-muted);
    }
    .score-chip b { font-weight: 500; font-size: 0.76rem; }
    .score-good b { color: #5FD08A; }
    .score-mid  b { color: var(--accent-amber); }
    .score-low  b { color: #E86A6A; }
    /* Marks the metric where a low number is the good outcome, so a green
       0.05 next to a green 0.95 doesn't read as a bug. */
    .score-chip i {
        font-style: normal; font-size: 0.62rem; color: var(--text-muted);
        opacity: 0.75;
    }
    .score-note { font-family: 'JetBrains Mono', monospace;
                  font-size: 0.68rem; color: var(--text-muted); }

    /* The judge's own justification per metric, inside the expander. */
    .why-row {
        display: grid; grid-template-columns: 1fr auto;
        gap: 2px 10px; margin-bottom: 12px;
    }
    .why-label { font-size: 0.8rem; color: var(--text-primary); }
    .why-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem; color: var(--accent-amber);
    }
    .why-bar {
        grid-column: 1 / -1; height: 4px; border-radius: 2px;
        background: var(--border); overflow: hidden;
    }
    .why-bar-fill {
        height: 100%; border-radius: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-amber));
    }
    .why-reason {
        grid-column: 1 / -1;
        font-size: 0.76rem; color: var(--text-muted);
        line-height: 1.5; margin: 3px 0 0 0;
    }
    .why-foot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; color: var(--text-muted); line-height: 1.7;
        margin-top: 2px; padding-top: 9px; border-top: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTED_QUESTIONS = [
    ("\U0001F4BC", "What services does PhotonX offer?"),
    ("\U0001F916", "What kind of AI work has PhotonX done?"),
    ("\U0001F91D", "How does PhotonX's engagement model work?"),
    ("\U0001F4C1", "What are some recent PhotonX projects?"),
]

# ---------------------------------------------------------------------------
# Resources (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up the copilot...")
def get_resources():
    try:
        return load_resources()
    except RuntimeError:
        # First run on a fresh deploy (Streamlit Cloud, HF Spaces, etc.) -- the
        # container has the repo's source_docs/ but no chroma_db/ yet, since
        # that's generated output, not something we commit. Build it once,
        # here, instead of requiring a manual `python ingest.py` step that's
        # easy to forget after every redeploy.
        import ingest
        with st.spinner("First run on this deployment: indexing PhotonX documents..."):
            ingest.run(source_dir=ingest.SOURCE_DIR)
        return load_resources()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "score_answers" not in st.session_state:
    # On by default; the sidebar toggle is the escape hatch when Groq starts
    # rate-limiting, since scoring adds one request per question.
    st.session_state.score_answers = True

with st.sidebar:
    st.toggle(
        "Score every answer",
        key="score_answers",
        help="Scores all six RAGAS metrics on each reply using one extra "
             "judge-LLM call. Adds a couple of seconds per question.",
    )


def queue_question(q: str):
    st.session_state.pending_query = q


def render_sources(sources: list[dict]):
    """One collapsed expander; opening it shows the excerpt each source
    contributed, so clicking actually surfaces the relevant document
    content instead of linking nowhere."""
    if not sources:
        return
    label = "Source" if len(sources) == 1 else "Sources"
    with st.expander(f"{label} ({len(sources)})"):
        parts = []
        for s in sources:
            heading = html.escape(s["label"][:80])
            excerpt = html.escape(s["excerpt"])
            parts.append(
                f'<div class="source-entry">'
                f'<span class="source-heading">{heading}</span>'
                f'<p class="source-excerpt">{excerpt}</p>'
                f"</div>"
            )
        st.markdown("".join(parts), unsafe_allow_html=True)


def _score_tone(value: float, direction: str) -> str:
    """Green/amber/red for one score. Noise Sensitivity is the one metric where
    low is the good outcome, so its thresholds are inverted rather than
    reusing the higher-is-better bands and colouring a good answer red."""
    if direction == "lower":
        return "score-good" if value <= 0.15 else ("score-mid" if value <= 0.30 else "score-low")
    return "score-good" if value >= 0.85 else ("score-mid" if value >= 0.70 else "score-low")


def render_answer_scores(scores: dict | None):
    """All six RAGAS metrics for the one answer directly above, as judged by
    llm_metrics.score_answer.

    Two tiers on purpose: the chip strip is always visible so every answer
    carries its own numbers, and the expander holds the judge's stated reason
    for each one -- which is the part that makes a score arguable rather than
    something you either trust or don't."""
    if not scores:
        return
    if scores.get("error"):
        st.markdown(
            f'<p class="score-note">Answer scoring unavailable: '
            f'{html.escape(str(scores["error"]))}</p>',
            unsafe_allow_html=True,
        )
        return

    values = scores.get("scores") or {}
    reasons = scores.get("reasons") or {}
    if not values:
        return

    chips = []
    for key, label, direction in METRICS:
        val = values.get(key)
        if val is None:
            continue
        arrow = ' <i>&darr;</i>' if direction == "lower" else ""
        chips.append(
            f'<span class="score-chip {_score_tone(val, direction)}">'
            f'{html.escape(label)} <b>{val:.2f}</b>{arrow}</span>'
        )
    if not chips:
        return

    st.markdown(
        '<div class="score-strip">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"How accurate is this? — RAGAS scores ({len(chips)} metrics)"):
        parts = []
        for key, label, direction in METRICS:
            val = values.get(key)
            if val is None:
                continue
            # Every metric is defined on 0-1; clamp anyway so an out-of-range
            # value from the judge can't blow the bar past its track.
            pct = max(0.0, min(1.0, val)) * 100
            hint = "lower is better" if direction == "lower" else "higher is better"
            reason = reasons.get(key)
            parts.append(
                f'<div class="why-row">'
                f'<span class="why-label">{html.escape(label)}</span>'
                f'<span class="why-score">{val:.2f}</span>'
                f'<div class="why-bar"><div class="why-bar-fill" style="width:{pct:.1f}%"></div></div>'
                + (f'<p class="why-reason">{html.escape(reason)}</p>' if reason else "")
                + f'<p class="why-reason">({hint})</p>'
                f"</div>"
            )
        parts.append(
            '<div class="why-foot">'
            f'Judged by {html.escape(JUDGE_MODEL)} in one call, scored from this '
            'answer and the context retrieved for it.<br/>'
            'Context Recall and Context Entity Recall have no human reference '
            'answer to compare against on a live question, so they are the '
            'judge&rsquo;s estimate of what a complete answer would need.'
            "</div>"
        )
        st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero (only before the first message)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-mark"><img src="data:image/png;base64,{LOGO_B64}" alt="PhotonX" /></div>
            <p class="hero-title">PhotonX Copilot</p>
            <p class="hero-sub">Ask anything about our services, projects, or how we work \u2014 answered straight from the source.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, (icon, question) in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            st.button(
                f"{icon}  {question}",
                key=f"suggest_{i}",
                on_click=queue_question,
                args=(question,),
            )

# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = str(LOGO_PATH) if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))
        render_answer_scores(msg.get("scores"))

# ---------------------------------------------------------------------------
# Input (typed or from a suggestion click)
# ---------------------------------------------------------------------------
typed_query = st.chat_input("Ask PhotonX Copilot...")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=str(LOGO_PATH)):
        try:
            resources = get_resources()
            chunks, stream = ask(resources, query, st.session_state.messages[:-1])
            full_answer = st.write_stream(stream)

            seen_keys, sources = set(), []
            for c in chunks:
                meta = c["metadata"]
                heading = meta.get("h2") or meta.get("h1") or ""
                key = (meta.get("source"), heading)
                if key not in seen_keys:
                    seen_keys.add(key)
                    label = meta.get("title", "Document")
                    if heading:
                        label += f" — {heading}"
                    excerpt = c["text"].strip().replace("\n", " ")
                    if len(excerpt) > 280:
                        excerpt = excerpt[:280].rsplit(" ", 1)[0] + "…"
                    sources.append({"label": label, "excerpt": excerpt})

            render_sources(sources)

            # Scored only after the answer is on screen, so the judge call
            # never delays the reply itself. Stored on the message so
            # Streamlit's reruns replay it instead of re-billing.
            scores = None
            if st.session_state.get("score_answers", True) and chunks:
                with st.spinner("Scoring this answer..."):
                    scores = score_answer(
                        question=query,
                        contexts=[c["text"] for c in chunks],
                        answer=full_answer,
                    ).as_dict()
                render_answer_scores(scores)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                    "scores": scores,
                }
            )
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
