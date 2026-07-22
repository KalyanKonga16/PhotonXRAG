"""
PhotonX Copilot - Centralized logging setup.

Every module in the app (app.py, rag_engine.py, metrics.py, ingest.py) pulls
its logger from here via get_logger(__name__), so the whole request
lifecycle -- app boot, resource loading, retrieval, generation, RAGAS
scoring, errors -- lands in ONE rotating log file with consistent
formatting, instead of being scattered or only visible in the terminal
Streamlit was launched from.

Design choice (worth knowing if you're wondering "why isn't this dumped
into the Streamlit UI?"):
  Full logs (stack traces, per-step timings, retrieval internals) go to
  disk only. That's deliberate -- end users of a chat UI shouldn't see
  raw tracebacks or internal timing spam, and a chat transcript is a bad
  place for unbounded log volume. The UI instead gets a small, curated,
  colourful metrics line per answer (see metrics.py + app.py) which is
  the "public" summary of what these logs capture in full underneath.
  For convenience while developing/demoing, app.py also offers an
  optional, OFF-by-default sidebar expander that tails the last N lines
  of this same log file -- so nothing is duplicated, it just gives you a
  peek at the background log without leaving the browser.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "photonx.log"

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call on every Streamlit rerun (module-level
    code in app.py re-executes each rerun, but we only want one set of
    handlers attached, or lines would be logged multiple times)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("photonx")
    root.setLevel(level)
    root.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler -- background, persistent, the source of truth.
    # 5 x 2MB files is plenty for a small app and keeps disk use bounded,
    # which matters on Streamlit Cloud's ephemeral/limited storage.
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Console handler too -- useful when running `streamlit run app.py`
    # locally and watching the terminal, and it's what shows up in
    # Streamlit Cloud's "Manage app" log panel for free.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Quiet down noisy third-party loggers so they don't drown out our
    # own lifecycle events at INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.info("Logging configured. Writing to %s", LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """Call this at the top of any module: logger = get_logger(__name__)"""
    configure_logging()
    return logging.getLogger(f"photonx.{name}")


def tail_log(n_lines: int = 60) -> str:
    """Used by the optional sidebar log viewer in app.py. Reads the last
    n_lines of the current log file. Returns '' if nothing logged yet."""
    if not LOG_FILE.exists():
        return ""
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return ""
