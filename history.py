"""
Council history — local SQLite archive of past runs.

Pure stdlib (sqlite3, json, re). No external deps, works offline.
DB lives next to this file: council_history.db
"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "council_history.db"

# Mode-midpoint estimate matching the help text in app.py
# (scout ≈ ₹2-3 · verdict ≈ ₹7-8 · deep ≈ ₹15-17).
COST_BY_MODE = {"scout": 2.5, "verdict": 7.5, "deep": 16.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS councils (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    chairman_model TEXT NOT NULL,
    advisor_models_json TEXT NOT NULL,
    advisor_responses_json TEXT NOT NULL,
    reviews_json TEXT NOT NULL,
    letter_map_json TEXT NOT NULL,
    chairman_verdict TEXT NOT NULL,
    cost_estimate REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_councils_timestamp ON councils(timestamp DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def cost_for_mode(mode: str) -> float:
    return COST_BY_MODE.get(mode, 0.0)


def save_council(result: dict, timestamp: datetime | None = None) -> int:
    """
    Persist a completed council run. `result` matches the dict produced by
    app.py's _run_pipeline (or the equivalent assembled in council.py main).

    Required keys: question, mode, topic, chairman_model, advisor_models,
                   advisor_responses, reviews, letter_map, chairman_verdict.
    Returns the new row id.
    """
    ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
    cost = cost_for_mode(result["mode"])

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO councils (
                timestamp, topic_slug, mode, question, chairman_model,
                advisor_models_json, advisor_responses_json, reviews_json,
                letter_map_json, chairman_verdict, cost_estimate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                result["topic"],
                result["mode"],
                result["question"],
                result["chairman_model"],
                json.dumps(result["advisor_models"]),
                json.dumps(result["advisor_responses"]),
                json.dumps(result["reviews"]),
                json.dumps(result["letter_map"]),
                result["chairman_verdict"],
                cost,
            ),
        )
        return cur.lastrowid


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("advisor_models", "advisor_responses", "reviews", "letter_map"):
        d[k] = json.loads(d.pop(f"{k}_json"))
    return d


def list_councils(limit: int | None = None) -> list[dict]:
    """Newest first."""
    sql = "SELECT * FROM councils ORDER BY timestamp DESC, id DESC"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    with _connect() as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def get_council(council_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM councils WHERE id = ?", (council_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None


def delete_council(council_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM councils WHERE id = ?", (council_id,))


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM councils")


# =============================================================================
# SIMILARITY (keyword overlap, Jaccard on stopword-filtered word sets)
# =============================================================================

# Common English stopwords + a few domain-specific ones. Kept small on purpose;
# Jaccard on full prose is already noisy enough without aggressive filtering.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "can", "may", "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "she",
    "him", "her", "his", "hers", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "out", "up", "down", "into",
    "over", "under", "again", "further", "once", "here", "there",
    # domain-ish — these appear in the boilerplate of most council questions
    "council", "this", "context", "question", "decision",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase word set, stopwords removed. Keeps 2-letter tokens because
    domain acronyms (CA, RL, ML, AI) carry the most signal."""
    words = re.findall(r"[a-zA-Z][a-zA-Z']*", text.lower())
    return {w for w in words if len(w) >= 2 and w not in _STOPWORDS}


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    """Containment / overlap coefficient: |A∩B| / min(|A|,|B|).
    Reads as 'what fraction of the shorter question's keywords appear in
    the longer one' — closer to a human's 'how similar do these feel'
    than Jaccard, which gets dragged down by length differences."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def find_similar(question: str, threshold: float = 0.6) -> list[tuple[float, dict]]:
    """
    Return councils whose question keyword-overlaps with `question` at >=
    threshold, sorted most-similar first. Uses overlap coefficient on
    stopword-filtered word sets. Each entry is (similarity, row_dict).
    """
    q_tokens = _tokenize(question)
    # Need enough keywords for a match to be meaningful — otherwise a
    # 2-word query trivially "100% matches" any stored question.
    if len(q_tokens) < 4:
        return []
    out: list[tuple[float, dict]] = []
    for row in list_councils():
        sim = _overlap_coeff(q_tokens, _tokenize(row["question"]))
        if sim >= threshold:
            out.append((sim, row))
    out.sort(key=lambda x: -x[0])
    return out
