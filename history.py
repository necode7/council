"""
Council history — durable archive of past runs.

Storage backend is chosen at runtime:
  · TURSO_DATABASE_URL + TURSO_AUTH_TOKEN env vars present → libSQL/Turso (cloud)
  · Otherwise → local sqlite3 file at council_history.db

Both backends share the exact same SQL schema (libSQL is SQLite-compatible).
"""

import json
import os
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
    cost_estimate REAL NOT NULL,
    interrupts_json TEXT NOT NULL DEFAULT '[]',
    clarifications_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_councils_timestamp ON councils(timestamp DESC);
"""

# Older local DBs may exist without the two new JSON columns. Add them
# idempotently on connect so upgrades don't require manual migration.
_BACKFILL_COLUMNS = [
    ("interrupts_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("clarifications_json", "TEXT NOT NULL DEFAULT '[]'"),
]


# =============================================================================
# CONNECTION (Turso if credentials present, else local sqlite)
# =============================================================================


def _turso_creds() -> tuple[str, str] | None:
    url = os.environ.get("TURSO_DATABASE_URL", "").strip()
    token = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
    if url and token:
        return url, token
    return None


def backend() -> str:
    """Returns 'turso' or 'sqlite' — useful for diagnostic UI."""
    return "turso" if _turso_creds() else "sqlite"


def _connect():
    """Open a connection to whichever backend is configured."""
    creds = _turso_creds()
    if creds:
        url, token = creds
        # libsql-experimental ships a sqlite3-like API; only available on PyPI,
        # so import lazily to avoid breaking environments that don't need it.
        import libsql_experimental as libsql
        conn = libsql.connect(database=url, auth_token=token)
    else:
        conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    # Backfill columns on pre-existing local DBs so historic rows still load.
    _ensure_columns(conn)
    return conn


def _ensure_columns(conn) -> None:
    """Add any missing columns from _BACKFILL_COLUMNS. Idempotent."""
    try:
        cur = conn.execute("PRAGMA table_info(councils)")
        existing = {row[1] for row in cur.fetchall()}
    except Exception:
        return
    for name, decl in _BACKFILL_COLUMNS:
        if name not in existing:
            try:
                conn.execute(f"ALTER TABLE councils ADD COLUMN {name} {decl}")
            except Exception:
                # Column race or backend quirk — schema in CREATE TABLE will
                # cover fresh installs anyway.
                pass


# =============================================================================
# BACKEND-AGNOSTIC HELPERS
# =============================================================================


def _rows_to_dicts(cursor) -> list[dict]:
    """Map cursor results to dicts using cursor.description. Works on both
    sqlite3 and libsql-experimental (neither uses sqlite3.Row by default)."""
    if cursor.description is None:
        return []
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# =============================================================================
# PUBLIC API
# =============================================================================


def cost_for_mode(mode: str) -> float:
    return COST_BY_MODE.get(mode, 0.0)


def save_council(result: dict, timestamp: datetime | None = None) -> int:
    """
    Persist a completed council run. `result` matches the dict produced by
    app.py's _run_pipeline (or the equivalent assembled in council.py main).

    Required keys: question, mode, topic, chairman_model, advisor_models,
                   advisor_responses, reviews, letter_map, chairman_verdict.
    Optional: interrupts (list[dict]), clarifications (list[dict]).
    Returns the new row id.
    """
    ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
    cost = cost_for_mode(result["mode"])
    interrupts = result.get("interrupts", []) or []
    clarifications = result.get("clarifications", []) or []

    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO councils (
                timestamp, topic_slug, mode, question, chairman_model,
                advisor_models_json, advisor_responses_json, reviews_json,
                letter_map_json, chairman_verdict, cost_estimate,
                interrupts_json, clarifications_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id""",
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
                json.dumps(interrupts),
                json.dumps(clarifications),
            ),
        )
        row = cur.fetchone()
        new_id = row[0] if row else None
        conn.commit()
        return int(new_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _decode(d: dict) -> dict:
    """Decode JSON columns and rename keys to drop the _json suffix."""
    out = dict(d)
    for k in (
        "advisor_models",
        "advisor_responses",
        "reviews",
        "letter_map",
        "interrupts",
        "clarifications",
    ):
        raw = out.pop(f"{k}_json", None)
        out[k] = json.loads(raw) if raw else ([] if k in ("interrupts", "clarifications") else {})
    return out


def list_councils(limit: int | None = None) -> list[dict]:
    """Newest first."""
    sql = "SELECT * FROM councils ORDER BY timestamp DESC, id DESC"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    conn = _connect()
    try:
        cur = conn.execute(sql, params)
        return [_decode(d) for d in _rows_to_dicts(cur)]
    finally:
        conn.close()


def get_council(council_id: int) -> dict | None:
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM councils WHERE id = ?", (council_id,))
        rows = _rows_to_dicts(cur)
        return _decode(rows[0]) if rows else None
    finally:
        conn.close()


def delete_council(council_id: int) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM councils WHERE id = ?", (council_id,))
        conn.commit()
    finally:
        conn.close()


def clear_all() -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM councils")
        conn.commit()
    finally:
        conn.close()


# =============================================================================
# SIMILARITY (overlap coefficient, stopword-filtered word sets)
# =============================================================================

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
    "council", "context", "question", "decision",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase word set, stopwords removed. Keeps 2-letter tokens because
    domain acronyms (CA, RL, ML, AI) carry the most signal."""
    words = re.findall(r"[a-zA-Z][a-zA-Z']*", text.lower())
    return {w for w in words if len(w) >= 2 and w not in _STOPWORDS}


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    """Containment / overlap coefficient: |A∩B| / min(|A|,|B|)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def find_similar(question: str, threshold: float = 0.6) -> list[tuple[float, dict]]:
    q_tokens = _tokenize(question)
    if len(q_tokens) < 4:
        return []
    out: list[tuple[float, dict]] = []
    for row in list_councils():
        sim = _overlap_coeff(q_tokens, _tokenize(row["question"]))
        if sim >= threshold:
            out.append((sim, row))
    out.sort(key=lambda x: -x[0])
    return out
