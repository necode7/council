"""
Council history — durable archive of past runs.

Storage backend is chosen at runtime:
  · TURSO_DATABASE_URL + TURSO_AUTH_TOKEN env vars present → Turso (cloud,
    over HTTPS using the Hrana HTTP API; no native dependencies, just httpx)
  · Otherwise → local sqlite3 file at council_history.db

Both backends share the exact same SQL schema (Turso is SQLite-compatible).
"""

import base64
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import httpx

DB_PATH = Path(__file__).parent / "council_history.db"

# Mode-midpoint estimate matching the help text in app.py
# (scout ≈ ₹2-3 · verdict ≈ ₹7-8 · deep ≈ ₹15-17).
COST_BY_MODE = {"scout": 2.5, "verdict": 7.5, "deep": 16.0}

SCHEMA_STATEMENTS = [
    """
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
    )
    """.strip(),
    "CREATE INDEX IF NOT EXISTS idx_councils_timestamp ON councils(timestamp DESC)",
]

# Older local DBs may exist without the two new JSON columns. Add them
# idempotently on connect so upgrades don't require manual migration.
_BACKFILL_COLUMNS = [
    ("interrupts_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("clarifications_json", "TEXT NOT NULL DEFAULT '[]'"),
]


# =============================================================================
# CONNECTION (Turso over HTTPS if creds present, else local sqlite)
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
        conn = _TursoConn(url, token)
    else:
        conn = sqlite3.connect(str(DB_PATH))
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
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
                pass


# =============================================================================
# TURSO HRANA-OVER-HTTP CLIENT (sqlite3-compatible thin shim)
# =============================================================================
#
# Turso speaks the "Hrana" protocol. The HTTP variant accepts batched JSON
# requests at /v2/pipeline (or /v3/pipeline on newer servers). We wrap it
# in a sqlite3-shaped Connection / Cursor so the rest of history.py doesn't
# need to know which backend is in play. No external dependency: just httpx
# (already in requirements.txt).
#
# Docs: https://github.com/tursodatabase/libsql/blob/main/docs/HRANA_3_SPEC.md


class TursoError(Exception):
    """Base class for Turso-related failures. UI catches this and renders
    via st.error so Streamlit doesn't redact the message."""


class TursoHTTPError(TursoError):
    def __init__(self, status: int, endpoint: str, token_preview: str, body_preview: str):
        self.status = status
        self.endpoint = endpoint
        self.token_preview = token_preview
        self.body_preview = body_preview
        super().__init__(
            f"Turso HTTP {status} from {endpoint}\n"
            f"Auth token (masked): {token_preview}\n"
            f"Response body: {body_preview}"
        )


class TursoHranaError(TursoError):
    def __init__(self, code: str, message: str, endpoint: str):
        self.code = code
        self.message = message
        self.endpoint = endpoint
        super().__init__(
            f"Turso Hrana error{f' [{code}]' if code else ''} at {endpoint}: {message}"
        )


def _encode_value(v):
    """Python value → Hrana value object."""
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": "1" if v else "0"}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray)):
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode("ascii")}
    return {"type": "text", "value": str(v)}


def _decode_value(v):
    """Hrana value object → Python value."""
    t = v.get("type")
    if t == "null":
        return None
    if t == "integer":
        # Hrana sends integers as strings to preserve full 64-bit range.
        return int(v.get("value", "0"))
    if t == "float":
        return float(v.get("value", 0.0))
    if t == "text":
        return v.get("value", "")
    if t == "blob":
        return base64.b64decode(v.get("base64", ""))
    return v.get("value")


class _TursoCursor:
    """Subset of sqlite3.Cursor: fetchone, fetchall, description, lastrowid."""

    def __init__(self, result: dict | None):
        if result is None:
            self._cols = []
            self._rows = []
            self.lastrowid = None
            self.description = None
            return
        self._cols = result.get("cols", []) or []
        rows_raw = result.get("rows", []) or []
        self._rows = [
            tuple(_decode_value(cell) for cell in row) for row in rows_raw
        ]
        self._idx = 0
        last = result.get("last_insert_rowid")
        self.lastrowid = int(last) if last not in (None, "") else None
        # PEP-249 description: 7-tuple (name, type_code, display_size,
        # internal_size, precision, scale, null_ok). We only fill name.
        self.description = (
            [(c.get("name", ""), None, None, None, None, None, None)
             for c in self._cols]
            if self._cols else None
        )

    def fetchone(self):
        if not getattr(self, "_rows", None) or self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def fetchall(self):
        rows = self._rows[self._idx:]
        self._idx = len(self._rows)
        return rows


class _TursoConn:
    """Minimum sqlite3-shaped connection. No transactions: every execute
    is auto-committed by Turso (the Hrana pipeline opens, runs, closes)."""

    def __init__(self, url: str, token: str):
        # Accept libsql:// or https://; Hrana lives at /vN/pipeline either way.
        host = url.replace("libsql://", "https://").rstrip("/")
        self._host = host
        self._token_preview = (token[:6] + "…" + token[-4:]) if len(token) > 14 else "<short>"
        # Try v2 first, then v3 if v2 says "not found". Initial value is v2;
        # _post() upgrades on a fresh-pipeline 404/405.
        self._endpoint = f"{host}/v2/pipeline"
        self._tried_v3 = False
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=30.0)

    def execute(self, sql: str, params: tuple | list = ()) -> _TursoCursor:
        stmt = {"sql": sql}
        if params:
            stmt["args"] = [_encode_value(p) for p in params]
        result = self._post([
            {"type": "execute", "stmt": stmt},
            {"type": "close"},
        ])
        return _TursoCursor(result)

    def commit(self) -> None:
        # Each pipeline auto-commits; no-op for compat.
        pass

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _post(self, requests: list) -> dict | None:
        body = {"requests": requests}
        resp = self._client.post(self._endpoint, headers=self._headers, json=body)

        # Endpoint discovery: if v2 isn't found, transparently retry on v3.
        if (resp.status_code in (404, 405)) and not self._tried_v3:
            self._tried_v3 = True
            self._endpoint = f"{self._host}/v3/pipeline"
            resp = self._client.post(self._endpoint, headers=self._headers, json=body)

        if resp.status_code != 200:
            # Surface enough to diagnose without leaking the auth token.
            body_preview = resp.text[:1000] if resp.text else "<empty body>"
            raise TursoHTTPError(
                status=resp.status_code,
                endpoint=self._endpoint,
                token_preview=self._token_preview,
                body_preview=body_preview,
            )

        try:
            data = resp.json()
        except Exception as e:
            raise TursoHTTPError(
                status=resp.status_code,
                endpoint=self._endpoint,
                token_preview=self._token_preview,
                body_preview=f"<non-JSON response: {e}>  raw={resp.text[:500]}",
            )

        results = data.get("results") or []
        for r in results:
            if r.get("type") == "error":
                err = r.get("error") or {}
                msg = err.get("message") or "unknown error"
                code = err.get("code") or ""
                raise TursoHranaError(
                    code=code, message=msg, endpoint=self._endpoint,
                )
        for r in results:
            if r.get("type") == "ok":
                inner = r.get("response") or {}
                if inner.get("type") == "execute":
                    return inner.get("result")
        return None


# =============================================================================
# BACKEND-AGNOSTIC HELPERS
# =============================================================================


def _rows_to_dicts(cursor) -> list[dict]:
    """Map cursor results to dicts using cursor.description. Works on both
    sqlite3 and our Turso cursor."""
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
    """Persist a completed council run. Returns the new row id."""
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
        new_id = row[0] if row else cur.lastrowid
        conn.commit()
        return int(new_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _decode(d: dict) -> dict:
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
        out[k] = json.loads(raw) if raw else (
            [] if k in ("interrupts", "clarifications") else {}
        )
    return out


def list_councils(limit: int | None = None) -> list[dict]:
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
