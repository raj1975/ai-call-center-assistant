import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

_DB_PATH = Path("data/memory/call_history.db")


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_history (
                thread_id     TEXT PRIMARY KEY,
                file_name     TEXT,
                agent_name    TEXT,
                customer_name TEXT,
                call_date     TEXT,
                sentiment     TEXT,
                overall_score REAL,
                summary       TEXT,
                qa_score      TEXT,
                transcript    TEXT,
                errors        TEXT,
                created_at    TEXT
            )
        """)


def save_call(thread_id: str, state: dict) -> None:
    meta = state.get("metadata") or {}
    summary = state.get("summary") or {}
    qa = state.get("qa_score") or {}
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO call_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                thread_id,
                meta.get("file_name", ""),
                meta.get("agent_name") or "",
                meta.get("customer_name") or "",
                meta.get("call_date", ""),
                summary.get("sentiment", ""),
                qa.get("overall_score", 0.0),
                json.dumps(summary),
                json.dumps(qa),
                state.get("transcript") or state.get("raw_content", ""),
                json.dumps(state.get("errors", [])),
                datetime.now().isoformat(),
            ),
        )


def list_calls(limit: int = 50) -> list:
    if not _DB_PATH.exists():
        return []
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM call_history ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_call(thread_id: str) -> Optional[dict]:
    if not _DB_PATH.exists():
        return None
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM call_history WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return dict(row) if row else None
