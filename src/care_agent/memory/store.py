"""Two-level memory backed by SQLite.

* **Session memory** (short-term): the turns of one conversation, replayed into the
  prompt so follow-up questions work ("그럼 그 약은요?"). A session is bound to the
  patient it was opened for; reusing it for another patient is refused, otherwise the
  previous patient's answers would leak into the new prompt.
* **Patient notes** (long-term): small facts the agent chose to remember about a patient
  across sessions (preferences, goals, context the EHR does not hold). Notes are
  agent-internal state, never written back to the EHR.

SQLite keeps this zero-ops for the demo; the interface is small enough to swap for
Postgres/Redis/a vector store later.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS patient_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    note TEXT NOT NULL,
    source TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS notes_patient ON patient_notes(patient_id, id);
"""


class SessionPatientMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    ts: float


@dataclass(frozen=True)
class Note:
    id: int
    note: str
    source: str | None
    ts: float


class MemoryStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # -- sessions -----------------------------------------------------------------
    def open_session(self, session_id: str, patient_id: str) -> None:
        with self._lock:
            row = self._db.execute(
                "SELECT patient_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?)", (session_id, patient_id, time.time())
                )
                self._db.commit()
            elif row[0] != patient_id:
                raise SessionPatientMismatch(
                    f"session {session_id!r} belongs to another patient; open a new session"
                )

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO turns (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
            self._db.commit()

    def recent_turns(self, session_id: str, limit: int = 10) -> list[Turn]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content, ts FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [Turn(*r) for r in reversed(rows)]

    # -- patient notes --------------------------------------------------------------
    def add_note(self, patient_id: str, note: str, source: str | None = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO patient_notes (patient_id, note, source, ts) VALUES (?, ?, ?, ?)",
                (patient_id, note, source, time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid or 0)

    def search_notes(self, patient_id: str, query: str | None = None, limit: int = 5) -> list[Note]:
        """Newest-first notes for a patient; with ``query``, notes matching any term."""
        sql = "SELECT id, note, source, ts FROM patient_notes WHERE patient_id = ?"
        params: list[object] = [patient_id]
        terms = [t for t in (query or "").split() if t]
        if terms:
            sql += " AND (" + " OR ".join("note LIKE ?" for _ in terms) + ")"
            params += [f"%{t}%" for t in terms]
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return [Note(*r) for r in self._db.execute(sql, params).fetchall()]
