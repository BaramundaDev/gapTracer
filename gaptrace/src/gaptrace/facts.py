"""Exact numbers live in a SQLite table and are looked up by key, never by
vector search."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List

from .schema import Note


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


class FactsTable:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS facts (note_id TEXT PRIMARY KEY, method TEXT, dataset TEXT,"
            " metric TEXT, value REAL, unit TEXT, paper_id TEXT, year INTEGER,"
            " method_n TEXT, dataset_n TEXT, metric_n TEXT)")

    def rebuild(self, notes: Iterable[Note]) -> int:
        self.db.execute("DELETE FROM facts")
        n = 0
        for note in notes:
            if note.type != "result" or note.fact is None:
                continue
            f = note.fact
            year = note.paper_date.year if note.paper_date else None
            self.db.execute("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (note.id, f.method, f.dataset, f.metric, f.value, f.unit, note.paper_id, year,
                             _norm(f.method), _norm(f.dataset), _norm(f.metric)))
            n += 1
        self.db.commit()
        return n

    def lookup(self, method: str = "", dataset: str = "", metric: str = "") -> List[dict]:
        clauses, args = [], []
        for col, val in (("method_n", method), ("dataset_n", dataset), ("metric_n", metric)):
            if val:
                clauses.append(f"{col} LIKE ?")
                args.append(f"%{_norm(val)}%")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.db.execute(
            "SELECT note_id, method, dataset, metric, value, unit, paper_id, year FROM facts" + where +
            " ORDER BY year DESC, note_id", args).fetchall()
        return [dict(r) for r in rows]

    def groups(self) -> List[List[dict]]:
        """Rows grouped by identical (method, dataset, metric), for conflict detection."""
        rows = self.db.execute("SELECT * FROM facts ORDER BY method_n, dataset_n, metric_n").fetchall()
        out, key, cur = [], None, []
        for r in rows:
            k = (r["method_n"], r["dataset_n"], r["metric_n"])
            if k != key and cur:
                out.append(cur)
                cur = []
            key = k
            cur.append(dict(r))
        if cur:
            out.append(cur)
        return out
