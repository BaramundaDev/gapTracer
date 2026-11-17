"""HTTP API for the web app: thin wrappers over the same toolbox."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .workspace import open_toolbox


def create_app(data: str | None = None) -> FastAPI:
    tb = open_toolbox(data or os.environ.get("GAPTRACE_DATA", "data"))
    app = FastAPI(title="GapTrace")

    @app.get("/notes/search")
    def search(q: str, note_type: str = "", k: int = 20):
        return [n.model_dump(mode="json", include={"id", "type", "name", "l0"})
                for n in tb.search.search(q, k, note_type or None)]

    @app.get("/notes/{note_id}")
    def note(note_id: str):
        n = tb.notes.get(note_id)
        if n is None:
            raise HTTPException(404, "no such note")
        return n.model_dump(mode="json")

    @app.get("/passages/{passage_id}")
    def passage(passage_id: str):
        p = tb.passages.get(passage_id)
        if p is None:
            raise HTTPException(404, "no such passage")
        return p.model_dump(mode="json")

    @app.get("/facts")
    def facts(method: str = "", dataset: str = "", metric: str = ""):
        return tb.facts.lookup(method, dataset, metric)

    @app.get("/gaps")
    def gaps(k: int = 10, gap_type: str = ""):
        cards = [c for c in tb.gaps() if not gap_type or c.type == gap_type][:k]
        return [dict(c.model_dump(mode="json"), verified_share=c.verified_share) for c in cards]

    @app.get("/gaps/{gap_id}")
    def gap(gap_id: str):
        c = next((c for c in tb.gaps() if c.id == gap_id), None)
        if c is None:
            raise HTTPException(404, "no such gap")
        return dict(c.model_dump(mode="json"), verified_share=c.verified_share)

    @app.post("/gap-reviews")
    def review_gaps(k: int = 3, gap_id: str = ""):
        """Run a bounded automated investigation; results include source evidence."""
        try:
            tb.review_gaps(k, gap_id)
            return tb.latest_review
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
