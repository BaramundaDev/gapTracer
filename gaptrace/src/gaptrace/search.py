"""Hybrid search: BM25 keyword ranking and an optional vector ranking, merged
by reciprocal rank fusion. A brute-force numpy vector index is enough for the
few thousand notes one topic produces, so no vector database is required."""
from __future__ import annotations

import re
import threading
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .schema import Note

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def note_text(note: Note) -> str:
    return " ".join([note.name, " ".join(note.aliases), note.l0, note.l1])


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal rank fusion of several ranked id lists."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class KeywordIndex:
    def __init__(self, notes: Iterable[Note], text_fn=note_text):
        self.ids = []
        corpus = []
        for n in notes:
            self.ids.append(n.id)
            corpus.append(tokenize(text_fn(n)) or ["_"])
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 20) -> List[str]:
        if not self.bm25:
            return []
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)
        qset = set(q)
        if len(scores) and max(scores) <= 0:
            # BM25 IDF can be zero/negative on tiny corpora. Preserve useful
            # query coverage instead of penalizing a document's extra matches.
            order = sorted(range(len(scores)), key=lambda i: (-len(qset & self.bm25.doc_freqs[i].keys()), -scores[i], i))
        out = []
        for i in order[: k * 3]:
            # BM25 can score 0 or below on tiny corpora; require a real term overlap
            if qset & set(self.bm25.doc_freqs[i].keys()):
                out.append(self.ids[i])
            if len(out) >= k:
                break
        return out


class VectorIndex:
    """embed: callable taking a list of strings and returning a 2-D array."""

    def __init__(self, notes: Iterable[Note], embed: Callable[[List[str]], np.ndarray], text_fn=note_text):
        notes = list(notes)
        self.ids = [n.id for n in notes]
        self.embed = embed
        vecs = np.asarray(embed([text_fn(n) for n in notes]), dtype=np.float32) if notes else np.zeros((0, 1))
        self.vecs = normalized_vectors(vecs, len(notes)) if notes else vecs

    def search(self, query: str, k: int = 20) -> List[str]:
        if not self.ids:
            return []
        q = normalized_vectors(self.embed([query]), 1)[0]
        if q.shape[0] != self.vecs.shape[1]:
            raise ValueError("Query embedding dimensions differ from the document index")
        order = np.argsort(-(self.vecs @ q), kind="stable")[:k]
        return [self.ids[i] for i in order]


class HybridSearch:
    def __init__(self, notes: Iterable[Note], embed: Optional[Callable] = None, rrf_k: int = 60,
                 hybrid: bool = True, text_fn=note_text):
        self.notes = {n.id: n for n in notes}
        self.keyword = KeywordIndex(self.notes.values(), text_fn)
        self._text_fn = text_fn
        self._embed = embed if hybrid else None
        self._vector = None
        self._vector_lock = threading.Lock()
        self.rrf_k = rrf_k

    @property
    def vector(self):
        if self._embed is not None and self._vector is None:
            with self._vector_lock:
                if self._vector is None:
                    self._vector = VectorIndex(self.notes.values(), self._embed, self._text_fn)
        return self._vector

    def search(self, query: str, k: int = 20, note_type: Optional[str] = None) -> List[Note]:
        if k <= 0:
            return []
        candidate_k = len(self.notes) if note_type else k * 2
        rankings = [self.keyword.search(query, candidate_k)]
        if self.vector is not None:
            rankings.append(self.vector.search(query, candidate_k))
        merged = [self.notes[i] for i, _ in rrf(rankings, self.rrf_k)]
        if note_type:
            merged = [n for n in merged if n.type == note_type]
        return merged[:k]


def sentence_transformer_embedder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """Optional: needs `pip install sentence-transformers`."""
    from sentence_transformers import SentenceTransformer  # imported lazily

    model = SentenceTransformer(model_name, device="cpu")
    return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def normalized_vectors(values, count: int) -> np.ndarray:
    """Reject malformed outputs before they can poison retrieval or its cache."""
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != count or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Embedding output must be a finite matrix with one row per input")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding output contains a zero vector")
    return vectors / norms
