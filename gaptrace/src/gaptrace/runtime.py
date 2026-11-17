"""Production model adapters, shared by CLI, API and MCP."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np


def ollama_json(base: str, endpoint: str, payload: dict | None = None):
    request = Request(base.rstrip("/") + endpoint,
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError(f"Ollama request failed at {base}{endpoint}: {exc}. Check ollama serve and installed models.") from exc
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"Invalid Ollama response from {endpoint}: {result}")
    return result


class OllamaEmbedder:
    """Bounded CPU batches with disk caching keyed by installed model digest."""

    def __init__(self, model: str, base: str, cache_dir: Path | None = None, truncate=False):
        self.model, self.base = model, base.rstrip("/")
        self.cache_path = Path(cache_dir) / "embeddings.sqlite" if cache_dir else None
        self.digest = None
        self.truncate = truncate
        self._lock = threading.Lock()

    def __call__(self, texts: list[str]) -> np.ndarray:
        from .search import normalized_vectors
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        with self._lock:
            if self.digest is None:
                tags = ollama_json(self.base, "/api/tags")
                tag = next((tag for tag in tags.get("models", [])
                            if self.model in {tag.get("name"), tag.get("model")}), None)
                if not tag or not tag.get("digest"):
                    raise RuntimeError(f"Embedding model {self.model} is not installed. Run ollama pull {self.model}.")
                self.digest = tag["digest"]
            namespace = json.dumps([self.base, self.model, self.digest, "cpu", "prefix-input-v1" if self.truncate else "full-input-v1"])
            keys = [hashlib.sha256((namespace + "\0" + text).encode()).hexdigest() for text in texts]
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.cache_path) if self.cache_path else ":memory:", timeout=30)
            try:
                db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
                vectors = {}
                for key in set(keys):
                    row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                    if row:
                        try:
                            vectors[key] = normalized_vectors([json.loads(row[0])], 1)[0].tolist()
                        except (ValueError, TypeError):
                            db.execute("DELETE FROM embeddings WHERE key=?", (key,))
                missing = list(dict.fromkeys(key for key in keys if key not in vectors))
                text_by_key = dict(zip(keys, texts))
                for offset in range(0, len(missing), 32):
                    batch = missing[offset:offset + 32]
                    result = ollama_json(self.base, "/api/embed", {
                        "model": self.model, "input": [text_by_key[key] for key in batch],
                        "truncate": self.truncate, "options": {"num_gpu": 0}, "keep_alive": "5m"})
                    matrix = normalized_vectors(result.get("embeddings"), len(batch))
                    for key, vector in zip(batch, matrix, strict=True):
                        vectors[key] = vector.tolist()
                        db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vectors[key])))
                db.commit()
                return normalized_vectors([vectors[key] for key in keys], len(texts))
            finally:
                db.close()


def create_embedder(settings, cache_dir: Path | None = None):
    model = settings.model.embedding
    if model.startswith("ollama/"):
        return OllamaEmbedder(model.split("/", 1)[1], getattr(settings.model, "local_api_base", "http://127.0.0.1:11434"), cache_dir)
    if model.startswith("sentence-transformers/"):
        from .search import sentence_transformer_embedder
        return sentence_transformer_embedder(model)
    raise ValueError("model.embedding must use ollama/MODEL or sentence-transformers/MODEL")
