"""Token counting. The default is a deliberately pessimistic estimate; call
set_counter() with the real tokenizer of the local model once it is known."""
from __future__ import annotations

import math
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Optional

_counter: Optional[Callable[[str], int]] = None
_description = "estimate"


def set_counter(fn: Optional[Callable[[str], int]]) -> None:
    global _counter, _description
    _counter = fn
    _description = "custom" if fn else "estimate"


@lru_cache(maxsize=8)
def _load_tokenizer(path: str, modified: int, size: int):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def configure(settings) -> str:
    """Product entry points require the actual model tokenizer; never fall back."""
    global _description
    configured = os.environ.get("GAPTRACE_TOKENIZER") or getattr(settings.model, "tokenizer", "assets/qwen35-tokenizer.json")
    if not configured:
        raise ValueError("Configure model.tokenizer or GAPTRACE_TOKENIZER with the local model's tokenizer.json")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    bundled = Path(__file__).resolve().parent / "assets" / "qwen35-tokenizer.json"
    if path.resolve() == bundled and not settings.model.local.startswith("ollama_chat/qwen3.5:4b"):
        raise ValueError("The bundled tokenizer is for Qwen3.5 4B; configure a matching tokenizer for the selected model")
    try:
        stat = path.stat()
        tokenizer = _load_tokenizer(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except ImportError as exc:
        raise ValueError("The real tokenizer requires tokenizers; install the project dependencies") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load the configured tokenizer {path}: {exc}") from exc
    set_counter(lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    _description = str(path.resolve())
    return _description


def description() -> str:
    return _description


def count(text: str) -> int:
    if not text:
        return 0
    if _counter is not None:
        return _counter(text)
    by_chars = math.ceil(len(text) / 3.5)
    by_words = math.ceil(len(re.findall(r"\S+", text)) * 1.3)
    return max(by_chars, by_words)


def iter_chunks(text: str, limit: int) -> Iterator[str]:
    """Split losslessly within the token limit, preferring sentence boundaries."""
    if limit < 1:
        raise ValueError("Chunk token limit must be positive")

    start = 0
    while start < len(text):
        remaining = len(text) - start
        low, high = 0, min(limit, remaining)
        while high < remaining and count(text[start:start + high]) <= limit:
            low, high = high, min(high * 2, remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if count(text[start:start + middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        if not low:
            raise ValueError("Chunk token limit cannot fit one character")

        chunk = text[start:start + low]
        if low < remaining:
            sentence_end = max((match.end() for match in re.finditer(r"(?<=[.!?])\s+", chunk)), default=0)
            word_end = max((match.end() for match in re.finditer(r"\s+", chunk)), default=0)
            end = sentence_end if sentence_end >= low // 2 and sentence_end else word_end
            if end and count(chunk[:end]) <= limit:
                chunk = chunk[:end]
        yield chunk
        start += len(chunk)


def truncate(text: str, limit: int, marker: str = " [...]") -> str:
    """Cut text at a word boundary so that count(result) <= limit."""
    if count(text) <= limit:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid]) + marker) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]) + marker
