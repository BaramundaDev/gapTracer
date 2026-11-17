import json
import zlib

import numpy as np
import pytest

from gaptrace import tokens
from gaptrace.bundle import PassageStore
from gaptrace.sample import sample_corpus, write_sample
from gaptrace.settings import load_settings


@pytest.fixture(autouse=True)
def _default_counter():
    tokens.set_counter(None)
    yield
    tokens.set_counter(None)


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def corpus():
    return sample_corpus()


@pytest.fixture
def data_dir(tmp_path):
    return write_sample(tmp_path / "data")


@pytest.fixture
def passages(data_dir):
    return PassageStore(data_dir / "passages.jsonl")


def fake_embed(texts):
    """Deterministic bag-of-words embedding, good enough to test the plumbing."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.strip('.,:;()').encode()) % 64] += 1.0
    return out


class FakeBackend:
    """Stands in for a model: answers from a list of (substring, reply) rules."""

    def __init__(self, rules, default="{}"):
        self.rules, self.default, self.calls = rules, default, []

    def __call__(self, prompt, schema, max_tokens):
        self.calls.append(prompt)
        for needle, reply in self.rules:
            if needle in prompt:
                return reply(prompt) if callable(reply) else (reply if isinstance(reply, str) else json.dumps(reply))
        return self.default
