"""Load settings.yaml into a plain object with attribute access."""
from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items()})
    return value


def load_settings(path: str | Path | None = None) -> SimpleNamespace:
    data = yaml.safe_load(Path(path or os.environ.get("GAPTRACE_SETTINGS") or DEFAULT_PATH).read_text(encoding="utf-8"))
    s = _ns(data)
    b = s.budget
    parts = b.instructions + b.state + b.notes + b.passage + b.output + b.margin
    if parts != b.total:
        raise ValueError(f"budget slots sum to {parts}, not total {b.total}")
    return s
