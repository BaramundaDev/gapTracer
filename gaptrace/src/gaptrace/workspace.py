"""Open a data directory: <data>/bundle, <data>/passages.jsonl, <data>/runs."""
from __future__ import annotations

from tempfile import mkdtemp
from pathlib import Path
from typing import Callable, Optional

from .bundle import PassageStore, read_bundle
from .settings import load_settings
from .tools import Toolbox
from . import tokens, runtime


def new_run_dir(data: Path) -> Path:
    runs = Path(data) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix="run-", dir=runs))


def open_toolbox(data: str | Path = "data", settings=None, embed: Optional[Callable] = None,
                 run_dir: Optional[Path] = None) -> Toolbox:
    data = Path(data)
    settings = settings or load_settings()
    tokens.configure(settings)
    if embed is None and settings.switches.hybrid_search:
        embed = runtime.create_embedder(settings, data / "cache")
    notes = read_bundle(data / "bundle")
    passages = PassageStore(data / "passages.jsonl")
    toolbox = Toolbox(notes, passages, settings, embed=embed, run_dir=run_dir)
    toolbox.data_dir = data
    return toolbox
