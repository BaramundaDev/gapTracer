"""Import dated, sectioned paper text into a new GapTrace workspace."""
from __future__ import annotations

import json
from contextlib import closing
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Callable

from pydantic import BaseModel, ConfigDict, Field

from . import tokens
from .bundle import PassageStore, write_bundle
from .facts import FactsTable
from .gateway import Gateway
from .pipeline import build_notes
from .schema import Note, Passage


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    heading: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SourcePaper(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-zA-Z0-9-]{1,100}$")
    title: str = Field(min_length=1)
    published: date
    sections: list[Section] = Field(min_length=1)


class PaperImport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    papers: Annotated[list[SourcePaper], Field(min_length=1)]


def ingest(source: PaperImport, destination: Path, cutoff: date, gateway: Gateway, settings,
           progress: Callable[[str], None] | None = None, checkpoint_dir: Path | None = None) -> dict:
    if destination.exists():
        raise ValueError(f"Choose a new data directory. {destination} already exists.")
    ids = set()
    for paper in source.papers:
        if paper.published > cutoff:
            raise ValueError(f"Paper {paper.id} was published after cutoff {cutoff}")
        if paper.id.casefold() in ids:
            raise ValueError(f"Duplicate paper ID {paper.id}")
        ids.add(paper.id.casefold())

    papers, passages = [], []
    for paper in source.papers:
        paper_passages = []
        for section_index, section in enumerate(paper.sections, start=1):
            for chunk_index, chunk in enumerate(tokens.iter_chunks(section.text, settings.limits.passage_tokens), start=1):
                if chunk.strip():
                    paper_passages.append(Passage(id=f"{paper.id}-{section_index}-{chunk_index}",
                                                  paper_id=paper.id, section=section.heading,
                                                  text=chunk, date=paper.published))
        papers.append(Note(id=paper.id, type="paper", name=paper.title, paper_date=paper.published,
                           l0=tokens.truncate(paper.title, settings.limits.l0_tokens),
                           passage_ids=[p.id for p in paper_passages]))
        passages.extend(paper_passages)

    notes, stats = build_notes(papers, passages, gateway, settings, progress=progress, checkpoint_dir=checkpoint_dir)
    provenance = {"cutoff": cutoff.isoformat(), "build_cloud_enabled": bool(settings.cloud.enabled),
                  "build_cloud_calls": gateway.cloud_calls, "model": settings.model.local, "stats": stats}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="gaptrace-import-", dir=destination.parent) as scratch:
        staged = Path(scratch) / "corpus"
        write_bundle(staged / "bundle", notes)
        store = PassageStore(staged / "passages.jsonl")
        store.add(passages)
        facts = FactsTable(staged / "facts.sqlite")
        with closing(facts.db):
            facts.rebuild(notes)
        (staged / "import.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        staged.rename(destination)
    if progress:
        progress(f"Saved {len(notes)} notes and {len(passages)} passages to {destination}")
    return provenance
