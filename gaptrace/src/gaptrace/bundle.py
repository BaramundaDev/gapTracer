"""The note bundle (markdown files with YAML front matter, OKF-style) and the
passage store (one JSON line per passage). These two are the sources of truth;
every index is rebuilt from them."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .schema import Note, Passage


def note_path(root: Path, note: Note) -> Path:
    return Path(root) / note.type / f"{note.id}.md"


def write_note(root: Path, note: Note) -> Path:
    path = note_path(root, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    front = note.model_dump(mode="json", exclude={"l1"}, exclude_none=True)
    front["description"] = front.pop("l0")
    body = f"# {note.name}\n\n{note.l1}\n"
    path.write_text("---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n\n" + body,
                    encoding="utf-8")
    return path


def read_note(path: Path) -> Note:
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing front matter")
    front_text, _, body = text[4:].partition("\n---\n")
    front = yaml.safe_load(front_text) or {}
    front["l0"] = front.pop("description", "")
    lines = [ln for ln in body.strip().splitlines() if not ln.startswith("# ")]
    front["l1"] = "\n".join(lines).strip()
    return Note.model_validate(front)


def write_bundle(root: Path, notes: Iterable[Note]) -> int:
    n = 0
    for note in notes:
        write_note(root, note)
        n += 1
    return n


def read_bundle(root: Path) -> List[Note]:
    return [read_note(p) for p in sorted(Path(root).glob("*/*.md"))]


class PassageStore:
    """Append-only JSONL store, loaded into memory by id."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._by_id: Dict[str, Passage] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    p = Passage.model_validate_json(line)
                    self._by_id[p.id] = p

    def add(self, passages: Iterable[Passage]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for p in passages:
                if p.id not in self._by_id:
                    self._by_id[p.id] = p
                    fh.write(json.dumps(p.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def get(self, passage_id: str) -> Optional[Passage]:
        return self._by_id.get(passage_id)

    def all(self) -> List[Passage]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)
