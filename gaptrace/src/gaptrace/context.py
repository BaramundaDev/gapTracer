"""The context manager: packs each step into the token budget, leaves pointer
stubs for what it removes, and keeps working memory in a file.

The window is scratch space. Nothing important lives only in it."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import tokens
from .schema import Note


class ContextBudgetExceeded(Exception):
    pass


@dataclass
class State:
    """Working memory for one run. Rewritten after every step."""
    goal: str = ""
    plan: List[str] = field(default_factory=list)
    findings: List[Dict] = field(default_factory=list)   # {"ids": [...], "text": "..."}
    earlier: List[str] = field(default_factory=list)      # ids of findings dropped from the view
    next_step: str = ""

    def add_finding(self, text: str, ids: Sequence[str] = ()) -> None:
        if text.strip():
            self.findings.append({"ids": list(ids), "text": text.strip()})

    def render(self, limit: int) -> str:
        """Text view within `limit` tokens. Oldest findings collapse to ids;
        the full record stays in the trail file (write before drop)."""
        findings = list(self.findings)
        earlier = list(self.earlier)
        while True:
            lines = [f"GOAL: {self.goal}"]
            if self.plan:
                lines.append("PLAN: " + " | ".join(self.plan))
            if findings:
                lines.append("FINDINGS:")
                lines += [f"- [{', '.join(f['ids'])}] {f['text']}" for f in findings]
            if earlier:
                lines.append("EARLIER (ids only, see trail): " + ", ".join(earlier[-40:]))
            if self.next_step:
                lines.append(f"NEXT: {self.next_step}")
            text = "\n".join(lines)
            if tokens.count(text) <= limit or not findings:
                break
            dropped = findings.pop(0)
            earlier += [i for i in dropped["ids"] if i not in earlier]
        return tokens.truncate(text, limit)

    def save(self, run_dir: Path, limit: int) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "state.json").write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=1), encoding="utf-8")
        (run_dir / "state.md").write_text(self.render(limit), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> "State":
        path = Path(run_dir) / "state.json"
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()


@dataclass
class Packed:
    prompt: str
    slot_tokens: Dict[str, int]
    full_ids: List[str]       # notes included as summaries
    stub_ids: List[str]       # notes left as one-line pointers
    omitted: int              # notes not shown at all

    @property
    def input_tokens(self) -> int:
        return sum(self.slot_tokens.values())


def stub_line(note: Note) -> str:
    return f"[{note.id}] ({note.type}) {note.l0}"


def pack(instructions: str, question: str, state_text: str, notes: Sequence[Note],
         passage: Optional[str], settings) -> Packed:
    """Build one prompt that fits the budget. `notes` must be ranked best first."""
    b, sw = settings.budget, settings.switches
    head = f"{instructions.strip()}\n\nSTEP QUESTION:\n{question.strip()}"
    head_t = tokens.count(head)
    if head_t > b.instructions:
        raise ContextBudgetExceeded(f"instructions + question use {head_t} tokens; slot is {b.instructions}")
    state_text = tokens.truncate(state_text or "", b.state)
    passage_text = tokens.truncate(passage, b.passage) if passage else ""

    note_lines: List[str] = []
    full_ids: List[str] = []
    stub_ids: List[str] = []
    used = 0
    rest = list(notes)
    # 1) summaries, best first. If everything will not fit as summaries, keep a
    #    quarter of the slot for pointer stubs so the rest stays reachable.
    use_stubs = sw.three_note_sizes and sw.pointer_stubs
    all_fit = sum(tokens.count(f"[{n.id}] ({n.type}) {n.name}: {n.l1 or n.l0}") + 1 for n in notes) <= b.notes
    summary_room = b.notes if (all_fit or not use_stubs) else int(b.notes * 0.75)
    while rest:
        n = rest[0]
        line = f"[{n.id}] ({n.type}) {n.name}: {n.l1 or n.l0}"
        t = tokens.count(line) + 1
        if used + t > summary_room:
            break
        note_lines.append(line)
        full_ids.append(n.id)
        used += t
        rest.pop(0)
    # 2) what does not fit stays as a pointer stub, not a silent omission
    if use_stubs:
        tail = "\n(+{} more; search again to see them)"
        reserve = tokens.count(tail.format(9999)) + 1
        stub_header = "STUBS (ask get_note for more):"
        header_t = tokens.count(stub_header) + 1
        while rest:
            n = rest[0]
            t = tokens.count(stub_line(n)) + 1
            extra = header_t if not stub_ids else 0
            if used + t + extra + reserve > b.notes:
                break
            if not stub_ids:
                note_lines.append(stub_header)
                used += header_t
            note_lines.append(stub_line(n))
            stub_ids.append(n.id)
            used += t
            rest.pop(0)
    omitted = len(rest)
    if omitted:
        note_lines.append(f"(+{omitted} more; search again to see them)")
    notes_text = "\n".join(note_lines)

    parts = [head, "STATE:\n" + (state_text or "(empty)"), "NOTES:\n" + (notes_text or "(none)"),
             "PASSAGE:\n" + (passage_text or "(none)")]
    prompt = "\n\n".join(parts)
    slot_tokens = {"instructions": head_t, "state": tokens.count(parts[1]),
                   "notes": tokens.count(parts[2]), "passage": tokens.count(parts[3])}
    packed = Packed(prompt, slot_tokens, full_ids, stub_ids, omitted)
    assert_within_budget(prompt, settings)
    return packed


def assert_within_budget(prompt: str, settings) -> int:
    """Hard gate, called before every local model call."""
    b = settings.budget
    used = tokens.count(prompt)
    allowed = b.total - b.output - b.margin
    if used > allowed:
        raise ContextBudgetExceeded(f"prompt is {used} tokens; at most {allowed} may be sent "
                                    f"(total {b.total} - output {b.output} - margin {b.margin})")
    return used


class Trail:
    """Append-only record of a run. Findings are written here before their
    sources leave the context."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "trail.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **data) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, **data}, ensure_ascii=False, default=str) + "\n")

    def read(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
