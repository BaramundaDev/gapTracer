"""The tools the agent, the MCP server, the API and the command line all share.
Each returns a short digest and ids; the full result is written to disk."""
from __future__ import annotations

import json
import re
from threading import RLock
from pathlib import Path
from typing import Callable, List, Optional

from . import tokens
from .bundle import PassageStore
from .facts import FactsTable
from .gaps import find_gaps
from .graph import build_graph, in_links, out_links
from .pipeline import _has_value
from .schema import GapCard, Note
from .search import HybridSearch


def cap_lines(lines: List[str], limit: int, hint: str = "narrow the query or ask by id") -> str:
    """Keep whole lines until the cap; say what was left out and how to get it."""
    out, used = [], 0
    for i, line in enumerate(lines):
        left = len(lines) - i
        tail = f"(+{left} more; {hint})"
        if used + tokens.count(line) + 1 + tokens.count(tail) > limit and i > 0:
            out.append(tail)
            break
        out.append(line)
        used += tokens.count(line) + 1
    return tokens.truncate("\n".join(out), limit)


class Toolbox:
    def __init__(self, notes: List[Note], passages: PassageStore, settings,
                 embed: Optional[Callable] = None, run_dir: Optional[Path] = None):
        self.settings = settings
        # Validate stored provenance before trusting flags from a saved bundle.
        notes = [n.model_copy(deep=True) for n in notes]
        self.notes = {n.id: n for n in notes}
        self.passages = passages
        self.evidence_errors = []
        for note in notes:
            for link in note.links:
                if not link.verified:
                    continue
                passage = passages.get(link.evidence_passage_id)
                target = self.notes.get(link.target)
                normalize = lambda text: re.sub(r"\s+", " ", text).strip().lower()
                valid = (passage is not None and target is not None and bool(link.evidence_sentence.strip())
                         and normalize(link.evidence_sentence) in normalize(passage.text))
                if valid and note.type == "paper":
                    valid = passage.paper_id == note.id
                if valid and link.type == "reports":
                    valid = (note.type == "paper" and target.type == "result" and target.paper_id == note.id
                             and passage.id in target.passage_ids and target.fact is not None
                             and _has_value(link.evidence_sentence, target.fact.value))
                if not valid:
                    link.verified = False
                    self.evidence_errors.append({"source": note.id, "target": link.target,
                                                 "passage_id": link.evidence_passage_id})
        self.search = HybridSearch(notes, embed, settings.retrieval.rrf_k, settings.switches.hybrid_search)
        self.graph = build_graph(notes, verified_only=settings.switches.link_verifier)
        self.facts = FactsTable()
        self.facts.rebuild(notes)
        self.run_dir = Path(run_dir) if run_dir else None
        self._gaps: Optional[List[GapCard]] = None
        self._n = 0
        self.data_dir = None
        self.review_gateway = None
        self.reviewer = None
        self.latest_review = None
        self.external_notes = {}
        self.external_passages = {}
        self._review_lock = RLock()
        self._passage_index = None

    def search_passages(self, query: str, k: int = 5) -> str:
        """Contextual retrieval; only original passages may substantiate claims."""
        if not 1 <= k <= 10:
            raise ValueError("Retrieve between 1 and 10 source passages")
        if self._passage_index is None:
            from .knowledge import passage_index, contextual_embedder
            embed = contextual_embedder(self) if self.settings.switches.hybrid_search else None
            self._passage_index = passage_index(self, embed=embed)
        hits = self._passage_index.search(query, k)
        result = [{"passage_id": n.id, "paper_id": n.paper_id, "retrieval_context": n.l1,
                   "context_is_evidence": False} for n in hits]
        self._save("search_passages", {"query": query, "k": k}, result)
        return cap_lines([f"[{r['passage_id']}] paper {r['paper_id']} | retrieval context (not evidence): {r['retrieval_context']}"
                          for r in result], 500) if result else "No source passages match."

    def source_note(self, note_id):
        return self.notes.get(note_id) or self.external_notes.get(note_id)

    def source_passage(self, passage_id):
        return self.passages.get(passage_id) or self.external_passages.get(passage_id)

    def gap_reviewer(self, provider="", as_of=""):
        if (self.reviewer is None or (provider and self.reviewer.provider != provider)
                or (as_of and self.reviewer.as_of.isoformat() != as_of)):
            from .gap_review import GapReviewer
            self.reviewer = GapReviewer(self, self.review_gateway, provider=provider or "openalex", as_of=as_of)
        return self.reviewer

    def search_literature(self, query: str, k: int = 5) -> str:
        from .schema import Passage
        record = self.gap_reviewer().search(query, k)
        self._save("search_literature", {"query": query, "k": k}, record)
        if record["error"]:
            return "Scholarly search incomplete: " + record["error"]
        lines = []
        for paper in record["works"]:
            pid = paper["id"] + "-abstract"
            self.external_notes[paper["id"]] = Note(id=paper["id"], type="paper", name=paper["title"],
                l0=paper["title"], l1=tokens.truncate(paper["abstract"], 100), passage_ids=[pid])
            self.external_passages[pid] = Passage(id=pid, paper_id=paper["id"], section="external abstract",
                                                text=paper["abstract"], date=paper["published"])
            lines.append(f"[{paper['id']}] {paper['title']} ({paper['published']}) {paper['url']}\n"
                         + tokens.truncate(paper["abstract"], 60) + f"\npassage: {pid}")
        return cap_lines(lines, 500, "narrow scholarly query; get_passage for source text") if lines else "No scholarly results; novelty remains unestablished."

    def review_gaps(self, k: int = 3, gap_id: str = "", provider: str = "", as_of: str = "") -> str:
        with self._review_lock:
            return self._review_gaps(k, gap_id, provider, as_of)

    def _review_gaps(self, k: int, gap_id: str, provider: str, as_of: str) -> str:
        from .gap_review import render, save
        self.latest_review = self.gap_reviewer(provider, as_of).run(self.gaps(), k, gap_id)
        self._gaps = [GapCard.model_validate(c) for c in self.latest_review["ranked"]]
        if self.run_dir:
            save(self.latest_review, self.run_dir / "gap-review.json")
        if self.data_dir:
            save(self.latest_review, self.data_dir / "gap-review.json")
        self._save("review_gaps", {"k": k, "gap_id": gap_id}, self.latest_review)
        # The assistant's context gets a bounded digest; complete decisions stay on disk.
        return tokens.truncate(render(self.latest_review), 1100)

    # -- plumbing --
    def _save(self, tool: str, args: dict, full) -> None:
        if not (self.run_dir and self.settings.switches.tool_results_to_disk):
            return
        self._n += 1
        d = self.run_dir / "tool_results"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self._n:04d}_{tool}.json").write_text(
            json.dumps({"tool": tool, "args": args, "result": full}, ensure_ascii=False, default=str, indent=1),
            encoding="utf-8")

    def _line(self, n: Note) -> str:
        return f"[{n.id}] ({n.type}) {n.l0}"

    # -- tools --
    def search_notes(self, query: str, note_type: str = "", k: int = 0) -> str:
        k = k or self.settings.retrieval.top_k
        hits = self.search.search(query, k, note_type or None)
        self._save("search_notes", {"query": query, "note_type": note_type, "k": k}, [h.id for h in hits])
        if not hits:
            return "No notes match."
        return cap_lines([self._line(h) for h in hits], self.settings.tool_caps.search_notes)

    def get_note(self, note_id: str, level: int = 1) -> str:
        n = self.source_note(note_id)
        if n is None:
            return f"No note with id {note_id}."
        self._save("get_note", {"note_id": note_id, "level": level}, n.model_dump(mode="json"))
        lines = [self._line(n)]
        if level >= 1 and n.l1:
            lines.append(n.l1)
        if n.fact:
            f = n.fact
            lines.append(f"fact: {f.method} | {f.dataset} | {f.metric} = {f.value:g}{f.unit}")
        links = [f"{d['type']}->{v}" for _, v, d in out_links(self.graph, n.id)]
        links += [f"{u}->{d['type']}" for u, _, d in in_links(self.graph, n.id)]
        if links:
            lines.append("links: " + ", ".join(links))
        if n.passage_ids:
            lines.append("passages: " + ", ".join(n.passage_ids))
        return cap_lines(lines, self.settings.tool_caps.get_note, "use neighbours or get_passage")

    def neighbours(self, note_id: str, link_type: str = "") -> str:
        if note_id not in self.notes:
            return f"No note with id {note_id}."
        lt = link_type or None
        lines = [f"{d['type']} -> {self._line(self.notes[v])}" for _, v, d in out_links(self.graph, note_id, lt)]
        lines += [f"{d['type']} <- {self._line(self.notes[u])}" for u, _, d in in_links(self.graph, note_id, lt)]
        self._save("neighbours", {"note_id": note_id, "link_type": link_type}, lines)
        return cap_lines(lines, self.settings.tool_caps.neighbours) if lines else "No verified links."

    def get_passage(self, passage_id: str) -> str:
        p = self.source_passage(passage_id)
        if p is None:
            return f"No passage with id {passage_id}."
        self._save("get_passage", {"passage_id": passage_id}, p.model_dump(mode="json"))
        head = f"[{p.id}] paper {p.paper_id}, section: {p.section or 'unknown'}"
        return tokens.truncate(head + "\n" + p.text, self.settings.tool_caps.get_passage)

    def lookup_fact(self, method: str = "", dataset: str = "", metric: str = "") -> str:
        rows = self.facts.lookup(method, dataset, metric)
        self._save("lookup_fact", {"method": method, "dataset": dataset, "metric": metric}, rows)
        if not rows:
            return "No reported value matches."
        lines = [f"[{r['note_id']}] {r['method']} | {r['dataset']} | {r['metric']} = {r['value']:g}{r['unit']}"
                 f" ({r['paper_id']}, {r['year']})" for r in rows]
        return cap_lines(lines, self.settings.tool_caps.lookup_fact, "add method, dataset or metric")

    def gaps(self) -> List[GapCard]:
        if self._gaps is None:
            self._gaps = find_gaps(list(self.notes.values()), self.settings)
        return self._gaps

    def find_gaps(self, k: int = 0, gap_type: str = "") -> str:
        k = k or self.settings.gaps.top_n
        cards = [c for c in self.gaps() if not gap_type or c.type == gap_type][:k]
        self._save("find_gaps", {"k": k, "gap_type": gap_type}, [c.model_dump(mode="json") for c in cards])
        if not cards:
            return "No gaps found."
        cap = self.settings.tool_caps.find_gaps
        return "\n".join(tokens.truncate(
            f"[{c.id}] {c.type} | status {c.status} | support {c.support_count}"
            f"{' | doability ' + str(c.doability) if c.doability is not None else ''}"
            f"{' | check: may already be handled' if c.needs_check else ''}\n  {c.statement}", cap) for c in cards)

    def get_trail(self, gap_id: str) -> str:
        card = next((c for c in self.gaps() if c.id == gap_id), None)
        if card is None:
            return f"No gap with id {gap_id}."
        lines = [f"{t.source} --{t.link}--> {t.target} | passage {t.passage_id or '-'} | "
                 f"{'verified' if t.verified else 'unverified'}" for t in card.trail]
        self._save("get_trail", {"gap_id": gap_id}, card.model_dump(mode="json"))
        return cap_lines(lines, self.settings.tool_caps.get_trail, "use get_passage for the wording")

    def call(self, tool: str, args: dict) -> str:
        allowed = {"search_notes", "get_note", "neighbours", "get_passage", "lookup_fact", "find_gaps", "get_trail",
                   "search_literature", "review_gaps", "search_passages"}
        if tool not in allowed:
            return f"Unknown tool {tool}. Tools: {', '.join(sorted(allowed))}."
        try:
            return getattr(self, tool)(**args)
        except TypeError as exc:
            return f"Bad arguments for {tool}: {exc}"
