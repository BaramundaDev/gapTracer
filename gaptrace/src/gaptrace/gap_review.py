"""Evidence-backed, bounded gap investigation and deterministic reranking."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import literature, tokens
from .gaps import rank
from .gateway import Gateway, ModelOutputInvalid
from .pipeline import fill, load_prompt
from .schema import GapCard, VerifierOutput
from .recovery import atomic_json, transient


class ClosureReview(BaseModel):
    verdict: Literal["fills", "partly", "does_not"]
    quote: str = ""
    reason: str


class Axis(BaseModel):
    score: int | None = Field(default=None, ge=0, le=2)
    source_id: str = ""
    quote: str = ""
    reason: str = ""


class FeasibilityReview(BaseModel):
    data: Axis
    compute: Axis
    blocks: Axis
    scope: Axis


def fingerprint(tb):
    value = {"notes": [n.model_dump(mode="json") for n in sorted(tb.notes.values(), key=lambda n: n.id)],
             "passages": [p.model_dump(mode="json") for p in sorted(tb.passages.all(), key=lambda p: p.id)],
             "model": vars(tb.settings.model), "gaps": vars(tb.settings.gaps),
             "prompts": {name: load_prompt(name) for name in ("gap_evidence", "gap_closure", "gap_feasibility", "gap_feasibility_check")}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def local_gateway(settings, cache=None):
    from urllib.parse import urlparse
    from .gateway import litellm_backend
    model, base = settings.model.local, settings.model.local_api_base
    if (not model.startswith("ollama_chat/") or model.endswith((":cloud", "-cloud"))
            or urlparse(base).hostname not in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("Gap review requires the configured local Ollama model and loopback endpoint")
    return Gateway(settings, {"local": litellm_backend(model, settings.model.local_num_ctx, api_base=base)}, cache)


def _quote(quote, text):
    return bool(quote.strip()) and quote in text


def _excerpt(text, limit):
    return next(tokens.iter_chunks(text, limit), "")


class GapReviewer:
    def __init__(self, toolbox, gateway=None, search=None, provider="openalex", as_of="", progress=None,
                 recover_transient=False):
        self.tb = toolbox
        self.gateway = gateway or local_gateway(toolbox.settings, toolbox.data_dir / "cache" / "gap-review" if toolbox.data_dir else None)
        self.searcher = search or literature.search
        self.provider = provider
        self.as_of = date.fromisoformat(as_of) if as_of else date.today()
        if self.as_of > date.today():
            raise ValueError("as_of cannot be in the future")
        if any(p.date and p.date > self.as_of for p in toolbox.passages.all()):
            raise ValueError("as_of precedes a paper in the source corpus")
        if provider not in {"openalex", "arxiv"}:
            raise ValueError("Provider must be openalex or arxiv")
        self.progress = progress or (lambda message: None)
        self.searches = []
        self.reviews = {}
        self.recover_transient = recover_transient
        self.original_cards = None

    def search(self, query, limit=5):
        if self.recover_transient:
            for record in self.searches:
                if record["query"] == query and record["limit"] == limit and not record["error"]:
                    return record
        if len(self.searches) >= 20:
            raise ValueError("This investigation has reached its 20-search allowance")
        record = {"query": query, "provider": self.provider, "as_of": self.as_of.isoformat(), "limit": limit,
                  "searched_utc": datetime.now(timezone.utc).isoformat()}
        self.progress(f"Searching {self.provider}: {query}")
        try:
            works = self.searcher(query, self.provider, limit, self.as_of.isoformat())
            if len(works) > limit or len({p["id"] for p in works}) != len(works):
                raise ValueError("Search returned duplicate papers or exceeded its allowance")
            for paper in works:
                if (not paper["title"] or not paper["abstract"] or not paper["url"].startswith("https://")
                        or date.fromisoformat(paper["published"]) > self.as_of):
                    raise ValueError("Invalid or out-of-scope search evidence")
            record.update(works=works, error=None)
        except (OSError, ValueError, KeyError) as exc:
            if self.recover_transient and transient(exc):
                raise
            record.update(works=[], error=f"{type(exc).__name__}: {exc}")
        self.searches.append(record)
        return record

    def _call(self, prompt, schema):
        try:
            # Provider diagnostics must not corrupt CLI JSON or MCP's stdio protocol.
            with redirect_stdout(sys.stderr):
                out, _ = self.gateway.call("local", prompt, schema)
            return out, None
        except (ModelOutputInvalid, OSError, ValueError, RuntimeError) as exc:
            if self.recover_transient and transient(exc):
                raise
            return None, f"{type(exc).__name__}: {exc}"

    def _queries(self, card):
        names = []
        for nid in card.note_ids:
            note = self.tb.notes.get(nid)
            if not note:
                continue
            if note.fact:
                names += [note.fact.method, note.fact.dataset]
            elif note.type in {"method", "dataset"}:
                names.append(note.name)
        names = list(dict.fromkeys(names))
        focus = " ".join(self.tb.notes[n].name for n in card.note_ids if n in self.tb.notes)
        main = " ".join(names) if names else (focus or card.statement)
        # Two fixed, visible formulations; do not let model text create arbitrary URLs.
        if card.type == "conflicting_results":
            extra = main + " reproducibility"
        elif card.type in {"unanswered_open_question", "unaddressed_limitation"}:
            owners = {edge.source for edge in card.trail}
            methods = [n.name for n in self.tb.notes.values() if n.type == "method" and n.paper_id in owners]
            extra = main + (" " + methods[0] if methods else " empirical study")
        else:
            extra = main + " evaluation"
        return list(dict.fromkeys([tokens.truncate(main, 70)[:500], tokens.truncate(extra, 70)[:500]]))[:2]

    def review(self, original):
        if original.id in self.reviews:
            return self.reviews[original.id]
        self.progress(f"Checking evidence for {original.id}")
        card = original.model_copy(deep=True)
        record = {"gap_id": card.id, "evidence": [], "candidates": [], "errors": [], "queries": self._queries(card)}
        for edge in card.trail[:8]:
            passage = self.tb.passages.get(edge.passage_id)
            source, target = self.tb.notes.get(edge.source), self.tb.notes.get(edge.target)
            intact = (edge.verified and passage is not None and source is not None and target is not None
                      and _quote(edge.sentence, passage.text))
            verdict, error = None, None
            if intact:
                payload = {"from": source.name, "relationship": edge.link, "to": target.name,
                           "from_type": source.type, "to_type": target.type,
                           "fact": target.fact.model_dump() if target.fact else None,
                           "quote": edge.sentence, "source_passage": _excerpt(passage.text, 500)}
                verdict, error = self._call(fill(load_prompt("gap_evidence"), payload=json.dumps(payload)), VerifierOutput)
            edge.verified = bool(intact and verdict and verdict.supported)
            record["evidence"].append({"source": edge.source, "target": edge.target, "passage_id": edge.passage_id,
                "paper_id": passage.paper_id if passage else None, "passage_text": passage.text if passage else None,
                "supported": edge.verified, "quote": edge.sentence, "reason": error or (verdict.reason if verdict else "Invalid source provenance")})
        for edge in card.trail[8:]:
            edge.verified = False
        complete_support = bool(card.trail) and all(edge.verified for edge in card.trail)
        card.needs_check = card.needs_check or not complete_support
        if len(card.trail) > 8:
            record["errors"].append("Evidence-check allowance exhausted; remaining links are unreviewed")

        used_searches = []
        external = {}
        for query in record["queries"]:
            search = self.search(query)
            used_searches.append(self.searches.index(search))
            if search["error"]:
                record["errors"].append(search["error"])
            for work in search["works"]:
                external.setdefault(work["id"], work)
        record["search_indices"] = list(dict.fromkeys(used_searches))
        # Local checks use actual passages, including papers missed by note extraction.
        origin_passages = {e.passage_id for e in card.trail}
        local = []
        for note in self.tb.search.search(record["queries"][0], 20):
            for pid in note.passage_ids:
                p = self.tb.passages.get(pid)
                if (p and p.id not in origin_passages and p.date and p.date <= self.as_of
                        and all(c["id"] != p.id for c in local)):
                    local.append({"id": p.id, "paper_id": p.paper_id, "title": self.tb.notes[p.paper_id].name if p.paper_id in self.tb.notes else p.paper_id,
                                  "text": _excerpt(p.text, 400), "evidence_level": "corpus passage"})
        words = set(record["queries"][0].lower().split())
        ordered = sorted(external.values(), key=lambda p: (-len(words & set((p["title"] + " " + p["abstract"]).lower().split())), p["id"]))
        candidates = [{**p, "text": _excerpt(p["abstract"], 400)} for p in ordered[:3]] + local[:2]
        positive, partial, negatives = [], [], 0
        for candidate in candidates:
            self.progress(f"Judging {card.id} against {candidate['id']}")
            payload = json.dumps({"gap": card.statement, "paper": candidate["title"], "text": candidate["text"]})
            item = {**candidate, "first": None, "recheck": None}
            verdict, error = self._call(fill(load_prompt("gap_closure"), strict="", payload=payload), ClosureReview)
            if verdict:
                item["first"] = verdict.model_dump()
            if verdict and verdict.verdict != "does_not":
                if not _quote(verdict.quote, candidate["text"]):
                    error = "Positive closure judgment has no exact source quote"
                else:
                    checked, error = self._call(fill(load_prompt("gap_closure"),
                        strict="STRICT RECHECK: independently reconsider every required part of the gap. Reject overclaims.", payload=payload), ClosureReview)
                    if checked:
                        item["recheck"] = checked.model_dump()
                    if checked and checked.verdict != "does_not" and not _quote(checked.quote, candidate["text"]):
                        error = "Closure recheck quote absent from source"
                    if checked and not error:
                        if verdict.verdict == checked.verdict == "fills":
                            positive.append(candidate)
                        elif checked.verdict in {"fills", "partly"}:
                            partial.append(candidate)
                        else:
                            error = "Initial positive closure judgment disagrees with recheck"
            elif verdict:
                negatives += 1
            item["error"] = error
            if error:
                record["errors"].append(error)
            record["candidates"].append(item)
        card.status, card.decided_by = "unchecked", ""
        if positive:
            card.status, card.decided_by = "closed", positive[0].get("paper_id", positive[0]["id"])
        elif partial:
            card.status, card.decided_by = "partly", partial[0].get("paper_id", partial[0]["id"])
        elif negatives and not record["errors"] and complete_support:
            card.status = "open"

        # Feasibility axes require quotes; unknown axes do not receive guessed values.
        self.progress(f"Assessing data, compute, code and scope for {card.id}")
        sources = {p.id: _excerpt(p.text, 180) for p in self.tb.passages.all() if p.id in origin_passages}
        sources.update({c["id"]: _excerpt(c["text"], 180) for c in candidates[:3]})
        sources = dict(list(sources.items())[:6])
        feasibility, error = self._call(fill(load_prompt("gap_feasibility"), payload=json.dumps({"gap": card.statement, "sources": sources})), FeasibilityReview)
        record["feasibility"] = {}
        claims = {"data": ["Data explicitly unavailable to the public", "Public data available but needs preparation", "Public ready-to-use data available"],
                  "compute": ["Cluster compute required", "Rented GPU required", "One consumer GPU suffices"],
                  "blocks": ["No public code available", "Partial public code available", "Public usable code available"],
                  "scope": ["Project takes over a year", "Project takes three to twelve months", "Project takes under three months"]}
        cues = {"data": r"\b(data|dataset|datasets|corpus|corpora)\b", "compute": r"\b(gpu|gpus|cpu|cpus|cluster|tpu|hardware)\b",
                "blocks": r"\b(code|github|repository|repositories|software)\b", "scope": r"\b(days?|weeks?|months?|years?)\b"}
        if feasibility:
            for axis, result in feasibility:
                if result.score is not None:
                    if not _quote(result.quote, sources.get(result.source_id, "")) or not re.search(cues[axis], result.quote, re.I):
                        result.score = None
                        result.reason = "Score withheld: no exact source quote explicitly discussing this resource"
                    else:
                        supported, error = self._call(fill(load_prompt("gap_feasibility_check"), payload=json.dumps({
                            "claim": claims[axis][result.score], "quote": result.quote, "text": sources[result.source_id]})), VerifierOutput)
                        if error or not supported or not supported.supported:
                            result.score = None
                            result.reason = error or (supported.reason if supported else "Resource claim unverified")
                record["feasibility"][axis] = result.model_dump()
        elif error:
            record["errors"].append("Feasibility: " + error)
        scores = [v["score"] for v in record["feasibility"].values()]
        card.doability = sum(scores) if len(scores) == 4 and all(s is not None for s in scores) else None
        card.judged_by = "local"
        record.update(card=card.model_dump(mode="json"), fully_supported=complete_support,
                      scope="bounded corpus-passage and scholarly-abstract search; open means no closure found in checked candidates",
                      assessment_type="automated; not independently adjudicated research labels")
        self.reviews[card.id] = record
        return record

    def run(self, cards, k=3, gap_id="", checkpoint=None):
        if not 1 <= k <= 10:
            raise ValueError("Review between 1 and 10 gaps per investigation")
        if self.original_cards is None:
            self.original_cards = [c.model_copy(deep=True) for c in cards]
        before = [c.model_copy(deep=True) for c in self.original_cards]
        selected = [c for c in before if not gap_id or c.id == gap_id][:k]
        if gap_id and not selected:
            raise ValueError(f"No gap with id {gap_id}")
        for card in selected:
            self.review(card)
            if checkpoint:
                checkpoint(self)
        after = rank([GapCard.model_validate(self.reviews[c.id]["card"]) if c.id in self.reviews else c for c in before])
        return {"kind": "gap-investigation", "version": 1, "as_of": self.as_of.isoformat(),
                "created_utc": datetime.now(timezone.utc).isoformat(), "provider": self.provider,
                "corpus_sha256": fingerprint(self.tb), "model": self.tb.settings.model.local,
                "ranking_rule": "status, needs_check, descending support, doability, verified share, stable id",
                "before": [c.model_dump(mode="json") for c in before], "ranked": [c.model_dump(mode="json") for c in after],
                "reviews": list(self.reviews.values()), "searches": self.searches, "gateway_log": self.gateway.log,
                "limits": {"gaps": k, "searches_per_gap": 2, "works_per_search": 5, "closure_candidates_per_gap": 5,
                           "evidence_links_per_gap": 8, "positive_closure_rechecks": 1, "feasibility_axis_checks": 4},
                "limitations": ["Automated judgments are not independent human labels.",
                    "Search results and abstracts are incomplete; open is a bounded finding, not proof of universal novelty.",
                    "Unknown feasibility remains null. Unreviewed cards retain unchecked status."]}


def render(report):
    reviewed = {r["gap_id"] for r in report["reviews"]}
    lines = [f"Gap investigation as of {report['as_of']} ({report['provider']}; automated assessments)"]
    for i, card in enumerate(report["ranked"], 1):
        score = str(card["doability"]) + "/8" if card["doability"] is not None else "unknown"
        lines.append(f"{i}. [{card['id']}] {card['status']} | support {card['support_count']} | feasibility {score}"
                     + (" | not reviewed this run" if card["id"] not in reviewed else "") + "\n   " + card["statement"])
        record = next((r for r in report["reviews"] if r["gap_id"] == card["id"]), None)
        if record:
            for candidate in record["candidates"]:
                verdict = candidate.get("recheck") or candidate.get("first")
                decision = verdict['verdict'] if verdict else 'unjudged'
                if candidate["error"]:
                    decision = "unconfirmed (" + decision + ")"
                lines.append(f"   Evidence: {candidate['title']} | {decision} | {candidate.get('url', candidate['id'])}")
                if verdict:
                    lines.append("   Reason: " + verdict["reason"])
                    if verdict.get("quote"):
                        lines.append('   Quote: "' + verdict["quote"] + '"')
            for error in record["errors"]:
                lines.append("   Incomplete: " + error)
    lines += ["Open = no closure found in this bounded search. Unknown feasibility is not a low-feasibility verdict."]
    return "\n".join(lines)


def save(report, path):
    atomic_json(path, report)


def load(path, tb):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("kind") != "gap-investigation" or report.get("corpus_sha256") != fingerprint(tb):
        raise ValueError("Saved review does not match the current corpus, model settings or prompts; review again")
    return report
