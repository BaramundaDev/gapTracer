"""Closure check and doability: the only places a model judges a gap."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from . import tokens
from .gateway import Gateway, ModelOutputInvalid
from .pipeline import fill, load_prompt
from .schema import ClosureOutput, DoabilityOutput, GapCard, Note

# a candidate paper: {"id": ..., "title": ..., "abstract": ..., "year": ...}
Candidates = Callable[[str], List[dict]]


def inside_candidates(toolbox, k: int = 3) -> Candidates:
    """Papers already in the map that might fill the gap."""
    def find(statement: str) -> List[dict]:
        out, seen = [], set()
        for n in toolbox.search.search(statement, k * 4):
            pid = n.id if n.type == "paper" else n.paper_id
            paper = toolbox.notes.get(pid) if pid else None
            if paper is None or paper.id in seen:
                continue
            seen.add(paper.id)
            out.append({"id": paper.id, "title": paper.name, "abstract": paper.l1 or paper.l0,
                        "year": paper.paper_date.year if paper.paper_date else "n.d."})
            if len(out) >= k:
                break
        return out
    return find


def check_closure(card: GapCard, gateway: Gateway, sources: Iterable[Candidates],
                  exclude: Iterable[str] = (), max_papers: int = 3) -> GapCard:
    """Judge the nearest papers one at a time. `sources` is a list of candidate
    finders: the map itself, then an outside scholarly index (phase 5)."""
    template = load_prompt("closure_judge")
    skip = set(exclude) | set(card.note_ids)       # papers that stated the gap cannot close it
    status, decided, tiers = "unchecked", "", set()
    judged = valid = failures = searches = 0
    diagnostics = {"kind": "closure", "gap_id": card.id, "max_papers": max_papers, "errors": []}
    if max_papers < 1:
        raise ValueError("max_papers must be positive")
    for find in sources:
        try:
            candidates = sorted(find(card.statement), key=lambda p: p["id"])
            searches += 1
        except (OSError, TimeoutError) as exc:
            failures += 1
            diagnostics["errors"].append(f"Search: {type(exc).__name__}: {exc}")
            continue
        except Exception as exc:
            gateway.log.append({**diagnostics, "unexpected_error": str(exc)})
            raise
        for paper in candidates:
            if paper["id"] in skip or judged >= max_papers:
                continue
            skip.add(paper["id"])
            judged += 1
            prompt = fill(template, gap_statement=card.statement, year=paper.get("year", "n.d."),
                          title=paper["title"], abstract=tokens.truncate(paper.get("abstract", ""), 400))
            try:
                out, tier = gateway.call("cloud_light", prompt, ClosureOutput)
            except (ModelOutputInvalid, OSError, TimeoutError) as exc:
                failures += 1
                diagnostics["errors"].append(f"Judge {paper['id']}: {type(exc).__name__}: {exc}")
                continue
            except Exception as exc:
                gateway.log.append({**diagnostics, "unexpected_error": str(exc)})
                raise
            valid += 1
            tiers.add(tier)
            if out.verdict == "fills":
                status, decided = "closed", paper["id"]
                break
            if out.verdict == "partly" and status != "partly":
                status, decided = "partly", paper["id"]
        if status == "closed":
            break
    if status == "unchecked" and valid and not failures:
        status = "open"
    diagnostics.update(searches_completed=searches, attempted=judged, valid=valid, failures=failures,
                       status=status, scope="bounded candidate search; not proof of global absence")
    gateway.log.append(diagnostics)
    card.status, card.decided_by = status, decided
    card.judged_by = "+".join(sorted(tiers)) or "none"
    return card


def score_doability(card: GapCard, gateway: Gateway, evidence: List[Note]) -> GapCard:
    summaries = "\n".join(f"- {n.name}: {n.l1 or n.l0}" for n in evidence)
    prompt = fill(load_prompt("doability"), gap_statement=card.statement,
                  l1_summaries=tokens.truncate(summaries, 1200))
    try:
        out, tier = gateway.call("cloud_light", prompt, DoabilityOutput)
        card.doability = out.total
        card.judged_by = "+".join(sorted(set(filter(None, card.judged_by.split("+"))) | {tier}))
    except ModelOutputInvalid:
        card.doability = None
    return card
