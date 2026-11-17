"""Find time-ordered old-blocker/new-enabler research hypotheses."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from . import literature, tokens
from .knowledge import digest, excerpt, load, source_fingerprint
from .pipeline import fill, load_prompt
from .recovery import atomic_json
from .search import rrf


class BridgeAssessment(BaseModel):
    mechanism_plausible: bool
    already_solved: bool
    quote: str = Field(min_length=1, max_length=1600)
    bridge: str = Field(min_length=1, max_length=1000)
    assumptions: list[str] = Field(default_factory=list, max_length=5)
    experiment: str = Field(min_length=1, max_length=1200)
    falsifier: str = Field(min_length=1, max_length=700)
    importance: int | None = Field(default=None, ge=0, le=4)
    importance_reason: str = Field(default="", max_length=700)


def task(tb, as_of="", provider="openalex", max_attempts=5, candidates_per_attempt=3):
    knowledge = load(tb)
    end = date.fromisoformat(as_of) if as_of else date.today()
    if end > date.today() or any(p.date and p.date > end for p in tb.passages.all()):
        raise ValueError("Analysis date must cover the source corpus and cannot be in the future")
    if not 1 <= max_attempts <= 10 or not 1 <= candidates_per_attempt <= 5:
        raise ValueError("Use 1-10 blocked attempts and 1-5 candidates per attempt")
    attempts = [o for o in knowledge["observations"] if o["role"] == "blocked_attempt" and o["verified"] and o["published"]]
    attempts.sort(key=lambda o: (o["published"], o["id"]))
    value = {"kind": "reactivation-task", "version": 1, "source_sha256": knowledge["source_sha256"],
             "knowledge_sha256": digest(knowledge), "as_of": end.isoformat(), "provider": provider,
             "max_attempts": max_attempts, "candidates_per_attempt": candidates_per_attempt,
             "attempts": attempts[:max_attempts],
             "capabilities": [o for o in knowledge["observations"] if o["role"] == "new_capability" and o["verified"]],
             "prompt": load_prompt("reactivation"), "prepared_utc": datetime.now(timezone.utc).isoformat(),
             "requirements": ["Quote actual passages, not generated retrieval metadata.",
                 "Search for both newer enabling capabilities and already completed solutions.",
                 "A mechanistic bridge is a hypothesis; propose an experiment that can refute it.",
                 "Report scientific importance separately from resource feasibility."]}
    value["task_sha256"] = digest(value)
    return value


def _source(tb, finding):
    p = tb.passages.get(finding["passage_id"])
    if not p or finding["quote"] not in p.text or p.paper_id != finding["paper_id"]:
        raise ValueError("Prepared observation has lost its source provenance")
    return {"id": p.id, "paper_id": p.paper_id, "published": str(p.date), "text": p.text,
            "title": tb.notes[p.paper_id].name if p.paper_id in tb.notes else p.paper_id,
            "evidence_level": "corpus passage"}


def validate_assessment(assessment, source):
    result = BridgeAssessment.model_validate(assessment)
    if not result.quote.strip() or result.quote not in source["text"]:
        raise ValueError("Assessment quote is absent from the exact supplied new source")
    if result.importance is not None and not result.importance_reason.strip():
        result.importance = None
    return result


def analyze(tb, gateway, as_of="", provider="openalex", max_attempts=5, candidates_per_attempt=3,
            search=None, progress=None):
    request = task(tb, as_of, provider, max_attempts, candidates_per_attempt)
    progress = progress or (lambda message: None)
    search = search or literature.search
    sources, searches, opportunities, errors = {}, [], [], []
    from .knowledge import passage_index, contextual_embedder
    prepared = load(tb)
    index = passage_index(tb, prepared, contextual_embedder(tb) if tb.settings.switches.hybrid_search else None)
    for old in request["attempts"]:
        old_source = _source(tb, old)
        candidates = {}
        for new in request["capabilities"]:
            if new["paper_id"] != old["paper_id"] and new["published"] and old["published"] < new["published"] <= request["as_of"]:
                source = _source(tb, new)
                candidates[source["id"]] = source
        rankings = [[n.id for n in index.search(old["goal"] + " " + old["mechanism"], 30) if n.id in candidates]]
        # Search both mechanism and direct goal; the second can reveal an already solved goal.
        for query in list(dict.fromkeys([old["goal"] + " " + old["mechanism"], old["goal"]]))[:2]:
            query = excerpt(query, 80)[:500]
            record = {"attempt_id": old["id"], "query": query, "provider": provider, "as_of": request["as_of"], "works": [], "error": None}
            progress("Searching current evidence: " + query)
            try:
                works = search(query, provider, 5, request["as_of"])
                if len(works) > 5:
                    raise ValueError("Search exceeded its five-paper allowance")
                for work in works:
                    if not old["published"] < work["published"] <= request["as_of"]:
                        continue
                    source = {**work, "text": excerpt(work["abstract"], 400), "evidence_level": "abstract"}
                    candidates.setdefault(source["id"], source)
                rankings.append([w["id"] for w in works if w["id"] in candidates])
                record["works"] = works
            except (OSError, ValueError, KeyError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            searches.append(record)
        ranked_ids = [cid for cid, _ in rrf(rankings)]
        ranked_ids.extend(cid for cid in sorted(candidates) if cid not in ranked_ids)
        ranked = [candidates[cid] for cid in ranked_ids]
        # Include a candidate from each query where possible; no model gets the unbounded pool.
        for source in ranked[:candidates_per_attempt]:
            progress(f"Testing old blocker {old['id']} against {source['id']}")
            sources[source["id"]] = source
            payload = {"old_objective": old, "old_source": old_source, "new_source": source}
            entry = {"id": "opp-" + digest([old["id"], source["id"]])[:20], "attempt_id": old["id"],
                     "old_source": old_source, "new_source": source, "status": "unchecked", "assessment": None,
                     "recheck": None, "error": None, "importance_estimate": None, "feasibility": None}
            try:
                first, _ = gateway.call("local", fill(request["prompt"], strict="", payload=json.dumps(payload)), BridgeAssessment)
                first = validate_assessment(first.model_dump(), source)
                entry["assessment"] = first.model_dump()
                # Positives and solved findings require an exact-quote recheck.
                if first.mechanism_plausible or first.already_solved:
                    second, _ = gateway.call("local", fill(request["prompt"],
                        strict="STRICT RECHECK: reject superficial links. Is the blocker actually addressed, and is the original goal already achieved?",
                        payload=json.dumps(payload)), BridgeAssessment)
                    second = validate_assessment(second.model_dump(), source)
                    entry["recheck"] = second.model_dump()
                    if first.already_solved and second.already_solved:
                        entry["status"] = "already_solved"
                    elif first.already_solved != second.already_solved:
                        entry["error"] = "Judgments disagree about whether the goal is already accomplished"
                    elif first.mechanism_plausible and second.mechanism_plausible:
                        entry["status"] = "reactivation_hypothesis"
                        if first.importance is not None and second.importance is not None:
                            entry["importance_estimate"] = min(first.importance, second.importance)
                    else:
                        entry["error"] = "Mechanism recheck did not confirm the proposed bridge"
                else:
                    entry["status"] = "weak_bridge"
            except (ValueError, RuntimeError, OSError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            if any(s["error"] for s in searches if s["attempt_id"] == old["id"]):
                entry["search_incomplete"] = True
                # A positive mechanism can still be recorded, but novelty remains unchecked.
            else:
                entry["search_incomplete"] = False
            entry["novelty"] = "unchecked"  # A bounded negative search never proves novelty.
            opportunities.append(entry)
        if not ranked:
            errors.append({"attempt_id": old["id"], "error": "No dated candidate capability found; not a negative judgment"})
    solved = {o["attempt_id"]: o["id"] for o in opportunities if o["status"] == "already_solved"}
    for opportunity in opportunities:
        if opportunity["attempt_id"] in solved and opportunity["status"] == "reactivation_hypothesis":
            opportunity["status"] = "already_solved"
            opportunity["closure_evidence"] = solved[opportunity["attempt_id"]]
            opportunity["importance_estimate"] = None
    order = {"reactivation_hypothesis": 0, "unchecked": 1, "weak_bridge": 2, "already_solved": 3}
    opportunities.sort(key=lambda o: (order[o["status"]], o["search_incomplete"],
                                     -(o["importance_estimate"] if o["importance_estimate"] is not None else -1), o["id"]))
    report = {"kind": "reactivation-analysis", "version": 1, "task_sha256": request["task_sha256"],
              "source_sha256": source_fingerprint(tb), "knowledge_sha256": request["knowledge_sha256"],
              "as_of": request["as_of"], "engine": "local", "model": tb.settings.model.local,
              "opportunities": opportunities, "searches": searches, "errors": errors,
              "gateway_log": gateway.log, "limits": {"attempts": max_attempts, "candidates_per_attempt": candidates_per_attempt},
              "limitations": ["Importance is an automated ordinal estimate, not a calibrated probability or independent label.",
                  "A reactivation hypothesis is not proof of success or novelty. Search scope is bounded.",
                  "Feasibility is intentionally separate and remains unknown without resource evidence."]}
    if tb.data_dir:
        atomic_json(tb.data_dir / "reactivation-task.json", request)
        atomic_json(tb.data_dir / "reactivation.json", report)
    return report


def render(report):
    lines = [f"Research opportunities as of {report['as_of']} ({report['engine']})"]
    if not report["opportunities"]:
        lines.append("No supported old-blocker/new-enabler pair was assessed. Inspect extraction coverage and source scope.")
    for rank, item in enumerate(report["opportunities"], 1):
        assessment = item["recheck"] or item["assessment"]
        lines += [f"{rank}. [{item['id']}] {item['status']} | importance {item['importance_estimate'] if item['importance_estimate'] is not None else 'unknown'}/4 | novelty unchecked",
                  f"   Old: {item['old_source']['title']} ({item['old_source']['published']})",
                  f"   New: {item['new_source']['title']} ({item['new_source']['published']})"]
        if assessment:
            lines += ["   Bridge hypothesis: " + assessment["bridge"], "   Test: " + assessment["experiment"],
                      "   Falsifier: " + assessment["falsifier"]]
        if item["error"]:
            lines.append("   Incomplete: " + item["error"])
    lines.append("Research hypotheses require experiments; importance and feasibility are separate.")
    return "\n".join(lines)
