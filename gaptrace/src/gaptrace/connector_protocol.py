"""Small, dependency-free knowledge connector shared by MCP and AWS Lambda.

It never runs an LLM. Source preparation remains on the user's local machine.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, timezone


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


TOOLS = [
    {"name": "search", "description": "Search the locally prepared research knowledge base. Context is a retrieval aid, not evidence.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "fetch", "description": "Fetch an original source passage or a previously retrieved external abstract by id.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}},
    {"name": "get_research_task", "description": "Get the analysis task, blockers, cutoff and evidence rules; extraction is already completed locally.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "get_attempt", "description": "Get one verified old objective/blocker and its exact source passage.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}},
    {"name": "search_literature", "description": "Search current scholarly abstracts up to the task's frozen analysis date. Do not infer novelty from no results.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "submit_analysis", "description": "Save model-generated hypotheses with exact evidence quotes and rechecks. These are not independent evaluation labels.",
     "inputSchema": {"type": "object", "properties": {"task_sha256": {"type": "string"}, "model": {"type": "string"},
         "assessments": {"type": "array", "maxItems": 10, "items": {"type": "object"}}},
         "required": ["task_sha256", "model", "assessments"], "additionalProperties": False}}
]


def validate_packet(packet):
    if packet.get("kind") != "gaptrace-connector-packet":
        raise ValueError("Not a prepared knowledge connector packet")
    request, knowledge = packet["task"], packet["knowledge"]
    if digest({k: v for k, v in request.items() if k != "task_sha256"}) != request["task_sha256"]:
        raise ValueError("Analysis task was modified")
    if digest(knowledge) != request["knowledge_sha256"] or knowledge["source_sha256"] != request["source_sha256"]:
        raise ValueError("Knowledge snapshot does not match the task")
    papers = sorted(packet["papers"], key=lambda p: p["id"])
    passages = sorted(packet["passages"], key=lambda p: p["id"])
    source = {"passages": passages, "papers": [{"id": p["id"], "title": p["title"], "published": p["published"]} for p in papers]}
    if digest(source) != request["source_sha256"]:
        raise ValueError("Connector source text differs from locally prepared sources")
    if any(p.get("date") and p["date"] > request["as_of"] for p in passages):
        raise ValueError("Connector packet contains future source text")
    return packet


class Connector:
    def __init__(self, packet, search=None, evidence_get=None, evidence_put=None, commit=None):
        self.packet = validate_packet(packet)
        self.task = packet["task"]
        self.passages = {p["id"]: p for p in packet["passages"]}
        self.papers = {p["id"]: p for p in packet["papers"]}
        self.searcher = search
        self.external = {}
        self.evidence_get = evidence_get or self.external.get
        self.evidence_put = evidence_put or (lambda key, value: self.external.__setitem__(key, value))
        self.commit = commit or (lambda value: None)
        self.search_calls = 0

    def call(self, name, args):
        definitions = {t["name"]: t for t in TOOLS}
        if name not in definitions:
            raise ValueError("Unknown tool")
        schema = definitions[name]["inputSchema"]
        if set(args) - schema["properties"].keys() or not set(schema.get("required", [])) <= args.keys():
            raise ValueError("Tool arguments do not match its schema")
        return getattr(self, name)(**args)

    def search(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Use a 1-500 character query")
        terms = set(re.findall(r"[\w-]+", query.lower()))
        scored = []
        for pid, p in self.passages.items():
            context = self.packet["knowledge"]["contexts"].get(pid, {}).get("text", "")
            text = p["text"] + " " + context + " " + self.papers[p["paper_id"]]["title"]
            words = set(re.findall(r"[\w-]+", text.lower()))
            overlap = terms & words
            if overlap:
                scored.append((len(overlap) / max(1, math.sqrt(len(words))), pid, context))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return {"results": [{"id": pid, "title": self.papers[self.passages[pid]["paper_id"]]["title"],
                              "url": "gaptrace://passage/" + pid, "retrieval_context": context, "context_is_evidence": False}
                             for _, pid, context in scored[:5]], "retrieval": "bounded contextual lexical search"}

    def fetch(self, id):
        if id in self.passages:
            p = self.passages[id]
            return {"id": id, "title": self.papers[p["paper_id"]]["title"], "text": p["text"],
                    "published": p["date"], "paper_id": p["paper_id"], "url": "gaptrace://passage/" + id,
                    "evidence_level": "original corpus passage"}
        value = self.evidence_get(id)
        if value is None:
            raise ValueError("Fetch only existing source ids or ids returned by scholarly search")
        return value

    def get_research_task(self):
        return {"task_sha256": self.task["task_sha256"], "as_of": self.task["as_of"],
                "requirements": self.task["requirements"], "analysis_prompt": self.task["prompt"],
                "attempts": [{k: o[k] for k in ("id", "goal", "mechanism", "published", "passage_id")} for o in self.task["attempts"]],
                "assessment_schema": {"attempt_id": "verified attempt id", "new_source_id": "fetched source id",
                    "assessment": {"mechanism_plausible": "bool", "already_solved": "bool", "quote": "exact source span",
                        "bridge": "hypothesis", "assumptions": ["..."], "experiment": "falsifiable test", "falsifier": "failure condition",
                        "importance": "null or 0..4", "importance_reason": "scientific value, separately from cost"},
                    "recheck": "same assessment schema, required for positive or already-solved claims"}}

    def get_attempt(self, id):
        match = next((a for a in self.task["attempts"] if a["id"] == id), None)
        if match is None:
            raise ValueError("Unknown prepared attempt")
        return {"observation": match, "source": self.fetch(match["passage_id"])}

    def search_literature(self, query):
        if self.searcher is None:
            raise ValueError("Scholarly search is not configured")
        if self.search_calls >= 20:
            raise ValueError("Connector process search allowance exhausted")
        self.search_calls += 1
        works = self.searcher(query, self.task["provider"], 5, self.task["as_of"])
        results = []
        for work in works[:5]:
            if work["published"] > self.task["as_of"] or not work["url"].startswith("https://"):
                raise ValueError("Out-of-scope scholarly evidence")
            value = {**work, "text": work["abstract"], "source_sha256": digest(work), "retrieved_utc": datetime.now(timezone.utc).isoformat()}
            self.evidence_put(work["id"], value)
            results.append({k: value[k] for k in ("id", "title", "published", "url")})
        return {"results": results, "as_of": self.task["as_of"], "scope": "bounded scholarly abstracts, not exhaustive novelty review"}

    def _assessment(self, value, source):
        fields = {"mechanism_plausible", "already_solved", "quote", "bridge", "assumptions", "experiment", "falsifier", "importance", "importance_reason"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Use the exact supplied assessment schema")
        if any(type(value[k]) is not bool for k in ("mechanism_plausible", "already_solved")):
            raise ValueError("Assessment flags must be booleans")
        if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 1800 for k in ("quote", "bridge", "experiment", "falsifier")):
            raise ValueError("Provide bounded source quote, bridge hypothesis, experiment and falsifier")
        if value["quote"] not in source["text"]:
            raise ValueError("The model quote is not in the fetched source text")
        if value["importance"] is not None and (type(value["importance"]) is not int or not 0 <= value["importance"] <= 4):
            raise ValueError("Importance must be null or an integer 0..4")
        if not isinstance(value["importance_reason"], str) or (value["importance"] is not None and not value["importance_reason"].strip()):
            raise ValueError("Scored importance needs a rationale")
        if not isinstance(value["assumptions"], list) or len(value["assumptions"]) > 5 or any(not isinstance(a, str) or len(a) > 1000 for a in value["assumptions"]):
            raise ValueError("Provide at most five bounded assumptions")
        return value

    def submit_analysis(self, task_sha256, model, assessments):
        if task_sha256 != self.task["task_sha256"] or not isinstance(model, str) or not 1 <= len(model) <= 150:
            raise ValueError("Submission must identify the exact prepared task and analyzing model")
        if not isinstance(assessments, list) or not 1 <= len(assessments) <= 10:
            raise ValueError("Submit 1-10 bounded assessments")
        opportunities, seen = [], set()
        for item in assessments:
            if set(item) != {"attempt_id", "new_source_id", "assessment", "recheck"}:
                raise ValueError("Each assessment requires attempt_id, new_source_id, assessment and recheck")
            old = self.get_attempt(item["attempt_id"])
            source = self.fetch(item["new_source_id"])
            if not old["observation"]["published"] < source["published"] <= self.task["as_of"]:
                raise ValueError("A new capability must postdate the blocked attempt and precede analysis")
            key = (item["attempt_id"], item["new_source_id"])
            if key in seen:
                raise ValueError("Duplicate opportunity")
            seen.add(key)
            first = self._assessment(item["assessment"], source)
            second = self._assessment(item["recheck"], source) if item["recheck"] is not None else None
            if (first["mechanism_plausible"] or first["already_solved"]) and second is None:
                raise ValueError("Positive and already-solved findings require an explicit recheck")
            status = "weak_bridge"
            if second:
                if first["already_solved"] and second["already_solved"]:
                    status = "already_solved"
                elif first["already_solved"] != second["already_solved"]:
                    status = "unchecked"
                elif first["mechanism_plausible"] and second["mechanism_plausible"]:
                    status = "reactivation_hypothesis"
                else:
                    status = "unchecked"
            importance = min(first["importance"], second["importance"]) if second and first["importance"] is not None and second["importance"] is not None else None
            opportunities.append({"id": "opp-" + digest(key)[:20], "attempt_id": item["attempt_id"], "old_source": old["source"],
                "new_source": source, "assessment": first, "recheck": second, "status": status,
                "importance_estimate": importance if status == "reactivation_hypothesis" else None,
                "feasibility": None, "novelty": "unchecked", "error": None, "search_incomplete": True})
        solved = {o["attempt_id"]: o["id"] for o in opportunities if o["status"] == "already_solved"}
        for opportunity in opportunities:
            if opportunity["attempt_id"] in solved and opportunity["status"] == "reactivation_hypothesis":
                opportunity["status"] = "already_solved"
                opportunity["closure_evidence"] = solved[opportunity["attempt_id"]]
                opportunity["importance_estimate"] = None
        order = {"reactivation_hypothesis": 0, "unchecked": 1, "weak_bridge": 2, "already_solved": 3}
        opportunities.sort(key=lambda p: (order[p["status"]], -(p["importance_estimate"] if p["importance_estimate"] is not None else -1), p["id"]))
        report = {"kind": "reactivation-analysis", "version": 1, "task_sha256": task_sha256,
                  "source_sha256": self.task["source_sha256"], "knowledge_sha256": self.task["knowledge_sha256"],
                  "as_of": self.task["as_of"], "engine": "connector", "model": model,
                  "opportunities": opportunities, "searches": [], "errors": [],
                  "provenance_checks": "Task, chronology and exact source quotes checked; semantic assessments are model-generated hypotheses, not independent labels"}
        self.commit(report)
        return {"submission_id": digest(report), "analysis": report}
