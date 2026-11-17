"""The step loop: one small question per call, rebuilt from the state file
each time, with findings written down before their sources are dropped."""
from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import List

from .context import ContextBudgetExceeded, State, Trail, pack
from .gateway import Gateway, ModelOutputInvalid
from . import tokens
from .pipeline import _numeric_values as _numbers, fill, load_prompt
from .schema import Note, StepOutput, VerifierOutput
from .tools import Toolbox
from .terminal_ui import emit

NOT_FOUND = "Not found in these papers."

# Provisional ceiling, measured by the development probe in evaluation/.
_MAX_ANSWER_CHECKS = 24
_MAX_SOURCES_PER_SENTENCE = 8


class Agent:
    def __init__(self, toolbox: Toolbox, gateway: Gateway, settings, run_dir: Path):
        self.tb, self.gw, self.s = toolbox, gateway, settings
        self.run_dir = Path(run_dir)
        self.tb.run_dir = self.run_dir
        self.trail = Trail(self.run_dir)
        self.instructions = load_prompt("step_agent")
        self.tb.review_gateway = gateway

    def _answer_sources(self, cited: List[str]) -> dict:
        sources = {}
        for note_id in sorted(set(cited)):
            note = self.tb.source_note(note_id)
            ids = set(note.passage_ids)
            for origin in self.tb.notes.values():
                for link in origin.links:
                    if link.verified and (origin.id == note_id or link.target == note_id):
                        ids.add(link.evidence_passage_id)
            # Associate further measurements only after the cited note has evidence.
            if any(self.tb.source_passage(pid) is not None for pid in ids) and note.type == "method":
                names = {name.lower().strip() for name in [note.name, *note.aliases]}
                for result in self.tb.notes.values():
                    if not result.fact or result.fact.method.lower().strip() not in names:
                        continue
                    paper = self.tb.notes.get(result.paper_id)
                    if paper:
                        ids.update(link.evidence_passage_id for link in paper.links
                                   if link.verified and link.type == "reports" and link.target == result.id)
            for pid in sorted(ids):
                passage = self.tb.source_passage(pid)
                if passage is None or (note.type == "result" and passage.paper_id != note.paper_id):
                    continue
                source = sources.setdefault(pid, {"passage": passage, "note_ids": set(), "facts": []})
                source["note_ids"].add(note_id)
        for source in sources.values():
            passage = source["passage"]
            paper = self.tb.notes.get(passage.paper_id)
            if not paper:
                continue
            for link in paper.links:
                result = self.tb.notes.get(link.target)
                if (link.verified and link.type == "reports" and link.evidence_passage_id == passage.id
                        and result and result.fact and result.paper_id == passage.paper_id):
                    source["facts"].append({"paper_id": passage.paper_id, **result.fact.model_dump()})
        return sources

    def _validate_answer(self, answer: str, cited: List[str], question: str = "") -> tuple[str, List[str]]:
        emit("Checking answer claims against original passages; unsupported claims will be removed")
        started = perf_counter()
        before = len(self.gw.log)
        sources = self._answer_sources(cited)
        kept, used_notes = [], set()
        checks = 0
        template = load_prompt("answer_verifier")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(re.findall(r"\w+", sentence.lower()))
            candidates = []
            for pid, source in sorted(sources.items()):
                for index, text in enumerate(tokens.iter_chunks(source["passage"].text, self.s.limits.passage_tokens)):
                    overlap = len(words & set(re.findall(r"\w+", text.lower())))
                    candidates.append((overlap, pid, index, text, source))
            candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
            candidates = candidates[:_MAX_SOURCES_PER_SENTENCE]
            required_numbers = _numbers(sentence)
            groups = [[candidate] for candidate in candidates]
            # A comparison can be supported by premises from different papers.
            if len(candidates) > 1:
                groups.append(candidates[:4])
            supported = False
            reason = "No supporting source or verification allowance exhausted"
            supporting_ids = []
            for group in groups:
                if checks >= _MAX_ANSWER_CHECKS:
                    break
                if not required_numbers <= set().union(*(_numbers(c[3]) for c in group)):
                    reason = "Claim numbers absent from candidate evidence"
                    continue
                evidence = [{"passage_id": c[1], "paper_id": c[4]["passage"].paper_id,
                             "text": c[3], "facts": c[4]["facts"]} for c in group]
                prompt = fill(template, evidence_json=json.dumps({"question": question, "claim": sentence, "sources": evidence}))
                checks += 1
                try:
                    verdict, _ = self.gw.call("local", prompt, VerifierOutput)
                    supported, reason = verdict.supported, verdict.reason
                except (ModelOutputInvalid, ContextBudgetExceeded, OSError) as exc:
                    supported, reason = False, f"{type(exc).__name__}: {exc}"
                if supported:
                    supporting_ids = sorted({c[1] for c in group})
                    used_notes.update(note_id for c in group for note_id in c[4]["note_ids"])
                    kept.append(sentence)
                    break
            self.trail.write("answer_validation", sentence=sentence, supported=supported,
                             passage_ids=supporting_ids, reason=reason)
            emit(f"Claim {'supported by ' + ', '.join(supporting_ids) if supported else 'removed: ' + reason}")
        logs = self.gw.log[before:]
        self.trail.write("answer_validation_summary", checks=checks, limit=_MAX_ANSWER_CHECKS,
                         sentences_kept=len(kept), seconds=perf_counter() - started,
                         dispatched_calls=sum(bool(entry.get("dispatch")) for entry in logs),
                         cache_hits=sum(bool(entry.get("cached")) for entry in logs),
                         repairs=sum(bool(entry.get("repair")) for entry in logs if entry.get("dispatch")),
                         input_tokens=sum(entry.get("input_tokens", 0) for entry in logs if entry.get("dispatch")),
                         output_tokens=sum(entry.get("output_tokens", 0) for entry in logs if entry.get("dispatch")),
                         token_count_kind="configured counter; server framing excluded")
        return (" ".join(kept), sorted(used_notes)) if kept else (NOT_FOUND, [])

    def _review_result(self, step):
        from .gap_review import render
        report = self.tb.latest_review
        answer = render(report)
        cited = sorted({nid for r in report["reviews"] for nid in r["card"]["note_ids"]})
        self.trail.write("gap_review_result", report=report)
        self.trail.write("final", answer=answer, cited=cited, steps=step)
        return {"answer": answer, "cited": cited, "steps": step, "run_dir": str(self.run_dir),
                "gap_review": str(self.run_dir / "gap-review.json")}

    def ask(self, question: str) -> dict:
        emit("Searching the active corpus for evidence relevant to the question")
        # An explicit action request must execute even if a small model would
        # prematurely abstain. This dispatch never uses model-generated evidence.
        action = re.match(r"\s*(?:(?:please|can you|could you|would you|i want you to|help me)\s+)?"
                          r"(?:validate|review|investigate|rank|check)\b", question, re.I)
        if action and re.search(r"\bgaps?\b|\bgap-[\w-]+", question, re.I):
            ids = re.findall(r"\bgap-[\w-]+", question)
            count = re.search(r"\b(?:top\s+)?(\d+)\s+(?:research\s+)?gaps?\b|\btop\s+(\d+)\b", question, re.I)
            historical = re.search(r"\bas\s+of\s+(\d{4}-\d{2}-\d{2})\b", question, re.I)
            args = {"k": int(next(g for g in count.groups() if g)) if count else 3,
                    "gap_id": ids[0] if len(ids) == 1 else "",
                    "provider": "arxiv" if re.search(r"\barxiv\b", question, re.I) else "openalex",
                    "as_of": historical.group(1) if historical else ""}
            if len(ids) <= 1:
                self.trail.write("step", n=1, tier="tool", input_tokens=0, slots={}, shown=[], stubs=[],
                                 tool_request={"tool": "review_gaps", "args": args}, reason="explicit gap investigation request")
                emit("Running review_gaps: explicit request to investigate research gaps")
                digest = self.tb.call("review_gaps", args)
                self.trail.write("tool", n=1, tool="review_gaps", args=args, digest=digest)
                if self.tb.latest_review is not None:
                    return self._review_result(1)
        b = self.s.budget
        state = State(goal=question)
        notes: List[Note] = self.tb.search.search(question, self.s.retrieval.top_k)
        emit(f"Retrieved {len(notes)} notes; preparing a bounded evidence context")
        tool_text = None
        transcript: List[str] = []
        external_visible = set()
        for step in range(1, self.s.limits.max_steps + 1):
            state_text = state.render(b.state)
            if not self.s.switches.stateless_steps and transcript:   # off switch: a growing transcript
                state_text = "\n".join(transcript)[-b.state * 3:]
            packed = pack(self.instructions, question, state_text, notes, tool_text, self.s)
            emit(f"Step {step}/{self.s.limits.max_steps} maximum: selecting an evidence action")
            out, tier = self.gw.call("local", packed.prompt, StepOutput)
            shown = set(packed.full_ids) | set(packed.stub_ids)
            cited = [c for c in out.cited if self.tb.source_note(c) is not None
                     and (c in shown or c in external_visible or any(c in f["ids"] for f in state.findings))]
            # write before drop: the finding is on disk before this context is discarded
            self.trail.write("step", n=step, tier=tier, input_tokens=packed.input_tokens,
                             slots=packed.slot_tokens, shown=packed.full_ids, stubs=packed.stub_ids,
                             answer=out.answer, cited=cited,
                             tool_request=out.tool_request.model_dump() if out.tool_request else None)
            if out.state_update and self.s.switches.write_before_drop:
                state.add_finding(out.state_update, cited)
            state.save(self.run_dir, b.state)
            transcript.append(f"step {step}: {out.answer or out.state_update}")
            if out.tool_request is not None:
                tr = out.tool_request
                emit(f"Running tool: {tr.tool} | {json.dumps(tr.args, ensure_ascii=False)}")
                digest = self.tb.call(tr.tool, tr.args)
                emit(f"Tool completed: {tr.tool}")
                self.trail.write("tool", n=step, tool=tr.tool, args=tr.args, digest=digest)
                tool_text = f"TOOL RESULT ({tr.tool}):\n{digest}"
                external_visible.update(nid for nid in self.tb.external_notes if f"[{nid}]" in digest)
                if tr.tool == "review_gaps" and self.tb.latest_review is not None:
                    # This is a completed tool action, rendered from recorded assessments,
                    # rather than a newly generated factual answer bypassing grounding.
                    return self._review_result(step)
                if tr.tool == "search_notes" and tr.args.get("query"):
                    notes = self.tb.search.search(tr.args["query"], self.s.retrieval.top_k,
                                                  tr.args.get("note_type") or None)
                continue
            answer = out.answer.strip()
            if not answer or answer.upper().startswith("NOT FOUND") or not cited:
                emit("No supported cited answer returned; abstaining")
                answer, cited = NOT_FOUND, []                 # no supported answer: say so
            else:
                answer, cited = self._validate_answer(answer, cited, question)
            self.trail.write("final", answer=answer, cited=cited, steps=step)
            return {"answer": answer, "cited": cited, "steps": step, "run_dir": str(self.run_dir)}
        self.trail.write("final", answer=NOT_FOUND, cited=[], steps=self.s.limits.max_steps)
        emit("Step limit reached without a supported answer; abstaining")
        return {"answer": NOT_FOUND, "cited": [], "steps": self.s.limits.max_steps, "run_dir": str(self.run_dir)}
