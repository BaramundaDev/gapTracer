"""Topic collection, grounded extraction, gap investigation and proposed experiments."""
from __future__ import annotations

import json
import hashlib
import re
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from . import literature
from .gap_review import GapReviewer, fingerprint, render, save
from .gateway import Gateway, ModelOutputInvalid
from .ingest import PaperImport, ingest
from .pipeline import fill, load_prompt
from .workspace import open_toolbox
from .recovery import recover, run_lock, read_checkpoint, write_checkpoint, transient, atomic_text

TOPICS = (
    "Retrieval augmented generation",
    "Continual learning",
    "Efficient small language models",
    "Graph neural networks",
    "AI for scientific discovery",
)


class Direction(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    proposed_test: str = Field(min_length=1, max_length=1500)
    success_measure: str = Field(min_length=1, max_length=600)
    passage_id: str
    quote: str = Field(min_length=1)


def new_destination(topic: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:55] or "topic"
    return Path("data/research") / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def _source_from_works(works):
    return PaperImport(papers=[{
        "id": work.get("corpus_id") or re.sub(r"[^a-zA-Z0-9-]", "-", work["id"]),
        "title": work["title"], "published": work["published"],
        "sections": [{"heading": "Abstract", "text": work["abstract"]}],
    } for work in works])


def run(topic: str, destination: Path, settings, gateway: Gateway, provider="openalex",
        papers=30, k=3, as_of="", progress=None, resume=False, auto_retries=2) -> dict:
    if not isinstance(auto_retries, int) or not 0 <= auto_retries <= 5:
        raise ValueError("Automatic retries must be between 0 and 5")
    with run_lock(destination):
        return _run(topic, destination, settings, gateway, provider, papers, k, as_of,
                    progress, resume, auto_retries)


def _run(topic, destination, settings, gateway, provider, papers, k, as_of, progress, resume, auto_retries):
    destination = Path(destination)
    progress = progress or (lambda message: None)
    def collection_retry(event):
        progress(f"Collection interrupted; automatically retrying in {event['delay_seconds']}s "
                 f"({event['attempt']}/{event['max_retries']}).")
    if resume:
        manifest = json.loads((destination / "research.json").read_text(encoding="utf-8"))
        if manifest.get("kind") != "topic-research":
            raise ValueError("This directory is not a saved topic research run")
        if topic.strip() and topic.strip() != manifest["topic"]:
            raise ValueError("The supplied topic does not match the saved research run")
        topic, provider, as_of = manifest["topic"], manifest["provider"], manifest["as_of"]
        papers, k = manifest["requested_papers"], manifest.get("review_k", k)
        works = manifest["sources"]
        source_path = destination / "papers.json"
        try:
            source = PaperImport.model_validate_json(source_path.read_bytes())
        except (FileNotFoundError, ValueError):
            source = _source_from_works(works)
            checksum = hashlib.sha256(source.model_dump_json().encode()).hexdigest()
            if manifest.get("source_sha256", checksum) != checksum:
                raise ValueError("Saved source metadata is inconsistent; refusing to replace the downloaded papers")
            if source_path.exists():
                source_path.replace(source_path.with_name(source_path.name + ".corrupt-" + uuid4().hex[:8]))
            save(source.model_dump(mode="json"), source_path)
            progress("Repaired downloaded-paper file from the saved source metadata; no download needed.")
    topic = topic.strip()
    if not topic or len(topic) > 500:
        raise ValueError("Enter a topic between 1 and 500 characters")
    if not 1 <= papers <= 100 or not 1 <= k <= 10:
        raise ValueError("Choose 1-100 papers and 1-10 gaps to review")
    cutoff = date.fromisoformat(as_of) if as_of else date.today()
    if cutoff > date.today():
        raise ValueError("as_of cannot be in the future")
    if destination.exists() and not resume:
        raise ValueError(f"Choose a new data directory. {destination} already exists.")
    if not resume:
        progress(f"[1/4] Collecting up to {papers} paper abstracts about: {topic}")
        works = recover(lambda: literature.collect(topic, provider, papers, cutoff.isoformat(), progress=progress),
                        "collection", auto_retries, collection_retry)
        if not works:
            raise ValueError("No papers with usable abstracts found. Try a broader topic or another provider.")
        source = _source_from_works(works)
        # Keep the downloaded evidence even if local inference fails later.
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {"kind": "topic-research", "topic": topic, "provider": provider,
                    "as_of": cutoff.isoformat(), "requested_papers": papers, "collected_papers": len(works),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "evidence_level": "abstract",
                    "sources": [{**work, "corpus_id": paper.id} for work, paper in zip(works, source.papers)],
                    "state": "collected", "directions": [], "errors": []}
        save(manifest, destination / "research.json")
        save(source.model_dump(mode="json"), destination / "papers.json")
    else:
        progress(f"[1/4] Reusing {len(source.papers)} downloaded papers from {destination}")
    source_hash = hashlib.sha256(source.model_dump_json().encode()).hexdigest()
    if manifest.get("source_sha256", source_hash) != source_hash:
        raise ValueError("Downloaded papers changed since this research run. Start a new run.")
    analysis_hash = hashlib.sha256(json.dumps([
        vars(settings.model), vars(settings.limits), settings.budget.output,
        {name: load_prompt(name) for name in ("extractor", "merge_confirm", "link_verifier")},
    ], sort_keys=True).encode()).hexdigest()
    corpus = destination / "corpus"
    if corpus.exists() and manifest.get("analysis_sha256") != analysis_hash:
        raise ValueError("The saved corpus used different or unknown model/extraction settings. Start a new run.")
    previous_log = manifest.get("gateway_log", [])
    if manifest.get("analysis_sha256") != analysis_hash:
        manifest.update(completed_passage_ids=[], completed_passages=0)
    manifest.update(source_sha256=source_hash, analysis_sha256=analysis_hash, review_k=k)
    manifest.pop("error", None)
    save(manifest, destination / "research.json")
    if gateway.cache_dir is None:
        gateway.cache_dir = destination / "checkpoints" / "model-responses"
    def stage_retry(event):
        manifest.setdefault("recoveries", []).append(event)
        manifest.update(state="recovering", recovery_stage=event["stage"])
        save(manifest, destination / "research.json")
        progress(f"Automatically recovering {event['stage']} in {event['delay_seconds']}s "
                 f"({event['attempt']}/{event['max_retries']}); completed work will be reused.")

    def passage_progress(message):
        if message.startswith("Extracting passage "):
            pid = message.split(": ", 1)[1].split(" (", 1)[0]
            completed = manifest.setdefault("completed_passage_ids", [])
            if pid in completed:
                completed.remove(pid)
                manifest.update(completed_passages=len(completed))
                save(manifest, destination / "research.json")
        if message.startswith(("Checkpoint saved:", "Reusing extracted passage")):
            pid = message.rsplit(": ", 1)[-1]
            completed = manifest.setdefault("completed_passage_ids", [])
            if pid not in completed:
                completed.append(pid)
            manifest.update(state="extracting", completed_passages=len(completed), last_completed_passage=pid)
            save(manifest, destination / "research.json")
        progress(message)

    try:
        if len(works) < papers:
            progress(f"Found {len(works)} usable unique papers; continuing with those available.")
        progress("[2/4] Reading abstracts: extracting methods, results, limitations and open questions")
        if corpus.exists():
            progress("Reusing the completed corpus; extraction and verification are already saved.")
            imported = json.loads((corpus / "import.json").read_text(encoding="utf-8"))
        else:
            imported = recover(lambda: ingest(source, corpus, cutoff, gateway, settings, progress=passage_progress,
                               checkpoint_dir=destination / "checkpoints" / "extraction"),
                               "extraction and verification", auto_retries, stage_retry)
        manifest.update(state="extracted", corpus=str(corpus.resolve()), extraction=imported["stats"])
        save(manifest, destination / "research.json")
        progress("[3/4] Comparing evidence and checking whether later work addresses the gap candidates")
        tb = recover(lambda: open_toolbox(corpus, settings), "opening corpus", auto_retries, stage_retry)
        cards = tb.gaps()
        context_hash = hashlib.sha256(json.dumps([fingerprint(tb), provider, cutoff.isoformat()]).encode()).hexdigest()
        review_path = destination / "checkpoints" / f"review-{context_hash}.json"
        reviewer = GapReviewer(tb, gateway=gateway, provider=provider, as_of=cutoff.isoformat(), progress=progress,
                               recover_transient=True)
        def validate_review(value):
            if not isinstance(value["reviews"], dict) or not isinstance(value["searches"], list):
                raise ValueError("Invalid review checkpoint")
            from .schema import GapCard
            for gid, record in value["reviews"].items():
                if GapCard.model_validate(record["card"]).id != gid:
                    raise ValueError("Invalid saved gap identity")
            return value
        saved_review = read_checkpoint(review_path, validate_review)
        if saved_review:
            reviewer.reviews = saved_review["reviews"]
            reviewer.searches = [record for record in saved_review["searches"] if not record["error"]]
            progress(f"Reusing {len(reviewer.reviews)} completed gap reviews.")
        def checkpoint_review(current):
            write_checkpoint(review_path, {"reviews": current.reviews, "searches": current.searches})
            manifest.update(state="reviewing", reviewed_gap_ids=list(current.reviews))
            save(manifest, destination / "research.json")
        review = recover(lambda: reviewer.run(cards, k, checkpoint=checkpoint_review),
                         "gap review", auto_retries, stage_retry)
        save(review, corpus / "gap-review.json")
        manifest.update(state="reviewed", gap_candidates=len(cards), review=review)
        manifest["directions"], manifest["errors"] = [], []
        save(manifest, destination / "research.json")
        progress("[4/4] Brainstorming proposed experiments from the reviewed evidence")
        reviewed = {item["gap_id"]: item for item in review["reviews"]}
        for card in review["ranked"]:
            record = reviewed.get(card["id"])
            if not record or not record["fully_supported"] or card["status"] != "open":
                continue
            evidence = [{"passage_id": edge["passage_id"], "quote": edge["quote"]}
                        for edge in record["evidence"] if edge["supported"]][:4]
            payload = {"topic": topic, "gap": card["statement"], "status": card["status"], "evidence": evidence}
            try:
                direction, _ = recover(lambda: gateway.call("local", fill(load_prompt("research_direction"),
                                       payload=json.dumps(payload)), Direction),
                                       "experiment proposal", auto_retries, stage_retry)
                if not any(direction.passage_id == edge["passage_id"] and direction.quote in edge["quote"]
                           and direction.quote.strip() for edge in evidence):
                    raise ValueError("Proposed experiment did not cite an exact supplied evidence quote")
                manifest["directions"].append({"gap_id": card["id"], "kind": "untested proposal",
                                               **direction.model_dump()})
                save(manifest, destination / "research.json")
            except (ModelOutputInvalid, ValueError, RuntimeError, OSError) as exc:
                if transient(exc):
                    raise
                manifest["errors"].append(f"Brainstorming {card['id']}: {exc}")
        partial = bool(manifest["errors"] or any(item["errors"] for item in review["reviews"]))
        manifest.update(state="partial" if partial else "complete", model=settings.model.local,
                        gateway_log=previous_log + gateway.log)
        manifest.pop("recovery_stage", None)
        atomic_text(destination / "report.md", render_report(manifest))
        save(manifest, destination / "research.json")
        progress(f"Research report saved to {destination / 'report.md'}")
        if partial:
            progress("Some assessments remain incomplete; inspect the report before using these candidates.")
        return manifest
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                        error=f"{type(exc).__name__}: {exc}", gateway_log=previous_log + gateway.log)
        save(manifest, destination / "research.json")
        if isinstance(exc, KeyboardInterrupt):
            progress(f'Interrupted safely. Resume with: python -m gaptrace.cli research --resume "{destination}"')
            raise
        raise RuntimeError(f"Research incomplete: {exc}. Collected papers and progress are saved in {destination}. "
                           f'Resume with: python -m gaptrace.cli research --resume "{destination}"') from exc


def render_report(report):
    lines = [f"# Research: {report['topic']}", "",
             f"Read {report['collected_papers']} of up to {report['requested_papers']} requested papers from "
             f"{report['provider']}, through {report['as_of']}.", "",
             "Evidence scope: abstracts only. These are research candidates and untested experiment proposals. "
             "A bounded search cannot establish universal novelty.", "",
             "## Gap investigation", "", render(report["review"]), "", "## Proposed experiments", ""]
    if not report["review"]["ranked"]:
        lines.append("No evidence-backed gap candidates were extracted. Try more papers, full sectioned texts via ingest, or a different topic.")
    if not report["directions"]:
        lines.append("No grounded experiment proposal was produced. Inspect the gap review and extraction statistics before drawing conclusions.")
    for item in report["directions"]:
        lines += [f"### {item['question']}", "", f"Gap: {item['gap_id']} (untested proposal)", "",
                  item["proposed_test"], "", "Success measure: " + item["success_measure"], "",
                  f"Evidence [{item['passage_id']}]: {item['quote']}", ""]
    lines += ["## Papers", ""]
    for paper in report["sources"]:
        lines.append(f"- [{paper['title']}]({paper['url']}) ({paper['published']}; {paper['corpus_id']})")
    stats = report["extraction"]
    lines += ["", "## Reading summary", "",
              f"Passages read: {stats['passages']} | Notes: {stats['notes']} | "
              f"Invalid extraction passages: {stats['invalid']} | Verified links: {stats['links']['verified']}", ""]
    if report["errors"]:
        lines += ["## Incomplete steps", "", *report["errors"], ""]
    return "\n".join(lines)
