"""Prepare locally first; only then choose local analysis or an external MCP client."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from . import literature
from .ingest import PaperImport, ingest
from .knowledge import digest, prepare, source_fingerprint
from .reactivation import analyze, task
from .recovery import atomic_json, run_lock
from .research import _source_from_works
from .workspace import open_toolbox


def export_packet(tb, request, destination):
    knowledge = json.loads((tb.data_dir / "knowledge.json").read_text(encoding="utf-8"))
    if request["source_sha256"] != source_fingerprint(tb) or request["knowledge_sha256"] != digest(knowledge):
        raise ValueError("Connector task is stale; prepare a new analysis task")
    packet = {"kind": "gaptrace-connector-packet", "version": 1, "task": request, "knowledge": knowledge,
              "papers": [{"id": n.id, "title": n.name, "published": str(n.paper_date)} for n in tb.notes.values() if n.type == "paper"],
              "passages": [p.model_dump(mode="json") for p in tb.passages.all()],
              "sharing": "Contains source excerpts. Upload only to your authenticated private workspace.",
              "model_weights_included": False}
    atomic_json(destination, packet)
    return packet


def run(topic, destination, settings, gateway, papers=100, provider="openalex", as_of="", source_file=None,
        metadata_tokens=50, engine="", choose_engine=None, resume=False, max_attempts=5, progress=None,
        historical_through="", full_text=True):
    destination = Path(destination)
    progress = progress or (lambda message: None)
    if not 1 <= papers <= 200:
        raise ValueError("Choose 1-200 papers; the default is a provisional 100-paper budget")
    if engine not in {"", "local", "connector", "prepare"}:
        raise ValueError("Choose local, connector or prepare")
    if settings.cloud.enabled or gateway.cloud_calls:
        raise ValueError("All collection-stage extraction and preparation must use local inference only")
    cutoff = date.fromisoformat(as_of) if as_of else date.today()
    if cutoff > date.today():
        raise ValueError("as_of cannot be in the future")
    with run_lock(destination):
        manifest_path = destination / "discovery.json"
        if resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("kind") != "opportunity-discovery":
                raise ValueError("This directory is not a prepared opportunity discovery run")
            if topic and topic != manifest["topic"]:
                raise ValueError("Topic differs from the saved discovery run")
            topic, provider, cutoff = manifest["topic"], manifest["provider"], date.fromisoformat(manifest["as_of"])
            source = PaperImport.model_validate_json((destination / "papers.json").read_bytes())
            if digest(source.model_dump(mode="json")) != manifest["source_sha256"]:
                raise ValueError("Source files changed; start a new discovery run")
            metadata_tokens = manifest["metadata_tokens"]
        else:
            if destination.exists():
                raise ValueError("Choose a new discovery directory or use --resume")
            if source_file:
                source = PaperImport.model_validate_json(Path(source_file).read_bytes())
                if len(source.papers) > papers:
                    raise ValueError("Source exceeds the chosen paper budget; raise --papers or supply a frozen subset")
                works, text_receipts = [], {}
                scope = "sectioned source text supplied by user"
            else:
                if not topic.strip() or len(topic) > 400:
                    raise ValueError("Supply a research topic or a sectioned --source file")
                progress(f"Collecting up to {papers} unique papers in historical and newer strata")
                if historical_through:
                    historical = date.fromisoformat(historical_through)
                else:
                    try:
                        historical = cutoff.replace(year=cutoff.year - 5)
                    except ValueError:
                        historical = date(cutoff.year - 5, 2, 28)
                if not date(1900, 1, 1) <= historical < cutoff:
                    raise ValueError("Historical cutoff must precede analysis and be on/after 1900-01-01")
                old_budget = min(papers - 1, max(1, papers * 2 // 5)) if papers > 1 else 1
                works, seen = [], set()
                for query, start, end, count in [(topic + " limitations", "1900-01-01", historical.isoformat(), old_budget),
                                                  (topic, (historical + timedelta(days=1)).isoformat(), cutoff.isoformat(), papers - old_budget)]:
                    if count < 1:
                        continue
                    for work in literature.collect_window(query, start, end, count, provider, progress=progress):
                        identity = work.get("doi") or work["title"].casefold().strip()
                        if identity not in seen and len(works) < papers:
                            seen.add(identity)
                            works.append(work)
                if not works:
                    raise ValueError("No usable papers found; try a different query or supply full sectioned texts")
                source = _source_from_works(works)
                text_receipts = {}
                if full_text:
                    from .paper_text import sections
                    value = source.model_dump(mode="json")
                    for index, (work, paper) in enumerate(zip(works, source.papers)):
                        progress(f"Retrieving source text {index + 1}/{len(works)}: {paper.title}")
                        parsed, receipt = sections(work, cutoff.isoformat())
                        value["papers"][index]["sections"] = parsed
                        text_receipts[paper.id] = receipt
                    source = PaperImport.model_validate(value)
                full_count = sum(r["evidence_level"] == "html full text" for r in text_receipts.values())
                scope = f"{full_count} parsed HTML texts, {len(source.papers) - full_count} abstract screening fallbacks; confirm original full sources"
            if any(p.published > cutoff for p in source.papers):
                raise ValueError("Source contains post-as_of papers")
            destination.mkdir(parents=True)
            atomic_json(destination / "papers.json", source.model_dump(mode="json"))
            manifest = {"kind": "opportunity-discovery", "version": 1, "topic": topic or "Imported research corpus",
                        "as_of": cutoff.isoformat(), "provider": provider, "requested_papers": papers,
                        "collected_papers": len(source.papers), "source_scope": scope, "sources": works,
                        "text_receipts": text_receipts,
                        "source_sha256": digest(source.model_dump(mode="json")), "metadata_tokens": metadata_tokens,
                        "state": "collected", "created_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(manifest_path, manifest)
        if gateway.cache_dir is None:
            gateway.cache_dir = destination / "checkpoints" / "model-responses"
        corpus = destination / "corpus"
        preparation_config = digest({"model": vars(settings.model), "limits": vars(settings.limits),
                                     "metadata_tokens": metadata_tokens})
        if manifest.get("preparation_config", preparation_config) != preparation_config:
            raise ValueError("Local preparation settings changed; start a new discovery run")
        manifest["preparation_config"] = preparation_config
        atomic_json(manifest_path, manifest)
        try:
            if not corpus.exists():
                progress("Extracting notes and checking original links using the local model")
                ingest(source, corpus, cutoff, gateway, settings, progress=progress,
                       checkpoint_dir=destination / "checkpoints" / "extraction")
            tb = open_toolbox(corpus, settings)
            progress("Preparing local contextual retrieval metadata and blocker/capability observations")
            knowledge = prepare(tb, gateway, metadata_tokens, progress, destination / "checkpoints" / "knowledge")
            manifest.update(state="awaiting_engine_choice", local_preparation_complete=True,
                            knowledge_coverage=knowledge["coverage"], preparation_errors=knowledge["errors"],
                            corpus=str(corpus.resolve()), local_model=settings.model.local,
                            extraction_cloud_calls=gateway.cloud_calls)
            atomic_json(manifest_path, manifest)
            # This is deliberately AFTER every source has gone through local preparation.
            selected = engine or (choose_engine(manifest) if choose_engine else "prepare")
            if selected not in {"local", "connector", "prepare"}:
                raise ValueError("Choose local, connector or prepare")
            manifest["engine"] = selected
            if selected == "local":
                result = analyze(tb, gateway, cutoff.isoformat(), provider, max_attempts=max_attempts, progress=progress)
                manifest.update(state="analysis_complete", opportunities=len(result["opportunities"]), analysis="corpus/reactivation.json")
            elif selected == "connector":
                request = task(tb, cutoff.isoformat(), provider, max_attempts=max_attempts)
                atomic_json(corpus / "reactivation-task.json", request)
                export_packet(tb, request, destination / "connector-packet.json")
                manifest.update(state="awaiting_external_analysis", task_sha256=request["task_sha256"],
                                connector_packet="connector-packet.json", external_model_called=False)
            else:
                manifest["state"] = "awaiting_engine_choice"
            atomic_json(manifest_path, manifest)
            return manifest
        except (Exception, KeyboardInterrupt) as exc:
            # Prepared corpora and checkpoints survive interruption or a cancelled choice.
            if manifest.get("state") != "awaiting_engine_choice":
                manifest["state"] = "interrupted"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(manifest_path, manifest)
            raise
