"""Command line. Everything here calls the same functions as the MCP server
and the API."""
from __future__ import annotations

import json
import re
import sys
from contextlib import redirect_stdout, nullcontext
from datetime import date
from pathlib import Path
from time import perf_counter

import typer

from .bundle import PassageStore, read_bundle
from .facts import FactsTable
from .settings import load_settings
from .workspace import new_run_dir, open_toolbox
from . import tokens
from .terminal_ui import Activity

app = typer.Typer(add_completion=False, help="GapTrace: find research gaps on a small local model.")
DATA = typer.Option("data", "--data", help="Data directory")


@app.callback(invoke_without_command=True)
def initialize(ctx: typer.Context):
    """Configure real token accounting for every ordinary product command."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    try:
        tokens.configure(load_settings())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if ctx.invoked_subcommand is None:
        from .terminal import run_shell
        run_shell(app, "data")


@app.command()
def shell(data: str = DATA):
    """Open a persistent terminal session using the same CLI commands."""
    from .terminal import run_shell
    run_shell(app, data)


@app.command()
def research(topic: str = typer.Argument("", help="Topic to investigate; omit for a topic picker"),
             data: str = typer.Option("", "--data", help="New output directory; defaults to data/research/TOPIC-TIMESTAMP"),
             papers: int = typer.Option(30, min=1, max=100, help="Maximum papers to collect"),
             k: int = typer.Option(3, min=1, max=10, help="Gap candidates to investigate"),
             provider: str = typer.Option("openalex", help="Scholarly index: openalex or arxiv"),
             as_of: str = typer.Option("", help="Latest publication date, YYYY-MM-DD; defaults to today"),
             resume: Path | None = typer.Option(None, "--resume", exists=True, file_okay=False,
                                               help="Continue a saved research directory without downloading again"),
             auto_retries: int = typer.Option(2, "--auto-retries", min=0, max=5,
                                             help="Automatically resume a stage after a recoverable failure"),
             as_json: bool = typer.Option(False, "--json")):
    """Choose a topic, collect papers online, and investigate research gaps locally."""
    from .gateway import Gateway
    from .research import TOPICS, new_destination, render_report, run

    if resume is not None and data:
        raise typer.BadParameter("Use --resume DIRECTORY without --data")
    if not topic.strip() and resume is None:
        if as_json:
            raise typer.BadParameter("Supply a topic when using --json")
        typer.echo("Choose a research topic or write your own:")
        for index, suggestion in enumerate(TOPICS, 1):
            typer.echo(f"  {index}. {suggestion}")
        choice = typer.prompt("Topic (number or text)").strip()
        if choice.isdecimal():
            if not 1 <= int(choice) <= len(TOPICS):
                raise typer.BadParameter(f"Choose a number from 1 to {len(TOPICS)}, or type a topic")
            topic = TOPICS[int(choice) - 1]
        else:
            topic = choice
    if provider not in {"openalex", "arxiv"}:
        raise typer.BadParameter("Provider must be openalex or arxiv")
    destination = resume if resume is not None else Path(data) if data else new_destination(topic)
    try:
        settings = load_settings()
        settings.cloud.enabled = False
        gateway = Gateway(settings, {"local": configured_backend(settings, True)})
        typer.echo(f"{'Resuming: ' + str(resume) if resume else 'Research: ' + topic} | "
                   f"local model: {settings.model.local}", err=True)
        with Activity("Research", settings.model.local) as activity, redirect_stdout(sys.stderr):
            report = run(topic, destination, settings, gateway, provider, papers, k, as_of,
                         progress=activity.event, resume=resume is not None, auto_retries=auto_retries)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        typer.echo(json.dumps(report, ensure_ascii=False))
    else:
        typer.echo(render_report(report))
        typer.echo(f"\nReport: {destination / 'report.md'}")
        typer.echo(f'Explore: gaptrace shell --data "{destination / "corpus"}"')
    return str(destination / "corpus")


@app.command()
def status(data: str = DATA, as_json: bool = typer.Option(False, "--json")):
    """Inspect corpus size, evidence verification, and the last import."""
    root = Path(data)
    if not (root / "bundle").is_dir() or not (root / "passages.jsonl").is_file():
        raise typer.BadParameter(f"No corpus at {root}. Use demo or ingest first.")
    tb = open_toolbox(data)
    links = [link for note in tb.notes.values() for link in note.links]
    metadata = root / "import.json"
    summary = {
        "data": str(root.resolve()),
        "papers": sum(n.type == "paper" for n in tb.notes.values()),
        "notes": len(tb.notes), "passages": len(tb.passages),
        "facts": sum(n.fact is not None for n in tb.notes.values()),
        "links": len(links), "verified_links": sum(link.verified for link in links),
        "gaps": len(tb.gaps()),
        "import": json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None,
    }
    if as_json:
        typer.echo(json.dumps(summary))
        return
    typer.echo(f"Corpus: {summary['data']}")
    typer.echo(f"Papers {summary['papers']} | Notes {summary['notes']} | Passages {summary['passages']} | Facts {summary['facts']}")
    typer.echo(f"Links {summary['links']} ({summary['verified_links']} verified) | Gap candidates {summary['gaps']}")
    if summary["import"]:
        imported = summary["import"]
        typer.echo(f"Cutoff: {imported['cutoff']} | Model: {imported['model']}")
        typer.echo(f"Extraction: {imported['stats'].get('invalid', 0)} invalid passages; inspect import.json for full statistics.")
    typer.echo("Gap candidates require closure review before treating them as current research opportunities.")


@app.command("discover")
def discover(topic: str = typer.Argument(""), data: str = typer.Option("", "--data"),
             source: Path | None = typer.Option(None, "--source", exists=True, dir_okay=False),
             papers: int = typer.Option(100, min=1, max=200), provider: str = "openalex", as_of: str = "",
             engine: str = typer.Option("", help="Choose after local extraction: local, connector or prepare"),
             metadata_tokens: int = typer.Option(50, "--metadata-tokens"),
             historical_through: str = typer.Option("", help="Historical attempt cutoff; default is five years before as-of"),
             abstracts_only: bool = typer.Option(False, "--abstracts-only", help="Skip supported arXiv HTML retrieval and screen abstracts only"),
             max_attempts: int = typer.Option(5, min=1, max=10),
             resume: Path | None = typer.Option(None, exists=True, file_okay=False),
             as_json: bool = typer.Option(False, "--json")):
    """Prepare papers locally, then choose who investigates newly enabled research."""
    from .discovery import run
    from .gateway import Gateway
    from .research import new_destination
    if resume is not None and data:
        raise typer.BadParameter("Use --resume without --data")
    if not topic and not source and not resume:
        if as_json:
            raise typer.BadParameter("Supply a topic or source file")
        topic = typer.prompt("Research topic").strip()
    destination = resume or (Path(data) if data else new_destination(topic or source.stem))
    def choose(manifest):
        typer.echo("\nLocal extraction and knowledge preparation are saved.", err=True)
        typer.echo("Choose local + scholarly search, connector for ChatGPT/Claude via MCP, or prepare to stop here.", err=True)
        return typer.prompt("Analysis engine", default="local").strip().lower()
    try:
        settings = load_settings()
        settings.cloud.enabled = False
        gateway = Gateway(settings, {"local": configured_backend(settings, True)})
        with redirect_stdout(sys.stderr) if as_json else nullcontext():
            report = run(topic, destination, settings, gateway, papers, provider, as_of, source,
                         metadata_tokens, engine or ("prepare" if as_json else ""), choose,
                         resume is not None, max_attempts, progress=lambda m: typer.echo(m, err=True),
                         historical_through=historical_through, full_text=not abstracts_only)
        if as_json:
            typer.echo(json.dumps(report, ensure_ascii=False))
        else:
            typer.echo(f"Knowledge base: {report['corpus']}\nState: {report['state']}")
            if report["engine"] == "local":
                from .reactivation import render
                typer.echo(render(json.loads((destination / "corpus/reactivation.json").read_text(encoding="utf-8"))))
            elif report["engine"] == "connector":
                typer.echo(f"MCP snapshot prepared: {destination / 'connector-packet.json'}")
                typer.echo("Run connector-serve for a local client, or follow docs/REACTIVATION.md to deploy the private AWS endpoint.")
            else:
                typer.echo(f"Continue: gaptrace discover --resume \"{destination}\" --engine local")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    return str(destination / "corpus")


@app.command("prepare-knowledge")
def prepare_knowledge(data: str = DATA, metadata_tokens: int = 50):
    """Prepare contextual retrieval and blocker/capability observations locally."""
    from .gateway import Gateway
    from .knowledge import prepare
    s = load_settings()
    s.cloud.enabled = False
    gateway = Gateway(s, {"local": configured_backend(s, True)}, Path(data) / "cache/knowledge")
    with redirect_stdout(sys.stderr):
        result = prepare(open_toolbox(data, s), gateway, metadata_tokens,
                         progress=lambda m: typer.echo(m, err=True), checkpoint_dir=Path(data) / "knowledge-checkpoints")
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("connector-serve")
def connector_serve(packet: Path = typer.Argument(..., exists=True, dir_okay=False), transport: str = "stdio", port: int = 8787):
    """Expose a prepared KB to a local MCP client; cloud deployment uses the AWS package."""
    from .connector import build_server
    server = build_server(packet)
    if transport == "stdio":
        server.run("stdio")
    elif transport == "http":
        typer.echo(f"Local MCP endpoint: http://127.0.0.1:{port}/mcp", err=True)
        server.run("streamable-http", host="127.0.0.1", port=port, stateless_http=True, json_response=True)
    else:
        raise typer.BadParameter("Use stdio or http; remote access requires the authenticated AWS deployment")


@app.command("opportunities")
def opportunities(data: str = DATA, provider: str = "openalex", as_of: str = "", k: int = 5,
                  as_json: bool = typer.Option(False, "--json")):
    """Investigate old research blockers against newer capabilities."""
    from .gateway import Gateway
    from .reactivation import analyze, render
    s = load_settings()
    gateway = Gateway(s, {"local": configured_backend(s, True)}, Path(data) / "cache/opportunities")
    with redirect_stdout(sys.stderr):
        result = analyze(open_toolbox(data, s), gateway, as_of, provider, max_attempts=k,
                         progress=lambda m: typer.echo(m, err=True))
    typer.echo(json.dumps(result, ensure_ascii=False) if as_json else render(result))


def configured_backend(settings, local_only: bool):
    from .gateway import litellm_backend

    model = settings.model.local
    options = {"api_base": getattr(settings.model, "local_api_base", "http://127.0.0.1:11434")}
    if local_only:
        if not model.startswith("ollama_chat/") or model.endswith(("-cloud", ":cloud")):
            raise typer.BadParameter("--local-only requires a local Ollama model")
        from urllib.parse import urlparse
        if urlparse(options["api_base"]).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise typer.BadParameter("--local-only requires a loopback Ollama endpoint")
    return litellm_backend(model, settings.model.local_num_ctx, **options)


@app.command()
def demo(data: str = DATA):
    """Write the small sample corpus so every other command can be tried."""
    from .sample import write_sample
    typer.echo(f"Sample corpus written to {write_sample(data)}")


@app.command()
def rebuild(data: str = DATA):
    """Rebuild every index from the note bundle and the passage store."""
    notes = read_bundle(Path(data) / "bundle")
    facts = FactsTable(Path(data) / "facts.sqlite")
    n_facts = facts.rebuild(notes)
    links = [l for n in notes for l in n.links]
    typer.echo(f"notes {len(notes)} | links {len(links)} (verified {sum(l.verified for l in links)}) | "
               f"facts {n_facts} | passages {len(PassageStore(Path(data) / 'passages.jsonl'))}")


@app.command()
def search(query: str, note_type: str = "", data: str = DATA):
    """Search notes; returns headlines with ids."""
    typer.echo(open_toolbox(data).search_notes(query, note_type))


@app.command()
def note(note_id: str, data: str = DATA):
    """Show one note with its links."""
    typer.echo(open_toolbox(data).get_note(note_id))


@app.command()
def neighbours(note_id: str, link_type: str = "", data: str = DATA):
    typer.echo(open_toolbox(data).neighbours(note_id, link_type))


@app.command()
def passage(passage_id: str, data: str = DATA, as_json: bool = typer.Option(False, "--json")):
    if as_json:
        item = PassageStore(Path(data) / "passages.jsonl").get(passage_id)
        if item is None:
            raise typer.BadParameter(f"No passage with ID {passage_id}")
        typer.echo(item.model_dump_json())
    else:
        typer.echo(open_toolbox(data).get_passage(passage_id))


@app.command()
def fact(method: str = "", dataset: str = "", metric: str = "", data: str = DATA):
    """Exact reported values."""
    typer.echo(open_toolbox(data).lookup_fact(method, dataset, metric))


@app.command()
def gaps(k: int = 10, gap_type: str = "", data: str = DATA, as_json: bool = typer.Option(False, "--json")):
    """Ranked gap cards (pattern search only; no model needed)."""
    toolbox = open_toolbox(data)
    if as_json:
        cards = [card.model_dump(mode="json") for card in toolbox.gaps()
                 if not gap_type or card.type == gap_type][:k or toolbox.settings.gaps.top_n]
        typer.echo(json.dumps(cards))
    else:
        typer.echo(toolbox.find_gaps(k, gap_type))


@app.command("literature")
def literature_search(query: str, provider: str = "openalex", k: int = 5, as_of: str = "", data: str = DATA):
    """Search scholarly titles/abstracts without a language-model call."""
    from .literature import search
    try:
        typer.echo(json.dumps(search(query, provider, k, as_of), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(f"Scholarly search incomplete: {exc}") from exc


@app.command("review-gaps")
def review_gaps(data: str = DATA, k: int = 3, gap_id: str = "", provider: str = "openalex", as_of: str = "",
                saved: bool = False, as_json: bool = typer.Option(False, "--json")):
    """Investigate source support, search for closure, and rerank gap candidates."""
    from .gap_review import GapReviewer, load, render, save
    try:
        if not (Path(data) / "bundle").is_dir() or not (Path(data) / "passages.jsonl").is_file():
            raise ValueError(f"No corpus at {data}. Use demo or ingest first.")
        run_dir = None if saved else new_run_dir(Path(data))
        tb = open_toolbox(data, run_dir=run_dir)
        if saved:
            report = load(Path(data) / "gap-review.json", tb)
        else:
            with Activity("Investigating gap candidates", tb.settings.model.local) as activity:
                reviewer = GapReviewer(tb, provider=provider, as_of=as_of, progress=activity.event)
                report = reviewer.run(tb.gaps(), k, gap_id)
            save(report, run_dir / "gap-review.json")
            save(report, Path(data) / "gap-review.json")
        if as_json:
            typer.echo(json.dumps(report, ensure_ascii=False))
        else:
            typer.echo(render(report))
            typer.echo(f"\nFull evidence and decisions: {Path(data) / 'gap-review.json'}")
            typer.echo("Evaluate/export this order: python -m evaluation.gap_review PATH_TO_REPORT --out NEW_DIRECTORY")
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def trail(gap_id: str, data: str = DATA):
    typer.echo(open_toolbox(data).get_trail(gap_id))


@app.command()
def ask(question: str, data: str = DATA, as_json: bool = typer.Option(False, "--json"),
        local_only: bool = typer.Option(False, "--local-only")):
    """Answer a question from the map with the local model (needs Ollama and litellm)."""
    from .agent import Agent
    from .gateway import Gateway

    s = load_settings()
    run_dir = new_run_dir(Path(data))
    tb = open_toolbox(data, s, run_dir=run_dir)
    cache = Path(data) / "cache"
    if local_only:
        cache /= "local"
    gw = Gateway(s, {"local": configured_backend(s, local_only)}, cache)
    agent = Agent(tb, gw, s, run_dir)
    start = perf_counter()
    with Activity("Answering from corpus evidence", s.model.local) as activity, redirect_stdout(sys.stderr):
        activity.event(f"Run files: {run_dir}")
        result = agent.ask(question)
    seconds = perf_counter() - start
    if as_json:
        trail = agent.trail.read()
        tool_records = [item for item in trail if item["kind"] == "tool"]
        seen_notes = {note_id for item in trail if item["kind"] == "step" for note_id in item["shown"]}
        seen_passages = set()
        validated_passages = {pid for entry in trail if entry["kind"] == "answer_validation"
                              and entry["supported"] for pid in entry["passage_ids"]}
        seen_passages.update(validated_passages)
        for item in tool_records:
            if item["n"] >= result["steps"]:
                continue
            if item["tool"] in {"get_note", "lookup_fact", "search_notes", "neighbours"}:
                seen_notes.update(re.findall(r"\[([\w.-]+)\]", item["digest"]))
            if item["tool"] == "get_passage" and item["digest"].startswith("["):
                seen_passages.add(item["args"]["passage_id"])
        retrieved = set()
        for note_id in seen_notes & tb.notes.keys():
            note = tb.source_note(note_id)
            seen_passages.update(note.passage_ids)
            if note.type == "paper" or note.paper_id:
                retrieved.add(note.paper_id or note.id)
        retrieved.update(p.paper_id for pid in seen_passages if (p := tb.source_passage(pid)))
        cited = validated_passages
        result.update(cited_passage_ids=sorted(cited), retrieved_paper_ids=sorted(retrieved),
                      tool_calls=len(tool_records), response_seconds=seconds, model=s.model.local)
        if tb.external_notes:
            result["external_sources"] = [{"id": n.id, "title": n.name, "passage_ids": n.passage_ids}
                                           for n in tb.external_notes.values()]
        typer.echo(json.dumps(result))
        return
    typer.echo(result["answer"])
    if result["cited"]:
        typer.echo("cited: " + ", ".join(result["cited"]))


@app.command()
def corpus(data: str = DATA):
    """Print a JSON inventory of the current corpus and import metadata."""
    root = Path(data)
    notes = read_bundle(root / "bundle")
    store = PassageStore(root / "passages.jsonl")
    metadata = root / "import.json"
    typer.echo(json.dumps({
        "papers": [{"id": n.id, "published": n.paper_date.isoformat() if n.paper_date else None}
                   for n in notes if n.type == "paper"],
        "passages": [p.model_dump(mode="json", exclude={"text"}) for p in store.all()],
        "import": json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None,
    }))


@app.command()
def ingest(source: Path, cutoff: str = typer.Option(..., help="Latest publication date, YYYY-MM-DD"),
           data: str = DATA, local_only: bool = typer.Option(False, "--local-only")):
    """Build a new corpus from dated papers with section text in JSON."""
    from .gateway import Gateway
    from .ingest import PaperImport, ingest as ingest_papers

    try:
        settings = load_settings()
        if local_only:
            settings.cloud.enabled = False
        dataset = PaperImport.model_validate_json(source.read_bytes())
        gateway = Gateway(settings, {"local": configured_backend(settings, local_only)})
        with Activity("Importing and verifying corpus", settings.model.local) as activity, redirect_stdout(sys.stderr):
            result = ingest_papers(dataset, Path(data), date.fromisoformat(cutoff), gateway, settings,
                                   progress=activity.event)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result))


@app.command()
def mcp(data: str = DATA):
    """Run the MCP server over standard input and output."""
    from .mcp_server import build_server
    build_server(data).run("stdio")


if __name__ == "__main__":
    app()
