import json
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

from gaptrace import cli, gateway, literature, research
from gaptrace.settings import load_settings


runner = CliRunner()


def paper(index=1):
    return {"id": f"oa-W{index}", "title": f"Study {index}", "published": "2024-01-01",
            "abstract": "Learning under distribution shift remains an open question.",
            "url": f"https://openalex.org/W{index}", "provider": "openalex", "evidence_level": "abstract"}


def test_collection_paginates_and_deduplicates():
    requests = []
    def read(url, headers):
        params = parse_qs(urlparse(url).query)
        requests.append(params)
        page = int(params["page"][0])
        indices = range(1, 26) if page == 1 else range(20, 45)
        return json.dumps({"results": [{"id": f"https://openalex.org/W{i}", "display_name": f"Paper {i}",
            "publication_date": "2024-01-01", "abstract_inverted_index": {"Evidence": [0]}}
            for i in indices]}).encode()
    papers = literature.collect("learning", limit=30, read=read)
    assert len(papers) == len({p["id"] for p in papers}) == 30
    assert [p["id"] for p in papers][-1] == "oa-W30"
    assert [r["page"] for r in requests] == [["1"], ["2"]]
    assert all(r["per_page"] == ["25"] for r in requests)


def test_collection_stops_on_repeated_page_and_rejects_invalid_count():
    calls = []
    def read(*args):
        calls.append(args)
        return json.dumps({"results": [{"id": "https://openalex.org/W1", "display_name": "One",
            "publication_date": "2024-01-01", "abstract_inverted_index": {"Evidence": [0]}}]}).encode()
    assert len(literature.collect("topic", limit=30, read=read)) == 1
    assert len(calls) == 2
    with pytest.raises(ValueError, match="between"):
        literature.collect("topic", limit=101, read=read)


def test_arxiv_collection_uses_offsets_and_safe_paper_ids(monkeypatch):
    monkeypatch.setattr(literature, "sleep", lambda _: None)
    offsets = []
    def read(url, headers):
        params = parse_qs(urlparse(url).query)
        offset = int(params["start"][0])
        offsets.append(offset)
        entries = "".join(f"<entry><id>https://arxiv.org/abs/2401.{i:05d}v1</id>"
            f"<title>Study {i}</title><summary>Research evidence.</summary>"
            "<published>2024-01-01T00:00:00Z</published><updated>2024-01-01T00:00:00Z</updated></entry>"
            for i in range(offset, offset + int(params["max_results"][0])))
        return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'.encode()
    papers = literature.collect("continual learning", "arxiv", 30, read=read)
    assert len(papers) == 30 and offsets == [0, 25]
    assert all(p["evidence_level"] == "abstract" for p in papers)


@pytest.fixture
def research_runtime(monkeypatch):
    settings = load_settings()
    settings.switches.hybrid_search = False
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(literature, "collect", lambda *args, **kwargs: [paper(1), paper(2)])
    monkeypatch.setattr(literature, "search", lambda *args: [{**paper(3), "abstract": "We describe an unrelated baseline."}])
    def backend(prompt, schema, limit):
        props = schema["properties"]
        if "notes" in props:
            return json.dumps({"notes": [{"type": "open_question", "name": "Learning under distribution shift",
                "l0": "Learning under distribution shift remains an open question.", "l1": "An unresolved question."}], "links": []})
        if "supported" in props:
            return json.dumps({"supported": True, "reason": "Supported by the source."})
        if "verdict" in props:
            return json.dumps({"verdict": "does_not", "reason": "Does not address distribution shift."})
        if "question" in props:
            return json.dumps({"question": "How does a baseline perform under distribution shift?",
                "proposed_test": "Compare a baseline across held-out shifted distributions.",
                "success_measure": "Measure held-out accuracy across shifts.", "passage_id": "oa-W1-1-1",
                "quote": paper()["abstract"]})
        return json.dumps({axis: {"score": None} for axis in ("data", "compute", "blocks", "scope")})
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: backend)
    return settings, backend


def test_research_cli_runs_real_pipeline_and_saves_evidence(tmp_path, research_runtime):
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "distribution shift", "--data", str(root), "--papers", "2", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["state"] == "complete" and report["collected_papers"] == 2
    assert report["extraction"]["passages"] == 2 and report["extraction"]["links"]["verified"] == 2
    assert report["review"]["ranked"][0]["status"] == "open"
    assert report["directions"][0]["kind"] == "untested proposal"
    assert report["directions"][0]["quote"] == paper()["abstract"]
    assert report["sources"][0]["corpus_id"] == "oa-W1"
    assert "[1/4]" in result.stderr and "[4/4]" in result.stderr
    assert "abstracts only" in (root / "report.md").read_text(encoding="utf-8")
    assert json.loads((root / "corpus/import.json").read_text())["build_cloud_enabled"] is False
    assert (root / "papers.json").exists() and (root / "corpus/gap-review.json").exists()


@pytest.mark.parametrize("choice,topic", [("2", research.TOPICS[1]), ("custom topic", "custom topic")])
def test_topic_picker(tmp_path, research_runtime, choice, topic):
    result = runner.invoke(cli.app, ["research", "--data", str(tmp_path / "topic")], input=choice + "\n")
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "topic/research.json").read_text())["topic"] == topic


def test_shell_switches_to_researched_corpus(tmp_path, research_runtime):
    root = tmp_path / "new topic"
    result = runner.invoke(cli.app, ["shell", "--data", str(tmp_path / "absent")],
                           input=f'research "distribution shift" --data "{root}" --papers 2\nstatus\nexit\n')
    assert result.exit_code == 0, result.output
    assert "Papers 2" in result.output and str((root / "corpus").resolve()) in result.output


def test_failed_extraction_preserves_downloads(tmp_path, research_runtime, monkeypatch):
    def fail(*args):
        raise RuntimeError("Inference unavailable")
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: fail)
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert result.exit_code != 0 and "Inference unavailable" in result.output
    assert json.loads((root / "research.json").read_text())["state"] == "incomplete"
    assert len(json.loads((root / "papers.json").read_text())["papers"]) == 2
    assert not (root / "corpus").exists() and not (root / "report.md").exists()


def test_no_papers_and_existing_destination_do_not_start_inference(tmp_path, research_runtime, monkeypatch):
    monkeypatch.setattr(literature, "collect", lambda *args, **kwargs: [])
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert result.exit_code != 0 and "No papers" in result.output and not root.exists()
    root.mkdir()
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert result.exit_code != 0 and "already exists" in result.output


def test_bad_brainstorm_quote_is_not_published(tmp_path, research_runtime, monkeypatch):
    _, backend = research_runtime
    def invalid(prompt, schema, limit):
        raw = backend(prompt, schema, limit)
        if "question" in schema["properties"]:
            value = json.loads(raw)
            value["quote"] = "This quotation was invented."
            return json.dumps(value)
        return raw
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: invalid)
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(tmp_path / "topic"), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert not report["directions"] and "exact supplied evidence quote" in report["errors"][0]


def test_no_extracted_gap_does_not_invent_proposals(tmp_path, research_runtime, monkeypatch):
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: lambda *args: '{"notes": [], "links": []}')
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["gap_candidates"] == 0 and report["directions"] == []
    assert "No evidence-backed gap candidates" in (root / "report.md").read_text()


def test_json_without_topic_does_not_prompt():
    result = runner.invoke(cli.app, ["research", "--json"])
    assert result.exit_code != 0 and "Supply a topic" in result.output


def test_closed_gap_is_not_used_for_brainstorming(tmp_path, research_runtime, monkeypatch):
    _, backend = research_runtime
    def closed(prompt, schema, limit):
        if "verdict" in schema["properties"]:
            return json.dumps({"verdict": "fills", "quote": "We describe an unrelated baseline.",
                               "reason": "Test closure decision."})
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: closed)
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(tmp_path / "topic"), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["review"]["ranked"][0]["status"] == "closed" and not report["directions"]


def test_resume_reuses_successful_passages_without_downloading(tmp_path, research_runtime, monkeypatch):
    _, backend = research_runtime
    calls = []
    def fail_second(prompt, schema, limit):
        if "notes" in schema["properties"]:
            calls.append(prompt)
            if "Study 2" in prompt:
                raise RuntimeError("Ollama stalled")
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: fail_second)
    root = tmp_path / "topic"
    failed = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert failed.exit_code != 0 and "--resume" in failed.output
    assert len(list((root / "checkpoints/extraction").glob("*.json"))) == 1
    assert not (root / "corpus").exists()
    def no_search(*args, **kwargs):
        pytest.fail("Resume must not download papers again")
    monkeypatch.setattr(literature, "collect", no_search)
    resumed_calls = []
    def record(prompt, schema, limit):
        if "notes" in schema["properties"]:
            resumed_calls.append(prompt)
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: record)
    result = runner.invoke(cli.app, ["research", "--resume", str(root), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["state"] == "complete" and report["topic"] == "topic"
    assert len(resumed_calls) == 1 and "Study 2" in resumed_calls[0]
    assert "Reusing extracted passage 1/2" in result.stderr
    assert report["extraction"]["links"]["verified"] == 2
    assert report["review"]["ranked"][0]["support_count"] == 2
    # A further resume uses the corpus, rather than running extraction again.
    resumed_calls.clear()
    result = runner.invoke(cli.app, ["research", "--resume", str(root), "--json"])
    assert result.exit_code == 0, result.output
    assert not resumed_calls and "Reusing the completed corpus" in result.stderr


def test_resume_refuses_changed_downloads(tmp_path, research_runtime):
    root = tmp_path / "topic"
    assert runner.invoke(cli.app, ["research", "topic", "--data", str(root)]).exit_code == 0
    path = root / "papers.json"
    source = json.loads(path.read_text())
    source["papers"][0]["sections"][0]["text"] = "Changed source text."
    path.write_text(json.dumps(source))
    result = runner.invoke(cli.app, ["research", "--resume", str(root)])
    assert result.exit_code != 0 and "papers changed" in result.output


def test_automatic_recovery_continues_after_a_failed_passage(tmp_path, research_runtime, monkeypatch):
    from gaptrace import recovery
    monkeypatch.setattr(recovery, "sleep", lambda _: None)
    _, backend = research_runtime
    calls = []
    def intermittent(prompt, schema, limit):
        if "notes" in schema["properties"]:
            calls.append("second" if "Study 2" in prompt else "first")
            if calls == ["first", "second"]:
                raise recovery.TransientFailure("Ollama stalled after request retry")
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: intermittent)
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["state"] == "complete" and calls == ["first", "second", "second"]
    assert report["completed_passages"] == 2 and len(report["recoveries"]) == 1
    assert "Automatically recovering" in result.stderr and "Reusing extracted passage 1/2" in result.stderr


def test_repeated_failures_stop_with_a_checkpoint_instead_of_looping(tmp_path, research_runtime, monkeypatch):
    from gaptrace import recovery
    monkeypatch.setattr(recovery, "sleep", lambda _: None)
    _, backend = research_runtime
    calls = []
    def unavailable(prompt, schema, limit):
        if "notes" in schema["properties"] and "Study 2" in prompt:
            calls.append(1)
            raise recovery.TransientFailure("service remains unavailable")
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: unavailable)
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert result.exit_code != 0 and "--resume" in result.output
    manifest = json.loads((root / "research.json").read_text())
    assert manifest["state"] == "incomplete" and manifest["completed_passages"] == 1
    assert len(calls) == 3 and len(manifest["recoveries"]) == 2
    assert len(list((root / "checkpoints/extraction").glob("*.json"))) == 1


def test_keyboard_interrupt_saves_status_and_does_not_restart(tmp_path, research_runtime, monkeypatch):
    _, backend = research_runtime
    def interrupt(prompt, schema, limit):
        if "notes" in schema["properties"] and "Study 2" in prompt:
            raise KeyboardInterrupt()
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: interrupt)
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert result.exit_code != 0
    manifest = json.loads((root / "research.json").read_text())
    assert manifest["state"] == "interrupted" and manifest["completed_passages"] == 1
    assert not manifest.get("recoveries")


def test_damaged_extraction_checkpoint_is_rebuilt_without_repeating_other_passages(tmp_path, research_runtime, monkeypatch):
    _, backend = research_runtime
    root = tmp_path / "topic"
    assert runner.invoke(cli.app, ["research", "topic", "--data", str(root)]).exit_code == 0
    (root / "corpus").rename(root / "previous-corpus")
    checkpoint = next(path for path in (root / "checkpoints/extraction").glob("*.json")
                      if json.loads(path.read_text())["payload"]["notes"][0]["paper_id"] == "oa-W1")
    checkpoint.write_text("{interrupted")
    for path in (root / "checkpoints/model-responses").glob("*.json"):
        path.write_text("{interrupted")
    calls = []
    def record(prompt, schema, limit):
        if "notes" in schema["properties"]:
            calls.append(prompt)
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: record)
    result = runner.invoke(cli.app, ["research", "--resume", str(root), "--json"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and "Study 1" in calls[0]
    assert "Repairing damaged checkpoint" in result.stderr
    assert list((root / "checkpoints/extraction").glob("*.corrupt-*"))


def test_invalid_extraction_is_not_published_as_completed_research(tmp_path, research_runtime, monkeypatch):
    _, backend = research_runtime
    def invalid(prompt, schema, limit):
        if "notes" in schema["properties"] and "Study 2" in prompt:
            return "invalid JSON"
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: invalid)
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root)])
    assert result.exit_code != 0
    manifest = json.loads((root / "research.json").read_text())
    assert manifest["state"] == "incomplete" and manifest["completed_passages"] == 1
    assert not (root / "corpus").exists()


def test_missing_download_file_is_restored_from_saved_metadata(tmp_path, research_runtime, monkeypatch):
    root = tmp_path / "topic"
    assert runner.invoke(cli.app, ["research", "topic", "--data", str(root)]).exit_code == 0
    (root / "papers.json").rename(root / "archived-papers.json")
    def no_download(*args, **kwargs):
        pytest.fail("Repair should use the saved source metadata")
    monkeypatch.setattr(literature, "collect", no_download)
    result = runner.invoke(cli.app, ["research", "--resume", str(root), "--json"])
    assert result.exit_code == 0, result.output
    assert "Repaired downloaded-paper file" in result.stderr
    assert json.loads((root / "papers.json").read_text()) == json.loads((root / "archived-papers.json").read_text())


def test_gap_reviews_survive_failure_and_resume(tmp_path, research_runtime, monkeypatch):
    from gaptrace.gap_review import GapReviewer
    from gaptrace.recovery import TransientFailure
    _, backend = research_runtime
    second = {**paper(2), "abstract": "Learning rate transfer remains an open question."}
    monkeypatch.setattr(literature, "collect", lambda *args, **kwargs: [paper(1), second])
    def two_questions(prompt, schema, limit):
        if "notes" in schema["properties"] and "Study 2" in prompt:
            return json.dumps({"notes": [{"type": "open_question", "name": "Learning rate transfer",
                "l0": second["abstract"], "l1": second["abstract"]}], "links": []})
        raw = backend(prompt, schema, limit)
        if "question" in schema["properties"] and "Learning rate transfer" in prompt:
            value = json.loads(raw)
            value.update(passage_id="oa-W2-1-1", quote=second["abstract"])
            return json.dumps(value)
        return raw
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: two_questions)
    original, started, failed = GapReviewer.review, [], []
    def interrupt(self, card):
        if card.id not in self.reviews:
            started.append(card.id)
            if len(self.reviews) == 1 and not failed:
                failed.append(True)
                raise TransientFailure("review connection lost")
        return original(self, card)
    monkeypatch.setattr(GapReviewer, "review", interrupt)
    root = tmp_path / "topic"
    first = runner.invoke(cli.app, ["research", "topic", "--data", str(root), "--auto-retries", "0"])
    assert first.exit_code != 0
    manifest = json.loads((root / "research.json").read_text())
    assert manifest["state"] == "incomplete" and len(manifest["reviewed_gap_ids"]) == 1
    result = runner.invoke(cli.app, ["research", "--resume", str(root), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert len(report["review"]["reviews"]) == 2 and report["state"] == "complete"
    assert started.count(started[0]) == 1 and "Reusing 1 completed gap reviews" in result.stderr


def test_verification_recovers_without_reextracting_passages(tmp_path, research_runtime, monkeypatch):
    from gaptrace import recovery
    monkeypatch.setattr(recovery, "sleep", lambda _: None)
    _, backend = research_runtime
    extractions, failed = [], []
    def interrupted(prompt, schema, limit):
        if "notes" in schema["properties"]:
            extractions.append(prompt)
        if prompt.startswith("Does the sentence support") and "Study 2" in prompt and not failed:
            failed.append(True)
            raise recovery.TransientFailure("verification request interrupted")
        return backend(prompt, schema, limit)
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: interrupted)
    root = tmp_path / "topic"
    result = runner.invoke(cli.app, ["research", "topic", "--data", str(root), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["state"] == "complete" and len(extractions) == 2 and failed
    assert report["extraction"]["links"]["verified"] == 2
