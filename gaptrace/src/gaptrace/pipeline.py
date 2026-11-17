"""Build pipeline after collection: passages -> extracted notes -> merged
notes -> verified links. Collection and PDF reading are separate (phase 1)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import tokens
from .gateway import Gateway, ModelOutputInvalid
from .recovery import read_checkpoint, write_checkpoint
from .schema import (ConfirmOutput, ExtractorOutput, Link, Note, Passage, VerifierOutput, link_allowed,
                     make_id)

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def fill(template: str, **values) -> str:
    """Replace {key} markers without touching the JSON braces in the prompts."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _result_id(note: Note) -> str:
    """Stable observation identity; names alone cannot identify measurements."""
    fact = note.fact.model_dump() if note.fact else None
    if fact:
        fact = {k: _squash(v) if isinstance(v, str) else v for k, v in fact.items()}
    identity = [note.paper_id, _squash(note.name), fact]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    return f"res-{digest}"


def _numeric_values(text: str) -> set[Decimal]:
    values = set()
    for number in re.findall(r"(?<![\w.\-])[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?!\w|\.\d)", text):
        try:
            values.add(Decimal(number.replace(",", "")))
        except InvalidOperation:
            continue
    return values


def _has_value(text: str, value: float) -> bool:
    return Decimal(str(value)) in _numeric_values(text)


def _best_sentence(passage: str, text: str) -> str:
    """The passage sentence sharing the most words with `text`."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return max(sentences, key=lambda s: len(words & set(re.findall(r"[a-z0-9]+", s.lower()))), default="")


def extract_passage(gateway: Gateway, passage: Passage, paper: Note, settings) -> Tuple[List[Note], dict]:
    """Read every chunk of a long passage while retaining its source ID."""
    limit = settings.limits.passage_tokens
    if tokens.count(passage.text) <= limit:
        return _extract_chunk(gateway, passage, paper, settings)

    notes: Dict[str, Note] = {}
    stats = {"notes": 0, "links": 0, "links_dropped": 0, "invalid": 0}
    for text in tokens.iter_chunks(passage.text, limit):
        if not text.strip():
            continue
        chunk = passage.model_copy(update={"text": text})
        extracted, chunk_stats = _extract_chunk(gateway, chunk, paper, settings)
        for note in extracted:
            if note.id in notes:
                _merge_into(notes[note.id], note)
            else:
                notes[note.id] = note
        for key in ("links", "links_dropped", "invalid"):
            stats[key] += chunk_stats[key]
    stats["notes"] = len(notes)
    return list(notes.values()), stats


def _extract_chunk(gateway: Gateway, passage: Passage, paper: Note, settings) -> Tuple[List[Note], dict]:
    lim = settings.limits
    prompt = fill(load_prompt("extractor"), section=passage.section or "unknown", paper_title=paper.name,
                  year=paper.paper_date.year if paper.paper_date else "n.d.",
                  passage=passage.text)
    stats = {"notes": 0, "links": 0, "links_dropped": 0, "invalid": 0}
    try:
        out, _ = gateway.call("local", prompt, ExtractorOutput)
    except ModelOutputInvalid:
        stats["invalid"] = 1
        return [], stats
    notes: Dict[str, Note] = {}
    by_name: Dict[str, List[Note]] = {"paper": [paper]}
    for en in out.notes:
        if en.type == "result" and en.fact is None:
            continue                                     # a result without its numbers is unusable
        note = Note(id=make_id(en.type, en.name), type=en.type, name=en.name.strip(),
                    l0=tokens.truncate(en.l0, lim.l0_tokens), l1=tokens.truncate(en.l1, lim.l1_tokens),
                    passage_ids=[passage.id], paper_id=paper.id, paper_date=paper.paper_date, fact=en.fact)
        if note.type == "result":
            note.id = _result_id(note)
        if note.id in notes:
            _merge_into(notes[note.id], note)
        else:
            notes[note.id] = note
            by_name.setdefault(_squash(en.name), []).append(note)
    paper_links: List[Link] = []
    passage_sq = _squash(passage.text)
    linked_from_paper = set()
    for el in out.links:
        sources = by_name.get(_squash(el.source), [])
        targets = by_name.get(_squash(el.target), [])
        if len(targets) > 1:
            targets = [n for n in targets if n.fact and _has_value(el.evidence_sentence, n.fact.value)]
        src = sources[0] if len(sources) == 1 else None
        dst = targets[0] if len(targets) == 1 else None
        ok = (src is not None and dst is not None and link_allowed(el.type, src.type, dst.type)
              and _squash(el.evidence_sentence) in passage_sq and len(el.evidence_sentence.split()) >= 3)
        if not ok:                                       # outside the schema, or the quote is not in the passage
            stats["links_dropped"] += 1
            continue
        link = Link(target=dst.id, type=el.type, evidence_passage_id=passage.id,
                    evidence_sentence=el.evidence_sentence.strip())
        if src is paper:
            paper_links.append(link)
            linked_from_paper.add(dst.id)
        else:
            src.links.append(link)
        stats["links"] += 1
    # a result, limitation or open question always belongs to the paper it was read from
    for n in notes.values():
        if n.id in linked_from_paper:
            continue
        kind = {"result": "reports", "limitation": "states", "open_question": "states"}.get(n.type)
        if kind:
            sentence = _best_sentence(passage.text, n.l0 + " " + n.name)
            if n.fact:
                candidates = [s for s in re.split(r"(?<=[.!?])\s+", passage.text)
                              if _has_value(s, n.fact.value)]
                if not candidates:
                    stats["links_dropped"] += 1
                    continue
                # A quote cannot select between same-name, same-value observations.
                siblings = by_name.get(_squash(n.name), [])
                if sum(x.fact is not None and x.fact.value == n.fact.value for x in siblings) > 1:
                    stats["links_dropped"] += 1
                    continue
                sentence = _best_sentence(" ".join(candidates), n.l0 + " " + n.name)
            paper_links.append(Link(target=n.id, type=kind, evidence_passage_id=passage.id,
                                    evidence_sentence=sentence))
            stats["links"] += 1
    paper.links.extend(paper_links)
    stats["notes"] = len(notes)
    return list(notes.values()), stats


def _merge_into(a: Note, b: Note) -> None:
    if (a.type == "result" or b.type == "result") and (a.type != b.type or _result_id(a) != _result_id(b)):
        raise ValueError("Cannot merge distinct result observations")
    if (a.type == "result" or b.type == "result") and (
            a.type != b.type or _result_id(a) != _result_id(b)):
        raise ValueError("Cannot merge distinct result observations")
    if _squash(b.name) != _squash(a.name) and b.name not in a.aliases:
        a.aliases.append(b.name)
    a.aliases += [x for x in b.aliases if x not in a.aliases and _squash(x) != _squash(a.name)]
    a.passage_ids += [p for p in b.passage_ids if p not in a.passage_ids]
    seen = {(l.target, l.type, l.evidence_passage_id) for l in a.links}
    a.links += [l for l in b.links if (l.target, l.type, l.evidence_passage_id) not in seen]
    if len(b.l1) > len(a.l1):
        a.l1 = b.l1


def merge_notes(notes: Iterable[Note], embed: Optional[Callable] = None, threshold: float = 0.86,
                confirm: Optional[Callable[[Note, Note], bool]] = None) -> Tuple[List[Note], Dict[str, str]]:
    """Join duplicates: identical ids first, then near-identical meaning within
    a type (never results or papers). Returns (notes, {old id -> kept id})."""
    notes = list(notes)
    observations: Dict[str, List[Note]] = {}
    for n in notes:
        if n.type == "result":
            observations.setdefault(n.id, []).append(n)
    result_remap = {}
    for old_id, group in observations.items():
        new_ids = {_result_id(n) for n in group}
        if len(new_ids) == 1 and old_id not in new_ids:
            result_remap[old_id] = next(iter(new_ids))
    for n in notes:
        for link in n.links:
            if link.target not in observations:
                continue
            candidates = observations[link.target]
            if n.type == "paper" and link.type == "reports":
                candidates = [x for x in candidates if x.paper_id == n.id]
            if link.evidence_passage_id:
                candidates = [x for x in candidates if link.evidence_passage_id in x.passage_ids]
            identities = {_result_id(x) for x in candidates}
            if len(identities) > 1:
                identities = {_result_id(x) for x in candidates
                              if x.fact and _has_value(link.evidence_sentence, x.fact.value)}
            if len(identities) == 1:
                link.target = next(iter(identities))
            else:
                link.verified = False
                logging.getLogger(__name__).warning("Ambiguous result link: %s -> %s", n.id, link.target)
        if n.type == "result":
            n.id = _result_id(n)
    kept: Dict[str, Note] = {}
    remap: Dict[str, str] = {}
    for n in notes:
        if n.id in kept:
            _merge_into(kept[n.id], n)
        else:
            kept[n.id] = n
    if embed is not None:
        for t in ("problem", "method", "dataset", "limitation", "open_question"):
            group = [n for n in kept.values() if n.type == t]
            if len(group) < 2:
                continue
            v = np.asarray(embed([f"{n.name}. {n.l0}" for n in group]), dtype=np.float32)
            v = v / np.where(np.linalg.norm(v, axis=1, keepdims=True) == 0, 1, np.linalg.norm(v, axis=1, keepdims=True))
            sim = v @ v.T
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a.id in remap or b.id in remap or sim[i, j] < threshold:
                        continue
                    if confirm is None or confirm(a, b):
                        _merge_into(a, b)
                        remap[b.id] = a.id
                        kept.pop(b.id, None)
    for n in kept.values():                                   # retarget links to the survivors
        for l in n.links:
            l.target = remap.get(l.target, l.target)
        uniq, seen = [], set()
        for l in n.links:
            k = (l.target, l.type, l.evidence_passage_id)
            if k not in seen and l.target != n.id:
                seen.add(k)
                uniq.append(l)
        n.links = uniq
    return list(kept.values()), {**result_remap, **remap}


def model_confirm(gateway: Gateway) -> Callable[[Note, Note], bool]:
    def confirm(a: Note, b: Note) -> bool:
        prompt = fill(load_prompt("merge_confirm"), type=a.type, name_a=a.name, l0_a=a.l0, name_b=b.name, l0_b=b.l0)
        try:
            out, _ = gateway.call("local", prompt, ConfirmOutput)
            return out.same
        except ModelOutputInvalid:
            return False
    return confirm


def verify_links(gateway: Gateway, notes: Iterable[Note], settings,
                 passages: Optional[Iterable[Passage]] = None) -> dict:
    """Keep a link only if its sentence supports it. Rejected links stay in the
    file with verified=false and are never used."""
    notes = list(notes)
    by_id = {n.id: n for n in notes}
    store = {p.id: p for p in passages} if passages is not None else None
    stats = {"proposed": 0, "verified": 0, "rejected": 0, "rejected_examples": []}
    template = load_prompt("link_verifier")
    for n in notes:
        for link in n.links:
            stats["proposed"] += 1
            target = by_id.get(link.target)
            passage = store.get(link.evidence_passage_id) if store is not None else None
            intact = True
            if store is not None:
                intact = (passage is not None and bool(link.evidence_sentence.strip())
                          and _squash(link.evidence_sentence) in _squash(passage.text))
                if intact and n.type == "paper":
                    intact = passage.paper_id == n.id
                if intact and link.type == "reports":
                    intact = (target is not None and target.paper_id == n.id
                              and passage.id in target.passage_ids)
            if intact and link.type == "reports" and target is not None:
                intact = (target.fact is not None and _has_value(link.evidence_sentence, target.fact.value))
            if target is None or not intact:
                link.verified = False
                stats["rejected"] += 1
                if len(stats["rejected_examples"]) < 20:
                    stats["rejected_examples"].append(f"Invalid provenance: {n.id} -> {link.target}")
                continue
            if not settings.switches.link_verifier:            # off switch: trust every link
                link.verified, link.confidence = True, 0.0
                stats["verified"] += 1
                continue
            target_description = target.name
            if link.type == "reports" and target.fact:
                target_description += " " + json.dumps(target.fact.model_dump(), sort_keys=True)
            prompt = fill(template, from_name=n.name, type=link.type, to_name=target_description,
                          evidence_sentence=link.evidence_sentence)
            try:
                out, _ = gateway.call("local", prompt, VerifierOutput)
                link.verified, link.confidence = out.supported, out.confidence
            except ModelOutputInvalid:
                link.verified = False
            if link.verified:
                stats["verified"] += 1
            else:
                stats["rejected"] += 1
                if len(stats["rejected_examples"]) < 20:
                    stats["rejected_examples"].append(
                        f"{n.name} --{link.type}--> {target.name} | {link.evidence_sentence[:120]}")
    return stats


def build_notes(papers: List[Note], passages: Iterable[Passage], gateway: Gateway, settings,
                embed: Optional[Callable] = None,
                progress: Optional[Callable[[str], None]] = None,
                checkpoint_dir: Optional[Path] = None) -> Tuple[List[Note], dict]:
    """passages + paper notes -> the full, merged, verified note list."""
    passages = list(passages)
    paper_by_id = {p.id: p for p in papers}
    collected: List[Note] = []
    stats = {"passages": 0, "notes": 0, "links_dropped": 0, "invalid": 0}
    for passage in passages:
        paper = paper_by_id.get(passage.paper_id)
        if paper is None:
            continue
        checkpoint = None
        if checkpoint_dir is not None:
            identity = [passage.model_dump(mode="json"), paper.name,
                        paper.paper_date.isoformat() if paper.paper_date else None,
                        settings.model.local, settings.model.local_num_ctx, settings.budget.output,
                        vars(settings.limits), load_prompt("extractor"), ExtractorOutput.model_json_schema()]
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            checkpoint = Path(checkpoint_dir) / f"{key}.json"
        def validate_saved(saved):
            extracted = [Note.model_validate(item) for item in saved["notes"]]
            links = [Link.model_validate(item) for item in saved["paper_links"]]
            for key in ("notes", "links", "links_dropped", "invalid"):
                if not isinstance(saved["stats"][key], int) or saved["stats"][key] < 0:
                    raise ValueError("Invalid checkpoint statistics")
            if saved["stats"]["invalid"] or any(passage.id not in n.passage_ids for n in extracted):
                raise ValueError("Invalid checkpoint provenance")
            if any(link.evidence_passage_id != passage.id for link in links):
                raise ValueError("Invalid checkpoint link provenance")
            return extracted, links, saved["stats"]
        had_checkpoint = checkpoint is not None and checkpoint.exists()
        saved = read_checkpoint(checkpoint, validate_saved) if checkpoint is not None else None
        if saved is not None:
            notes, links, s = saved
            paper.links.extend(links)
            if progress:
                progress(f"Reusing extracted passage {stats['passages'] + 1}/{len(passages)}: {passage.id}")
        else:
            if had_checkpoint and progress:
                progress(f"Repairing damaged checkpoint for {passage.id}; rebuilding this passage only.")
            if progress:
                progress(f"Extracting passage {stats['passages'] + 1}/{len(passages)}: {passage.id} ({passage.section})")
            before = len(paper.links)
            notes, s = extract_passage(gateway, passage, paper, settings)
            if checkpoint is not None and not s["invalid"]:
                saved = {"notes": [n.model_dump(mode="json") for n in notes], "stats": s,
                         "paper_links": [link.model_dump(mode="json") for link in paper.links[before:]]}
                write_checkpoint(checkpoint, saved)
            if checkpoint is not None and progress:
                progress(f"{'Checkpoint saved' if not s['invalid'] else 'Extraction invalid; passage remains pending'}: {passage.id}")
            if checkpoint is not None and s["invalid"]:
                raise ModelOutputInvalid(f"Extraction is still invalid for {passage.id}; completed passages are saved.")
        collected += notes
        stats["passages"] += 1
        for k in ("links_dropped", "invalid"):
            stats[k] += s[k]
    if progress:
        progress(f"Merging {len(papers) + len(collected)} notes")
    merged, remap = merge_notes(papers + collected, embed, settings.merge.similarity_threshold,
                                model_confirm(gateway) if embed is not None else None)
    stats["notes"] = len(merged)
    stats["merged_away"] = len(papers) + len(collected) - len(merged)
    if progress:
        progress(f"Verifying {sum(len(n.links) for n in merged)} evidence links")
    stats["links"] = verify_links(gateway, merged, settings, passages)
    return merged, stats
