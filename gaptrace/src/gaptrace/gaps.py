"""Gap discovery. Code searches the graph for four fixed patterns; models are
used only to judge the candidates found here (see closure.py)."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List

from .facts import FactsTable
from .graph import build_graph, in_links, out_links
from .schema import GapCard, Note, TrailStep


def _gid(kind: str, *parts: str) -> str:
    return "gap-" + hashlib.sha1("|".join((kind,) + parts).encode()).hexdigest()[:10]


def _step(g, u, v, d) -> TrailStep:
    return TrailStep(source=u, link=d["type"], target=v, passage_id=d.get("passage_id", ""),
                     sentence=d.get("sentence", ""), verified=bool(d.get("verified")))


def find_unaddressed(notes: Iterable[Note], kind: str) -> List[GapCard]:
    """Limitations (kind='limitation') or open questions (kind='open_question')
    that no paper or method addresses."""
    notes = list(notes)
    by_id = {n.id: n for n in notes}
    g_all = build_graph(notes, verified_only=False)
    gap_type = "unaddressed_limitation" if kind == "limitation" else "unanswered_open_question"
    cards = []
    for n in notes:
        if n.type != kind:
            continue
        addressed = in_links(g_all, n.id, "addresses")
        if any(d["verified"] for _, _, d in addressed):
            continue                                   # verifiably handled: not a gap
        stated = [e for e in in_links(g_all, n.id, "states") if e[2]["verified"]]
        if not stated:
            continue                                   # no verified evidence that anyone said it
        papers = sorted({u for u, _, _ in stated})
        label = "limitation" if kind == "limitation" else "open question"
        cards.append(GapCard(
            id=_gid(gap_type, n.id), type=gap_type,
            statement=f"No later work addresses the {label}: {n.l0}",
            note_ids=[n.id] + papers, support_count=len(papers),
            needs_check=bool(addressed),               # an unverified 'addresses' link exists
            trail=[_step(g_all, *e) for e in stated]))
    return cards


def find_untested_combinations(notes: Iterable[Note], min_users: int = 2) -> List[GapCard]:
    """A method that targets a problem but was never evaluated on a dataset
    that at least `min_users` other methods for the same problem use."""
    notes = list(notes)
    by_id = {n.id: n for n in notes}
    g = build_graph(notes, verified_only=True)
    g_all = build_graph(notes, verified_only=False)
    cards = []
    for p in (n for n in notes if n.type == "problem"):
        targeting = in_links(g, p.id, "targets")
        methods = sorted({u for u, _, _ in targeting})
        users: Dict[str, List[tuple]] = {}
        for m in methods:
            for e in out_links(g, m, "evaluated_on"):
                users.setdefault(e[1], []).append(e)
        for d_id, edges in sorted(users.items()):
            using = {u for u, _, _ in edges}
            if len(using) < min_users:
                continue
            for m in methods:
                if m in using:
                    continue
                maybe = any(v == d_id for _, v, _ in out_links(g_all, m, "evaluated_on"))
                m_edge = next(e for e in targeting if e[0] == m)
                cards.append(GapCard(
                    id=_gid("untested_combination", m, d_id), type="untested_combination",
                    statement=(f"{by_id[m].name} has not been evaluated on {by_id[d_id].name}, which "
                               f"{len(using)} other methods for '{p.name}' use"),
                    note_ids=[m, d_id, p.id], support_count=len(using), needs_check=maybe,
                    trail=[_step(g, *m_edge)] + [_step(g, *e) for e in edges]))
    return cards


def find_conflicts(notes: Iterable[Note], tolerance: float = 0.5) -> List[GapCard]:
    """Two papers reporting different values for the same method, dataset and metric."""
    notes = list(notes)
    by_id = {n.id: n for n in notes}
    g = build_graph(notes, verified_only=True)
    eligible = []
    for n in notes:
        if n.type != "result" or not n.fact or not n.paper_id or not math.isfinite(n.fact.value):
            continue
        reports = [(u, v, d) for u, v, d in in_links(g, n.id, "reports")
                   if u == n.paper_id and by_id[u].type == "paper"
                   and d.get("passage_id") in n.passage_ids and d.get("sentence", "").strip()]
        if reports:
            eligible.append(n)
    table = FactsTable()
    table.rebuild(sorted(eligible, key=lambda n: n.id))
    cards = []
    groups = []
    for rows in table.groups():
        for unit in sorted({r["unit"].strip().lower() for r in rows}):
            groups.append([r for r in rows if r["unit"].strip().lower() == unit])
    for rows in groups:
        rows.sort(key=lambda r: (r["paper_id"], r["note_id"]))
        papers = {r["paper_id"] for r in rows}
        values = [r["value"] for r in rows]
        if len(rows) < 2 or len(papers) < 2 or max(values) - min(values) <= tolerance:
            continue
        r0 = rows[0]
        ids = [r["note_id"] for r in rows]
        trail = []
        for nid in ids:
            trail += [_step(g, *e) for e in in_links(g, nid, "reports")
                      if e[0] == by_id[nid].paper_id and e[2].get("passage_id")
                      and e[2].get("sentence", "").strip()]
        trail.sort(key=lambda t: (t.source, t.target, t.passage_id, t.sentence))
        shown = ", ".join(f"{r['value']:g}{r['unit']}" for r in rows)
        cards.append(GapCard(
            id=_gid("conflicting_results", *sorted(ids)), type="conflicting_results",
            statement=(f"Reported {r0['metric']} for {r0['method']} on {r0['dataset']} disagrees "
                       f"across papers ({shown}); no work explains the difference"),
            note_ids=ids, support_count=len(papers), trail=trail))
    return cards


_STATUS_ORDER = {"open": 0, "unchecked": 1, "partly": 2, "closed": 3}


def rank(cards: Iterable[GapCard]) -> List[GapCard]:
    """Code-only ranking: open first, then support, doability, verified share."""
    return sorted(cards, key=lambda c: (_STATUS_ORDER[c.status], c.needs_check, -c.support_count,
                                        -(c.doability or 0), -c.verified_share, c.id))


def find_gaps(notes: Iterable[Note], settings) -> List[GapCard]:
    notes = list(notes)
    cards = (find_unaddressed(notes, "limitation") + find_unaddressed(notes, "open_question") +
             find_untested_combinations(notes, settings.gaps.min_dataset_users) +
             find_conflicts(notes, settings.gaps.conflict_tolerance))
    return rank(cards)
