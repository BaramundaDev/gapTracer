"""The note graph. Answers use verified links only; absence checks look at
every link so that an unverified link is not mistaken for a missing one."""
from __future__ import annotations

from typing import Dict, Iterable, List

import networkx as nx

from .schema import Note, link_allowed


def build_graph(notes: Iterable[Note], verified_only: bool = True) -> nx.MultiDiGraph:
    notes = list(notes)
    by_id: Dict[str, Note] = {n.id: n for n in notes}
    g = nx.MultiDiGraph()
    for n in notes:
        g.add_node(n.id, type=n.type, name=n.name, l0=n.l0, paper_date=n.paper_date, paper_id=n.paper_id)
    for n in notes:
        for link in n.links:
            target = by_id.get(link.target)
            if target is None or not link_allowed(link.type, n.type, target.type):
                continue
            if verified_only and not link.verified:
                continue
            g.add_edge(n.id, link.target, key=link.type, type=link.type, verified=link.verified,
                       passage_id=link.evidence_passage_id, sentence=link.evidence_sentence,
                       confidence=link.confidence)
    return g


def out_links(g: nx.MultiDiGraph, node: str, link_type: str | None = None) -> List[tuple]:
    return [(u, v, d) for u, v, d in g.out_edges(node, data=True) if link_type in (None, d["type"])]


def in_links(g: nx.MultiDiGraph, node: str, link_type: str | None = None) -> List[tuple]:
    return [(u, v, d) for u, v, d in g.in_edges(node, data=True) if link_type in (None, d["type"])]
