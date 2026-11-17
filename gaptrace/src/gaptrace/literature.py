"""Bounded scholarly-index search. Returned text is evidence data, never instructions."""
from __future__ import annotations

import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from datetime import date
from time import monotonic, sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_lock = threading.Lock()
_last_arxiv = 0.0


def _read(url, headers=None):
    request = Request(url, headers={"User-Agent": "GapTrace/0.1 literature review", **(headers or {})})
    with urlopen(request, timeout=25) as response:
        return response.read(4_000_001)


def search(query: str, provider="openalex", limit=5, as_of: str = "", read=None) -> list[dict]:
    if not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("Search limit must be between 1 and 10")
    return _search_page(query, provider, limit, as_of, read)


def _search_page(query, provider, limit, as_of, read=None, offset=0, start=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("Use a nonempty scholarly query of at most 500 characters")
    end = date.fromisoformat(as_of) if as_of else date.today()
    if end > date.today():
        raise ValueError("as_of cannot be in the future")
    read = read or _read
    papers = {}
    if provider == "openalex":
        params = {"search": query.strip(), "filter": f"to_publication_date:{end},has_abstract:true",
                  "per_page": limit, "page": offset // limit + 1,
                  "select": "id,doi,publication_date,display_name,abstract_inverted_index"}
        if start:
            params["filter"] += f",from_publication_date:{start}"
        key = os.environ.get("OPENALEX_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        raw = read("https://api.openalex.org/works?" + urlencode(params), headers)
        if len(raw) > 4_000_000:
            raise ValueError("Scholarly search response exceeded its size limit")
        for work in json.loads(raw)["results"][:limit]:
            positions = sorted((index, word) for word, indices in (work.get("abstract_inverted_index") or {}).items()
                               for index in indices)
            if not work.get("display_name") or not positions:
                continue
            pid = "oa-" + work["id"].rsplit("/", 1)[-1]
            papers[pid] = {"id": pid, "title": work["display_name"], "abstract": " ".join(w for _, w in positions),
                           "published": work["publication_date"], "url": work["id"], "doi": work.get("doi")}
    elif provider == "arxiv":
        stopwords = {"does", "has", "have", "been", "not", "no", "later", "work", "addresses", "the", "on", "in",
                     "and", "a", "an", "only", "whether", "is", "are", "for", "of", "to", "with", "help"}
        terms = [term for term in re.findall(r"[\w-]+", query, flags=re.UNICODE) if term.lower() not in stopwords][:16]
        expression = " AND ".join(f'all:"{term}"' for term in terms)
        if not expression:
            raise ValueError("arXiv query has no searchable terms")
        beginning = date.fromisoformat(start).strftime("%Y%m%d") if start else "19000101"
        expression += f" AND submittedDate:[{beginning}0000 TO {end:%Y%m%d}2359]"
        global _last_arxiv
        with _lock:
            sleep(max(0, 3 - (monotonic() - _last_arxiv)))
            _last_arxiv = monotonic()
            raw = read("https://export.arxiv.org/api/query?" + urlencode({
                "search_query": expression, "start": offset, "max_results": limit, "sortBy": "relevance"}), {})
        if len(raw) > 4_000_000:
            raise ValueError("Scholarly search response exceeded its size limit")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        feed = ET.fromstring(raw)
        for entry in feed.findall("a:entry", ns)[:limit]:
            get = lambda name: " ".join(entry.findtext("a:" + name, default="", namespaces=ns).split())
            identifier = get("id").partition("/abs/")[2]
            if not identifier:
                raise ValueError("arXiv returned an error or invalid entry")
            if get("updated")[:10] > end.isoformat():
                continue  # No historical claims from a later revision's abstract.
            pid = "arxiv-" + re.sub(r"v\d+$", "", identifier)
            papers[pid] = {"id": pid, "title": get("title"), "abstract": get("summary"),
                           "published": get("published")[:10], "url": "https://arxiv.org/abs/" + identifier}
    else:
        raise ValueError("Provider must be openalex or arxiv")
    for paper in papers.values():
        if (date.fromisoformat(paper["published"]) > end or (start and paper["published"] < start)
                or not paper["title"] or not paper["abstract"]):
            raise ValueError("Search returned undated, empty or out-of-scope evidence")
        paper.update(provider=provider, evidence_level="abstract", year=paper["published"][:4])
    return list(papers.values())


def collect(query: str, provider="openalex", limit=30, as_of="", progress=None, read=None):
    """Collect up to 100 unique abstracts, scanning at most ten index pages."""
    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Paper count must be between 1 and 100")
    progress = progress or (lambda message: None)
    size = min(limit, 25)
    papers, seen = [], set()
    for page in range(10):
        progress(f"Searching {provider}, page {page + 1}: {len(papers)}/{limit} papers collected")
        batch = _search_page(query, provider, size, as_of, read, offset=page * size)
        if not batch:
            break
        added = 0
        for paper in batch:
            identities = {paper["id"], "title:" + re.sub(r"\W+", "", paper["title"].casefold())}
            if paper.get("doi"):
                identities.add("doi:" + paper["doi"].casefold().removeprefix("https://doi.org/"))
            if identities & seen:
                continue
            seen.update(identities)
            papers.append(paper)
            added += 1
            if len(papers) == limit:
                return papers
        if not added:
            break
    return papers


def collect_window(query, start, end, limit=50, provider="openalex", progress=None, read=None):
    """Collect a dated stratum for old attempts or newer candidate enablers."""
    if not 1 <= limit <= 200 or date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("Use a valid date window and 1-200 papers")
    progress = progress or (lambda message: None)
    works, seen = [], set()
    page_size = min(limit, 25)
    for page in range(10):
        progress(f"Collecting {start} through {end}: {len(works)}/{limit}")
        batch = _search_page(query, provider, page_size, end, read, page * page_size, start=start)
        if not batch:
            break
        added = 0
        for work in batch:
            key = work.get("doi") or re.sub(r"\W+", "", work["title"].casefold())
            if key not in seen:
                seen.add(key)
                works.append(work)
                added += 1
                if len(works) == limit:
                    return works
        if not added:
            break
    return works
