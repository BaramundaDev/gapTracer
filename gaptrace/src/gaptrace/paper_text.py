"""Retrieve bounded arXiv HTML text; never label an abstract fallback full text."""
from __future__ import annotations

import re
import hashlib
from datetime import date, datetime, timezone
from html.parser import HTMLParser

from .knowledge import digest
from .literature import _read


class Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = 0
        self.skip = 0
        self.capture = None
        self.buffer = []
        self.heading = "Main text"
        self.sections = []

    def handle_starttag(self, tag, attrs):
        if tag == "article": self.in_article += 1
        if tag in {"script", "style", "nav", "footer"}: self.skip += 1
        if self.in_article and not self.skip and tag in {"h2", "h3", "p"} and self.capture is None:
            self.capture, self.buffer = tag, []

    def handle_data(self, data):
        if self.capture and not self.skip: self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == self.capture:
            text = " ".join(" ".join(self.buffer).split())
            if tag in {"h2", "h3"}:
                self.heading = text
            elif text and not re.search(r"references|bibliography|acknowledg", self.heading, re.I):
                if self.sections and self.sections[-1]["heading"] == self.heading:
                    self.sections[-1]["text"] += "\n\n" + text
                else:
                    self.sections.append({"heading": self.heading or "Main text", "text": text})
            self.capture = None
        if tag == "article": self.in_article = max(0, self.in_article - 1)
        if tag in {"script", "style", "nav", "footer"}: self.skip = max(0, self.skip - 1)


def sections(work, as_of, read=None):
    fallback = [{"heading": "Abstract", "text": work["abstract"]}]
    urls = work.get("url", "") + " " + str(work.get("doi", ""))
    match = re.search(r"(?:/abs/|arxiv[.:/])(\d{4}\.\d{4,5}(?:v\d+)?)", urls, re.I)
    if not match:
        return fallback, {"evidence_level": "abstract", "reason": "No supported full-text HTML source; supply sectioned text to confirm"}
    identifier = match.group(1)
    if date.fromisoformat(as_of) < date.today() and not re.search(r"v\d+$", identifier):
        return fallback, {"evidence_level": "abstract", "reason": "Historical full-text retrieval requires a pinned revision or supplied archive"}
    url = "https://arxiv.org/html/" + identifier
    try:
        raw = (read or _read)(url, {})
        if len(raw) > 4_000_000:
            raise ValueError("HTML source exceeds retrieval size bound")
        parser = Article()
        parser.feed(raw.decode("utf-8"))
        if not parser.sections or sum(len(s["text"]) for s in parser.sections) < 800:
            raise ValueError("No substantial article text was parsed")
        return parser.sections, {"evidence_level": "html full text", "url": url, "revision": identifier,
                                 "retrieved_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashlib.sha256(raw).hexdigest(),
                                 "limitation": "HTML parsing may omit figures, tables or mathematical formatting; verify original sources"}
    except (OSError, ValueError, UnicodeError) as exc:
        return fallback, {"evidence_level": "abstract", "url": url, "reason": f"Full-text fallback: {type(exc).__name__}: {exc}"}
