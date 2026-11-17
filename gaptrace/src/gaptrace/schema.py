"""The fixed data model. Anything outside it is rejected."""
from __future__ import annotations

import hashlib
import re
from datetime import date as Date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

NoteType = Literal["paper", "problem", "method", "dataset", "result", "limitation", "open_question"]
LinkType = Literal["proposes", "targets", "evaluated_on", "reports", "states", "addresses", "extends", "contradicts"]

# link type -> (allowed source note types, allowed target note types)
ALLOWED_LINKS = {
    "proposes": ({"paper"}, {"method"}),
    "targets": ({"method"}, {"problem"}),
    "evaluated_on": ({"method"}, {"dataset"}),
    "reports": ({"paper"}, {"result"}),
    "states": ({"paper"}, {"limitation", "open_question"}),
    "addresses": ({"paper", "method"}, {"limitation", "open_question"}),
    "extends": ({"method"}, {"method"}),
    "contradicts": ({"result"}, {"result"}),
}


def link_allowed(link_type: str, source_type: str, target_type: str) -> bool:
    src, dst = ALLOWED_LINKS.get(link_type, (set(), set()))
    return source_type in src and target_type in dst


def make_id(note_type: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    digest = hashlib.sha1(f"{note_type}:{name.lower().strip()}".encode()).hexdigest()[:6]
    return f"{note_type[:3]}-{slug}-{digest}"


class Link(BaseModel):
    target: str
    type: LinkType
    evidence_passage_id: str = ""
    evidence_sentence: str = ""
    verified: bool = False
    confidence: float = 0.0


class Fact(BaseModel):
    method: str
    dataset: str
    metric: str
    value: float
    unit: str = ""


class Note(BaseModel):
    id: str
    type: NoteType
    name: str
    aliases: List[str] = Field(default_factory=list)
    l0: str                      # headline, <= limits.l0_tokens
    l1: str = ""                 # summary, <= limits.l1_tokens
    passage_ids: List[str] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)
    paper_id: Optional[str] = None       # the paper a result/limitation came from
    paper_date: Optional[Date] = None
    fact: Optional[Fact] = None          # only for type == "result"


class Passage(BaseModel):
    id: str
    paper_id: str
    section: str = ""
    text: str
    date: Optional[Date] = None


# ---- model outputs (validated by the gateway) ----

class ExtractedNote(BaseModel):
    type: Literal["problem", "method", "dataset", "result", "limitation", "open_question"]
    name: str
    l0: str
    l1: str = ""
    fact: Optional[Fact] = None


class ExtractedLink(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    source: str = Field(alias="from")    # a note name, or "PAPER" for the paper itself
    target: str = Field(alias="to")
    type: LinkType
    evidence_sentence: str


class ExtractorOutput(BaseModel):
    notes: List[ExtractedNote] = Field(default_factory=list)
    links: List[ExtractedLink] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    supported: bool
    confidence: float = 0.0
    reason: str = ""


class ToolRequest(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class StepOutput(BaseModel):
    answer: str = ""
    cited: List[str] = Field(default_factory=list)
    tool_request: Optional[ToolRequest] = None
    state_update: str = ""


class ClosureOutput(BaseModel):
    verdict: Literal["fills", "partly", "does_not"]
    reason: str = ""


class DoabilityOutput(BaseModel):
    data: int = Field(ge=0, le=2)
    compute: int = Field(ge=0, le=2)
    blocks: int = Field(ge=0, le=2)
    scope: int = Field(ge=0, le=2)
    reasons: dict = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.data + self.compute + self.blocks + self.scope


class ConfirmOutput(BaseModel):
    same: bool


# ---- gaps ----

class TrailStep(BaseModel):
    source: str
    link: str
    target: str
    passage_id: str = ""
    sentence: str = ""
    verified: bool = False


GapType = Literal["unaddressed_limitation", "unanswered_open_question", "untested_combination", "conflicting_results"]
GapStatus = Literal["open", "partly", "closed", "unchecked"]


class GapCard(BaseModel):
    id: str
    type: GapType
    statement: str
    note_ids: List[str] = Field(default_factory=list)
    support_count: int = 1
    status: GapStatus = "unchecked"
    needs_check: bool = False        # an unverified link suggests it may already be handled
    decided_by: str = ""             # paper that decided the closure verdict
    doability: Optional[int] = None  # 0..8
    trail: List[TrailStep] = Field(default_factory=list)
    judged_by: str = ""              # which model tier made the judgements

    @property
    def verified_share(self) -> float:
        if not self.trail:
            return 0.0
        return sum(1 for t in self.trail if t.verified) / len(self.trail)
