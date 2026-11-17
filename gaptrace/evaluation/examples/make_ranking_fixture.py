"""Rebuild the synthetic annotated example; no source literature or model calls.

Run from the repository root: python -m evaluation.examples.make_ranking_fixture
"""
import copy
import json
from pathlib import Path

from evaluation.metrics import Evaluation


def build():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    data["ranking_ks"] = [1, 3, 5, 10]
    data["topics"][0]["gold_gaps"] = [
        {"id": "gold-both", "statement": "A study testing red and blue together.", "relevance": 3},
        {"id": "gold-third", "statement": "A study testing a third color.", "relevance": 2},
    ]
    data["topics"][1]["gold_gaps"] = []
    for run in data["runs"]:
        for topic in run["topics"]:
            topic["gold_mapping_complete"] = True
            for gap in topic["gaps"]:
                good = gap["id"] == "both-colors"
                medium = gap["id"] == "third-color"
                gap.update(type="untested_combination", canonical_id=gap["id"],
                           gold_id="gold-both" if good else "gold-third" if medium else None,
                           predicted_status="open", reference_status="open" if good or medium else "closed",
                           predicted_doability=6, future_probability=0.8 if good else 0.25 if medium else 0.7,
                           judgment={"grader": "synthetic-adjudicator", "real": 5 if good or medium else 2,
                                     "open": 5 if good or medium else 1, "doable": 5 if good else 3,
                                     "reason": "Synthetic label for exercising the scorer only.",
                                     "doability_rubric": {"data": 2, "compute": 1, "blocks": 2, "scope": 2}})
                gap["evidence"] = [{"passage_id": "a-1", "paper_id": "a",
                                    "passage_text": "Synthetic source: this study tests red only.",
                                    "quote": "tests red only", "supported": good or medium,
                                    "grader": "synthetic-evidence-reviewer"}]
                gap["ratings"] = [
                    {"rater": "synthetic-rater-a", "real": 5 if good or medium else 2,
                     "open": 5 if good or medium else 1, "doable": 5 if good else 3,
                     "reason": "Synthetic independent rating A."},
                    {"rater": "synthetic-rater-b", "real": 4 if good or medium else 2,
                     "open": 5 if good or medium else 2, "doable": 4 if good else 3,
                     "reason": "Synthetic independent rating B."},
                ]
    originals = copy.deepcopy(data["runs"])
    for number in (2, 3):
        for original in originals:
            repeat = copy.deepcopy(original)
            repeat["id"] = f"{repeat['system']}-{number}"
            data["runs"].append(repeat)
    raw = json.dumps(data, indent=2) + "\n"
    Evaluation.model_validate_json(raw)
    (root / "examples" / "ranking-fixture.json").write_text(raw, encoding="utf-8")
    (root / "manifest.schema.json").write_text(json.dumps(Evaluation.model_json_schema(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
