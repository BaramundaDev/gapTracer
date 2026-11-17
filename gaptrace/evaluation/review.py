"""Export blinded human rating forms and merge completed forms into a manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

if __package__:
    from .metrics import Evaluation, HumanRating
else:
    from metrics import Evaluation, HumanRating


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def export_reviews(raw: bytes, sample_size: int = 50, seed: int = 1729):
    evaluation = Evaluation.model_validate_json(raw)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    units = {}
    for run in evaluation.runs:
        for topic in run.topics:
            for gap in topic.gaps:
                unit = units.setdefault((topic.id, gap.statement), {"topic": topic.id, "statement": gap.statement,
                                                                  "evidence": [], "references": []})
                unit["references"].append({"run": run.id, "topic": topic.id, "gap": gap.id})
                for link in gap.evidence or []:
                    evidence = {key: getattr(link, key) for key in ("passage_id", "paper_id", "passage_text", "quote")}
                    if evidence not in unit["evidence"]:
                        unit["evidence"].append(evidence)
    if sample_size > len(units):
        raise ValueError(f"Requested {sample_size} reviews but only {len(units)} unique gaps are available")
    population = [units[key] for key in sorted(units)]
    rng = random.Random(seed)
    selected = rng.sample(population, sample_size)
    public, mapping = [], {}
    for index, unit in enumerate(selected, 1):
        review_id = f"review-{index:04d}"
        # No original gap IDs, ranks, predictions, labels, matches, or system names.
        public.append({"review_id": review_id, "topic": unit["topic"], "statement": unit["statement"],
                       "evidence": unit["evidence"]})
        mapping[review_id] = unit["references"]
    key = {"manifest_sha256": digest(raw), "seed": seed, "population_size": len(population),
           "sample_size": sample_size, "sampling": "uniform without replacement over unique topic and statement",
           "mapping": mapping}
    forms = []
    for number in (1, 2):
        records = [{"review_id": item["review_id"], "real": None, "open": None, "doable": None, "reason": ""}
                   for item in public]
        rng.shuffle(records)
        forms.append({"rater": f"rater-{number}", "manifest_sha256": digest(raw), "ratings": records})
    return {"items": public}, key, forms


def merge_reviews(raw: bytes, key: dict, forms: list[dict]):
    if key["manifest_sha256"] != digest(raw):
        raise ValueError("Review key does not match this exact manifest")
    data = json.loads(raw)
    targets = {(run["id"], topic["id"], gap["id"]): gap
               for run in data["runs"] for topic in run["topics"] for gap in topic["gaps"]}
    used_raters = set()
    for form in forms:
        rater = form["rater"]
        if rater in used_raters:
            raise ValueError(f"Duplicate rater form {rater}")
        used_raters.add(rater)
        if form["manifest_sha256"] != digest(raw):
            raise ValueError("Rating form does not match this manifest")
        seen = set()
        for record in form["ratings"]:
            review_id = record["review_id"]
            if review_id in seen or review_id not in key["mapping"]:
                raise ValueError(f"Duplicate or unknown review ID {review_id}")
            seen.add(review_id)
            rating = HumanRating.model_validate_json(json.dumps({"rater": rater, **{k: v for k, v in record.items() if k != "review_id"}}))
            for ref in key["mapping"][review_id]:
                gap = targets[(ref["run"], ref["topic"], ref["gap"])]
                if any(existing["rater"] == rater for existing in gap.get("ratings", [])):
                    raise ValueError(f"Rating by {rater} already exists for {ref['gap']}")
                gap.setdefault("ratings", []).append(rating.model_dump())
        if seen != key["mapping"].keys():
            raise ValueError("Every sampled gap needs a completed rating in each form")
    # Validate without injecting defaults or changing the original gap order.
    Evaluation.model_validate_json(json.dumps(data))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("manifest", type=Path)
    export.add_argument("--out", type=Path, required=True, help="New output directory")
    export.add_argument("--sample-size", type=int, default=50)
    export.add_argument("--seed", type=int, default=1729)
    merge = commands.add_parser("merge")
    merge.add_argument("manifest", type=Path)
    merge.add_argument("--key", type=Path, required=True)
    merge.add_argument("--ratings", type=Path, nargs="+", required=True)
    merge.add_argument("--out", type=Path, required=True)
    for command in (export, merge):
        command.add_argument("--split", choices=("development", "held-out"), default="development")
    args = parser.parse_args()
    try:
        raw = args.manifest.read_bytes()
        evaluation = Evaluation.model_validate_json(raw)
        if evaluation.split != args.split:
            raise ValueError(f"Use --split {evaluation.split} for this manifest")
        if args.command == "export":
            public, key, forms = export_reviews(raw, args.sample_size, args.seed)
            args.out.mkdir(parents=True, exist_ok=False)
            for name, content in (("items.json", public), ("private-key.json", key),
                                  ("rater-1.json", forms[0]), ("rater-2.json", forms[1])):
                (args.out / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            key = json.loads(args.key.read_text(encoding="utf-8"))
            forms = [json.loads(path.read_text(encoding="utf-8")) for path in args.ratings]
            merged = merge_reviews(raw, key, forms)
            with args.out.open("x", encoding="utf-8") as destination:
                destination.write(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Review error: {exc}\n")


if __name__ == "__main__":
    main()
