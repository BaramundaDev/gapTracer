"""Pure evaluation calculations. No product imports, models, or extra dependencies."""
from __future__ import annotations

import itertools
import math
import random
import statistics


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def average(values):
    return ratio(math.fsum(values), len(values))


def dcg(grades, k):
    return math.fsum((2 ** grade - 1) / math.log2(index + 2)
                     for index, grade in enumerate(grades[:k]))


def ndcg(grades, ideal_grades, k):
    ideal = dcg(sorted(ideal_grades, reverse=True), k)
    # Store a topic-level observation so aggregation is macro, not IDCG-weighted.
    return ratio(dcg(grades, k) / ideal, 1) if ideal else ratio(0, 0)


def relevance(judgment):
    """Prespecified ordinal utility: validity gates feasibility."""
    if judgment.real < 4 or judgment.open < 4:
        return 0
    return 3 if judgment.doable >= 4 else 2 if judgment.doable == 3 else 1


def score_ranking(gaps, hits, gold_gaps, ks, gold_mapping_complete=False):
    """List order is the frozen product order; never sort using evaluation labels."""
    n = len(gaps)
    metrics = {}
    incomplete = []
    seen = set()
    unique = []
    for gap in gaps:
        key = ("canonical", gap.canonical_id) if gap.canonical_id else ("gap", gap.id)
        unique.append(key not in seen)
        seen.add(key)
    future = [int(gap.id in hits and fresh) for gap, fresh in zip(gaps, unique)]
    quality_complete = all(gap.judgment is not None for gap in gaps)
    grades = [relevance(gap.judgment) * fresh for gap, fresh in zip(gaps, unique)] if quality_complete else None
    gold = {gap.id: gap.relevance for gap in gold_gaps or []}
    catalogue_ready = gold_gaps is not None and gold_mapping_complete
    gold_seen = set()
    gold_grades = []
    for gap in gaps:
        gold_grades.append(gold.get(gap.gold_id, 0) if gap.gold_id not in gold_seen else 0)
        if gap.gold_id is not None:
            gold_seen.add(gap.gold_id)
    metrics["gap_yield"] = ratio(n, 1)
    metrics["gap_judgment_coverage"] = ratio(sum(gap.judgment is not None for gap in gaps), n)
    metrics["gap_type_coverage"] = ratio(sum(gap.type is not None for gap in gaps), n)
    metrics["canonical_label_coverage"] = ratio(sum(gap.canonical_id is not None for gap in gaps), n)
    metrics["duplicate_rate"] = ratio(n - sum(unique), n) if all(gap.canonical_id for gap in gaps) else ratio(0, 0)
    if any(gap.canonical_id is None for gap in gaps):
        incomplete.append("duplicate_rate")
    first = next((index for index, hit in enumerate(future, 1) if hit), None)
    metrics["future_mrr"] = ratio(1 / first if first else 0, 1)
    total_hits = sum(future)
    for k in ks:
        top = gaps[:k]
        prefix = future[:k]
        metrics[f"list_fill@{k}"] = ratio(len(top), k)
        metrics[f"future_precision@{k}"] = ratio(sum(prefix), k)
        metrics[f"future_success@{k}"] = ratio(int(any(prefix)), 1)
        metrics[f"future_recall_within_list@{k}"] = ratio(sum(prefix), total_hits)
        metrics[f"future_ndcg_within_list@{k}"] = ndcg(future, future, k)
        ap = math.fsum(sum(prefix[:index]) / index for index, hit in enumerate(prefix, 1) if hit)
        metrics[f"future_ap_within_list@{k}"] = ratio(ap, min(k, total_hits))
        metrics[f"judgment_coverage@{k}"] = ratio(sum(gap.judgment is not None for gap in top), len(top))
        complete = all(gap.judgment is not None for gap in top)
        actionable = sum(fresh and gap.judgment.real >= 4 and gap.judgment.open >= 4 and gap.judgment.doable >= 4
                         for gap, fresh in zip(top, unique)) if complete else 0
        metrics[f"actionable_precision@{k}"] = ratio(actionable, k) if complete else ratio(0, 0)
        metrics[f"quality_ndcg_within_list@{k}"] = ndcg(grades, grades, k) if grades is not None else ratio(0, 0)
        if not complete:
            incomplete.extend([f"actionable_precision@{k}", *(f"{axis}_utility@{k}" for axis in ("real", "open", "doable"))])
        if not quality_complete:
            incomplete.append(f"quality_ndcg_within_list@{k}")
        for axis in ("real", "open", "doable"):
            # Missing output slots score zero; 1..5 labels map linearly to 0..1.
            utility = math.fsum((getattr(gap.judgment, axis) - 1) / 4 * fresh
                                for gap, fresh in zip(top, unique)) if complete else 0
            metrics[f"{axis}_utility@{k}"] = ratio(utility, k) if complete else ratio(0, 0)
        metrics[f"gold_recall@{k}"] = ratio(sum(grade > 0 for grade in gold_grades[:k]),
                                             sum(grade > 0 for grade in gold.values())) if catalogue_ready else ratio(0, 0)
        metrics[f"gold_ndcg@{k}"] = ndcg(gold_grades, list(gold.values()), k) if catalogue_ready else ratio(0, 0)
        if not catalogue_ready:
            incomplete.extend([f"gold_recall@{k}", f"gold_ndcg@{k}"])

    annotated = [gap for gap in gaps if gap.evidence is not None]
    links = [link for gap in annotated for link in gap.evidence]
    metrics["gap_evidence_coverage"] = ratio(len(annotated), n)
    metrics["gap_evidence_precision"] = ratio(sum(link.supported for link in links), len(links))
    metrics["gap_unsupported_link_rate"] = ratio(sum(not link.supported for link in links), len(links))
    metrics["fully_supported_gap_rate"] = (ratio(sum(bool(gap.evidence) and all(link.supported for link in gap.evidence)
                                                    for gap in gaps), n)
                                            if len(annotated) == n else ratio(0, 0))
    if len(annotated) != n:
        incomplete.append("fully_supported_gap_rate")
    metrics["independent_support_papers"] = average([len({link.paper_id for link in gap.evidence if link.supported})
                                                       for gap in annotated])
    status_pairs = [gap for gap in gaps if gap.reference_status is not None and gap.predicted_status is not None]
    closed = [gap for gap in status_pairs if gap.reference_status == "closed"]
    metrics["closure_label_coverage"] = ratio(len(status_pairs), n)
    metrics["closure_accuracy"] = ratio(sum(gap.reference_status == gap.predicted_status for gap in status_pairs), len(status_pairs))
    metrics["false_open_rate"] = ratio(sum(gap.predicted_status == "open" for gap in closed), len(closed))
    rubric_pairs = [gap for gap in gaps if gap.predicted_doability is not None and gap.judgment is not None
                    and gap.judgment.doability_rubric is not None]
    metrics["doability_label_coverage"] = ratio(len(rubric_pairs), n)
    metrics["doability_mae"] = average([abs(gap.predicted_doability - sum(gap.judgment.doability_rubric.model_dump().values()))
                                         for gap in rubric_pairs])
    probabilistic = [gap for gap in gaps if gap.future_probability is not None]
    metrics["future_probability_coverage"] = ratio(len(probabilistic), n)
    metrics["future_brier"] = average([(gap.future_probability - int(gap.id in hits)) ** 2 for gap in probabilistic])
    bins = []
    for index in range(10):
        members = [gap for gap in probabilistic if min(9, int(gap.future_probability * 10)) == index]
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": len(members),
                     "mean_probability": statistics.mean(gap.future_probability for gap in members) if members else None,
                     "observed_rate": sum(gap.id in hits for gap in members) / len(members) if members else None})
    ece_sum = math.fsum(b["count"] * abs(b["mean_probability"] - b["observed_rate"]) for b in bins if b["count"])
    metrics["future_ece"] = ratio(ece_sum, len(probabilistic))
    # Diagnostics retain the raw status confusion and the four gap-pattern slices.
    confusion = {reference: {prediction: sum(gap.reference_status == reference and gap.predicted_status == prediction
                                            for gap in status_pairs)
                             for prediction in ("open", "partly", "closed", "unchecked")}
                 for reference in ("open", "partly", "closed")}
    by_type = {}
    for kind in ("unaddressed_limitation", "unanswered_open_question", "untested_combination", "conflicting_results"):
        members = [gap for gap in gaps if gap.type == kind]
        by_type[kind] = {"gap_count": len(members), "hit_rate": ratio(sum(gap.id in hits for gap in members), len(members))}
    items = [{"rank": rank, "gap_id": gap.id, "canonical_id": gap.canonical_id,
              "gold_id": gap.gold_id, "first_canonical_occurrence": fresh,
              "future_confirmed": gap.id in hits, "future_gain": gain,
              "quality_grade": relevance(gap.judgment) if gap.judgment else None,
              "gold_gain": gold_gain if catalogue_ready else None}
             for rank, (gap, fresh, gain, gold_gain) in enumerate(zip(gaps, unique, future, gold_grades), 1)]
    return {"metrics": metrics, "incomplete_metrics": incomplete,
            "order": [gap.id for gap in gaps], "items": items, "calibration_bins": bins,
            "closure_confusion": confusion, "by_type": by_type,
            "counts": {"gaps": n, "judged": sum(gap.judgment is not None for gap in gaps),
                       "evidence_links": len(links), "probability_predictions": len(probabilistic),
                       "gold_catalogue": len(gold) if gold_gaps is not None else None}}


def weighted_kappa(pairs):
    """Linear-weighted Cohen kappa on the fixed 1..5 scale."""
    n = len(pairs)
    if not n:
        return None
    observed = math.fsum(abs(a - b) / 4 for a, b in pairs) / n
    left = [sum(a == value for a, _ in pairs) for value in range(1, 6)]
    right = [sum(b == value for _, b in pairs) for value in range(1, 6)]
    expected = math.fsum(left[a] * right[b] * abs(a - b) / 4 for a in range(5) for b in range(5)) / n ** 2
    return 1 - observed / expected if expected else None


def human_agreement(evaluation):
    # Identical topic+statement is one review unit, even across systems/repeats.
    units = {}
    system_units = {}
    for run in evaluation.runs:
        for topic in run.topics:
            for gap in topic.gaps:
                unit = units.setdefault((topic.id, gap.statement), {})
                system_units.setdefault(run.system, {})[(topic.id, gap.statement)] = unit
                for rating in gap.ratings:
                    previous = unit.setdefault(rating.rater, rating)
                    if previous != rating:
                        raise ValueError(f"Conflicting repeated rating by {rating.rater} for {topic.id}/{gap.id}")
    raters = sorted({rater for unit in units.values() for rater in unit})
    results = []
    for left, right in itertools.combinations(raters, 2):
        matched = [unit for unit in units.values() if left in unit and right in unit]
        axes = {}
        for axis in ("real", "open", "doable"):
            pairs = [(getattr(unit[left], axis), getattr(unit[right], axis)) for unit in matched]
            axes[axis] = {"linear_weighted_kappa": weighted_kappa(pairs),
                          "exact_agreement": ratio(sum(a == b for a, b in pairs), len(pairs)),
                          "within_one_agreement": ratio(sum(abs(a - b) <= 1 for a, b in pairs), len(pairs))}
        results.append({"raters": [left, right], "paired_gaps": len(matched), "axes": axes})
    def rating_summary(selected):
        reviewed = [unit for unit in selected.values() if unit]
        axes = {}
        for axis in ("real", "open", "doable"):
            # Average raters within gaps first; gaps, not rating forms, get equal weight.
            values = [statistics.mean(getattr(rating, axis) for rating in unit.values()) for unit in reviewed]
            axes[axis] = {"gap_mean": statistics.mean(values) if values else None,
                          "sample_sd_across_gap_means": statistics.stdev(values) if len(values) > 1 else None,
                          "gaps": len(values),
                          "rating_histogram": {str(score): sum(getattr(rating, axis) == score
                                                                for unit in reviewed for rating in unit.values())
                                               for score in range(1, 6)}}
        return {"reviewed_unique_gaps": len(reviewed), "axes": axes}
    return {"reviewed_unique_gaps": sum(bool(unit) for unit in units.values()),
            "double_rated_unique_gaps": sum(len(unit) >= 2 for unit in units.values()), "pairs": results,
            "rating_summary": rating_summary(units),
            "rating_summary_by_system": {system: rating_summary(selected) for system, selected in system_units.items()}}


def bootstrap_interval(values, samples=2000, seed=1729):
    """Resample topics, never claims/gaps/repeats as independent observations."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    def quantile(p):
        position = (len(draws) - 1) * p
        lo = math.floor(position)
        hi = math.ceil(position)
        return draws[lo] + (draws[hi] - draws[lo]) * (position - lo)
    return {"lower": quantile(0.025), "upper": quantile(0.975), "samples": samples, "seed": seed}


def metric_direction(name):
    if name in {"hallucination_rate", "retrieval_steps", "response_seconds", "duplicate_rate",
                "gap_unsupported_link_rate", "false_open_rate", "doability_mae", "future_brier", "future_ece"}:
        return "lower"
    if name in {"lead_time_months", "gap_yield", "independent_support_papers"} or "coverage" in name or name.startswith("list_fill"):
        return "diagnostic"
    return "higher"
