"""Section 3.3: counterfactual pairs with semantic answer correspondence.

Two errors in the original pipeline are fixed here and the source pairs are
kept only as *template-matched comparisons*, never as audited counterfactuals:

1. ``pair_gap_summary`` scored both members of a pair at the FIRST member's
   label index, even when the second member's candidate order or correct answer
   differed. The estimand "probability of the correct answer" must use each
   example's own label.
2. BBQ "pairs" were rows sharing a template and polarity whose correct answers
   belong to different groups. That does not make them minimal-change
   counterfactuals: the context, the question wording and the answer set can all
   differ. Discrim-Eval profiles of one decision question ARE matched by
   construction; WinoBias pro/anti share a sentence with one pronoun swapped.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .records import Row


def _softmax_at(scores, i):
    m = max(scores)
    z = [math.exp(s - m) for s in scores]
    return z[i] / sum(z)


@dataclass
class PairStatus:
    benchmark: str
    kind: str                    # "matched_by_construction" | "template_matched_unaudited"
    same_candidates: bool
    same_label: bool
    same_cluster: bool
    audited: bool = False        # only a human audit can set this


def classify(a: Row, b: Row) -> PairStatus:
    same_cands = tuple(a.meta.get("permutation", ())) == tuple(b.meta.get("permutation", ())) and \
        len(a.logprob_sum) == len(b.logprob_sum)
    if a.benchmark == "discrim_eval":
        kind = "matched_by_construction"      # same decision question, profile varies
    elif a.benchmark == "winobias":
        kind = "matched_by_construction"      # same sentence, pronoun swapped
    else:
        kind = "template_matched_unaudited"   # BBQ: NOT a minimal-change pair
    return PairStatus(a.benchmark, kind, same_cands, a.label == b.label, a.cluster == b.cluster)


def correct_answer_gap(a: Row, b: Row, norm: str = "sum") -> float | None:
    """P_a(correct_a) - P_b(correct_b): each example scored at ITS OWN label."""
    if a.label is None or b.label is None:
        return None
    sa = a.logprob_sum if norm == "sum" else a.logprob_mean
    sb = b.logprob_sum if norm == "sum" else b.logprob_mean
    return _softmax_at(sa, a.label) - _softmax_at(sb, b.label)


def shared_decision_gap(a: Row, b: Row, decision_idx: int = 0, norm: str = "sum") -> float:
    """P_a(decision) - P_b(decision) for a fixed shared decision (Discrim-Eval 'yes')."""
    sa = a.logprob_sum if norm == "sum" else a.logprob_mean
    sb = b.logprob_sum if norm == "sum" else b.logprob_mean
    return _softmax_at(sa, decision_idx) - _softmax_at(sb, decision_idx)


def rebuild_pairs(rows: dict[str, Row]) -> list[tuple[Row, Row, PairStatus]]:
    """Reconstruct pairs from records alone (no dataset download).

    Discrim-Eval: every profile vs the (60, male, white) baseline of its question.
    WinoBias: pro vs anti of the same cluster.
    BBQ: disambiguated rows of the same cluster+polarity whose correct answer
    belongs to different groups -- kept, but flagged unaudited.
    """
    out: list[tuple[Row, Row, PairStatus]] = []
    de: dict[str, list[Row]] = defaultdict(list)
    wb: dict[str, dict[str, Row]] = defaultdict(dict)
    bbq: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows.values():
        if r.benchmark == "discrim_eval":
            de[r.cluster].append(r)
        elif r.benchmark == "winobias":
            wb[r.cluster][r.meta.get("condition", "?")] = r
        elif r.benchmark == "bbq" and r.context_condition == "disambig" and r.label is not None:
            bbq[(r.cluster, r.meta.get("question_polarity"))].append(r)
    for cl, lst in de.items():
        base = next((r for r in lst if r.meta.get("is_baseline")), None)
        if base is None:
            continue
        for r in lst:
            if r is not base:
                out.append((base, r, classify(base, r)))
    for cl, d in wb.items():
        if "pro" in d and "anti" in d:
            out.append((d["pro"], d["anti"], classify(d["pro"], d["anti"])))
    import itertools
    for key, lst in bbq.items():
        lst = sorted(lst, key=lambda r: r.uid)
        for x, y in itertools.islice(itertools.combinations(lst, 2), 8):   # same cap as the source run
            if x.answer_groups and y.answer_groups and x.label is not None and y.label is not None:
                if x.answer_groups[x.label] != y.answer_groups[y.label]:
                    out.append((x, y, classify(x, y)))
    return out


def pair_gap_change(dense: dict[str, Row], comp: dict[str, Row], norm: str = "sum") -> dict[str, Any]:
    """Change in the (correctly mapped) pair gap from dense to compressed, split
    by whether the pair is matched by construction or merely template-matched."""
    pairs = rebuild_pairs(dense)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for a, b, st in pairs:
        ca, cb = comp.get(a.uid), comp.get(b.uid)
        if ca is None or cb is None:
            continue
        if a.benchmark == "discrim_eval":
            gd, gc = shared_decision_gap(a, b, 0, norm), shared_decision_gap(ca, cb, 0, norm)
            estimand = "shared_decision_P(yes)"
        else:
            gd, gc = correct_answer_gap(a, b, norm), correct_answer_gap(ca, cb, norm)
            estimand = "own_correct_answer_probability"
            if gd is None or gc is None:
                continue
        buckets[f"{a.benchmark}:{st.kind}"].append({
            "gap_dense": gd, "gap_comp": gc, "signed_change": gc - gd,
            "abs_gap_change": abs(gc) - abs(gd), "estimand": estimand})
    out: dict[str, Any] = {}
    for k, rows in buckets.items():
        n = len(rows)
        out[k] = {"n": n, "estimand": rows[0]["estimand"],
                  "confirmatory_eligible": not k.endswith("unaudited"),
                  "mean_abs_gap_dense": sum(abs(r["gap_dense"]) for r in rows) / n,
                  "mean_abs_gap_comp": sum(abs(r["gap_comp"]) for r in rows) / n,
                  "mean_signed_change": sum(r["signed_change"] for r in rows) / n,
                  "mean_abs_gap_change": sum(r["abs_gap_change"] for r in rows) / n}
    return out
