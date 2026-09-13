"""Section 8: choose the smallest effect of practical interest BEFORE running,
then size the evaluation by simulation on this design's actual structure.

The restoration pilot (n=200, k=4 sites) returned identical arms. Before
spending GPU on a larger test, this asks: on the full final split, what
reduction in harmful transitions could a paired comparison actually detect?
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .common import RESULTS, log, write_json, provenance
from .records import load_config_rows


def paired_power(dense_correct: np.ndarray, base_harm: np.ndarray, clusters: list[str],
                 relative_reduction: float, churn: float, n_sim: int = 400, n_boot: int = 300,
                 seed: int = 0, alpha: float = 0.05) -> float:
    """Power to detect that arm B has `relative_reduction` fewer harmful
    transitions than arm A, with a Dirichlet-multiplier paired interval.

    Alternative model: B repairs a random `relative_reduction` share of A's
    harmful rows AND, independently, churns a `churn` fraction of dense-correct
    rows in both directions (a real restoration changes unrelated answers too).
    A repair-only model gives power 1.0 trivially and is not used."""
    from .uncertainty import multiplier_bootstrap
    rng = np.random.default_rng(seed)
    idx_harm = np.where(base_harm)[0]
    idx_dc = np.where(dense_correct)[0]
    hits = 0
    for s in range(n_sim):
        b = base_harm.copy()
        k = int(round(len(idx_harm) * relative_reduction))
        if k:
            b[rng.choice(idx_harm, k, replace=False)] = False
        # net-zero churn: repair m harmful rows at random AND break m non-harmful
        # dense-correct rows at random. Flipping a random subset on an imbalanced
        # base would add net harm and is not a null.
        m = int(round(len(idx_harm) * churn))
        if m:
            still_harm = np.where(b)[0]; not_harm = np.setdiff1d(idx_dc, np.where(b)[0])
            m = min(m, len(still_harm), len(not_harm))
            b[rng.choice(still_harm, m, replace=False)] = False
            b[rng.choice(not_harm, m, replace=False)] = True
        diff = base_harm.astype(float) - b.astype(float)   # >0 where B repaired
        res = multiplier_bootstrap(lambda w: float((diff * w).sum() / w.sum()), clusters, n_boot, seed + s, alpha)
        hits += int(res["ci_low"] > 0)
    return hits / n_sim


def main(tag: str = "mistral_7b_v0_1", config: str = "rtn4") -> dict[str, Any]:
    dense = load_config_rows("e1", tag, "dense")
    comp = load_config_rows("e1", tag, config)
    rows = [(dense[u], comp[u]) for u in dense if u in comp and dense[u].benchmark == "bbq"
            and dense[u].context_condition == "disambig" and dense[u].label is not None]
    dc = np.array([d.correct_under("sum") for d, _ in rows])
    harm = np.array([d.correct_under("sum") and not c.correct_under("sum") for d, c in rows])
    clusters = [d.cluster for d, _ in rows]
    # churn: of A's harmful rows, what share does a comparable 4-bit method
    # (gptq4 vs rtn4) get right instead? -> the arbitrary repair/break rate two
    # equal-cost interventions show against each other, as a fraction of harm.
    other = load_config_rows("e1", tag, "gptq4" if config == "rtn4" else "rtn4")
    hrows = [d for d, c in rows if d.correct_under("sum") and not c.correct_under("sum") and d.uid in other]
    churn = float(np.mean([other[d.uid].correct_under("sum") for d in hrows])) if hrows else 0.0
    out: dict[str, Any] = {"tag": tag, "reference_config": config, "n_rows": len(rows),
                           "churn_between_comparable_4bit_methods": churn,
                           "n_dense_correct": int(dc.sum()), "n_harmful_events": int(harm.sum()),
                           "harm_rate_per_dense_correct": float(harm.sum() / max(1, dc.sum())),
                           "n_clusters": len(set(clusters)), "power": {}}
    log(f"{tag}/{config}: {len(rows)} BBQ-disambig rows, {int(harm.sum())} harmful events over "
        f"{len(set(clusters))} clusters -> simulating power")
    for rr in (0.15, 0.25, 0.35, 0.50):
        p = paired_power(dc, harm, clusters, rr, churn, n_sim=150, n_boot=200)
        out["power"][f"reduce_{int(rr*100)}pct"] = p
        log(f"  detect a {rr:.0%} reduction in harmful transitions: power = {p:.2f}")
    smallest = next((rr for rr in (0.15, 0.25, 0.35, 0.50) if out["power"][f"reduce_{int(rr*100)}pct"] >= 0.8), None)
    out["smallest_effect_at_80pct_power"] = smallest
    out["decision"] = (f"the full final split can detect a >= {smallest:.0%} reduction; a restoration "
                       "that does less than that is not resolvable here and must not be claimed"
                       if smallest else "no tested effect reaches 80% power; more independent templates needed")
    out["provenance"] = provenance()
    write_json(RESULTS / "power" / f"{tag}_{config}.json", out)
    log(out["decision"])
    return out


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:3]))
