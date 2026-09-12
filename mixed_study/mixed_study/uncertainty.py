"""Section 3.5: uncertainty that preserves group support and compares methods
by PAIRED differences on the same examples.

The source bootstrap resampled template clusters, returned NaN whenever any
included group vanished from a draw, and then discarded that draw -- keeping
293 of 2000 draws in the E7 runs. Its interval was therefore conditional on a
restrictive event. Here:

  * a Bayesian-bootstrap / multiplier scheme draws Dirichlet cluster weights,
    so every cluster (and so every group) stays present in every draw;
  * method-vs-method comparisons are paired differences over the same rows,
    resampled jointly, not two separate intervals held up side by side;
  * a small simulation checks finite-sample coverage of the interval under
    the actual cluster/group structure before it is trusted.

Overlapping separate intervals never established indistinguishability, and
a failure to show superiority is not equivalence; both are stated as such.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np


def _cluster_index(clusters: Sequence[str]):
    idx: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        idx[c].append(i)
    keys = sorted(idx)
    return keys, [np.asarray(idx[k]) for k in keys]


def multiplier_bootstrap(stat: Callable[[np.ndarray], float], clusters: Sequence[str],
                         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """``stat(weights)`` receives one non-negative weight per ROW (cluster-level
    Dirichlet weights broadcast to rows, mean 1). Every group keeps support."""
    rng = np.random.default_rng(seed)
    keys, groups = _cluster_index(clusters)
    n_rows = len(clusters)
    base = np.ones(n_rows)
    point = float(stat(base))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        w_c = rng.dirichlet(np.ones(len(keys))) * len(keys)   # mean 1 over clusters
        w = np.empty(n_rows)
        for wc, g in zip(w_c, groups):
            w[g] = wc
        draws[b] = stat(w)
    ok = draws[np.isfinite(draws)]
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi),
            "se": float(ok.std(ddof=1)), "n_boot": n_boot, "n_boot_valid": int(len(ok)),
            "n_clusters": len(keys), "n_rows": n_rows, "alpha": alpha, "scheme": "dirichlet-multiplier"}


def weighted_group_error_delta(dense_correct: np.ndarray, comp_correct: np.ndarray, group: np.ndarray,
                               g: str) -> Callable[[np.ndarray], float]:
    d = np.asarray(dense_correct, float); c = np.asarray(comp_correct, float); m = np.asarray(group) == g
    def f(w):
        ww = w[m]
        if ww.sum() == 0:
            return np.nan
        return float(((1 - c[m]) * ww).sum() / ww.sum() - ((1 - d[m]) * ww).sum() / ww.sum())
    return f


def paired_method_difference(correct_a: np.ndarray, correct_b: np.ndarray, clusters: Sequence[str],
                             n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Accuracy(A) - Accuracy(B) on the SAME rows, jointly resampled.

    This is the comparison the plan asks for instead of overlapping intervals.
    A CI excluding zero says the methods differ on these examples; a CI that
    includes zero says the data cannot tell them apart -- it does not say they
    are equivalent."""
    a = np.asarray(correct_a, float); b = np.asarray(correct_b, float)
    diff = a - b
    def f(w):
        return float((diff * w).sum() / w.sum())
    res = multiplier_bootstrap(f, clusters, n_boot, seed, alpha)
    res["interpretation"] = ("A better than B" if res["ci_low"] > 0 else
                             "B better than A" if res["ci_high"] < 0 else
                             "not distinguishable on these examples (NOT equivalence)")
    return res


def paired_H_difference(dense_correct, comp_a, comp_b, group, clusters, groups: Sequence[str],
                        n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """H(A) - H(B) where H = max_g dE_g, computed jointly per draw over the same rows."""
    d = np.asarray(dense_correct, float); a = np.asarray(comp_a, float); b = np.asarray(comp_b, float)
    gr = np.asarray(group)
    masks = [gr == g for g in groups]
    def H(c, w):
        vals = []
        for m in masks:
            ww = w[m]
            if ww.sum() == 0:
                return np.nan
            vals.append(((1 - c[m]) * ww).sum() / ww.sum() - ((1 - d[m]) * ww).sum() / ww.sum())
        return max(vals)
    def f(w):
        return H(a, w) - H(b, w)
    res = multiplier_bootstrap(f, clusters, n_boot, seed, alpha)
    res["note"] = ("H is a maximum; the bootstrap of a maximum is biased upward and its point "
                   "estimate can fall outside its own interval. Treat as secondary.")
    return res


def coverage_simulation(clusters: Sequence[str], group: Sequence[str], p_dense: float = 0.2,
                        true_delta: float = 0.05, icc: float = 0.3, n_sim: int = 200,
                        n_boot: int = 300, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Does the multiplier interval cover a known group-delta at the nominal rate
    under THIS design's cluster/group structure? Cheap, and required before the
    interval is reported as 95%."""
    rng = np.random.default_rng(seed)
    keys, idx = _cluster_index(clusters)
    gr = np.asarray(group)
    groups = sorted(set(gr.tolist()))
    target = groups[0]
    n = len(clusters)
    covered = 0
    widths = []
    for s in range(n_sim):
        # cluster random effect on the logit scale, then per-row Bernoulli
        re = rng.normal(0, np.sqrt(icc / (1 - icc)) * 0.8, size=len(keys))
        row_re = np.empty(n)
        for r, g in zip(re, idx):
            row_re[g] = r
        base = 1 / (1 + np.exp(-(np.log(p_dense / (1 - p_dense)) + row_re)))
        dense_err = rng.random(n) < base
        comp_err = dense_err.copy()
        m = gr == target
        flip = (rng.random(n) < true_delta) & m & ~dense_err
        comp_err[flip] = True
        dc, cc = ~dense_err, ~comp_err
        res = multiplier_bootstrap(weighted_group_error_delta(dc, cc, gr, target), clusters,
                                   n_boot, seed + s, alpha)
        truth = float((comp_err[m].mean() - dense_err[m].mean()))
        covered += int(res["ci_low"] <= truth <= res["ci_high"])
        widths.append(res["ci_high"] - res["ci_low"])
    return {"nominal": 1 - alpha, "empirical_coverage": covered / n_sim, "n_sim": n_sim,
            "mean_width": float(np.mean(widths)), "target_group": target,
            "n_clusters_target": len({clusters[i] for i in range(n) if gr[i] == target}),
            "verdict": "adequate" if covered / n_sim >= (1 - alpha) - 0.05 else "UNDER-COVERS: report with caution"}
