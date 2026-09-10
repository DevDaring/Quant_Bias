"""Uncertainty: paired cluster bootstrap, Holm correction, equivalence, MDE.

Units of resampling are template clusters, never tokens or single examples
(plan, Section 6). numpy only; no torch dependency.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np


def _cluster_index(cluster_ids: Sequence[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    idx: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        idx[c].append(i)
    keys = sorted(idx)
    return keys, {k: np.asarray(idx[k]) for k in keys}


def paired_cluster_bootstrap(stat_fn: Callable[[np.ndarray], float], cluster_ids: Sequence[str],
                             n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                             strata: Sequence[str] | None = None) -> dict[str, Any]:
    """Percentile CI for ``stat_fn(row_indices)`` under cluster resampling.

    ``stat_fn`` receives the resampled row indices (with repeats) and returns a
    scalar. Pairing is implicit: dense and quantized values for the same row
    are looked up inside ``stat_fn``. ``strata`` (e.g. calibration seed)
    resamples clusters within each stratum, giving the hierarchical variant.
    """
    rng = np.random.default_rng(seed)
    keys, idx = _cluster_index(cluster_ids)
    point = float(stat_fn(np.arange(len(cluster_ids))))
    if strata is None:
        strata_keys = {None: keys}
    else:
        strat_of = {}
        for c, s in zip(cluster_ids, strata):
            strat_of[c] = s
        strata_keys = defaultdict(list)
        for k in keys:
            strata_keys[strat_of[k]].append(k)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        rows = []
        for s, ks in strata_keys.items():
            pick = rng.integers(0, len(ks), size=len(ks))
            rows.extend(idx[ks[p]] for p in pick)
        r = np.concatenate(rows)
        v = stat_fn(r)
        draws[b] = v if v is not None and np.isfinite(v) else np.nan
    ok = draws[~np.isnan(draws)]
    lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(ok) else (np.nan, np.nan))
    # two-sided bootstrap p-value for H0: stat == 0
    p = 2 * min((ok <= 0).mean(), (ok >= 0).mean()) if len(ok) else np.nan
    return {"estimate": point, "ci_low": float(lo), "ci_high": float(hi), "se": float(ok.std(ddof=1)) if len(ok) > 1 else np.nan,
            "p_value": float(min(1.0, p)), "n_clusters": len(keys), "n_boot_valid": int(len(ok)), "alpha": alpha}


def delta_error_stat(correct_dense: np.ndarray, correct_quant: np.ndarray, group: np.ndarray, g: str):
    """Returns stat_fn for ΔE_g."""
    cd = np.asarray(correct_dense, dtype=float)
    cq = np.asarray(correct_quant, dtype=float)
    gm = np.asarray(group) == g

    def fn(rows: np.ndarray):
        m = gm[rows]
        if m.sum() == 0:
            return np.nan
        return float((1 - cq[rows][m]).mean() - (1 - cd[rows][m]).mean())
    return fn


def disparity_stat(correct_dense: np.ndarray, correct_quant: np.ndarray, group: np.ndarray, groups: Sequence[str],
                   which: str = "A"):
    """stat_fn for A (spread increase) or H (worst added group harm)."""
    cd = np.asarray(correct_dense, dtype=float)
    cq = np.asarray(correct_quant, dtype=float)
    gr = np.asarray(group)

    def fn(rows: np.ndarray):
        ed, eq = [], []
        for g in groups:
            m = gr[rows] == g
            if m.sum() == 0:
                return np.nan
            ed.append((1 - cd[rows][m]).mean())
            eq.append((1 - cq[rows][m]).mean())
        ed, eq = np.array(ed), np.array(eq)
        if which == "A":
            return float((eq.max() - eq.min()) - (ed.max() - ed.min()))
        return float((eq - ed).max())
    return fn


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm step-down correction; returns adjusted p and rejection flag per key."""
    items = sorted(((k, p) for k, p in p_values.items() if p is not None and not math.isnan(p)), key=lambda x: x[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = {"p": p, "p_holm": running, "reject": running < alpha}
    for k, p in p_values.items():
        if k not in out:
            out[k] = {"p": p, "p_holm": None, "reject": False}
    return out


def tost_equivalence(diffs: Sequence[float], cluster_ids: Sequence[str], margin: float,
                     n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Two one-sided tests via cluster bootstrap: equivalent if the (1-2α) CI of
    the mean difference lies inside (-margin, +margin)."""
    d = np.asarray(diffs, dtype=float)
    res = paired_cluster_bootstrap(lambda rows: float(d[rows].mean()), cluster_ids, n_boot, seed, alpha=2 * alpha)
    eq = res["ci_low"] > -margin and res["ci_high"] < margin
    return {**res, "margin": margin, "equivalent": bool(eq)}


def minimum_detectable_effect(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true effect a paired test with standard error ``se`` detects."""
    from scipy.stats import norm
    return float(se * (norm.ppf(1 - alpha / 2) + norm.ppf(power)))


def group_ci_table(correct_dense: Sequence[bool], correct_quant: Sequence[bool], group: Sequence[str],
                   cluster_ids: Sequence[str], min_n: int = 20, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> dict[str, Any]:
    """ΔE_g with CI for every group plus A and H, Holm-corrected across groups."""
    cd, cq, gr = np.asarray(correct_dense, float), np.asarray(correct_quant, float), np.asarray(group)
    counts = {g: int((gr == g).sum()) for g in set(gr.tolist())}
    groups = sorted(g for g, n in counts.items() if n >= min_n)
    rows: dict[str, Any] = {}
    for g in groups:
        rows[g] = paired_cluster_bootstrap(delta_error_stat(cd, cq, gr, g), cluster_ids, n_boot, seed, alpha)
        rows[g]["n"] = counts[g]
    corr = holm({g: rows[g]["p_value"] for g in groups}, alpha)
    for g in groups:
        rows[g].update(corr[g])
    out = {"groups": rows, "excluded_sparse": {g: n for g, n in counts.items() if n < min_n}}
    if groups:
        out["A"] = paired_cluster_bootstrap(disparity_stat(cd, cq, gr, groups, "A"), cluster_ids, n_boot, seed, alpha)
        out["H"] = paired_cluster_bootstrap(disparity_stat(cd, cq, gr, groups, "H"), cluster_ids, n_boot, seed, alpha)
        se = np.nanmean([rows[g]["se"] for g in groups])
        out["mde_delta_E"] = minimum_detectable_effect(float(se), alpha) if np.isfinite(se) else None
    return out


# ----------------------------------------------------------------------------
# Sampling design: how much precision does a smaller per-cluster sample cost?
# ----------------------------------------------------------------------------

def design_effect(m: float, icc: float) -> float:
    """Kish design effect for cluster sampling: DEFF = 1 + (m - 1) * icc."""
    return 1.0 + (max(m, 1.0) - 1.0) * icc


def effective_n(n_clusters: int, m: float, icc: float) -> float:
    """Effective sample size of ``n_clusters`` clusters of size ``m``."""
    return n_clusters * m / design_effect(m, icc)


def subsample_cost(n_clusters: int, m_full: float, m_cut: float, icc: float) -> dict[str, float]:
    """Precision lost by keeping ``m_cut`` instead of ``m_full`` examples per cluster.

    CI half-width scales as 1/sqrt(effective n), so ``ci_inflation`` is the factor
    by which every interval widens. ``compute_ratio`` is what is saved.
    """
    n_full = effective_n(n_clusters, m_full, icc)
    n_cut = effective_n(n_clusters, m_cut, icc)
    return {"icc": icc, "m_full": m_full, "m_cut": m_cut,
            "eff_n_full": n_full, "eff_n_cut": n_cut,
            "eff_n_retained": n_cut / n_full,
            "ci_inflation": (n_full / n_cut) ** 0.5,
            "compute_ratio": m_cut / m_full}


def icc_from_records(correct: Sequence[float], cluster_ids: Sequence[str]) -> float:
    """One-way ANOVA estimate of the intra-cluster correlation of an outcome.

    Run this on the pilot's per-example records to replace the assumed ICC with
    a measured one before fixing the final sample size.
    """
    import numpy as np
    y = np.asarray(correct, dtype=float)
    keys, idx = _cluster_index(cluster_ids)
    k = len(keys)
    if k < 2:
        return float("nan")
    sizes = np.array([len(idx[c]) for c in keys], dtype=float)
    means = np.array([y[idx[c]].mean() for c in keys])
    grand = y.mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    within = sum(float(((y[idx[c]] - means[i]) ** 2).sum()) for i, c in enumerate(keys))
    dfw = len(y) - k
    if dfw <= 0:
        return float("nan")
    msw = within / dfw
    m0 = (sizes.sum() - (sizes ** 2).sum() / sizes.sum()) / (k - 1)
    denom = msb + (m0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else 0.0
