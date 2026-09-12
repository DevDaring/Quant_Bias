"""Section 6: nested predictor evaluation at FIXED granularity, on saved records.

Runs on CPU from e2_sites.json. Evaluates the nested baseline ladder the plan
lists, at one granularity at a time, with leave-layer-out validation so the
predictor is never scored on the layer it was fit on. The point is to show
whether group conditioning adds anything beyond intervention size and generic
damage -- not to maximise a single correlation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .common import QB_RESULTS, RESULTS, log, read_json, write_json, source_tags, provenance


def _rows(sites: dict[str, Any], granularity: str) -> list[dict[str, Any]]:
    out = []
    for sid, v in sites.items():
        if v.get("granularity") != granularity:
            continue
        f = v.get("features_selection") or {}
        gs = list(f)
        if not gs:
            continue
        ff = v["observed_final"]["full_score_flips"]
        Vg = np.array([f[g]["V_final"] for g in gs])
        Lg = np.array([f[g]["logit_linf"] for g in gs])
        out.append({
            "site": sid, "layer": int(sid[1:].split(".")[0]),
            "depth_frac": None, "n_components": len(v.get("components", [])),
            "V_macro": float(Vg.mean()), "V_max": float(Vg.max()), "V_spread": float(Vg.max() - Vg.min()),
            "linf_macro": float(Lg.mean()), "linf_max": float(Lg.max()),
            "ppl_regret": v.get("ppl_utility"),
            "y_harmful": ff.get("harmful_flip_rate"), "y_flip": ff.get("flip_rate"),
        })
    if out:
        L = max(r["layer"] for r in out) or 1
        for r in out:
            r["depth_frac"] = r["layer"] / L
    return out


LADDER = [
    ("size_depth",        ["n_components", "depth_frac"]),
    ("local_energy",      ["V_macro"]),
    ("ppl_regret",        ["ppl_regret"]),
    ("global_energy_linf",["V_macro", "linf_macro"]),
    ("group_energy",      ["V_macro", "V_max", "V_spread"]),
    ("group_energy_linf", ["V_macro", "V_max", "V_spread", "linf_macro", "linf_max"]),
]


def _fit_predict(X_tr, y_tr, X_te):
    """Ridge on standardised features; tiny n, so keep it linear and regularised."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    Xt = (X_tr - mu) / sd; Xe = (X_te - mu) / sd
    lam = 1.0
    A = Xt.T @ Xt + lam * np.eye(Xt.shape[1]); b = Xt.T @ (y_tr - y_tr.mean())
    w = np.linalg.solve(A, b)
    return Xe @ w + y_tr.mean()


def leave_layer_out(rows: list[dict], feats: list[str], target: str) -> dict[str, Any]:
    from scipy.stats import spearmanr
    layers = sorted({r["layer"] for r in rows})
    X = np.array([[r[f] if r[f] is not None else np.nan for f in feats] for r in rows], float)
    y = np.array([r[target] if r[target] is not None else np.nan for r in rows], float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y, lay = X[ok], y[ok], np.array([r["layer"] for r in rows])[ok]
    if len(y) < 6 or len(set(lay)) < 3:
        return {"n": int(len(y)), "rho": None, "note": "too few sites/layers"}
    preds = np.full(len(y), np.nan)
    for L in sorted(set(lay)):
        te = lay == L
        if te.sum() == len(y):
            continue
        preds[te] = _fit_predict(X[~te], y[~te], X[te])
    m = np.isfinite(preds)
    if m.sum() < 4 or np.allclose(y[m], y[m][0]) or np.allclose(preds[m], preds[m][0]):
        return {"n": int(m.sum()), "rho": None, "note": "degenerate"}
    r = spearmanr(preds[m], y[m])
    return {"n": int(m.sum()), "rho": float(r.correlation), "p": float(r.pvalue)}


def evaluate(tag: str) -> dict[str, Any]:
    sites = read_json(QB_RESULTS / "e2" / tag / "e2_sites.json")
    out: dict[str, Any] = {"tag": tag}
    for gran in ("layer", "component"):
        rows = _rows(sites, gran)
        res = {"n_sites": len(rows)}
        for name, feats in LADDER:
            res[name] = {t: leave_layer_out(rows, feats, t) for t in ("y_harmful", "y_flip")}
        out[gran] = res
    return out


def main() -> None:
    OUT = RESULTS / "group_prediction"; OUT.mkdir(parents=True, exist_ok=True)
    allres = {"provenance": provenance()}
    for tag in source_tags("e2"):
        allres[tag] = evaluate(tag)
        lay = allres[tag]["layer"]
        log(f"  {tag} (layer, harmful, leave-layer-out rho): " +
            "  ".join(f"{n}={lay[n]['y_harmful'].get('rho')!s:>6.6}" for n, _ in LADDER))
    write_json(OUT / "ladder.json", allres)
    # markdown
    L = ["# Nested predictor ladder, leave-layer-out, fixed granularity\n",
         "Target: harmful flip rate on the held-out FINAL split. Spearman of out-of-layer predictions vs observed.\n",
         "| model | granularity | n | " + " | ".join(n for n, _ in LADDER) + " |",
         "|---|---|---|" + "|".join("---" for _ in LADDER) + "|"]
    for tag in source_tags("e2"):
        for gran in ("layer", "component"):
            r = allres[tag][gran]
            cells = []
            for n, _ in LADDER:
                v = r[n]["y_harmful"].get("rho")
                cells.append("--" if v is None else f"{v:+.2f}")
            L.append(f"| {tag} | {gran} | {r['n_sites']} | " + " | ".join(cells) + " |")
    L.append("\nRead across a row: if `group_energy` does not beat `local_energy`/`global_energy_linf` "
             "out of layer, group conditioning has not shown added predictive value on that model.\n")
    (OUT / "LADDER.md").write_text("\n".join(L))
    log(f"report: {OUT / 'LADDER.md'}")


if __name__ == "__main__":
    main()
