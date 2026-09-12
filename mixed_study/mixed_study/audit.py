"""P0: reanalyse the existing records under the corrected definitions.

Runs entirely on CPU from the E1/E2/E7 records the GPU run stored. Produces,
under results/v2/audit/:

  fixed_granularity.json    E6-style correlations at fixed intervention type
  outcomes_<exp>.json       the four separate harm outcomes per model/config
  margins_<exp>.json        corrected argmax / gold margin checks
  pairs_<exp>.json          semantically-mapped pair gaps, audited vs unaudited
  competence.json           dense competence flags per model, sum vs mean rule
  method_pairs_<tag>.json   paired method differences with multiplier bootstrap
  coverage.json             finite-sample coverage of the interval on this design
  AUDIT.md                  the reconciled tables

Source result files are never modified.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

from . import harm, margins, pairs as pairs_mod, competence as comp_mod, uncertainty as unc
from .common import QB_RESULTS, RESULTS, TAGS, log, read_json, write_json, source_tags, provenance
from .records import load_config_rows, load_paired, available_configs

OUT = RESULTS / "audit"


# ----------------------------------------------------------------------------
# 1. fixed-granularity correlations (Next_Plan §2.1)
# ----------------------------------------------------------------------------

def _spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.allclose(x[ok], x[ok][0]) or np.allclose(y[ok], y[ok][0]):
        return {"n": int(ok.sum()), "rho": None, "p": None}
    r = spearmanr(x[ok], y[ok])              # average ranks for ties
    return {"n": int(ok.sum()), "rho": float(r.correlation), "p": float(r.pvalue)}


def fixed_granularity() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag in source_tags("e2"):
        sites = read_json(QB_RESULTS / "e2" / tag / "e2_sites.json")
        rows = []
        for sid, v in sites.items():
            f = v.get("features_selection") or {}
            groups = list(f)
            if not groups:
                continue
            ff = v["observed_final"]["full_score_flips"]
            rows.append({
                "site": sid, "granularity": v.get("granularity"),
                "n_components": len(v.get("components", [])),
                "V_final_macro": float(np.mean([f[g]["V_final"] for g in groups])),
                "V_final_max": float(np.max([f[g]["V_final"] for g in groups])),
                "logit_linf_macro": float(np.mean([f[g]["logit_linf"] for g in groups])),
                "ppl_utility": v.get("ppl_utility"),
                "harmful_flip_rate": ff.get("harmful_flip_rate"),
                "flip_rate": ff.get("flip_rate"),
            })
        res: dict[str, Any] = {"n_sites_total": len(rows)}
        for name, sel in (("all", rows), ("layer", [r for r in rows if r["granularity"] == "layer"]),
                          ("component", [r for r in rows if r["granularity"] == "component"])):
            res[name] = {
                "n": len(sel),
                "V_macro_vs_harmful": _spearman([r["V_final_macro"] for r in sel], [r["harmful_flip_rate"] for r in sel]),
                "V_macro_vs_anyflip": _spearman([r["V_final_macro"] for r in sel], [r["flip_rate"] for r in sel]),
                "ppl_vs_harmful": _spearman([r["ppl_utility"] for r in sel], [r["harmful_flip_rate"] for r in sel]),
                "ncomp_vs_harmful": _spearman([r["n_components"] for r in sel], [r["harmful_flip_rate"] for r in sel]),
            }
        res["note"] = ("'all' pools whole-layer and single-component interventions of very different size; "
                       "the component subset is selection-biased (drawn from the four layers the predictor "
                       "shortlisted). Only 'layer' is an unselected, fixed-size sample.")
        out[tag] = res
    return out


# ----------------------------------------------------------------------------
# 2-4. outcomes, margins, pairs for every model/config
# ----------------------------------------------------------------------------

def outcomes_for(exp: str, norms=("sum", "mean")) -> tuple[dict, dict, dict]:
    outc, marg, prs = {}, {}, {}
    for tag in source_tags(exp):
        cfgs = [c for c in available_configs(exp, tag) if c != "dense"]
        if "dense" not in available_configs(exp, tag):
            continue
        outc[tag], marg[tag], prs[tag] = {}, {}, {}
        for cfg in cfgs:
            p = load_paired(exp, tag, cfg)
            outc[tag][cfg] = {n: harm.all_outcomes(p, n) for n in norms}
            marg[tag][cfg] = margins.check_all(p.rows(), "sum")
            prs[tag][cfg] = pairs_mod.pair_gap_change(p.dense, p.comp, "sum")
            log(f"  {exp}/{tag}/{cfg}: harmful(bbq disambig, per dense-correct)="
                f"{outc[tag][cfg]['sum']['task_damage_bbq_disambig']['harmful_per_dense_correct']:.4f}  "
                f"new stereotype errs={outc[tag][cfg]['sum']['stereotype_damage']['new_stereotype_errors']}  "
                f"argmax viol={marg[tag][cfg]['argmax']['n_violations']} gold viol={marg[tag][cfg]['gold']['n_violations']}")
    return outc, marg, prs


# ----------------------------------------------------------------------------
# 5. competence
# ----------------------------------------------------------------------------

def competence_all() -> dict[str, Any]:
    out = {}
    for tag in source_tags("e1"):
        rows = load_config_rows("e1", tag, "dense")
        out[tag] = comp_mod.competence(rows)
        log(f"  competence {tag}: " + " | ".join(out[tag]["verdict"])[:160])
    return out


# ----------------------------------------------------------------------------
# 6. paired method differences (E7) with support-preserving bootstrap
# ----------------------------------------------------------------------------

def method_pairs(tag: str, n_boot: int = 2000, min_n: int = 20, min_clusters: int = 3) -> dict[str, Any]:
    cfgs = [c for c in available_configs("e7", tag) if c != "dense"]
    if "gptq_uniform" not in cfgs:
        return {}
    dense = load_config_rows("e1", tag, "dense") if "dense" not in available_configs("e7", tag) \
        else load_config_rows("e7", tag, "dense")
    rows = {c: load_config_rows("e7", tag, c) for c in cfgs}
    # common BBQ-disambiguated rows with labels
    common = sorted(set(dense) & set.intersection(*(set(r) for r in rows.values())))
    common = [u for u in common if dense[u].benchmark == "bbq" and dense[u].context_condition == "disambig"
              and dense[u].label is not None]
    clusters = [dense[u].cluster for u in common]
    group = np.array([dense[u].group for u in common])
    dc = np.array([dense[u].correct_under("sum") for u in common], float)
    # eligible groups by n AND independent clusters
    gc: dict[str, set] = defaultdict(set); gn: dict[str, int] = defaultdict(int)
    for u in common:
        gc[dense[u].group].add(dense[u].cluster); gn[dense[u].group] += 1
    eligible = sorted(g for g in gn if gn[g] >= min_n and len(gc[g]) >= min_clusters)
    out: dict[str, Any] = {"n_rows": len(common), "n_clusters": len(set(clusters)),
                           "eligible_groups": {g: {"n": gn[g], "clusters": len(gc[g])} for g in eligible},
                           "excluded_groups": {g: {"n": gn[g], "clusters": len(gc[g])} for g in gn if g not in eligible},
                           "vs_gptq_uniform": {}}
    base = np.array([rows["gptq_uniform"][u].correct_under("sum") for u in common], float)
    for c in cfgs:
        if c == "gptq_uniform":
            continue
        other = np.array([rows[c][u].correct_under("sum") for u in common], float)
        acc = unc.paired_method_difference(other, base, clusters, n_boot, 0)
        H = unc.paired_H_difference(dc, other, base, group, clusters, eligible, n_boot, 0) if eligible else None
        out["vs_gptq_uniform"][c] = {"accuracy_diff": acc, "H_diff": H}
        log(f"  {tag} {c} vs gptq_uniform: dAcc={acc['estimate']:+.4f} [{acc['ci_low']:+.4f},{acc['ci_high']:+.4f}] "
            f"-> {acc['interpretation']}")
    return out


# ----------------------------------------------------------------------------
# 7. coverage of the interval on this design
# ----------------------------------------------------------------------------

def coverage(tag: str = "mistral_7b_v0_1") -> dict[str, Any]:
    dense = load_config_rows("e1", tag, "dense")
    rows = [r for r in dense.values() if r.benchmark == "bbq" and r.context_condition == "disambig" and r.label is not None]
    clusters = [r.cluster for r in rows]
    group = [r.group for r in rows]
    return unc.coverage_simulation(clusters, group, n_sim=150, n_boot=200)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def _md_table(rows, cols):
    def c(v):
        return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
    return ("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
            + "".join("| " + " | ".join(c(r.get(k)) for k in cols) + " |\n" for r in rows))


def write_report(fg, outc, marg, prs, compt, mp, cov) -> None:
    L = ["# P0 audit: corrected reanalysis of the completed run\n",
         f"Generated {provenance()['generated']}. Source records under `quant-bias/results/` are unchanged.\n"]
    L.append("## 1. Fixed-granularity correlations (V_final macro vs harmful flip rate)\n")
    rows = []
    for tag, r in fg.items():
        rows.append({"model": tag, "all n/rho": f"{r['all']['n']} / {_f(r['all']['V_macro_vs_harmful']['rho'])}",
                     "layer n/rho": f"{r['layer']['n']} / {_f(r['layer']['V_macro_vs_harmful']['rho'])}",
                     "component n/rho": f"{r['component']['n']} / {_f(r['component']['V_macro_vs_harmful']['rho'])}",
                     "ppl vs harm (layer)": _f(r['layer']['ppl_vs_harmful']['rho']),
                     "#comp vs harm (all)": _f(r['all']['ncomp_vs_harmful']['rho'])})
    L.append(_md_table(rows, ["model", "all n/rho", "layer n/rho", "component n/rho", "ppl vs harm (layer)", "#comp vs harm (all)"]))
    L.append("\nThe last column is the intervention-size confound: whole layers touch many more weights than one component. "
             "Only the layer column is an unselected, fixed-size sample.\n")
    L.append("## 2. Separate outcomes, BBQ disambiguated, sum rule (E1)\n")
    rows = []
    for tag, cfgs in outc.items():
        for cfg, o in cfgs.items():
            s = o["sum"]; td = s["task_damage_bbq_disambig"]; sd = s["stereotype_damage"]; gd = s["group_disparity_bbq"]
            H = gd.get("H_worst_added_error", {})
            rows.append({"model": tag, "config": cfg, "dense acc": 1 - td["n_dense_correct"] / td["n_labelled"] if False else td["n_dense_correct"] / td["n_labelled"],
                         "harmful/dense-correct": td["harmful_per_dense_correct"], "beneficial/dense-wrong": td["beneficial_per_dense_wrong"],
                         "new stereo errs": sd["new_stereotype_errors"], "new other errs": sd["new_other_errors"],
                         "H (elig groups)": H.get("value"), "H group n/clusters": f"{H.get('n')}/{H.get('n_clusters')}" if H else "",
                         "macro +dE": gd.get("macro_positive_added_error"),
                         "#elig/#restricted": f"{len(gd['eligible'])}/{len(gd['restricted_descriptive_only'])}"})
    L.append(_md_table(rows, ["model", "config", "dense acc", "harmful/dense-correct", "beneficial/dense-wrong",
                              "new stereo errs", "new other errs", "H (elig groups)", "H group n/clusters", "macro +dE", "#elig/#restricted"]))
    L.append("\n## 3. Corrected margin checks (E1, sum rule)\n")
    rows = []
    for tag, cfgs in marg.items():
        for cfg, m in cfgs.items():
            rows.append({"model": tag, "config": cfg, "n": m["n"], "argmax certified": m["argmax"]["n_certified"],
                         "argmax changed": m["argmax"]["n_changed_overall"], "ARGMAX VIOLATIONS": m["argmax"]["n_violations"],
                         "gold certified": m["gold"]["n_certified"], "gold lost": m["gold"]["n_lost_overall"],
                         "GOLD VIOLATIONS": m["gold"]["n_violations"], "ties": m["ties_at_top"], "mean eps": m["eps"]["mean"]})
    L.append(_md_table(rows, ["model", "config", "n", "argmax certified", "argmax changed", "ARGMAX VIOLATIONS",
                              "gold certified", "gold lost", "GOLD VIOLATIONS", "ties", "mean eps"]))
    L.append("\nA violation column of 0 everywhere means the recorded 'margin-lemma violations' in the source run "
             "were artefacts of testing |gold margin| against argmax change, exactly the counter-example in Next_Plan §3.2.\n")
    L.append("## 4. Pair gaps with semantic answer correspondence (E1, sum rule)\n")
    rows = []
    for tag, cfgs in prs.items():
        for cfg, pr in cfgs.items():
            for k, v in pr.items():
                rows.append({"model": tag, "config": cfg, "pair type": k, "n": v["n"], "confirmatory": v["confirmatory_eligible"],
                             "|gap| dense": v["mean_abs_gap_dense"], "|gap| comp": v["mean_abs_gap_comp"], "d|gap|": v["mean_abs_gap_change"]})
    L.append(_md_table(rows, ["model", "config", "pair type", "n", "confirmatory", "|gap| dense", "|gap| comp", "d|gap|"]))
    L.append("\n## 5. Dense competence under the scoring adaptation\n")
    rows = []
    for tag, c in compt.items():
        for norm in ("sum", "mean"):
            s = c[norm]
            rows.append({"model": tag, "rule": norm, "bbq disambig": s["bbq_disambig"]["acc"], "bbq ambig": s["bbq_ambig"]["acc"],
                         "ambig unknown-rate": s["bbq_ambig"]["unknown_rate"], "wb pro": s["winobias_pro"]["acc"], "wb anti": s["winobias_anti"]["acc"],
                         "picks longest": s.get("bbq_length_confound", {}).get("pick_longest_rate")})
        rows.append({"model": tag, "rule": "verdict", "bbq disambig": "; ".join(c["verdict"])[:200]})
    L.append(_md_table(rows, ["model", "rule", "bbq disambig", "bbq ambig", "ambig unknown-rate", "wb pro", "wb anti", "picks longest"]))
    L.append("\n## 6. E7 comparators: PAIRED differences vs uniform GPTQ (support-preserving bootstrap)\n")
    rows = []
    for tag, m in mp.items():
        for c, v in m.get("vs_gptq_uniform", {}).items():
            a = v["accuracy_diff"]; H = v.get("H_diff") or {}
            rows.append({"model": tag, "method": c, "dAcc": a["estimate"], "dAcc CI": f"[{a['ci_low']:+.4f}, {a['ci_high']:+.4f}]",
                         "verdict": a["interpretation"], "dH": H.get("estimate"), "dH CI": f"[{H.get('ci_low',0):+.3f}, {H.get('ci_high',0):+.3f}]" if H else "",
                         "draws kept": f"{a['n_boot_valid']}/{a['n_boot']}"})
    L.append(_md_table(rows, ["model", "method", "dAcc", "dAcc CI", "verdict", "dH", "dH CI", "draws kept"]))
    L.append(f"\nEligible groups (n>=20 AND >=3 independent templates): "
             + "; ".join(f"{t}: {len(m.get('eligible_groups',{}))} eligible, {len(m.get('excluded_groups',{}))} restricted" for t, m in mp.items()) + "\n")
    L.append("\n## 7. Interval coverage on this design (simulation)\n")
    L.append(f"Nominal {cov['nominal']:.2f}, empirical {cov['empirical_coverage']:.3f} over {cov['n_sim']} simulations "
             f"(target group backed by {cov['n_clusters_target']} clusters): **{cov['verdict']}**\n")
    (OUT / "AUDIT.md").write_text("\n".join(L))
    log(f"report: {OUT / 'AUDIT.md'}")


def _f(x):
    return "--" if x is None else f"{x:+.3f}"


def main(quick: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("1/7 fixed-granularity correlations")
    fg = fixed_granularity(); write_json(OUT / "fixed_granularity.json", {"provenance": provenance(), **fg})
    log("2-4/7 outcomes, margins, pairs (E1)")
    outc, marg, prs = outcomes_for("e1")
    write_json(OUT / "outcomes_e1.json", outc); write_json(OUT / "margins_e1.json", marg); write_json(OUT / "pairs_e1.json", prs)
    log("5/7 competence")
    compt = competence_all(); write_json(OUT / "competence.json", compt)
    log("6/7 paired method differences (E7)")
    nb = 300 if quick else 2000
    mp = {t: method_pairs(t, nb) for t in source_tags("e7")}
    write_json(OUT / "method_pairs_e7.json", mp)
    log("7/7 coverage simulation")
    cov = coverage(); write_json(OUT / "coverage.json", cov)
    write_report(fg, outc, marg, prs, compt, mp, cov)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
