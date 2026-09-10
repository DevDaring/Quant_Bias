"""Tables and figures from per-example records and summary JSON.

Figures follow the plan's main-figure list: paired subgroup change, layer-by-
group sensitivity heat-map, predicted-vs-observed flips, restoration controls,
fairness-utility frontier annotated with bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .common import RESULTS_DIR, read_json, write_json, log


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def md_table(rows: Sequence[dict[str, Any]], cols: Sequence[str], fmt: str = "{:.4g}") -> str:
    def cell(v):
        if isinstance(v, float):
            return fmt.format(v)
        return "" if v is None else str(v)
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "".join("| " + " | ".join(cell(r.get(c)) for c in cols) + " |\n" for r in rows)


def subgroup_table(harm: dict[str, Any], ci: dict[str, Any] | None = None) -> str:
    rows = []
    for g in harm.get("groups", []):
        r = {"group": g, "n": harm["n_per_group"][g], "E_dense": harm["E_dense"][g],
             "E_quant": harm["E_quant"][g], "delta_E": harm["delta_E"][g]}
        if ci and g in ci.get("groups", {}):
            c = ci["groups"][g]
            r["ci"] = f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
            r["p_holm"] = c.get("p_holm")
        rows.append(r)
    cols = ["group", "n", "E_dense", "E_quant", "delta_E"] + (["ci", "p_holm"] if ci else [])
    tail = f"\nA = {harm.get('A_disparity_increase'):.4f}, H = {harm.get('H_worst_added_harm'):.4f} ({harm.get('H_group')})\n"
    return md_table(rows, cols) + tail


def plot_subgroup_change(harm: dict[str, Any], ci: dict[str, Any] | None, out: Path, title: str = "") -> Path:
    plt = _plt()
    groups = harm.get("groups", [])
    if not groups:
        return out
    d = [harm["delta_E"][g] for g in groups]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(groups)), 3.5))
    ax.axhline(0, color="k", lw=0.8)
    if ci:
        lo = [harm["delta_E"][g] - ci["groups"][g]["ci_low"] if g in ci["groups"] else 0 for g in groups]
        hi = [ci["groups"][g]["ci_high"] - harm["delta_E"][g] if g in ci["groups"] else 0 for g in groups]
        ax.errorbar(range(len(groups)), d, yerr=[lo, hi], fmt="o", capsize=3)
    else:
        ax.plot(range(len(groups)), d, "o")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔE_g (quantized − dense)")
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_layer_group_heatmap(matrix: dict[str, dict[str, float]], out: Path, title: str = "",
                             value_label: str = "flip rate") -> Path:
    """matrix: component_id -> group -> value."""
    plt = _plt()
    import numpy as np
    comps = list(matrix)
    groups = sorted({g for v in matrix.values() for g in v})
    M = np.array([[matrix[c].get(g, np.nan) for g in groups] for c in comps], dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(groups) + 2), min(30, max(4, 0.18 * len(comps)))))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(comps, fontsize=6 if len(comps) <= 120 else 3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_label)
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_frontier(points: Sequence[dict[str, Any]], out: Path, x: str = "bytes_total", y: str = "objective",
                  label: str = "name", util: str = "acc") -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p in points:
        ax.scatter(p[x] / 2**20, p[y], s=40)
        ax.annotate(f"{p[label]}\n{util}={p['metrics'].get(util, float('nan')):.3f}", (p[x] / 2**20, p[y]), fontsize=7)
    ax.set_xlabel("serialized MiB (accounted)")
    ax.set_ylabel("bias objective J (lower is better)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def plot_predicted_vs_observed(rows: Sequence[dict[str, Any]], out: Path, pred_key: str, obs_key: str = "flip_rate") -> Path:
    plt = _plt()
    import numpy as np
    from scipy.stats import spearmanr
    xs = np.array([r[pred_key] for r in rows], float)
    ys = np.array([r[obs_key] for r in rows], float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    rho = spearmanr(xs[ok], ys[ok]).correlation if ok.sum() > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(xs[ok], ys[ok], s=18)
    ax.set_xlabel(pred_key)
    ax.set_ylabel(obs_key)
    ax.set_title(f"Spearman ρ = {rho:.3f} (n={int(ok.sum())})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def e1_summary_markdown(root: Path = RESULTS_DIR / "e1") -> str:
    """Collect every E1 summary.json under results/bias/e1/<model>/ into one table."""
    rows = []
    for f in sorted(root.glob("*/summary_*.json")):
        s = read_json(f)
        m = s.get("metrics", {})
        rows.append({"model": f.parent.name, "config": s.get("config"), "bytes_MiB": s.get("bytes", {}).get("total_MiB"),
                     "ppl": s.get("ppl", {}).get("ppl"),
                     "bbq_dis_acc": m.get("bbq", {}).get("disambig", {}).get("acc"),
                     "bbq_dis_bias": m.get("bbq", {}).get("disambig", {}).get("bias_score"),
                     "bbq_amb_bias": m.get("bbq", {}).get("ambig", {}).get("bias_score"),
                     "wb_gap": m.get("winobias", {}).get("gap_pro_minus_anti"),
                     "H": s.get("harm", {}).get("bbq", {}).get("H_worst_added_harm"),
                     "A": s.get("harm", {}).get("bbq", {}).get("A_disparity_increase")})
    return md_table(rows, ["model", "config", "bytes_MiB", "ppl", "bbq_dis_acc", "bbq_dis_bias", "bbq_amb_bias", "wb_gap", "H", "A"])


def write_report(experiment_id: str, model_tag: str, sections: dict[str, str]) -> Path:
    out = RESULTS_DIR / experiment_id / model_tag / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {experiment_id} — {model_tag}\n\n" + "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
    out.write_text(body)
    log(f"report written: {out}")
    return out


def comparator_table(results: dict[str, Any]) -> str:
    """E7 headline table: every method at its accounted memory."""
    rows = []
    for k, v in results.items():
        if not isinstance(v, dict) or "name" not in v:
            continue
        harm = (v.get("harm") or {}).get("bbq", {})
        m = v.get("metrics", {})
        bbq = (m.get("bbq") or {}).get("disambig", {})
        rows.append({
            "method": v["name"],
            "source": ("this study" if not v.get("reimplementation")
                       else v.get("paper", "").split(",")[0]),
            "reimpl": "yes" if v.get("reimplementation") else "no",
            "MiB": v.get("bytes_MiB"),
            "ppl": (v.get("ppl") or {}).get("ppl"),
            "bbq_acc": bbq.get("acc"),
            "bbq_bias": bbq.get("bias_score"),
            "H": harm.get("H_worst_added_harm"),
            "A": harm.get("A_disparity_increase"),
            "gap_change": (v.get("pair_gap") or {}).get("mean_abs_gap_change"),
        })
    note = ("\n\nEvery row marked reimpl=yes is this project's re-implementation from the "
            "published description, not the original authors' code; see each method's "
            "`assumptions` field in e7_results.json.\n")
    return md_table(rows, ["method", "source", "reimpl", "MiB", "ppl", "bbq_acc",
                           "bbq_bias", "H", "A", "gap_change"]) + note
