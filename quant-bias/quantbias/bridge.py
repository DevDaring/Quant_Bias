"""Bridge to the prior compression study in ``Codes/living-inference``.

The earlier work measured, for each layer, two utility-only quantities:

  * ``rho``    per-layer Lyapunov contraction of a *random* embedding
               perturbation, from `colab_unified_eval.py` and friends;
  * ``regret`` the perplexity cost of compressing one layer or component
               alone, from the single-layer and per-group ablations.

Neither was conditioned on a demographic group, and both were produced under
*pruning* or random noise rather than quantization. This module loads them and
lines them up with the group-conditioned quantities this study measures, so the
question "does utility-derived sensitivity already predict who gets hurt?" can
be answered rather than assumed.

A negative answer is the interesting one: it is the evidence for contribution
axes A3 and A4 in Future_Plan.md, and it is why the bias-aware allocator is not
simply the utility-aware allocator that living-inference already produced.

Nothing here writes to the living-inference tree; it is opened read-only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import LIVING_INFERENCE_DIR, PROJECT_DIR, log, read_json

# The prior study lives in its own repository. Its result JSONs are vendored
# (code-free) under provenance/ so a fresh clone can reproduce E6; when the
# sibling checkout is present locally that is preferred, since it is the source.
_VENDORED = PROJECT_DIR / "provenance" / "living_inference_results"
_SIBLING = LIVING_INFERENCE_DIR / "python" / "results"
LI_RESULTS = _SIBLING if _SIBLING.exists() else _VENDORED


def results_root() -> Path:
    """Directory the prior results are being read from, for the manifest."""
    return LI_RESULTS

# Where each model's prior per-layer rho lives. Several files may carry it; the
# first that parses wins, and the file actually used is recorded in the output.
RHO_SOURCES: dict[str, list[tuple[str, str]]] = {
    "gpt2_small":  [("h100/gpt2_small_unified.json", "lyapunov.rho_per_layer"),
                    ("lyapunov_gpt2.json", "rho_per_layer")],
    "gpt2_medium": [("h100/gpt2_medium_unified.json", "lyapunov.rho_per_layer"),
                    ("lyapunov_gpt2_medium.json", "rho_per_layer")],
    "lfm2_2.6b":   [("h100/lfm2_2.6b_unified.json", "lyapunov.rho_per_layer"),
                    ("h100/unified_summary.json", "lfm2_2.6b.lyapunov.rho_per_layer")],
    "mistral_7b":  [("h100/mistral_7b_unified.json", "lyapunov.rho_per_layer"),
                    ("h100/unified_summary.json", "mistral_7b.lyapunov.rho_per_layer"),
                    ("mistral_micro/mistral_7b_v0_1_lyapunov.json", "rho_per_layer")],
    "qwen3_8b":    [("h100/qwen3_8b_lyapunov.json", "rho_per_layer"),
                    ("h100/unified_summary.json", "qwen3_8b.lyapunov.rho_per_layer")],
    "qwen3_5_2b":  [("qwen35/qwen3_5_2b_lyapunov.json", "rho_per_layer")],
    "llama_2_7b":  [("lyapunov_deep_gpt2.json", "rho_per_layer")],   # may be absent
}

# Prior *utility* sensitivity: perplexity regret of compressing one site alone.
REGRET_SOURCES: dict[str, list[tuple[str, str]]] = {
    "mistral_7b": [("mistral_micro/mistral_7b_v0_1_single_layer.json", ""),
                   ("h100/mistral7b_fixed.json", "single_layer_ablation_50pct_wanda")],
    "qwen3_5_2b": [("qwen35/qwen3_5_2b_per_group.json", "")],
}


def _dig(obj: Any, dotted: str) -> Any:
    if not dotted:
        return obj
    for part in dotted.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def load_prior_rho(li_key: str) -> dict[str, Any] | None:
    """Per-layer Lyapunov rho measured by the prior study, if it exists."""
    for rel, path in RHO_SOURCES.get(li_key, []):
        f = LI_RESULTS / rel
        if not f.exists():
            continue
        try:
            v = _dig(read_json(f), path)
        except Exception:
            continue
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return {"rho_per_layer": [float(x) for x in v], "source": rel,
                    "n_layers_measured": len(v)}
    return None


def load_prior_regret(li_key: str) -> dict[str, Any] | None:
    """Per-layer or per-component perplexity regret from the prior ablations."""
    for rel, path in REGRET_SOURCES.get(li_key, []):
        f = LI_RESULTS / rel
        if not f.exists():
            continue
        try:
            v = _dig(read_json(f), path)
        except Exception:
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict) and "regret" in v[0]:
            key = "layer" if "layer" in v[0] else "component"
            return {"by": key, "regret": {str(r[key]): float(r["regret"]) for r in v},
                    "source": rel}
        if isinstance(v, dict) and v and isinstance(next(iter(v.values())), dict):
            inner = next(iter(v.values()))
            if "regret" in inner:
                return {"by": "component", "source": rel,
                        "regret": {k: float(d["regret"]) for k, d in v.items() if "regret" in d}}
    return None


def _layer_of(site_id: str) -> int | None:
    """'L12.mlp.up_proj' or 'L12.all' -> 12."""
    if not site_id.startswith("L"):
        return None
    head = site_id.split(".", 1)[0][1:]
    return int(head) if head.isdigit() else None


def align_with_sites(li_key: str, per_site: dict[str, Any]) -> dict[str, Any]:
    """Attach prior rho/regret to each E2 site and correlate with observed harm.

    ``per_site`` is the ``e2_sites.json`` written by E2. Returns per-site rows
    plus Spearman correlations of the prior utility-only quantities against the
    group-conditioned harm this study measures.
    """
    from scipy.stats import spearmanr
    import numpy as np

    prior_rho = load_prior_rho(li_key)
    prior_regret = load_prior_regret(li_key)
    rows: list[dict[str, Any]] = []
    for sid, v in per_site.items():
        li = _layer_of(sid)
        obs = v.get("observed_final", {})
        flips = obs.get("full_score_flips", {}) or {}
        ftr = obs.get("first_token_flip_rate", {}) or {}
        feats = v.get("features_selection", {}) or {}
        groups = list(feats)
        row: dict[str, Any] = {
            "site": sid, "layer": li, "granularity": v.get("granularity"),
            # what this study measures
            "harmful_flip_rate": flips.get("harmful_flip_rate"),
            "worst_group_flip": max(ftr.values()) if ftr else None,
            "flip_spread": (max(ftr.values()) - min(ftr.values())) if len(ftr) > 1 else None,
            "V_final_mean": float(np.mean([feats[g]["V_final"] for g in groups])) if groups else None,
            "ppl_utility": v.get("ppl_utility"),
            # what the prior study measured
            "prior_rho": None, "prior_regret": None,
        }
        if prior_rho and li is not None:
            r = prior_rho["rho_per_layer"]
            # rho[t] is the ratio between layer t and t+1, so layer L maps to index L-1
            idx = li - 1
            if 0 <= idx < len(r):
                row["prior_rho"] = r[idx]
        if prior_regret:
            reg = prior_regret["regret"]
            if prior_regret["by"] == "layer" and li is not None:
                row["prior_regret"] = reg.get(str(li))
            else:
                comp = sid.split(".", 1)[1] if "." in sid else sid
                row["prior_regret"] = next((x for k, x in reg.items() if k.endswith(comp)), None)
        rows.append(row)

    def corr(xk: str, yk: str) -> dict[str, Any]:
        xy = [(r[xk], r[yk]) for r in rows
              if r.get(xk) is not None and r.get(yk) is not None]
        if len(xy) < 4:
            return {"n": len(xy), "spearman": None, "p": None}
        x, y = np.array([a for a, _ in xy], float), np.array([b for _, b in xy], float)
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return {"n": len(xy), "spearman": None, "p": None, "note": "constant input"}
        res = spearmanr(x, y)
        return {"n": len(xy), "spearman": float(res.correlation), "p": float(res.pvalue)}

    out = {
        "li_key": li_key,
        "prior_rho": prior_rho, "prior_regret": prior_regret,
        "rows": rows,
        "correlations": {
            # Does the prior random-perturbation contraction predict bias harm?
            "prior_rho_vs_harmful_flips": corr("prior_rho", "harmful_flip_rate"),
            "prior_rho_vs_worst_group_flip": corr("prior_rho", "worst_group_flip"),
            "prior_rho_vs_flip_spread": corr("prior_rho", "flip_spread"),
            # Does the prior pruning-utility regret predict bias harm?
            "prior_regret_vs_harmful_flips": corr("prior_regret", "harmful_flip_rate"),
            "prior_regret_vs_flip_spread": corr("prior_regret", "flip_spread"),
            # Sanity: prior rho should track this study's own utility measure.
            "prior_rho_vs_ppl_utility": corr("prior_rho", "ppl_utility"),
            "prior_regret_vs_ppl_utility": corr("prior_regret", "ppl_utility"),
            # This study's own group-conditioned energy vs its own harm.
            "V_final_vs_harmful_flips": corr("V_final_mean", "harmful_flip_rate"),
        },
    }
    return out


def interpret(bridge: dict[str, Any], alpha: float = 0.05,
              min_util_rho: float = 0.30) -> str:
    """Plain statement of what the bridge shows, for the report."""
    c = bridge["correlations"]
    lines = []
    util = c["prior_rho_vs_ppl_utility"]
    bias = c["prior_rho_vs_harmful_flips"]
    own = c["V_final_vs_harmful_flips"]
    if util.get("spearman") is None:
        lines.append("Prior rho could not be aligned with this study's sites; no bridge claim is made.")
        return " ".join(lines)

    # The alignment sanity check has to actually pass before anything is read
    # into the bias result. If the prior study's rho does not even track this
    # study's own utility measure at the same layers, then a null relation to
    # harm is uninterpretable: it cannot be told apart from the two pipelines
    # having been mis-aligned or measuring different things in the first place.
    # An earlier version asserted "confirming the two pipelines measure the
    # same layers" unconditionally, which stated the opposite of what a near
    # zero correlation shows.
    # A *positive* correlation is what alignment means here: prior rho is a
    # contraction/amplification factor (higher = more sensitive) and ppl_utility
    # is the perplexity when that site is quantized (higher = worse), so aligned
    # pipelines should agree in direction. A strong negative correlation is an
    # anomaly to investigate, not evidence of agreement, so abs() is wrong.
    sane = util["spearman"] >= min_util_rho
    if not sane:
        lines.append(f"ALIGNMENT CHECK FAILED: prior Lyapunov rho does not track this study's own "
                     f"utility measure at the same layers (rho_s={util['spearman']:.2f}, n={util['n']}, "
                     f"threshold {min_util_rho:.2f}). The two pipelines cannot be shown to be measuring "
                     "comparable quantities, so no claim is made about whether prior utility sensitivity "
                     "predicts group-conditioned harm: a null result here is indistinguishable from a "
                     "mis-aligned comparison.")
        if own.get("spearman") is not None:
            lines.append(f"Independently of that, this study's group-conditioned residual energy relates "
                         f"to harm at rho_s={own['spearman']:.2f} (n={own['n']}); this stands on its own "
                         "measurements and does not depend on the prior study.")
        return " ".join(lines)

    lines.append(f"Prior Lyapunov rho tracks this study's utility measure at rho_s={util['spearman']:.2f} "
                 f"(n={util['n']}), so the two pipelines are measuring comparable quantities at the same "
                 "layers and the comparison below is meaningful.")
    if bias.get("spearman") is None:
        lines.append("Its relation to group-conditioned harm could not be estimated.")
    elif bias.get("p") is not None and bias["p"] > alpha:
        lines.append(f"It does not predict group-conditioned harmful flips (rho_s={bias['spearman']:.2f}, "
                     f"p={bias['p']:.2f}), so utility-derived layer sensitivity does not identify where "
                     "quantization changes socially relevant decisions.")
    else:
        lines.append(f"It also predicts harmful flips (rho_s={bias['spearman']:.2f}, p={bias['p']:.3g}); "
                     "the prior utility machinery transfers, and the contribution must rest on the "
                     "allocation result rather than on the diagnostic being new.")
    if own.get("spearman") is not None:
        lines.append(f"This study's group-conditioned residual energy relates to harm at "
                     f"rho_s={own['spearman']:.2f} (n={own['n']}).")
    return " ".join(lines)


def prior_dense_ppl() -> dict[str, Any]:
    """Dense perplexities recorded by the prior study, for the provenance table.

    These come from runs with different token budgets and chunking conventions
    and must not be pooled or compared with this study's re-measured baselines;
    they are reported only to show which checkpoints the prior work covered.
    """
    out: dict[str, Any] = {}
    for rel in ["h100/unified_summary.json", "h100/gpt2_small_unified.json",
                "h100/gpt2_medium_unified.json", "h100/qwen3_8b_results.json",
                "h100/lfm2_2.6b_unified.json", "h100/mistral7b_fixed.json"]:
        f = LI_RESULTS / rel
        if not f.exists():
            continue
        try:
            d = read_json(f)
        except Exception:
            continue
        if "baseline_ppl" in d:
            out[d.get("short", rel)] = {"ppl": d["baseline_ppl"], "source": rel,
                                        "model": d.get("model")}
        else:
            for k, v in d.items():
                if isinstance(v, dict) and "baseline_ppl" in v:
                    out.setdefault(k, {"ppl": v["baseline_ppl"], "source": rel,
                                       "model": v.get("model")})
    return out
