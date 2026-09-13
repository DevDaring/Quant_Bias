"""E8: the score-gradient predictor, evaluated at fixed granularity on held-out
examples, against the energy predictors on the SAME sites (Next_Plan §6).

For each whole-layer site L and each held-out example x:

    r_L(x)      actual RTN-4 residual at block L's output, prompt positions
    g_L(x)      gradient of the dense answer contrast s(x) = score(winner) -
                score(runner-up) w.r.t. that same state
    pred(x)     <g_L(x), r_L(x)>            first-order predicted change in s
    actual(x)   s_quantized(x) - s_dense(x) measured by rescoring with L quantized

Site-level predictor: fraction of examples whose predicted margin goes
negative. Outcome: fraction whose answer actually flips / flips harmfully.
The predictor has no fitted parameters, so there is nothing to leave out; it
is compared directly with V_final and logit-linf computed on the same runs.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from . import directional as D, residuals as R
from .common import RESULTS, log, write_json, provenance
from quantbias.model_adapters import ModelAdapter
from quantbias.quantization import PrecisionMap, Quantizer


@torch.no_grad()
def _batch(adapter, examples, max_len=512):
    tok = adapter.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    enc = [tok(e.prompt, add_special_tokens=False)["input_ids"][-max_len:] for e in examples]
    L = max(len(x) for x in enc)
    ids = torch.full((len(enc), L), pad, dtype=torch.long); mask = torch.zeros((len(enc), L), dtype=torch.long)
    for r, x in enumerate(enc):
        ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
    return {"input_ids": ids, "attention_mask": mask}, [len(x) for x in enc]


def run(adapter: ModelAdapter, quantizer: Quantizer, examples, layers: list[int], bits: int = 4,
        out_dir=None, tag: str = "model", batch_size: int = 8) -> dict[str, Any]:
    from quantbias.evaluate import score_candidates
    ex = list(examples)
    batch, lens = _batch(adapter, ex)
    dense_scores = {s.uid: s for s in score_candidates(adapter, ex, batch_size=batch_size, progress_every=0)}
    # dense gradients are per boundary, so they are computed inside the site loop
    sites: dict[str, Any] = {}
    t0 = time.time()
    for li in layers:
        b = R.Boundary(li)
        comp_ids = [c.id for c in adapter.components_of(li)]
        pm = PrecisionMap.single_site(adapter, comp_ids, bits)
        # residuals at the boundary (batched, right-padded -> prompt positions identical to single runs)
        dense_states = R.capture_states(adapter, batch)
        quantizer.apply_rtn(pm, record_stats=False)
        try:
            q_states = R.capture_states(adapter, batch)
            q_scores = {s.uid: s for s in score_candidates(adapter, ex, batch_size=batch_size, progress_every=0)}
        finally:
            quantizer.restore()
        resid = (q_states[b.state_index] - dense_states[b.state_index])      # (N, L, H)
        rows = []
        tg = 0.0
        for i, e in enumerate(ex):
            cg = D.contrast_and_gradient(adapter, e, b)                        # dense, grad at boundary
            tg += cg["grad_seconds"]
            rp = resid[i, :lens[i]]
            g = cg["grad_contrast"]
            n = min(g.shape[0], rp.shape[0])
            pred = float((g[:n].to(rp.device) * rp[:n]).sum())
            ds, qs = dense_scores[e.uid], q_scores[e.uid]
            w, r_ = cg["winner"], cg["runner_up"]
            actual = (qs.logprob_sum[w] - qs.logprob_sum[r_]) - (ds.logprob_sum[w] - ds.logprob_sum[r_])
            rows.append({"uid": e.uid, "group": e.group, "margin": cg["margin"],
                         "pred_delta_s": pred, "actual_delta_s": actual,
                         "pred_flip": (cg["margin"] + pred) < 0, "actual_flip": qs.pred != ds.pred,
                         "harmful": bool(ds.correct and qs.correct is False),
                         "resid_energy": float(rp[:n].pow(2).sum() / (dense_states[b.state_index][i, :n].pow(2).sum() + 1e-8))})
        P = np.array([r["pred_delta_s"] for r in rows]); A = np.array([r["actual_delta_s"] for r in rows])
        from scipy.stats import spearmanr, pearsonr
        sites[f"L{li}.all"] = {
            "layer": li, "n": len(rows),
            # site-level predictors
            "pred_flip_rate": float(np.mean([r["pred_flip"] for r in rows])),
            "pred_mean_margin_loss": float(np.mean([-r["pred_delta_s"] / max(abs(r["margin"]), 1e-6) for r in rows])),
            "V_final_proxy": float(np.mean([r["resid_energy"] for r in rows])),
            # site-level outcomes (held-out examples)
            "obs_flip_rate": float(np.mean([r["actual_flip"] for r in rows])),
            "obs_harmful_rate": float(np.mean([r["harmful"] for r in rows])),
            # example-level validation of the first-order approximation
            "example_pearson_pred_vs_actual": float(pearsonr(P, A)[0]) if len(P) > 3 and P.std() > 0 and A.std() > 0 else None,
            "example_spearman_pred_vs_actual": float(spearmanr(P, A).correlation) if len(P) > 3 else None,
            "flip_agreement": float(np.mean([r["pred_flip"] == r["actual_flip"] for r in rows])),
            "grad_seconds": tg,
            "rows": rows if len(rows) <= 64 else rows[:64],
        }
        s = sites[f"L{li}.all"]
        log(f"  L{li:2d}: pred_flip={s['pred_flip_rate']:.3f} obs_flip={s['obs_flip_rate']:.3f} harmful={s['obs_harmful_rate']:.3f} "
            f"ex-corr={s['example_pearson_pred_vs_actual']!s:>6.6} agree={s['flip_agreement']:.2f} ({time.time()-t0:.0f}s)")
        if out_dir:
            write_json(out_dir / "directional_sites.json", {"tag": tag, "bits": bits, "sites": sites})
    out = {"tag": tag, "bits": bits, "layers": layers, "n_examples": len(ex), "sites": sites,
           "site_level": site_level(sites), "provenance": provenance()}
    if out_dir:
        write_json(out_dir / "directional_sites.json", out)
    return out


def site_level(sites: dict[str, Any]) -> dict[str, Any]:
    """Across sites: which predictor ranks layers by harm best? No fitting."""
    from scipy.stats import spearmanr
    S = list(sites.values())
    if len(S) < 4:
        return {"n_sites": len(S), "note": "too few sites"}
    y_h = np.array([s["obs_harmful_rate"] for s in S]); y_f = np.array([s["obs_flip_rate"] for s in S])
    out = {"n_sites": len(S)}
    for name in ("pred_flip_rate", "pred_mean_margin_loss", "V_final_proxy"):
        x = np.array([s[name] for s in S])
        def rho(y):
            if np.allclose(x, x[0]) or np.allclose(y, y[0]): return None
            r = spearmanr(x, y); return {"rho": float(r.correlation), "p": float(r.pvalue)}
        out[name] = {"vs_harmful": rho(y_h), "vs_anyflip": rho(y_f)}
    ex_corr = [s["example_pearson_pred_vs_actual"] for s in S if s["example_pearson_pred_vs_actual"] is not None]
    out["example_level_first_order_validity"] = {"mean_pearson": float(np.mean(ex_corr)) if ex_corr else None,
                                                 "n_sites": len(ex_corr)}
    out["total_grad_seconds"] = float(sum(s["grad_seconds"] for s in S))
    return out
