"""Section 5.1: reproduce the legacy `living-inference` random-perturbation
trace numerically, on the same model and inputs, before any transfer claim.

`living-inference/python/cfi_validation.py::measure_lyapunov` (and the same
routine in `colab_unified_eval.py`) does:

    noise  = randn_like(embed_out) * eps * ||embed_out|| / sqrt(numel)   (one draw)
    v[l]   = mean_positions  ||e_l||^2 / ||h_l||^2      over BLOCK OUTPUTS l = 0..L-1
    rho[t] = v[t] / v[t-1]                             for t = 1..L-1   (L-1 values)

So ``rho[i]`` is the ratio between the outputs of block i and block i+1. It is
NOT indexed by "layer" in either of the two natural ways, and E6 in quant-bias
mapped site L -> rho[L-1] without checking. That convention is fixed here as a
named constant and both candidate mappings are exposed so the choice is tested,
not assumed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .common import li_results_root, log, read_json
from .residuals import _forward_capture
from quantbias.model_adapters import ModelAdapter

# rho[i] = v[i+1] / v[i], v indexed by block output. A site at block L therefore
# has an "into" ratio rho[L-1] (from block L-1's output to block L's output) and
# an "out of" ratio rho[L] (from block L's output to block L+1's output).
RHO_INDEX_CONVENTION = "rho[i] = v[i+1]/v[i]; v[i] = relative energy at block i OUTPUT"


@torch.no_grad()
def reproduce_lyapunov(adapter: ModelAdapter, text: str, eps: float = 0.01, seq_len: int = 512,
                       seed: int = 0) -> dict[str, Any]:
    """Exact replication of the legacy routine using explicit-hook capture."""
    tok = adapter.tokenizer
    ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=seq_len).to(adapter.device)
    batch = {"input_ids": ids}
    clean, _ = _forward_capture(adapter, batch)          # index 0 = block-0 input, l+1 = block l output
    # legacy perturbs the token-embedding module output with a single global-norm-scaled draw
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = {"flag": False}

    def perturb(m, i, out):
        if done["flag"]:
            return out
        done["flag"] = True
        of = out.float()
        noise = torch.randn(of.shape, generator=g).to(of.device) * eps * of.norm() / (of.numel() ** 0.5)
        return (of + noise).to(out.dtype)
    h = adapter.embed.register_forward_hook(perturb)
    try:
        pert, _ = _forward_capture(adapter, batch)
    finally:
        h.remove()
    # legacy v is over BLOCK OUTPUTS only (its layer hooks), i.e. our indices 1..L
    v = []
    for l in range(1, len(clean)):
        hc, hp = clean[l], pert[l]
        e = hp - hc
        v.append(float(torch.mean(torch.sum(e ** 2, dim=-1) / torch.sum(hc ** 2, dim=-1))))
    rho = [v[t] / v[t - 1] if v[t - 1] > 1e-15 else 0.0 for t in range(1, len(v))]
    return {"v_trajectory": v, "rho_per_layer": [round(r, 4) for r in rho],
            "convention": RHO_INDEX_CONVENTION, "eps": eps, "seq_len": int(ids.shape[1]),
            "n_blocks": len(v), "n_rho": len(rho), "seed": seed}


def load_saved_rho(li_key: str) -> dict[str, Any] | None:
    from quantbias.bridge import load_prior_rho
    return load_prior_rho(li_key)


def compare_to_saved(adapter: ModelAdapter, li_key: str, text: str, eps: float = 0.01,
                     seq_len: int = 512, seeds=(0, 1, 2)) -> dict[str, Any]:
    """Fresh reproduction vs the saved profile. The saved run used ONE unrecorded
    noise draw, so agreement is judged on the profile shape (Spearman across
    layers) and on whether the saved values fall inside the fresh seed spread,
    not on exact equality."""
    from scipy.stats import spearmanr
    saved = load_saved_rho(li_key)
    fresh = [reproduce_lyapunov(adapter, text, eps, seq_len, s)["rho_per_layer"] for s in seeds]
    out: dict[str, Any] = {"li_key": li_key, "convention": RHO_INDEX_CONVENTION,
                           "fresh_seeds": seeds, "fresh_rho": fresh}
    if saved is None:
        out["verdict"] = "no saved profile for this model"
        return out
    sr = saved["rho_per_layer"]
    F = np.array(fresh)
    n = min(len(sr), F.shape[1])
    out["saved_source"] = saved["source"]
    out["n_saved"] = len(sr); out["n_fresh"] = F.shape[1]
    out["length_matches"] = len(sr) == F.shape[1]
    fm = F[:, :n].mean(0); fs = F[:, :n].std(0)
    out["shape_spearman"] = float(spearmanr(sr[:n], fm).correlation) if n > 3 else None
    out["max_abs_diff_vs_fresh_mean"] = float(np.max(np.abs(np.array(sr[:n]) - fm)))
    inside = np.abs(np.array(sr[:n]) - fm) <= 3 * fs + 1e-3
    out["frac_saved_within_3sd_of_fresh"] = float(inside.mean())
    out["verdict"] = ("REPRODUCED" if out["length_matches"] and (out["shape_spearman"] or 0) > 0.8
                      and out["frac_saved_within_3sd_of_fresh"] > 0.8 else "NOT REPRODUCED -- do not use for transfer")
    return out


def site_rho(rho: list[float], layer: int, mapping: str) -> float | None:
    """Both candidate layer->rho mappings, so a transfer test can try each.
    'into' : rho[layer-1]  (ratio ending at this block's output)
    'out'  : rho[layer]    (ratio starting at this block's output)"""
    i = layer - 1 if mapping == "into" else layer
    return rho[i] if 0 <= i < len(rho) else None
