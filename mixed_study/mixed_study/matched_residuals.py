"""B1: matched residual-source and direction experiment (Next_Plan §5).

For each prespecified site (whole layer, or one component in a fixed panel)
and each compression source (RTN-4/8, GPTQ-4, per-row Wanda), capture the
actual residual at the block boundary, then propagate the actual residual, its
sign-reversal, and k norm-matched random directions through the dense
downstream network on the SAME inputs. Per-example answer-score changes are
recorded so behaviour, not just drift, is compared across sources.

Sites are fixed in advance from the config -- never chosen by an outcome.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import torch

from . import residuals as R
from .common import RESULTS, TAGS, log, write_json, provenance
from quantbias.model_adapters import ModelAdapter
from quantbias.quantization import PrecisionMap, Quantizer
from quantbias.pruning import collect_feature_norms, wanda_per_row


def prespecified_layers(n_layers: int, k: int = 8) -> list[int]:
    """k layers spread across depth including both endpoints."""
    if n_layers <= k:
        return list(range(n_layers))
    return sorted({round(i * (n_layers - 1) / (k - 1)) for i in range(k)})


def component_panel(adapter: ModelAdapter, layers: list[int], kinds=("attn", "mlp"), per_kind: int = 1) -> list[str]:
    """An equal, prespecified attention/MLP panel at each chosen layer."""
    out = []
    for li in layers:
        for kind in kinds:
            cs = adapter.components_of(li, kind)
            out.extend(c.id for c in cs[:per_kind])
    return out


@torch.no_grad()
def apply_source(adapter, quantizer, source: str, comp_ids: list[str], calib_batches) -> dict[str, Any]:
    """Apply one compression source to the given components; returns metadata."""
    if source in ("rtn4", "rtn8"):
        bits = int(source[-1])
        pm = PrecisionMap.single_site(adapter, comp_ids, bits)
        quantizer.apply_rtn(pm, record_stats=False)
        return {"kind": "quant", "bits": bits}
    if source == "gptq4":
        pm = PrecisionMap.single_site(adapter, comp_ids, 4)
        quantizer.apply_gptq(pm, calib_batches, progress=None, record_stats=False)
        return {"kind": "quant", "bits": 4}
    if source.startswith("wanda"):
        sp = float(source[5:]) / 100 if len(source) > 5 else 0.5
        for cid in comp_ids:
            quantizer._stash(adapter.component_by_id[cid])
        fn = collect_feature_norms(adapter, calib_batches)
        rep = wanda_per_row(adapter, fn, sp, comp_ids)
        return {"kind": "prune", "sparsity": rep["sparsity_achieved"]}
    raise ValueError(source)


@torch.no_grad()
def score_answers(adapter: ModelAdapter, examples, norm="sum", bs=8):
    from quantbias.evaluate import score_candidates
    return {s.uid: s for s in score_candidates(adapter, examples, norm=norm, batch_size=bs, progress_every=0)}


def run_b1(adapter: ModelAdapter, quantizer: Quantizer, examples, calib_batches,
           sources=("rtn4", "rtn8", "gptq4", "wanda50"), n_random: int = 3,
           granularity: str = "layer", k_layers: int = 8, batch_size: int = 8,
           norm: str = "sum", out_dir=None, tag: str = "model") -> dict[str, Any]:
    """The core panel: sites x sources x {actual, reversed, random_k}."""
    from quantbias.trace import capture_hidden  # noqa: F401  (kept for parity)
    tok = adapter.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    # one fixed input batch for propagation (per-token comparison needs the same tokens)
    enc = [tok(e.prompt, add_special_tokens=False)["input_ids"][-512:] for e in examples[:batch_size]]
    L = max(len(x) for x in enc)
    ids = torch.full((len(enc), L), pad, dtype=torch.long); mask = torch.zeros((len(enc), L), dtype=torch.long)
    for r, x in enumerate(enc):
        ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
    batch = {"input_ids": ids, "attention_mask": mask}

    layers = prespecified_layers(adapter.n_layers, k_layers)
    if granularity == "layer":
        sites = {f"L{li}.all": ([c.id for c in adapter.components_of(li)], li) for li in layers}
    else:
        sites = {cid: ([cid], adapter.component_by_id[cid].layer) for cid in component_panel(adapter, layers)}
    dense_scores = score_answers(adapter, examples, norm, batch_size)
    results: dict[str, Any] = {"tag": tag, "granularity": granularity, "layers": layers,
                               "sources": list(sources), "n_random": n_random, "cells": {}}
    t0 = time.time()
    for sid, (comp_ids, li) in sites.items():
        boundary = R.Boundary(li)
        for src in sources:
            key = f"{sid}|{src}"
            try:
                meta = apply_source(adapter, quantizer, src, comp_ids, calib_batches)
                comp_states = R.capture_states(adapter, batch)
                comp_scores = score_answers(adapter, examples, norm, batch_size)
            finally:
                quantizer.restore()
            dense_states = R.capture_states(adapter, batch)
            residual = comp_states[boundary.state_index] - dense_states[boundary.state_index]
            srcs = R.make_sources(residual, n_random, seed0=hash(key) % 10_000, mask=mask)
            props = R.run_sources(adapter, boundary, srcs, batch, mask)
            # behavioural: per-example score change under the REAL compression
            flips = sum(1 for u, d in dense_scores.items() if d.pred != comp_scores[u].pred)
            harmful = sum(1 for u, d in dense_scores.items()
                          if d.correct and comp_scores[u].correct is False)
            results["cells"][key] = {
                "site": sid, "layer": li, "source": src, "n_components": len(comp_ids), **meta,
                "propagation": {k: asdict(v) for k, v in props.items()},
                "behaviour": {"n": len(dense_scores), "flips": flips, "harmful": harmful,
                              "flip_rate": flips / max(1, len(dense_scores)),
                              "harmful_rate": harmful / max(1, len(dense_scores))},
            }
            a, rnd = props["actual"], [props[k] for k in props if k.startswith("random_")]
            log(f"  {key}: local_rel={a.local_rel:.2e} amp(actual)={a.amplification:.2f} "
                f"amp(random)={sum(r.amplification for r in rnd)/max(1,len(rnd)):.2f} "
                f"cos(rand,actual)={sum((r.cos_final_vs_actual or 0) for r in rnd)/max(1,len(rnd)):+.2f} "
                f"flips={flips} ({time.time()-t0:.0f}s)")
            if out_dir:
                write_json(out_dir / "b1_cells.json", results)
    results["provenance"] = provenance()
    results["summary"] = summarize(results)
    if out_dir:
        write_json(out_dir / "b1_cells.json", results)
    return results


def summarize(res: dict[str, Any]) -> dict[str, Any]:
    """Does direction matter? Compare actual vs random at equal per-token norm."""
    import numpy as np
    by_src: dict[str, dict[str, list]] = {}
    for cell in res["cells"].values():
        p = cell["propagation"]
        d = by_src.setdefault(cell["source"], {"amp_actual": [], "amp_random": [], "amp_reversed": [],
                                                "cos_random": [], "linf_actual": [], "linf_random": []})
        rnd = [v for k, v in p.items() if k.startswith("random_")]
        d["amp_actual"].append(p["actual"]["amplification"])
        d["amp_reversed"].append(p["sign_reversed"]["amplification"])
        d["amp_random"].append(float(np.mean([r["amplification"] for r in rnd])) if rnd else np.nan)
        d["cos_random"].append(float(np.mean([r["cos_final_vs_actual"] or 0 for r in rnd])) if rnd else np.nan)
        d["linf_actual"].append(p["actual"]["logit_linf"])
        d["linf_random"].append(float(np.mean([r["logit_linf"] for r in rnd])) if rnd else np.nan)
    out = {}
    for src, d in by_src.items():
        aa, ar = np.array(d["amp_actual"]), np.array(d["amp_random"])
        out[src] = {
            "n_sites": len(aa),
            "amp_actual_mean": float(np.nanmean(aa)), "amp_random_mean": float(np.nanmean(ar)),
            "amp_reversed_mean": float(np.nanmean(d["amp_reversed"])),
            "actual_over_random_ratio": float(np.nanmean(aa / np.where(ar == 0, np.nan, ar))),
            "cos_random_vs_actual_final": float(np.nanmean(d["cos_random"])),
            "logit_linf_actual_over_random": float(np.nanmean(np.array(d["linf_actual"]) / np.where(np.array(d["linf_random"]) == 0, np.nan, d["linf_random"]))),
            "direction_matters": bool(abs(np.nanmean(aa / np.where(ar == 0, np.nan, ar)) - 1.0) > 0.25),
        }
    return out
