"""F5: strengthened residual-direction controls (plan section 6).

Eight relative depths (the completed B1 mapping), three sources (RTN4, RTN8, GPTQ4),
64 fixed prompts balanced pro/anti from the mechanism split, and per cell: the actual
compression residual, its sign reversal and ten norm-matched random directions injected
at the block's output boundary into the otherwise-dense model. Every measurement is
stored per prompt so the analysis can bootstrap over occupation-pair clusters.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import synthbias as S
from .common import MODELS, SEED, Manifest, log, read_json, read_jsonl, results_root, seed_from, write_json
from .directional import _batch, _prompt_ids, capture_batched

SOURCES = ("rtn4", "rtn8", "gptq4")


def relative_depths(n_layers: int, k: int = 8) -> list[int]:
    from mixed_study.matched_residuals import prespecified_layers
    return prespecified_layers(n_layers, k)


def select_prompts(rows: Sequence[S.Row], dense_recs: Sequence[dict], n: int = 64, seed: int = SEED) -> list[str]:
    """n/2 pro + n/2 anti dense-correct mechanism rows, one row per cluster where possible."""
    correct = {r["uid"] for r in dense_recs if r["correct"]}
    rng = random.Random(seed)
    out = []
    for stratum in ("pro", "anti"):
        cand = sorted([r for r in rows if r.split == "mechanism" and r.type == "type2" and r.stereo == stratum and r.uid in correct],
                      key=lambda r: r.uid)
        rng.shuffle(cand)
        seen, pick = set(), []
        for r in cand:
            if r.cluster not in seen:
                seen.add(r.cluster)
                pick.append(r.uid)
            if len(pick) == n // 2:
                break
        for r in cand:
            if len(pick) == n // 2:
                break
            if r.uid not in pick:
                pick.append(r.uid)
        out.extend(pick)
    return out


@torch.no_grad()
def norm_matched_random(residual: torch.Tensor, seed: int) -> torch.Tensor:
    """Random direction with the same per-token norm as ``residual`` (prompt positions)."""
    g = torch.Generator(device="cpu").manual_seed(seed % (2**63 - 1))
    z = torch.randn(residual.shape, generator=g, dtype=torch.float32).to(residual.device)
    zn = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rn = residual.float().norm(dim=-1, keepdim=True)
    return z / zn * rn


@torch.no_grad()
def propagate_one(adapter, li: int, batch: dict[str, torch.Tensor], lens: list[int], delta: torch.Tensor,
                  dense_states: list[torch.Tensor], dense_logits: torch.Tensor) -> list[dict[str, float]]:
    """Inject ``delta`` (B, L, H) at block li's output; per-prompt local/final errors."""
    from mixed_study.residuals import Boundary, inject, _forward_capture
    with inject(adapter, Boundary(li), delta):
        states, logits = _forward_capture(adapter, batch)
    out = []
    for i, n in enumerate(lens):
        d_loc = delta[i, :n].float()
        base_loc = dense_states[li + 1][i, :n].float()
        d_fin = states[-1][i, :n].float() - dense_states[-1][i, :n].float()
        base_fin = dense_states[-1][i, :n].float()
        la = float(d_loc.norm()); lr = float(la / (base_loc.norm() + 1e-8))
        fa = float(d_fin.norm()); fr = float(fa / (base_fin.norm() + 1e-8))
        linf = float((logits[i, :n].float() - dense_logits[i, :n].float()).abs().max())
        out.append({"local_abs": la, "local_rel": lr, "final_abs": fa, "final_rel": fr,
                    "amplification": float(fa / (la + 1e-12)), "logit_linf": linf, "_final_drift": d_fin})
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def run(adapter, key: str, rows: Sequence[S.Row], smoke: bool, n_prompts: int = 64, n_random: int = 10,
        layers: Sequence[int] | None = None, sources: Sequence[str] = SOURCES, batch_size: int = 16,
        calib_n_seq: int | None = None) -> dict[str, Any]:
    from .confirmation import calibration, quantizer_for, QUANT
    from mixed_study.matched_residuals import apply_source
    root = results_root(smoke) / "residual_controls" / key
    root.mkdir(parents=True, exist_ok=True)
    done = root / "residual_COMPLETE.json"
    if done.exists():
        log(f"{key}: residual controls already complete")
        return read_json(done)
    conf = results_root(smoke) / "confirmation" / key
    dense = read_json(conf / "dense_COMPLETE.json")
    template = dense["template"]
    mech_recs = read_jsonl(conf / "mechanism_dense.jsonl")
    by = {r.uid: r for r in rows}
    uids = select_prompts(rows, mech_recs, n_prompts)
    sel = [by[u] for u in uids]
    examples = S.build_examples(sel, template, key, adapter.tokenizer, chat=MODELS[key]["chat"])
    layers = list(layers) if layers is not None else relative_depths(adapter.n_layers)
    man = Manifest(done, model=key, stage="F5_residual_controls", smoke=smoke, template=template, layers=layers,
                   sources=list(sources), n_prompts=len(examples), n_random=n_random,
                   depth_mapping={f"{i}/7": li for i, li in enumerate(layers)})
    write_json(root / "prompts.json", {"uids": uids, "layers": layers, "sources": list(sources), "seed": SEED})
    rev = read_json(root.parents[2] / "protocol" / "LOCKED_PROTOCOL.json")["models"][key]["revision"] \
        if (root.parents[2] / "protocol" / "LOCKED_PROTOCOL.json").exists() else "main"
    q = quantizer_for(adapter)
    cal = None
    if "gptq4" in sources:
        c = QUANT["calib"]
        cal = calibration(adapter, calib_n_seq or c["n_seq"], c["max_tokens"]).batches(
            adapter.tokenizer.pad_token_id, str(adapter.device), batch_size=1)
    prompts = [_prompt_ids(adapter, e.prompt) for e in examples]
    t0 = time.time()
    for li in layers:
        for src in sources:
            cell_p = root / f"cell_L{li:02d}_{src}.json"
            if cell_p.exists():
                continue
            comp_ids = [c.id for c in adapter.components_of(li)]
            per_prompt: list[dict[str, Any]] = [{"uid": e.uid, "cluster": e.cluster_id, "stereo": e.group} for e in examples]
            for b0 in range(0, len(examples), batch_size):
                ex_b = examples[b0:b0 + batch_size]
                lens = [len(p) for p in prompts[b0:b0 + batch_size]]
                batch = _batch(adapter, prompts[b0:b0 + batch_size])
                from mixed_study.residuals import _forward_capture
                dense_states, dense_logits = _forward_capture(adapter, batch)
                apply_source(adapter, q, src, comp_ids, cal)
                try:
                    q_states = capture_batched(adapter, batch)
                finally:
                    q.restore()
                assert all(q.is_restored(c) for c in comp_ids), "block not restored exactly"
                resid = (q_states[li + 1] - dense_states[li + 1]).float()
                # zero the padding positions so injected deltas never touch them
                for i, n in enumerate(lens):
                    resid[i, n:] = 0
                arms: dict[str, torch.Tensor] = {"actual": resid, "sign_reversed": -resid}
                for k in range(n_random):
                    z = torch.zeros_like(resid)
                    for i, e in enumerate(ex_b):
                        n = lens[i]
                        z[i, :n] = norm_matched_random(resid[i, :n], seed_from(rev, e.uid, li, src, k))
                    arms[f"random_{k}"] = z
                actual_drift = None
                for arm, delta in arms.items():
                    res = propagate_one(adapter, li, batch, lens, delta, dense_states, dense_logits)
                    if arm == "actual":
                        actual_drift = [r.pop("_final_drift") for r in res]
                    for i, r in enumerate(res):
                        drift = r.pop("_final_drift", None)
                        if drift is not None and actual_drift is not None:
                            r["cos_final_vs_actual"] = _cos(drift, actual_drift[i])
                        per_prompt[b0 + i][arm] = r
            # per-prompt ratios and percentile of the actual residual among the random controls
            for p in per_prompt:
                rnd = [p[f"random_{k}"] for k in range(n_random)]
                for m in ("amplification", "logit_linf", "final_rel"):
                    mean_r = float(np.mean([x[m] for x in rnd]))
                    p[f"ratio_{m}"] = p["actual"][m] / mean_r if mean_r > 0 else None
                    p[f"percentile_{m}"] = float(np.mean([x[m] <= p["actual"][m] for x in rnd]))
            cell = {"layer": li, "source": src, "components": comp_ids, "n_random": n_random, "per_prompt": per_prompt,
                    "summary": {m: {"actual_mean": float(np.mean([p["actual"][m] for p in per_prompt])),
                                    "random_mean": float(np.mean([p[f"random_{k}"][m] for p in per_prompt for k in range(n_random)])),
                                    "reversed_mean": float(np.mean([p["sign_reversed"][m] for p in per_prompt])),
                                    "ratio_mean_of_prompt_ratios": float(np.mean([p[f"ratio_{m}"] for p in per_prompt if p[f"ratio_{m}"] is not None])),
                                    "percentile_mean": float(np.mean([p[f"percentile_{m}"] for p in per_prompt]))}
                                for m in ("amplification", "logit_linf", "final_rel")}}
            write_json(cell_p, cell)
            s = cell["summary"]
            log(f"  L{li:2d} {src:5s}: amp ratio={s['amplification']['ratio_mean_of_prompt_ratios']:.3f} "
                f"logit ratio={s['logit_linf']['ratio_mean_of_prompt_ratios']:.3f} pct={s['amplification']['percentile_mean']:.2f} ({time.time() - t0:.0f}s)")
    return man.finish(cells=[f"L{li:02d}_{src}" for li in layers for src in sources])
