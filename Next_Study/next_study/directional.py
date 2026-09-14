"""F4: expanded directional ladder (plan section 5).

For every transformer block of a mechanistic model: quantize that block alone with
RTN4, compute the first-order margin change <grad_h m, delta_h> at the block's output
boundary, rescore the actual single-layer model, restore the block and verify exact
equality. Gradients for every boundary come from one backward pass per candidate
(the score is a scalar, so all boundary gradients are available at once); they are
mathematically the per-boundary gradients the completed study computed one at a time.

FP32 everywhere the plan asks: log-softmax and sums are fp32 on the GPU, gradients and
residuals are cast to fp32 before the inner product, inner products run in fp32.
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import synthbias as S
from .common import MODELS, SEED, Manifest, log, read_json, read_jsonl, results_root, write_json


# ----------------------------------------------------------------------------- sample

def select_sample(rows: Sequence[S.Row], dense_recs: Sequence[dict], n_per_stratum: int = 256, seed: int = SEED) -> dict[str, Any]:
    """512 dense-correct type-2 rows from the mechanism split: 256 pro + 256 anti,
    occupation-pair clusters sampled first, rows within clusters next, pronoun and
    answer position balanced greedily."""
    correct = {r["uid"] for r in dense_recs if r["correct"]}
    pos = {r["uid"]: r["gold_position"] for r in dense_recs}
    pool = [r for r in rows if r.split == "mechanism" and r.type == "type2" and r.uid in correct]
    rng = random.Random(seed)
    out: dict[str, list[S.Row]] = {}
    for stratum in ("pro", "anti"):
        cand = [r for r in pool if r.stereo == stratum]
        by_cluster: dict[str, list[S.Row]] = defaultdict(list)
        for r in cand:
            by_cluster[r.cluster].append(r)
        clusters = sorted(by_cluster)
        rng.shuffle(clusters)
        for c in clusters:
            by_cluster[c].sort(key=lambda r: r.uid)
            rng.shuffle(by_cluster[c])
        chosen: list[S.Row] = []
        counts: Counter = Counter()
        target = min(n_per_stratum, len(cand))
        while len(chosen) < target:
            progressed = False
            for c in clusters:
                if not by_cluster[c] or len(chosen) >= target:
                    continue
                # within the cluster, take the row that best balances pronoun family and answer position
                def score(r: S.Row) -> tuple:
                    fam = "f" if r.pronoun.lower() in ("she", "her", "herself") else "m"
                    return (counts[("fam", fam)] + counts[("pos", pos[r.uid])], r.uid)
                by_cluster[c].sort(key=score)
                r = by_cluster[c].pop(0)
                chosen.append(r)
                fam = "f" if r.pronoun.lower() in ("she", "her", "herself") else "m"
                counts[("fam", fam)] += 1
                counts[("pos", pos[r.uid])] += 1
                progressed = True
            if not progressed:
                break
        out[stratum] = chosen
    n = min(len(out["pro"]), len(out["anti"]))
    sel = out["pro"][:n] + out["anti"][:n]
    summary = {"seed": seed, "n_per_stratum_target": n_per_stratum, "n_selected": len(sel),
               "n_pro": n, "n_anti": n, "clusters": len({r.cluster for r in sel}),
               "pronouns": dict(Counter(r.pronoun.lower() for r in sel)),
               "gold_positions": dict(Counter(pos[r.uid] for r in sel)),
               "dense_correct_pool": {"pro": sum(r.stereo == "pro" for r in pool), "anti": sum(r.stereo == "anti" for r in pool)},
               "uids": [r.uid for r in sel]}
    return summary


# ----------------------------------------------------------------------------- encoding

def _encode(adapter, prompt: str, cand: str) -> tuple[list[int], int]:
    from quantbias.evaluate import _encode_pair, _prepend_bos
    ids, start, _ = _encode_pair(adapter.tokenizer, prompt, cand)
    ids, off = _prepend_bos(adapter.tokenizer, ids)
    return ids, start + off


def _prompt_ids(adapter, prompt: str) -> list[int]:
    from quantbias.evaluate import _prepend_bos
    ids = adapter.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids, _ = _prepend_bos(adapter.tokenizer, ids)
    return ids


def _batch(adapter, prompts: list[list[int]]):
    pad = adapter.tokenizer.pad_token_id if adapter.tokenizer.pad_token_id is not None else 0
    L = max(len(x) for x in prompts)
    ids = torch.full((len(prompts), L), pad, dtype=torch.long)
    mask = torch.zeros((len(prompts), L), dtype=torch.long)
    for r, x in enumerate(prompts):
        ids[r, :len(x)] = torch.tensor(x)
        mask[r, :len(x)] = 1
    return {"input_ids": ids, "attention_mask": mask}


# ----------------------------------------------------------------------------- gradients

def all_boundary_gradients(adapter, ids: list[int], start: int) -> tuple[float, list[torch.Tensor]]:
    """Score of the continuation ids[start:] and its gradient w.r.t. every block
    output at the prompt positions [0, start). Returned as fp32 CPU tensors."""
    model = adapter.model
    captured: list[torch.Tensor | None] = [None] * adapter.n_layers
    handles = []
    for i, blk in enumerate(adapter.layers):
        def mk(i):
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[i] = h
            return hook
        handles.append(blk.register_forward_hook(mk(i)))
    x = torch.tensor([ids], device=adapter.device)
    try:
        with torch.enable_grad():
            logits = model(input_ids=x).logits.float()
            lp = torch.log_softmax(logits[0], -1)
            tgt = torch.tensor(ids[start:], device=adapter.device)
            pos = torch.arange(start - 1, len(ids) - 1, device=adapter.device)
            score = lp[pos, tgt].sum()
            assert all(h is not None for h in captured), "a block did not fire"
            # gradients w.r.t. the boundary states only: no parameter gradients are
            # materialised (15 GB for a 7B model), and one call serves every layer
            gs = torch.autograd.grad(score, captured)
    finally:
        for h in handles:
            h.remove()
    grads = [g[0, :start].detach().float().cpu() for g in gs]
    return float(score.detach()), grads


def gradient_phase(adapter, examples: Sequence, log_every: int = 64) -> dict[str, Any]:
    """Dense scores, winner/runner-up, gold-minus-other margin and contrast gradients
    (winner minus runner-up) at every boundary for every example."""
    out: dict[str, Any] = {}
    t0 = time.time()
    for n, e in enumerate(examples):
        scores, grads = [], []
        for c in e.candidates:
            ids, start = _encode(adapter, e.prompt, c)
            s, g = all_boundary_gradients(adapter, ids, start)
            scores.append(s)
            grads.append(g)
        w = 0 if scores[0] >= scores[1] else 1
        r = 1 - w
        contrast = [gw - gr for gw, gr in zip(grads[w], grads[r])]
        out[e.uid] = {"dense_scores": scores, "winner": w, "runner_up": r, "margin_wr": scores[w] - scores[r],
                      "margin_gold": scores[e.label] - scores[1 - e.label], "grad_contrast": contrast,
                      "prompt_len": len(_prompt_ids(adapter, e.prompt))}
        if log_every and (n + 1) % log_every == 0:
            log(f"    gradients {n + 1}/{len(examples)} ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------- per-layer

@torch.no_grad()
def capture_batched(adapter, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    from mixed_study.residuals import capture_states
    return capture_states(adapter, batch)


def layer_step(adapter, quantizer, examples: Sequence, grads: dict[str, Any], li: int, bits: int,
               batch_size: int) -> dict[str, Any]:
    """Quantize block ``li`` alone, measure residual and final drift at prompt positions,
    rescore, restore and verify."""
    from quantbias.evaluate import score_candidates
    from quantbias.quantization import PrecisionMap
    comp_ids = [c.id for c in adapter.components_of(li)]
    pm = PrecisionMap.single_site(adapter, comp_ids, bits)
    prompts = [_prompt_ids(adapter, e.prompt) for e in examples]
    t0 = time.time()
    resid_energy, final_energy, preds = {}, {}, {}
    dense_final_norm = {}
    # dense states per batch, then quantized states per batch; boundary index li+1
    for b0 in range(0, len(examples), batch_size):
        ex_b = examples[b0:b0 + batch_size]
        batch = _batch(adapter, prompts[b0:b0 + batch_size])
        dense_states = capture_batched(adapter, batch)
        quantizer.apply_rtn(pm, record_stats=False)
        try:
            q_states = capture_batched(adapter, batch)
        finally:
            quantizer.restore()
        for i, e in enumerate(ex_b):
            n = len(prompts[b0 + i])
            d_b = dense_states[li + 1][i, :n].float()
            r_b = q_states[li + 1][i, :n].float() - d_b
            d_f = dense_states[-1][i, :n].float()
            r_f = q_states[-1][i, :n].float() - d_f
            g = grads[e.uid]["grad_contrast"][li].to(r_b.device)
            m = min(g.shape[0], r_b.shape[0])
            preds[e.uid] = float((g[:m] * r_b[:m]).sum())
            resid_energy[e.uid] = float(r_b.pow(2).sum() / (d_b.pow(2).sum() + 1e-8))
            final_energy[e.uid] = float(r_f.pow(2).sum() / (d_f.pow(2).sum() + 1e-8))
    inject_seconds = time.time() - t0
    # actual rescoring under the single-layer quantized model
    t1 = time.time()
    quantizer.apply_rtn(pm, record_stats=True)
    try:
        q_scored = {s.uid: s for s in score_candidates(adapter, examples, batch_size=batch_size, progress_every=0)}
    finally:
        quantizer.restore()
    restored_exact = all(quantizer.is_restored(c) for c in comp_ids)
    score_seconds = time.time() - t1
    rows = []
    for e in examples:
        g = grads[e.uid]
        qs = q_scored[e.uid]
        w, r = g["winner"], g["runner_up"]
        actual = (qs.logprob_sum[w] - qs.logprob_sum[r]) - g["margin_wr"]
        pred = preds[e.uid]
        anti = e.group == "anti"
        other = 1 - e.label
        rows.append({"uid": e.uid, "cluster": e.cluster_id, "stereo": e.group, "pronoun": e.group_fields["pronoun"],
                     "gold_position": e.label, "margin_dense": g["margin_wr"], "margin_gold_dense": g["margin_gold"],
                     "pred_delta": pred, "actual_delta": actual,
                     "pred_flip": bool(g["margin_wr"] + pred < 0), "actual_flip": bool(qs.pred != w),
                     "pred_stereo": bool(anti and g["margin_wr"] + pred < 0 and w == e.label),
                     "actual_stereo": bool(anti and qs.pred == other),
                     "q_pred": qs.pred, "q_correct": bool(qs.pred == e.label),
                     "resid_energy": resid_energy[e.uid], "final_energy": final_energy[e.uid]})
    P = np.array([x["pred_delta"] for x in rows]); A = np.array([x["actual_delta"] for x in rows])
    nz = (P != 0) & (A != 0)
    from scipy.stats import pearsonr, spearmanr
    anti_rows = [x for x in rows if x["stereo"] == "anti"]
    summary = {"layer": li, "n": len(rows), "n_anti": len(anti_rows), "bits": bits, "components": comp_ids,
               "pearson": float(pearsonr(P, A)[0]) if P.std() > 0 and A.std() > 0 else None,
               "spearman": float(spearmanr(P, A).correlation) if len(P) > 3 else None,
               "sign_agreement_nonzero": float(np.mean(np.sign(P[nz]) == np.sign(A[nz]))) if nz.any() else None,
               "n_nonzero": int(nz.sum()),
               "pred_flip_rate": float(np.mean([x["pred_flip"] for x in rows])),
               "obs_flip_rate": float(np.mean([x["actual_flip"] for x in rows])),
               "pred_stereo_rate": float(np.mean([x["pred_stereo"] for x in anti_rows])) if anti_rows else None,
               "obs_stereo_rate": float(np.mean([x["actual_stereo"] for x in anti_rows])) if anti_rows else None,
               "obs_stereo_events": int(sum(x["actual_stereo"] for x in anti_rows)),
               "dense_margin_quantiles": {q: float(np.percentile([x["margin_dense"] for x in rows], q)) for q in (5, 25, 50, 75, 95)},
               "mean_resid_energy": float(np.mean([x["resid_energy"] for x in rows])),
               "mean_final_energy": float(np.mean([x["final_energy"] for x in rows])),
               "restored_exact": restored_exact, "inject_seconds": round(inject_seconds, 1), "score_seconds": round(score_seconds, 1)}
    return {"summary": summary, "rows": rows}


def run(adapter, key: str, rows: Sequence[S.Row], smoke: bool, layers: Sequence[int] | None = None,
        n_per_stratum: int = 256, batch_size: int = 16, bits: int = 4) -> dict[str, Any]:
    from .confirmation import quantizer_for
    root = results_root(smoke) / "directional" / key
    root.mkdir(parents=True, exist_ok=True)
    done = root / "directional_COMPLETE.json"
    if done.exists():
        log(f"{key}: directional already complete")
        return read_json(done)
    conf = results_root(smoke) / "confirmation" / key
    dense = read_json(conf / "dense_COMPLETE.json")
    template = dense["template"]
    mech_recs = read_jsonl(conf / "mechanism_dense.jsonl")
    sample_p = root / "sample.json"
    if sample_p.exists():
        sample = read_json(sample_p)
    else:
        sample = select_sample(rows, mech_recs, n_per_stratum)
        write_json(sample_p, sample)
    by = {r.uid: r for r in rows}
    sel_rows = [by[u] for u in sample["uids"]]
    examples = S.build_examples(sel_rows, template, key, adapter.tokenizer, chat=MODELS[key]["chat"])
    layers = list(layers) if layers is not None else list(range(adapter.n_layers))
    man = Manifest(done, model=key, stage="F4_directional", smoke=smoke, template=template, n_examples=len(examples), layers=layers)
    index_p = root / "index.json"
    index = read_json(index_p) if index_p.exists() else {"model": key, "layers_done": [], "n_examples": len(examples)}
    todo = [li for li in layers if li not in index["layers_done"]]
    log(f"{key}: directional ladder on {len(examples)} rows, {len(todo)}/{len(layers)} layers to do")
    if not todo:
        return man.finish(layers_done=index["layers_done"])
    t0 = time.time()
    grads = gradient_phase(adapter, examples)
    grad_seconds = time.time() - t0
    log(f"  gradients for {len(examples)} examples x {adapter.n_layers} boundaries in {grad_seconds:.0f}s")
    q = quantizer_for(adapter)
    for li in todo:
        res = layer_step(adapter, q, examples, grads, li, bits, batch_size)
        res["summary"]["grad_seconds_total"] = round(grad_seconds, 1)
        res["provenance"] = {"model": key, "template": template, "bits": bits}
        write_json(root / f"layer_{li:02d}.json", res)
        index["layers_done"] = sorted(set(index["layers_done"]) | {li})
        write_json(index_p, index)
        s = res["summary"]
        log(f"  L{li:2d}: r={s['pearson']!s:>6.6} pred_flip={s['pred_flip_rate']:.3f} obs_flip={s['obs_flip_rate']:.3f} "
            f"stereo pred={s['pred_stereo_rate']!s:>6.6} obs={s['obs_stereo_rate']!s:>6.6} events={s['obs_stereo_events']} "
            f"restored={s['restored_exact']} ({time.time() - t0:.0f}s)")
        if not s["restored_exact"]:
            raise RuntimeError(f"block {li} was not restored exactly; aborting to protect the dense model")
    return man.finish(layers_done=index["layers_done"], grad_seconds=round(grad_seconds, 1))
