"""Section 6: task-relevant directional prediction.

Hypothesis:  delta_s(j, x)  ~=  grad_{h_j} s(x) . residual_j(x)

where s(x) is a differentiable answer contrast (dense winner minus runner-up
on the FULL candidate score) and residual_j is the compression residual at
boundary j. The gradient is taken w.r.t. the whole (seq, hidden) state the
score depends on, not only the last prompt token, and the first-order estimate
is validated against the actual score change (finite difference) before being
used as a predictor. Its compute cost is recorded.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from . import residuals as R
from quantbias.model_adapters import ModelAdapter


def _candidate_logprob(adapter: ModelAdapter, prompt_ids: list[int], cand_ids: list[int],
                       boundary: R.Boundary, delta: torch.Tensor | None = None,
                       grad_state: bool = False):
    """Sum log-prob of ``cand_ids`` after ``prompt_ids``, optionally with a
    perturbation injected at ``boundary`` and optionally returning the gradient
    of that score w.r.t. the boundary state."""
    ids = torch.tensor([prompt_ids + cand_ids], device=adapter.device)
    start = len(prompt_ids)
    captured = {}

    def grab(m, i, out):
        h = out[0] if isinstance(out, tuple) else out
        if delta is not None:
            h = h + delta.to(h.dtype)
        if grad_state:
            h = h.detach().requires_grad_(True)
        captured["h"] = h
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    mod = adapter.embed if boundary.layer < 0 else adapter.layers[boundary.layer]
    hk = mod.register_forward_hook(grab)
    try:
        with torch.set_grad_enabled(grad_state):
            logits = adapter.model(input_ids=ids).logits.float()
            lp = torch.log_softmax(logits[0], -1)
            tgt = torch.tensor(cand_ids, device=adapter.device)
            pos = torch.arange(start - 1, len(prompt_ids) + len(cand_ids) - 1, device=adapter.device)
            score = lp[pos, tgt].sum()
            g = None
            if grad_state:
                g, = torch.autograd.grad(score, captured["h"])
                g = g.detach().float()
    finally:
        hk.remove()
    return float(score.detach()), g, captured["h"].detach().float()


def contrast_and_gradient(adapter: ModelAdapter, example, boundary: R.Boundary) -> dict[str, Any]:
    """s(x) = score(winner) - score(runner-up) on the dense model, with grad_h s.

    The two candidate sequences may differ in length, so the gradient is taken
    at the boundary state of the PROMPT positions only (shared by both), which is
    exactly the part a residual at that boundary perturbs identically."""
    tok = adapter.tokenizer
    p_ids = tok(example.prompt, add_special_tokens=False)["input_ids"]
    c_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in example.candidates]
    t0 = time.time()
    scores, grads = [], []
    for cid in c_ids:
        s, g, h = _candidate_logprob(adapter, p_ids, cid, boundary, grad_state=True)
        scores.append(s); grads.append(g[0, :len(p_ids)])       # prompt positions only
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    w, r = order[0], order[1]
    return {"uid": example.uid, "winner": w, "runner_up": r, "label": example.label,
            "dense_scores": scores, "margin": scores[w] - scores[r],
            "grad_contrast": grads[w] - grads[r],            # (prompt_len, hidden)
            "prompt_ids": p_ids, "cand_ids": c_ids, "grad_seconds": time.time() - t0}


def predict_and_verify(adapter: ModelAdapter, cg: dict[str, Any], boundary: R.Boundary,
                       residual_prompt: torch.Tensor) -> dict[str, Any]:
    """First-order prediction  <grad s, residual>  vs the actual contrast change
    when the residual is really injected (finite difference on the same inputs)."""
    g = cg["grad_contrast"]
    rp = residual_prompt[:g.shape[0]].to(g.device)
    pred = float((g * rp).sum())
    # actual: inject residual (padded to full sequence length with zeros beyond the prompt)
    def full_delta(n_total):
        d = torch.zeros(1, n_total, rp.shape[-1], device=rp.device)
        d[0, :rp.shape[0]] = rp
        return d
    p_ids, c_ids = cg["prompt_ids"], cg["cand_ids"]
    w, r = cg["winner"], cg["runner_up"]
    sw, _, _ = _candidate_logprob(adapter, p_ids, c_ids[w], boundary, full_delta(len(p_ids) + len(c_ids[w])))
    sr, _, _ = _candidate_logprob(adapter, p_ids, c_ids[r], boundary, full_delta(len(p_ids) + len(c_ids[r])))
    actual = (sw - sr) - cg["margin"]
    return {"uid": cg["uid"], "margin_dense": cg["margin"], "predicted_delta_s": pred,
            "actual_delta_s": actual, "abs_error": abs(pred - actual),
            "predicted_flip": (cg["margin"] + pred) < 0, "actual_flip": (cg["margin"] + actual) < 0,
            "gold_is_winner": cg["label"] == w}


def batch_predict(adapter: ModelAdapter, examples: Sequence, boundary: R.Boundary,
                  residual_fn, max_examples: int = 64) -> dict[str, Any]:
    """``residual_fn(prompt_ids) -> (prompt_len, hidden)`` residual at the boundary.
    Returns per-example rows plus agreement statistics of predicted vs actual."""
    rows = []
    t0 = time.time()
    for e in list(examples)[:max_examples]:
        cg = contrast_and_gradient(adapter, e, boundary)
        rp = residual_fn(cg["prompt_ids"])
        rows.append(predict_and_verify(adapter, cg, boundary, rp))
    import numpy as np
    pred = np.array([r["predicted_delta_s"] for r in rows]); act = np.array([r["actual_delta_s"] for r in rows])
    from scipy.stats import spearmanr, pearsonr
    ok = np.isfinite(pred) & np.isfinite(act)
    return {"n": len(rows), "seconds": time.time() - t0,
            "spearman_pred_vs_actual": float(spearmanr(pred[ok], act[ok]).correlation) if ok.sum() > 3 else None,
            "pearson_pred_vs_actual": float(pearsonr(pred[ok], act[ok])[0]) if ok.sum() > 3 else None,
            "mean_abs_error": float(np.mean([r["abs_error"] for r in rows])),
            "flip_agreement": float(np.mean([r["predicted_flip"] == r["actual_flip"] for r in rows])),
            "rows": rows}
