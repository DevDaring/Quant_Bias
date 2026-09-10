"""Group-conditioned error propagation for single-site quantization.

For a component j quantized alone (all other weights dense), record at every
layer l the residual e_l(x) = h_l^q(x) - h_l(x) at an aligned token position,
then compute (plan, Section 7):

    V_l,g  = mean_{x in g} ||e_l(x)||^2 / (||h_l(x)||^2 + eps)
    rho_l,g = V_{l+1,g} / V_{l,g}          (undefined when V_{l,g} < floor)

plus absolute drift, the systematic fraction ||mean_g e_l|| / mean_g ||e_l||,
and the first-token logit margin before/after (the quantity in the margin lemma).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import torch

from .common import log
from .data import Example
from .model_adapters import ModelAdapter
from .quantization import PrecisionMap, Quantizer


@dataclass
class HiddenCapture:
    uids: list[str]
    groups: list[str]
    hidden: torch.Tensor          # [n, n_layers+1, d]  (index 0 = embeddings, last = pre-final-norm block output)
    logits: torch.Tensor          # [n, vocab] at the aligned position
    cand_first_ids: list[list[int]]  # first token id of every candidate, per example


@torch.no_grad()
def capture_hidden(adapter: ModelAdapter, examples: Sequence[Example], batch_size: int = 8,
                   max_len: int = 1024, position: str = "last") -> HiddenCapture:
    """Hidden states at the last prompt token (default) for every example."""
    tok, model, dev = adapter.tokenizer, adapter.model, adapter.device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    enc = [tok(e.prompt, add_special_tokens=False)["input_ids"][-max_len:] for e in examples]
    if tok.bos_token_id is not None and adapter.family != "gpt2":
        enc = [[tok.bos_token_id] + x if not x or x[0] != tok.bos_token_id else x for x in enc]
    cand_first = [[tok(c, add_special_tokens=False)["input_ids"][:1] or [pad] for c in e.candidates] for e in examples]
    cand_first = [[c[0] for c in cf] for cf in cand_first]
    hs_all, lg_all = [], []
    for bi in range(0, len(enc), batch_size):
        chunk = enc[bi:bi + batch_size]
        L = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
        store: list[list[torch.Tensor]] = [[] for _ in adapter.layers]
        with adapter.layer_output_capture(store):
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
        last = torch.tensor([len(x) - 1 for x in chunk], device=dev)
        rows = torch.arange(len(chunk), device=dev)
        emb = out.hidden_states[0][rows, last].float()
        blocks = [store[i][0][rows, last].float() for i in range(len(adapter.layers))]
        hs_all.append(torch.stack([emb] + blocks, dim=1).cpu())
        lg_all.append(out.logits[rows, last].float().cpu())
    return HiddenCapture(uids=[e.uid for e in examples], groups=[e.group for e in examples],
                         hidden=torch.cat(hs_all), logits=torch.cat(lg_all), cand_first_ids=cand_first)


def _first_token_margin(logits: torch.Tensor, cand_ids: list[int], label: int | None) -> tuple[float, int]:
    """m = z_correct - max_other on first candidate tokens (or top1-top2 when no label)."""
    z = logits[cand_ids]
    if label is None or label >= len(cand_ids):
        top = torch.topk(z, 2).values
        return (top[0] - top[1]).item(), int(torch.argmax(z))
    others = torch.cat([z[:label], z[label + 1:]])
    return (z[label] - others.max()).item(), int(torch.argmax(z))


@dataclass
class TraceResult:
    component_id: str
    bits: int
    method: str
    n_examples: int
    groups: list[str]
    V: dict[str, list[float]]              # group -> per-layer relative energy
    rho: dict[str, list[float | None]]     # group -> per-layer ratio (None when undefined)
    abs_drift: dict[str, list[float]]      # group -> per-layer mean ||e_l||
    systematic_fraction: dict[str, list[float]]  # group -> ||mean e|| / mean ||e||
    logit_linf: dict[str, float]           # group -> mean max|Δlogit| (the eps_x of the lemma)
    margin_dense: dict[str, float]
    margin_quant: dict[str, float]
    flip_rate: dict[str, float]
    lemma_violations: int                  # flips where |m| > 2 eps_x (must be 0)
    per_example: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_captures(dense: HiddenCapture, quant: HiddenCapture, labels: Sequence[int | None],
                     eps: float = 1e-8, floor: float = 1e-12, keep_examples: bool = True) -> dict[str, Any]:
    assert dense.uids == quant.uids
    e = quant.hidden - dense.hidden                      # [n, L+1, d]
    h2 = dense.hidden.pow(2).sum(-1) + eps                # [n, L+1]
    rel = e.pow(2).sum(-1) / h2                           # [n, L+1]
    absn = e.norm(dim=-1)                                 # [n, L+1]
    groups = sorted(set(dense.groups))
    V, rho, drift, sysf = {}, {}, {}, {}
    linf, md, mq, flips = {}, {}, {}, {}
    per_ex = []
    dl = (quant.logits - dense.logits).abs().amax(dim=-1)   # eps_x per example
    viol = 0
    ex_m = []
    for i, uid in enumerate(dense.uids):
        m_d, p_d = _first_token_margin(dense.logits[i], dense.cand_first_ids[i], labels[i])
        m_q, p_q = _first_token_margin(quant.logits[i], quant.cand_first_ids[i], labels[i])
        flip = p_d != p_q
        if flip and abs(m_d) > 2 * dl[i].item() + 1e-6:
            viol += 1
        ex_m.append((m_d, m_q, flip))
        if keep_examples:
            per_ex.append({"uid": uid, "group": dense.groups[i], "margin_dense": m_d, "margin_quant": m_q,
                           "flip": flip, "eps_x": dl[i].item(), "final_rel_energy": rel[i, -1].item()})
    for g in groups:
        idx = [i for i, gg in enumerate(dense.groups) if gg == g]
        t = torch.tensor(idx)
        Vg = rel[t].mean(0)
        V[g] = Vg.tolist()
        rho[g] = [None if Vg[l].item() < floor else (Vg[l + 1] / Vg[l]).item() for l in range(len(Vg) - 1)]
        drift[g] = absn[t].mean(0).tolist()
        mean_e = e[t].mean(0).norm(dim=-1)                # [L+1]
        sysf[g] = (mean_e / absn[t].mean(0).clamp(min=eps)).tolist()
        linf[g] = dl[t].mean().item()
        md[g] = sum(ex_m[i][0] for i in idx) / len(idx)
        mq[g] = sum(ex_m[i][1] for i in idx) / len(idx)
        flips[g] = sum(ex_m[i][2] for i in idx) / len(idx)
    return {"groups": groups, "V": V, "rho": rho, "abs_drift": drift, "systematic_fraction": sysf,
            "logit_linf": linf, "margin_dense": md, "margin_quant": mq, "flip_rate": flips,
            "lemma_violations": viol, "per_example": per_ex}


def single_site_trace(adapter: ModelAdapter, quantizer: Quantizer, comp_id: str, bits: int,
                      examples: Sequence[Example], dense_capture: HiddenCapture | None = None,
                      method: str = "rtn", calib_batches=None, batch_size: int = 8,
                      keep_examples: bool = False) -> TraceResult:
    """Quantize one component, pass its residual through the dense network, restore."""
    if dense_capture is None:
        dense_capture = capture_hidden(adapter, examples, batch_size)
    pm = PrecisionMap.single_site(adapter, [comp_id], bits)
    quantizer.apply(pm, method=method, calib_batches=calib_batches, record_stats=False, progress=None) \
        if method.startswith("gptq") else quantizer.apply(pm, method=method, record_stats=False)
    try:
        qc = capture_hidden(adapter, examples, batch_size)
    finally:
        quantizer.restore()
    cmp = compare_captures(dense_capture, qc, [e.label for e in examples], keep_examples=keep_examples)
    return TraceResult(component_id=comp_id, bits=bits, method=method, n_examples=len(examples), **cmp)


def propagation_features(tr: TraceResult, final_layer: int = -1) -> dict[str, dict[str, float]]:
    """Per-group scalar predictors for E2: injected energy, propagated energy,
    amplification, systematic fraction at the output, logit L-inf, dense margin."""
    feats: dict[str, dict[str, float]] = {}
    for g in tr.groups:
        V = tr.V[g]
        inject_layer = next((l for l, v in enumerate(V) if v > 0), 0)
        rhos = [r for r in tr.rho[g][inject_layer:] if r is not None]
        amp = math.exp(sum(math.log(max(r, 1e-12)) for r in rhos) / max(1, len(rhos))) if rhos else float("nan")
        feats[g] = {"V_inject": V[inject_layer], "V_final": V[final_layer],
                    "geo_mean_rho": amp, "max_rho": max(rhos) if rhos else float("nan"),
                    "sys_frac_final": tr.systematic_fraction[g][final_layer],
                    "logit_linf": tr.logit_linf[g], "margin_dense": tr.margin_dense[g],
                    "flip_rate": tr.flip_rate[g], "abs_drift_final": tr.abs_drift[g][final_layer]}
    return feats


def residual_moments(adapter: ModelAdapter, quantizer: Quantizer, pm: PrecisionMap, examples: Sequence[Example],
                     layer: int, method: str = "rtn", calib_batches=None, batch_size: int = 8) -> dict[str, Any]:
    """Group-conditioned mean and covariance trace of e_layer under a full map
    (used in E3 to test whether *systematic* error predicts harm)."""
    d = capture_hidden(adapter, examples, batch_size)
    if method.startswith("gptq"):
        quantizer.apply(pm, method=method, calib_batches=calib_batches, record_stats=False, progress=None)
    else:
        quantizer.apply(pm, method=method, record_stats=False)
    try:
        q = capture_hidden(adapter, examples, batch_size)
    finally:
        quantizer.restore()
    e = (q.hidden - d.hidden)[:, layer]
    out: dict[str, Any] = {"layer": layer, "groups": {}}
    for g in sorted(set(d.groups)):
        idx = torch.tensor([i for i, gg in enumerate(d.groups) if gg == g])
        eg = e[idx]
        mu = eg.mean(0)
        cov_tr = (eg - mu).pow(2).sum(-1).mean().item()
        out["groups"][g] = {"n": len(idx), "mean_norm": mu.norm().item(), "cov_trace": cov_tr,
                            "mean_sq_norm": eg.pow(2).sum(-1).mean().item()}
    return out
