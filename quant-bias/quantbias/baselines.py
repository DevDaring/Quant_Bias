"""Comparator methods from the closest 2025-2026 literature, at matched budget.

Every method here is a **re-implementation from the published description**, not
the authors' code. Where only an abstract was available the design choice made
is stated in the docstring so a reviewer can see exactly what was assumed. Each
result records ``reimplementation: true`` and the assumptions, and the paper must
say the same. These are run to position this study's allocator, not to claim the
original authors' numbers.

  Fair-GPTQ            Proskurina, Metzler, Velcin. arXiv:2509.15206
  Debias-SparseGPT     Proskurina, Metzler, Gourru, Velcin. EMNLP 2026, arXiv:2609.02496
  Critical Weight
    Protection (CWP)   Al Hakim, Wicaksono, Koto. arXiv:2601.12033
  SparseGPT            Frantar and Alistarh, ICML 2023 (substrate for Debias-SparseGPT)

The shared mechanism in the two Proskurina methods is a second-order term built
from *demographically contrasting* inputs. Both are implemented on one helper,
``contrastive_hessian``, so the comparison isolates where the term is applied
(rounding vs pruning) rather than how it was built.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .common import log
from .data import Example, Pair
from .model_adapters import Component, ModelAdapter
from .quantization import (DENSE_BITS, PrecisionMap, Quantizer, ByteFormat,
                           account_bytes, quantize_rtn, gptq_quantize, _grid)


# ----------------------------------------------------------------------------
# Shared: activations and the contrastive second-order term
# ----------------------------------------------------------------------------

@torch.no_grad()
def _last_token_inputs(adapter: ModelAdapter, examples: Sequence[Example],
                       comp_ids: Sequence[str], batch_size: int = 8,
                       max_len: int = 512) -> dict[str, torch.Tensor]:
    """Input activation at the final prompt token of each example, per component.

    Returns ``{component_id: (n_examples, in_features)}``. The final token is used
    because it is the position the answer is read from, and because it makes
    activations from prompts of different length directly comparable -- the
    alignment requirement stated in Section 7 of the plan.
    """
    tok, dev = adapter.tokenizer, adapter.device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    want = {adapter.component_by_id[c].module: c for c in comp_ids}
    store: dict[str, list[torch.Tensor]] = {c: [] for c in comp_ids}
    order: list[int] = []

    for bi in range(0, len(examples), batch_size):
        chunk = examples[bi:bi + batch_size]
        enc = [tok(e.prompt, add_special_tokens=False)["input_ids"][-max_len:] for e in chunk]
        L = max(len(x) for x in enc)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, x in enumerate(enc):
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
        last = torch.tensor([len(x) - 1 for x in enc], device=dev)
        rows = torch.arange(len(chunk), device=dev)

        def grab(mod, x, _last=last, _rows=rows):
            cid = want.get(mod)
            if cid is None:
                return
            xd = x.detach()
            if xd.dim() == 3:
                store[cid].append(xd[_rows, _last].float().cpu())

        with adapter.module_input_capture(list(want), grab):
            adapter.model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        order.extend(range(bi, bi + len(chunk)))

    return {c: torch.cat(v) for c, v in store.items() if v}


def contrastive_hessian(acts_a: torch.Tensor, acts_b: torch.Tensor,
                        ridge: float = 0.0) -> torch.Tensor:
    """Second-order term from demographically contrasting inputs.

    ``H_c = (2/n) D^T D`` where ``D`` holds the paired differences
    ``x_a - x_b``. Adding ``lambda * H_c`` to the ordinary GPTQ/SparseGPT
    Hessian penalises the component of the weight error that acts *differently*
    on the two groups, which is the stated aim of both Proskurina methods.
    ``H_c`` is positive semi-definite by construction, so the Cholesky step in
    the solvers stays valid.
    """
    n = min(acts_a.shape[0], acts_b.shape[0])
    D = (acts_a[:n] - acts_b[:n]).float()
    H = 2.0 * D.t() @ D / max(n, 1)
    if ridge:
        H = H + ridge * torch.eye(H.shape[0], device=H.device)
    return H


def pair_group_examples(pairs: Sequence[Pair], max_pairs: int = 256
                        ) -> tuple[list[Example], list[Example]]:
    """Split audited counterfactual pairs into two aligned example lists."""
    sel = list(pairs)[:max_pairs]
    return [p.a for p in sel], [p.b for p in sel]


# ----------------------------------------------------------------------------
# SparseGPT (substrate for Debias-SparseGPT)
# ----------------------------------------------------------------------------

def sparsegpt_prune(W: torch.Tensor, H: torch.Tensor, sparsity: float,
                    blocksize: int = 128, percdamp: float = 0.01,
                    prune_n: int = 0, prune_m: int = 0) -> torch.Tensor:
    """SparseGPT (Frantar and Alistarh 2023): mask by w^2 / [H^-1]_ii^2 with
    error compensation inside each block. ``prune_n:prune_m`` gives the
    semi-structured variant (e.g. 2:4) when both are non-zero."""
    W = W.clone().float()
    H = H.clone().float()
    out_f, in_f = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(in_f, device=H.device) * damp
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        d = torch.diag(Hinv1).reshape(1, -1)
        if prune_n == 0:
            score = W1 ** 2 / (d ** 2)
            k = int(round((i2 - i1) * sparsity))
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
            if k > 0:
                idx = torch.topk(score, k, dim=1, largest=False).indices
                mask1.scatter_(1, idx, True)
        else:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(i2 - i1):
            w = W1[:, i]
            dcol = Hinv1[i, i]
            if prune_n != 0 and i % prune_m == 0:
                blk = W1[:, i:i + prune_m] ** 2 / (torch.diag(Hinv1)[i:i + prune_m].reshape(1, -1) ** 2)
                mask1.scatter_(1, i + torch.topk(blk, prune_n, dim=1, largest=False).indices, True)
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / dcol
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            E1[:, i] = err
        W[:, i1:i2] = Q1
        W[:, i2:] -= E1.matmul(Hinv[i1:i2, i2:])
    return W


# ----------------------------------------------------------------------------
# The comparator runner
# ----------------------------------------------------------------------------

@dataclass
class BaselineResult:
    name: str
    paper: str
    reimplementation: bool
    assumptions: str
    precision_map: dict[str, int] = field(default_factory=dict)
    bytes_total: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Comparators:
    """Applies each comparator to a live model. Always restore() between runs."""

    def __init__(self, adapter: ModelAdapter, quantizer: Quantizer,
                 fmt: ByteFormat | None = None, batch_size: int = 8):
        self.a = adapter
        self.q = quantizer
        self.fmt = fmt or ByteFormat()
        self.bs = batch_size

    # ---- shared Hessian accumulation over a layer's components
    @torch.no_grad()
    def _layer_hessians(self, layer: int, inps, kwargs_list, comps: list[Component]
                        ) -> dict[str, torch.Tensor]:
        H: dict[str, torch.Tensor] = {}
        n: dict[str, int] = {}
        block = self.a.layers[layer]

        def acc(mod, x):
            c = next((cc for cc in comps if cc.module is mod), None)
            if c is None:
                return
            x2 = x.detach().reshape(-1, x.shape[-1]).float()
            if c.id not in H:
                H[c.id] = torch.zeros(x2.shape[1], x2.shape[1], device=x2.device)
                n[c.id] = 0
            m = x2.shape[0]
            H[c.id] *= n[c.id] / (n[c.id] + m)
            n[c.id] += m
            x2 = math.sqrt(2 / n[c.id]) * x2
            H[c.id] += x2.t() @ x2

        with self.a.module_input_capture([c.module for c in comps], acc):
            for x, kw in zip(inps, kwargs_list):
                block(x, **kw)
        return H

    def _sequential(self, calib_batches, apply_fn, comp_filter=None, label="baseline"):
        """Walk the stack layer by layer, applying ``apply_fn(comp, W, H)``."""
        a = self.a
        captured = a.catch_layer_inputs(calib_batches, layer=0)
        inps = [x[0] for x, _ in captured]
        kwargs_list = [k for _, k in captured]
        for li, block in enumerate(a.layers):
            comps = [c for c in a.components_of(li)
                     if comp_filter is None or comp_filter(c)]
            if comps:
                H = self._layer_hessians(li, inps, kwargs_list, comps)
                for c in comps:
                    if c.id not in H:
                        continue
                    self.q._stash(c)
                    W = a.get_weight(c).to(H[c.id].device)
                    a.set_weight(c, apply_fn(c, W, H[c.id]))
                del H
            new = []
            with torch.no_grad():
                for x, kw in zip(inps, kwargs_list):
                    o = block(x, **kw)
                    new.append(o[0] if isinstance(o, tuple) else o)
            inps = new
            if li % 8 == 0:
                log(f"  [{label}] layer {li}/{a.n_layers}")

    # ------------------------------------------------------------------ methods
    def fair_gptq(self, bits: int, calib_batches, pairs: Sequence[Pair],
                  lam: float = 1.0, max_pairs: int = 256) -> BaselineResult:
        """Fair-GPTQ (arXiv:2509.15206), re-implemented.

        Assumption, stated because only the abstract was available: the
        "explicit group-fairness constraint" is realised as an added term in the
        layerwise rounding objective,
            ||(W - W_q) X||^2  +  lam * ||(W - W_q) (X_a - X_b)||^2,
        i.e. ``H_fair = H + lam * H_contrastive``. This penalises quantization
        error that acts unequally on the two demographic sides of a matched pair,
        which is what "guides rounding toward reduced stereotype generation"
        must mean at the layer level. The grid, group size and damping are
        identical to this study's GPTQ so only the objective differs.
        """
        ex_a, ex_b = pair_group_examples(pairs, max_pairs)
        ids = [c.id for c in self.a.components]
        acts_a = _last_token_inputs(self.a, ex_a, ids, self.bs)
        acts_b = _last_token_inputs(self.a, ex_b, ids, self.bs)
        gs, pd = self.q.group_size, self.q.percdamp

        def apply(c: Component, W, H):
            if c.id in acts_a and c.id in acts_b:
                Hc = contrastive_hessian(acts_a[c.id].to(H.device), acts_b[c.id].to(H.device))
                scale = (torch.diag(H).mean() / torch.diag(Hc).mean().clamp(min=1e-12)).clamp(max=1e6)
                H = H + lam * scale * Hc
            return gptq_quantize(W, H, bits, gs, 128, pd, False, False).to(W.device)

        self._sequential(calib_batches, apply, label="fair-gptq")
        for c in self.a.components:
            self.q.current[c.id] = bits
        self.q.method = f"fair-gptq{bits}"
        pm = PrecisionMap({c.id: bits for c in self.a.components})
        return BaselineResult(
            name=f"fair_gptq{bits}", paper="Proskurina et al., arXiv:2509.15206",
            reimplementation=True,
            assumptions=("group-fairness constraint realised as H_fair = H + lam*H_contrastive "
                         "from last-token activations of matched counterfactual pairs; "
                         f"lam={lam}, {len(ex_a)} pairs"),
            precision_map=dict(pm), bytes_total=account_bytes(self.a, pm, self.fmt).total,
            extra={"lam": lam, "n_pairs": len(ex_a)})

    def debias_sparsegpt(self, sparsity: float, calib_batches, pairs: Sequence[Pair],
                         lam: float = 1.0, max_pairs: int = 256,
                         prune_n: int = 0, prune_m: int = 0) -> BaselineResult:
        """Debias-SparseGPT (EMNLP 2026, arXiv:2609.02496), re-implemented.

        The paper modifies the SparseGPT Hessian to account for demographic input
        differences. Implemented as ``H_debias = H + lam * H_contrastive`` with
        the same contrastive term as Fair-GPTQ, so that the pruning-vs-rounding
        comparison isolates the application, not the construction. Sparsity is
        unstructured by default; pass prune_n/prune_m for the 2:4 variant.
        """
        ex_a, ex_b = pair_group_examples(pairs, max_pairs)
        ids = [c.id for c in self.a.components]
        acts_a = _last_token_inputs(self.a, ex_a, ids, self.bs)
        acts_b = _last_token_inputs(self.a, ex_b, ids, self.bs)

        def apply(c: Component, W, H):
            if c.id in acts_a and c.id in acts_b:
                Hc = contrastive_hessian(acts_a[c.id].to(H.device), acts_b[c.id].to(H.device))
                scale = (torch.diag(H).mean() / torch.diag(Hc).mean().clamp(min=1e-12)).clamp(max=1e6)
                H = H + lam * scale * Hc
            return sparsegpt_prune(W, H, sparsity, 128, self.q.percdamp, prune_n, prune_m).to(W.device)

        self._sequential(calib_batches, apply, label="debias-sparsegpt")
        self.q.method = f"debias-sparsegpt{sparsity}"
        return BaselineResult(
            name=f"debias_sparsegpt_sp{sparsity}", paper="Proskurina et al., EMNLP 2026, arXiv:2609.02496",
            reimplementation=True,
            assumptions=("H_debias = H + lam*H_contrastive on the SparseGPT objective; "
                         f"lam={lam}, {len(ex_a)} pairs, "
                         f"{'unstructured' if not prune_n else f'{prune_n}:{prune_m}'} sparsity"),
            bytes_total=0,   # zeros in dense tensors are not a memory saving; see plan Section 10
            extra={"sparsity": sparsity, "lam": lam, "n_pairs": len(ex_a),
                   "memory_note": "sparsity is not byte-accounted: zeros in dense tensors save nothing"})

    def sparsegpt(self, sparsity: float, calib_batches, prune_n: int = 0, prune_m: int = 0
                  ) -> BaselineResult:
        """Plain SparseGPT, the un-debiased control for Debias-SparseGPT."""
        def apply(c: Component, W, H):
            return sparsegpt_prune(W, H, sparsity, 128, self.q.percdamp, prune_n, prune_m).to(W.device)

        self._sequential(calib_batches, apply, label="sparsegpt")
        self.q.method = f"sparsegpt{sparsity}"
        return BaselineResult(
            name=f"sparsegpt_sp{sparsity}", paper="Frantar and Alistarh, ICML 2023",
            reimplementation=True, assumptions="standard SparseGPT; control for the debiased variant",
            bytes_total=0, extra={"sparsity": sparsity})

    def critical_weight_protection(self, bits: int, pairs: Sequence[Pair],
                                   protect_frac: float = 0.01, max_pairs: int = 256
                                   ) -> BaselineResult:
        """Critical Weight Protection (arXiv:2601.12033), re-implemented.

        The paper identifies fairness-critical weights and keeps them at higher
        precision. Assumption, stated because only the abstract was available:
        criticality of ``w_ij`` is scored by how much it moves the *group
        difference* of the layer output, ``|w_ij| * |mean_a(x_j) - mean_b(x_j)|``.
        The top ``protect_frac`` of weights per component stay at 16 bits and the
        rest are quantized to ``bits``.

        Unlike this study's allocator this is *within-tensor* protection, so its
        cost is a sparse index, not a whole component at higher precision. The
        byte accounting below charges that index honestly (a 32-bit position per
        protected weight plus the 16-bit value), which is why CWP is not free.
        """
        ex_a, ex_b = pair_group_examples(pairs, max_pairs)
        ids = [c.id for c in self.a.components]
        acts_a = _last_token_inputs(self.a, ex_a, ids, self.bs)
        acts_b = _last_token_inputs(self.a, ex_b, ids, self.bs)
        n_prot_total = 0
        per_comp: dict[str, int] = {}

        for c in self.a.components:
            if c.id not in acts_a or c.id not in acts_b:
                continue
            self.q._stash(c)
            W = self.a.get_weight(c)
            d = (acts_a[c.id].mean(0) - acts_b[c.id].mean(0)).abs().to(W.device)
            score = W.abs() * d.unsqueeze(0)
            k = int(round(W.numel() * protect_frac))
            Wq = quantize_rtn(W, bits, self.q.group_size, self.q.sym)
            if k > 0:
                flat = torch.topk(score.flatten(), k, largest=True).indices
                Wq_flat, W_flat = Wq.flatten(), W.flatten()
                Wq_flat[flat] = W_flat[flat]          # keep critical weights dense
                Wq = Wq_flat.view_as(W)
            self.a.set_weight(c, Wq)
            self.q.current[c.id] = bits
            per_comp[c.id] = k
            n_prot_total += k

        self.q.method = f"cwp{bits}"
        pm = PrecisionMap({c.id: bits for c in self.a.components})
        base = account_bytes(self.a, pm, self.fmt)
        # sparse high-precision overlay: 16-bit value + 32-bit index per weight
        overlay = n_prot_total * (16 + 32) // 8
        return BaselineResult(
            name=f"cwp{bits}_p{protect_frac}", paper="Al Hakim et al., arXiv:2601.12033",
            reimplementation=True,
            assumptions=("criticality = |w| * |mean_a(x) - mean_b(x)| on matched counterfactual "
                         f"pairs; top {protect_frac:.1%} per component kept at 16 bits; "
                         "sparse overlay charged 48 bits per protected weight"),
            precision_map=dict(pm), bytes_total=base.total + overlay,
            extra={"protect_frac": protect_frac, "n_protected": n_prot_total,
                   "overlay_bytes": overlay, "base_bytes": base.total,
                   "n_pairs": len(ex_a)})
