"""Verified Wanda: activation-norm scores with per-output-row sparsity.

Sun et al. (2024) prune each output row to the target sparsity using
score_ij = |w_ij| * ||x_j||_2 with x from calibration inputs. The repository's
earlier scripts used weight-column magnitudes and a global threshold; this
module raises instead of falling back when activations are missing
(plan, Section 10, correction 1).
"""
from __future__ import annotations

from typing import Any

import torch

from .common import log
from .model_adapters import ModelAdapter, Component


@torch.no_grad()
def collect_feature_norms(adapter: ModelAdapter, calib_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """||x_j||_2 over all calibration tokens for every component's input."""
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    mods = {c.module: c.id for c in adapter.components}

    def acc(mod, x):
        cid = mods[mod]
        x2 = x.detach().reshape(-1, x.shape[-1]).float()
        s = (x2 ** 2).sum(0)
        sums[cid] = sums.get(cid, torch.zeros_like(s)) + s
        counts[cid] = counts.get(cid, 0) + x2.shape[0]

    with adapter.module_input_capture(list(mods), acc):
        for b in calib_batches:
            adapter.model(**{k: v.to(adapter.device) for k, v in b.items()})
    missing = [c.id for c in adapter.components if c.id not in sums]
    if missing:
        raise RuntimeError(f"no activations captured for {len(missing)} components, e.g. {missing[:3]}; "
                           "check hooks (Conv1D modules must be included)")
    return {k: v.sqrt() for k, v in sums.items()}


@torch.no_grad()
def wanda_per_row(adapter: ModelAdapter, feat_norms: dict[str, torch.Tensor], sparsity: float,
                  comp_ids: list[str] | None = None) -> dict[str, Any]:
    """Zero the lowest-scoring ``sparsity`` fraction of each output row in place.

    Returns exact pruned cardinality per component so tests can check it.
    """
    report: dict[str, Any] = {"sparsity_target": sparsity, "components": {}}
    total_zero = total = 0
    for c in adapter.components:
        if comp_ids is not None and c.id not in comp_ids:
            continue
        W = adapter.get_weight(c)                       # (out, in)
        fn = feat_norms[c.id].to(W.device)
        if fn.numel() != W.shape[1]:
            raise RuntimeError(f"{c.id}: activation dim {fn.numel()} != in_features {W.shape[1]}")
        score = W.abs() * fn.unsqueeze(0)
        k = int(round(W.shape[1] * sparsity))
        if k > 0:
            idx = torch.topk(score, k, dim=1, largest=False).indices
            mask = torch.ones_like(W, dtype=torch.bool)
            mask.scatter_(1, idx, False)
            W = W * mask
        adapter.set_weight(c, W)
        nz = int((W == 0).sum())
        report["components"][c.id] = {"zeros": nz, "numel": W.numel(), "sparsity": nz / W.numel(),
                                      "per_row_exact": all(int((W[r] == 0).sum()) >= k for r in range(min(4, W.shape[0])))}
        total_zero += nz
        total += W.numel()
    report["sparsity_achieved"] = total_zero / max(1, total)
    log(f"Wanda per-row {sparsity:.0%}: achieved {report['sparsity_achieved']:.4f} over {total} weights")
    return report
