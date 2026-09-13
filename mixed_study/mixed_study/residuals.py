"""B1 mechanics: inject a chosen residual at one block boundary and propagate it
through an otherwise DENSE downstream network (Next_Plan §5, items 4-6).

Three residual sources share one pipeline so they are compared on identical
model, tensors and inputs:

  actual        the residual compression really produces at that boundary,
                r = h_compressed - h_dense, captured per token
  sign_reversed -r  (same magnitude, opposite direction)
  random_k      norm-matched random directions, per token, k seeds

If `actual` and `random_k` propagate differently at equal per-token norm, then
residual DIRECTION carries information that magnitude-only diagnostics miss --
the mechanism the earlier random-noise profile could not see.

Everything is on the full (batch, seq, hidden) state with a per-token norm
convention, never a single-token summary, so the sequence shape is preserved.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import torch

from . import common as _common  # noqa: F401  -- puts ../quant-bias on sys.path
from quantbias.model_adapters import ModelAdapter
from quantbias.quantization import PrecisionMap, Quantizer


@dataclass
class Boundary:
    """Where the injection happens: after block ``layer`` (0-based), i.e. the
    hidden state that block ``layer+1`` reads. ``layer=-1`` means the embedding
    output. Recorded explicitly so a trace can never be mis-indexed."""
    layer: int

    @property
    def state_index(self) -> int:
        # hidden_states[0] is the embedding output, hidden_states[l+1] is block l's output
        return self.layer + 1


@torch.no_grad()
def _forward_capture(adapter: ModelAdapter, batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """(states, logits) with states captured by explicit hooks registered at
    call time. Index 0 = INPUT to block 0, index l+1 = OUTPUT of block l.

    Not ``output_hidden_states``: the library records a block's output before
    any user forward hook has modified it, so an injected residual would be
    absent from the recorded boundary state while still propagating
    downstream. Hooks registered here run after the injection hook and see the
    tensor the next block actually reads."""
    states: list[torch.Tensor | None] = [None] * (adapter.n_layers + 1)
    handles = []

    def pre0(m, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        states[0] = x.detach().float()
    handles.append(adapter.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
    for i, blk in enumerate(adapter.layers):
        def mk(i):
            def hook(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                states[i + 1] = h.detach().float()
            return hook
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        out = adapter.model(**{k: v.to(adapter.device) for k, v in batch.items()})
    finally:
        for h in handles:
            h.remove()
    assert all(x is not None for x in states), "a block did not fire; state mapping would be wrong"
    return states, out.logits.detach().float()  # type: ignore[return-value]


@torch.no_grad()
def capture_states(adapter: ModelAdapter, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """All (batch, seq, hidden) states: index 0 = input to block 0, l+1 = block l output."""
    return _forward_capture(adapter, batch)[0]


@torch.no_grad()
def residual_at(adapter: ModelAdapter, quantizer: Quantizer, pm: PrecisionMap,
                batch: dict[str, torch.Tensor], boundary: Boundary,
                method: str = "rtn", calib_batches=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(dense_state, residual) at the boundary for the compression in ``pm``.

    Applies the map, captures, restores exactly. The residual is what the
    compressed weights *upstream of and including* the boundary induce."""
    dense = capture_states(adapter, batch)[boundary.state_index]
    if method.startswith("gptq"):
        quantizer.apply_gptq(pm, calib_batches, progress=None, record_stats=False)
    else:
        quantizer.apply_rtn(pm, record_stats=False)
    try:
        comp = capture_states(adapter, batch)[boundary.state_index]
    finally:
        quantizer.restore()
    return dense, comp - dense


def per_token_norm(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """(batch, seq) L2 norm over hidden; padding positions zeroed if mask given."""
    n = x.norm(dim=-1)
    if mask is not None:
        n = n * mask.to(n.device, n.dtype)
    return n


def norm_matched_random(residual: torch.Tensor, seed: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Random direction per token with EXACTLY the actual residual's per-token norm."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(residual.shape, generator=g).to(residual.device, residual.dtype)
    z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    out = z * residual.norm(dim=-1, keepdim=True)
    if mask is not None:
        # mask arrives as the CPU batch tensor; residual lives wherever the model
        # ran. A CPU-only test can never catch this, the GPU smoke run did.
        out = out * mask.unsqueeze(-1).to(out.device, out.dtype)
    return out


def make_sources(residual: torch.Tensor, n_random: int, seed0: int = 0,
                 mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    src = {"actual": residual, "sign_reversed": -residual}
    for k in range(n_random):
        src[f"random_{k}"] = norm_matched_random(residual, seed0 + k, mask)
    return src


@contextlib.contextmanager
def inject(adapter: ModelAdapter, boundary: Boundary, delta: torch.Tensor) -> Iterator[None]:
    """Add ``delta`` to block ``boundary.layer``'s output during the forward.

    ``boundary.layer == -1`` adds to the embedding output instead. The hook adds
    to the tensor the next block actually reads, so the downstream network is
    dense and only the injected perturbation differs from the dense run."""
    d = delta.to(adapter.device)
    if boundary.layer < 0:
        mod = adapter.embed
        def hook(m, i, out):
            return out + d.to(out.dtype)
    else:
        mod = adapter.layers[boundary.layer]
        def hook(m, i, out):
            if isinstance(out, tuple):
                return (out[0] + d.to(out[0].dtype),) + tuple(out[1:])
            return out + d.to(out.dtype)
    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@dataclass
class Propagation:
    source: str
    boundary: int
    local_abs: float             # mean per-token ||delta|| at injection (tokens with mask)
    local_rel: float             # mean ||delta|| / ||h_dense||
    final_abs: float             # mean per-token ||drift|| at the last block output
    final_rel: float
    amplification: float         # final_abs / local_abs
    per_layer_rel: list[float]   # relative drift at every downstream boundary
    logit_linf: float            # max |dlogit| at the last prompt token, averaged over batch
    cos_final_vs_actual: float | None = None   # direction similarity to the ACTUAL source's final drift


@torch.no_grad()
def propagate(adapter: ModelAdapter, boundary: Boundary, delta: torch.Tensor,
              batch: dict[str, torch.Tensor], dense_states: list[torch.Tensor],
              dense_logits_last: torch.Tensor, mask: torch.Tensor,
              reference_final: torch.Tensor | None = None, eps: float = 1e-8) -> tuple[Propagation, torch.Tensor]:
    """Run once with ``delta`` injected; measure drift at every downstream boundary."""
    with inject(adapter, boundary, delta):
        states, logits = _forward_capture(adapter, batch)
    last = (mask.sum(1) - 1).long().to(adapter.device)
    rows = torch.arange(mask.shape[0], device=adapter.device)
    logits_last = logits[rows, last]
    m = mask.to(adapter.device).float()
    ntok = m.sum().clamp(min=1)

    def mean_norm(x):
        return float((per_token_norm(x) * m).sum() / ntok)

    def mean_rel(x, ref):
        return float(((per_token_norm(x) / (per_token_norm(ref) + eps)) * m).sum() / ntok)

    si = boundary.state_index
    local_abs = mean_norm(delta.to(adapter.device))
    local_rel = mean_rel(delta.to(adapter.device), dense_states[si])
    per_layer = []
    for j in range(si, len(states)):
        per_layer.append(mean_rel(states[j] - dense_states[j], dense_states[j]))
    drift_final = states[-1] - dense_states[-1]
    final_abs = mean_norm(drift_final)
    cos = None
    if reference_final is not None:
        mb = m.bool()
        a = drift_final[mb]; b = reference_final.to(drift_final.device)[mb]
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
    return Propagation(
        source="", boundary=boundary.layer, local_abs=local_abs, local_rel=local_rel,
        final_abs=final_abs, final_rel=per_layer[-1] if per_layer else float("nan"),
        amplification=final_abs / max(local_abs, eps), per_layer_rel=per_layer,
        logit_linf=float((logits_last - dense_logits_last).abs().amax(dim=-1).mean()),
        cos_final_vs_actual=cos), drift_final


@torch.no_grad()
def run_sources(adapter: ModelAdapter, boundary: Boundary, sources: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, Propagation]:
    """Propagate every source at the same boundary on the same inputs; the
    'actual' source's final drift is the direction reference for the others."""
    dense_states, dense_logits = _forward_capture(adapter, batch)
    last = (mask.sum(1) - 1).long().to(adapter.device)
    rows = torch.arange(mask.shape[0], device=adapter.device)
    dense_logits_last = dense_logits[rows, last]
    out: dict[str, Propagation] = {}
    ref = None
    order = ["actual"] + [k for k in sources if k != "actual"]
    for name in order:
        p, drift = propagate(adapter, boundary, sources[name], batch, dense_states, dense_logits_last, mask, ref)
        p.source = name
        out[name] = p
        if name == "actual":
            ref = drift
    return out
