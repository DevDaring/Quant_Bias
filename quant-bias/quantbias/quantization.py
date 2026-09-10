"""Weight-only quantization: RTN, reference GPTQ, precision maps, byte accounting.

Every quantizer here is *simulated*: weights are quantized and immediately
dequantized back to the storage dtype. Memory is therefore *accounted*, not
realized (plan, Section 8). Efficiency numbers must come from a packed backend.

Precision map convention: ``{component_id: bits}`` with 16 meaning "leave dense".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch

from .common import log, stable_hash
from .model_adapters import Component, ModelAdapter

DENSE_BITS = 16
ALLOWED_BITS = (3, 4, 8, 16)


# ----------------------------------------------------------------------------
# Precision map
# ----------------------------------------------------------------------------

class PrecisionMap(dict):
    """component_id -> bits. Missing ids are treated as dense."""

    @classmethod
    def uniform(cls, adapter: ModelAdapter, bits: int) -> "PrecisionMap":
        return cls({c.id: bits for c in adapter.components})

    @classmethod
    def dense(cls, adapter: ModelAdapter) -> "PrecisionMap":
        return cls.uniform(adapter, DENSE_BITS)

    @classmethod
    def single_site(cls, adapter: ModelAdapter, comp_ids: Iterable[str], bits: int) -> "PrecisionMap":
        pm = cls.dense(adapter)
        for cid in comp_ids:
            pm[cid] = bits
        return pm

    def bits_of(self, cid: str) -> int:
        return int(self.get(cid, DENSE_BITS))

    def quantized_ids(self) -> list[str]:
        return [k for k, v in self.items() if v < DENSE_BITS]

    def hash(self) -> str:
        return stable_hash(dict(self))

    def validate(self) -> None:
        bad = {k: v for k, v in self.items() if v not in ALLOWED_BITS}
        if bad:
            raise ValueError(f"bits must be in {ALLOWED_BITS}: {bad}")


# ----------------------------------------------------------------------------
# Byte accounting  C(b) = sum_j n_j b_j / 8 + metadata + uncompressed
# ----------------------------------------------------------------------------

@dataclass
class ByteFormat:
    group_size: int = 128       # groups along the input dimension
    scale_bits: int = 16        # fp16 scale per group
    zero_bits: int | None = None  # None -> same as weight bits (packed); 16 -> fp16
    row_align_bytes: int = 4    # packed rows padded to 32-bit words
    dense_bits: int = 16


@dataclass
class ByteReport:
    total: int
    components: int
    metadata: int
    padding: int
    uncompressed: int
    per_component: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"total_bytes": self.total, "component_bytes": self.components,
                "metadata_bytes": self.metadata, "padding_bytes": self.padding,
                "uncompressed_bytes": self.uncompressed, "total_MiB": round(self.total / 2**20, 2)}


def component_bytes(comp: Component, bits: int, fmt: ByteFormat) -> tuple[int, int, int]:
    """Return (weight_bytes, metadata_bytes, padding_bytes) for one component."""
    n_out, n_in = comp.out_features, comp.in_features
    if bits >= fmt.dense_bits:
        w = n_out * n_in * fmt.dense_bits // 8
        return w, 0, 0
    n_groups = n_out * math.ceil(n_in / fmt.group_size)
    row_bits = n_in * bits
    row_bytes = math.ceil(row_bits / 8)
    padded = math.ceil(row_bytes / fmt.row_align_bytes) * fmt.row_align_bytes
    weight = n_out * row_bytes
    padding = n_out * (padded - row_bytes)
    zbits = bits if fmt.zero_bits is None else fmt.zero_bits
    meta = math.ceil(n_groups * (fmt.scale_bits + zbits) / 8)
    return weight, meta, padding


def account_bytes(adapter: ModelAdapter, pm: PrecisionMap, fmt: ByteFormat | None = None,
                  include_uncompressed: bool = True) -> ByteReport:
    fmt = fmt or ByteFormat()
    comp_total = meta_total = pad_total = 0
    per: dict[str, int] = {}
    for c in adapter.components:
        w, m, p = component_bytes(c, pm.bits_of(c.id), fmt)
        b = getattr(c.module, "bias", None)
        bias_bytes = (b.numel() * fmt.dense_bits // 8) if b is not None else 0
        per[c.id] = w + m + p + bias_bytes
        comp_total += w + bias_bytes
        meta_total += m
        pad_total += p
    unc = 0
    if include_uncompressed:
        for e in adapter.excluded_tensors():
            if e.tied_to is None:
                unc += e.numel * fmt.dense_bits // 8
    return ByteReport(total=comp_total + meta_total + pad_total + unc, components=comp_total,
                      metadata=meta_total, padding=pad_total, uncompressed=unc, per_component=per)


# ----------------------------------------------------------------------------
# Round-to-nearest (asymmetric min/max, per group along input dim)
# ----------------------------------------------------------------------------

def _grid(bits: int, sym: bool) -> tuple[float, float]:
    if sym:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0.0, 2 ** bits - 1


def quantize_rtn(w: torch.Tensor, bits: int, group_size: int = 128, sym: bool = False,
                 return_params: bool = False):
    """Fake-quantize ``w`` (out, in) in groups along the input dimension."""
    assert w.dim() == 2
    out_f, in_f = w.shape
    g = in_f if group_size <= 0 else min(group_size, in_f)
    pad = (-in_f) % g
    wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
    wg = wp.reshape(out_f, -1, g)
    qmin, qmax = _grid(bits, sym)
    if sym:
        amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / qmax
        zero = torch.zeros_like(scale)
    else:
        wmin = wg.amin(dim=-1, keepdim=True)
        wmax = wg.amax(dim=-1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
        zero = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(wg / scale) + zero, qmin, qmax)
    deq = ((q - zero) * scale).reshape(out_f, -1)[:, :in_f]
    if return_params:
        return deq, {"scale": scale.squeeze(-1), "zero": zero.squeeze(-1), "q": q.reshape(out_f, -1)[:, :in_f]}
    return deq


def quantization_grid_check(w: torch.Tensor, deq: torch.Tensor, bits: int, group_size: int = 128,
                            sym: bool = False) -> dict[str, float]:
    """Verify that ``deq`` lies on a grid with at most 2**bits levels per group."""
    out_f, in_f = w.shape
    g = in_f if group_size <= 0 else min(group_size, in_f)
    pad = (-in_f) % g
    d = torch.nn.functional.pad(deq, (0, pad)) if pad else deq
    d = d.reshape(out_f, -1, g)
    max_levels = 0
    for r in range(min(out_f, 8)):
        for gi in range(d.shape[1]):
            max_levels = max(max_levels, int(torch.unique(d[r, gi]).numel()))
    err = (w - deq).abs()
    return {"max_levels_seen": max_levels, "levels_allowed": 2 ** bits,
            "grid_ok": max_levels <= 2 ** bits,
            "mean_abs_err": err.mean().item(), "max_abs_err": err.max().item(),
            "rel_fro_err": ((w - deq).norm() / w.norm().clamp(min=1e-12)).item()}


# ----------------------------------------------------------------------------
# Reference GPTQ (Frantar et al. 2023), per-group asymmetric grid
# ----------------------------------------------------------------------------

def gptq_quantize(w: torch.Tensor, H: torch.Tensor, bits: int, group_size: int = 128,
                  blocksize: int = 128, percdamp: float = 0.01, sym: bool = False,
                  actorder: bool = False) -> torch.Tensor:
    """Return the dequantized (out, in) weight found by GPTQ column updates.

    ``H`` is the (in, in) Hessian proxy 2 X^T X / n accumulated from calibration
    inputs. Group scales are recomputed when a new group starts, as in the
    reference implementation (static groups, no act-order by default).
    """
    W = w.clone().float()
    H = H.clone().float()
    out_f, in_f = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(in_f, device=H.device) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    g = in_f if group_size <= 0 else min(group_size, in_f)
    qmin, qmax = _grid(bits, sym)
    scale = zero = None

    def fit(cols: torch.Tensor):
        if sym:
            s = cols.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
            return s, torch.zeros_like(s)
        wmin = cols.amin(dim=1, keepdim=True)
        wmax = cols.amax(dim=1, keepdim=True)
        s = ((wmax - wmin) / qmax).clamp(min=1e-8)
        return s, torch.round(-wmin / s)

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % g == 0:
                scale, zero = fit(W[:, col:col + g])
            wcol = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(wcol.unsqueeze(1) / scale) + zero, qmin, qmax)
            qcol = ((q - zero) * scale).squeeze(1)
            Q1[:, i] = qcol
            err = (wcol - qcol) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    if actorder:
        Q = Q[:, invperm]
    return Q


# ----------------------------------------------------------------------------
# Applying maps to a live model, with exact restoration
# ----------------------------------------------------------------------------

class Quantizer:
    """Applies a PrecisionMap to an adapter and can restore any component exactly."""

    def __init__(self, adapter: ModelAdapter, group_size: int = 128, sym: bool = False,
                 percdamp: float = 0.01, blocksize: int = 128, keep_originals_on: str = "cpu"):
        self.adapter = adapter
        self.group_size = group_size
        self.sym = sym
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.keep_on = keep_originals_on
        self._orig: dict[str, torch.Tensor] = {}
        self.current = PrecisionMap.dense(adapter)
        self.method = "dense"
        self.last_stats: dict[str, dict] = {}

    # ----- originals
    def _stash(self, comp: Component) -> None:
        if comp.id not in self._orig:
            self._orig[comp.id] = comp.module.weight.data.detach().to(self.keep_on).clone()

    def restore(self, comp_ids: Iterable[str] | None = None) -> list[str]:
        ids = list(comp_ids) if comp_ids is not None else list(self._orig.keys())
        done = []
        for cid in ids:
            if cid in self._orig:
                c = self.adapter.component_by_id[cid]
                c.module.weight.data.copy_(self._orig[cid].to(c.module.weight.device, c.module.weight.dtype))
                self.current[cid] = DENSE_BITS
                done.append(cid)
        if comp_ids is None:
            self.method = "dense"
            self.last_stats = {}
        return done

    def is_restored(self, cid: str) -> bool:
        if cid not in self._orig:
            return True
        c = self.adapter.component_by_id[cid]
        return torch.equal(c.module.weight.data.to(self.keep_on), self._orig[cid].to(c.module.weight.dtype))

    # ----- RTN
    def apply_rtn(self, pm: PrecisionMap, record_stats: bool = True) -> PrecisionMap:
        pm.validate()
        self.restore()
        for cid, bits in pm.items():
            if bits >= DENSE_BITS:
                continue
            c = self.adapter.component_by_id[cid]
            self._stash(c)
            w = self.adapter.get_weight(c)
            deq = quantize_rtn(w, bits, self.group_size, self.sym)
            if record_stats:
                self.last_stats[cid] = quantization_grid_check(w, deq, bits, self.group_size, self.sym)
            self.adapter.set_weight(c, deq)
            self.current[cid] = bits
        self.method = "rtn"
        return self.current

    # ----- GPTQ (sequential, layer by layer)
    def apply_gptq(self, pm: PrecisionMap, calib_batches: list[dict[str, torch.Tensor]],
                   actorder: bool = False, record_stats: bool = True,
                   progress: Callable[[str], None] | None = log) -> PrecisionMap:
        """Standard sequential GPTQ: quantized outputs of layer l feed layer l+1.

        ``calib_batches`` are tokenizer outputs (input_ids, attention_mask). Only
        components with bits < 16 in ``pm`` are changed; dense ones pass through.
        """
        pm.validate()
        self.restore()
        adapter = self.adapter
        wanted = set(pm.quantized_ids())
        if not wanted:
            self.method = "gptq"
            return self.current
        captured = adapter.catch_layer_inputs(calib_batches, layer=0)
        inps = [a[0] for a, _ in captured]
        kwargs_list = [k for _, k in captured]
        for li, block in enumerate(adapter.layers):
            comps = [c for c in adapter.components_of(li) if c.id in wanted]
            if comps:
                H: dict[str, torch.Tensor] = {}
                n: dict[str, int] = {}

                def acc(mod, x, _H=H, _n=n, _comps=comps):
                    c = next(cc for cc in _comps if cc.module is mod)
                    x2 = x.detach().reshape(-1, x.shape[-1]).float()
                    if c.id not in _H:
                        _H[c.id] = torch.zeros(x2.shape[1], x2.shape[1], device=x2.device)
                        _n[c.id] = 0
                    m = x2.shape[0]
                    _H[c.id] *= _n[c.id] / (_n[c.id] + m)
                    _n[c.id] += m
                    x2 = math.sqrt(2 / _n[c.id]) * x2
                    _H[c.id] += x2.t() @ x2

                with adapter.module_input_capture([c.module for c in comps], acc):
                    with torch.no_grad():
                        for x, kw in zip(inps, kwargs_list):
                            block(x, **kw)
                for c in comps:
                    self._stash(c)
                    w = adapter.get_weight(c)
                    bits = pm.bits_of(c.id)
                    deq = gptq_quantize(w.to(H[c.id].device), H[c.id], bits, self.group_size,
                                        self.blocksize, self.percdamp, self.sym, actorder)
                    if record_stats:
                        self.last_stats[c.id] = quantization_grid_check(w, deq.to(w.device), bits, self.group_size, self.sym)
                    adapter.set_weight(c, deq)
                    self.current[c.id] = bits
                del H
                if progress:
                    progress(f"  GPTQ layer {li}: {len(comps)} components")
            # propagate (quantized) outputs to next layer
            new_inps = []
            with torch.no_grad():
                for x, kw in zip(inps, kwargs_list):
                    out = block(x, **kw)
                    new_inps.append(out[0] if isinstance(out, tuple) else out)
            inps = new_inps
        self.method = "gptq" + ("-actorder" if actorder else "")
        return self.current

    def apply(self, pm: PrecisionMap, method: str = "rtn", calib_batches=None, **kw) -> PrecisionMap:
        if method == "rtn":
            return self.apply_rtn(pm, **kw)
        if method.startswith("gptq"):
            if calib_batches is None:
                raise ValueError("GPTQ needs calib_batches")
            return self.apply_gptq(pm, calib_batches, actorder=method.endswith("actorder"), **kw)
        if method == "awq":
            raise NotImplementedError(
                "AWQ is an external backend. Install autoawq and quantize the checkpoint "
                "offline, then load it through ModelAdapter.from_pretrained; do not fall "
                "back silently to RTN (plan, Section 10, correction 1).")
        raise ValueError(f"unknown method {method}")

    def stats_summary(self) -> dict[str, Any]:
        if not self.last_stats:
            return {}
        rel = [s["rel_fro_err"] for s in self.last_stats.values()]
        return {"n_quantized": len(self.last_stats),
                "grid_ok_all": all(s["grid_ok"] for s in self.last_stats.values()),
                "mean_rel_fro_err": sum(rel) / len(rel), "max_rel_fro_err": max(rel)}
