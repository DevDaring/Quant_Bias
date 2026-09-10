"""Architecture adapters: one interface over GPT-2 and Llama-style models.

Fixes the three failure modes seen in the saved runs (plan, Section 10):
  * GPT-2 stores projections as Conv1D with weight shape (in, out); every
    weight is exposed here in (out, in) orientation and written back with the
    matching transpose.
  * ``'GPT2LMHeadModel' has no attribute 'model'`` -- layer lists are located
    by family, not by a hard-coded path.
  * Tied lm_head/embedding weights are reported once in byte accounting and
    excluded from quantization by default.
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from .common import log, resolve_hub_revision

try:
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover - very old transformers
    Conv1D = type("Conv1D", (), {})


ATTN_PAT = re.compile(r"(attn|attention|self_attn)")
MLP_PAT = re.compile(r"(mlp|feed_forward|ffn|w1|w2|w3)")
CONV_PAT = re.compile(r"(conv|short_conv|in_proj|out_proj)")
# Names of projections inside a block. Order fixes component ids.
_CACHE_KWARGS = {"past_key_value", "past_key_values", "layer_past", "cache_position_ids"}
KNOWN_PROJ = ("c_attn", "q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj",
              "gate_proj", "up_proj", "down_proj", "qkv_proj", "out_proj",
              "fc1", "fc2", "dense", "dense_h_to_4h", "dense_4h_to_h")


@dataclass
class Component:
    """One quantizable weight matrix inside a transformer block."""
    id: str            # e.g. "L3.mlp.up_proj"
    layer: int
    kind: str          # "attn" | "conv" | "mlp" | "other"
    name: str          # module path relative to the block
    module: nn.Module = field(repr=False)
    is_conv1d: bool = False
    out_features: int = 0
    in_features: int = 0

    @property
    def n_weights(self) -> int:
        return self.out_features * self.in_features

    @property
    def has_bias(self) -> bool:
        b = getattr(self.module, "bias", None)
        return b is not None


@dataclass
class ExcludedTensor:
    name: str
    numel: int
    dtype_bits: int
    tied_to: str | None = None


class ModelAdapter:
    """Wraps a causal LM and exposes blocks, components and weight access."""

    def __init__(self, model: nn.Module, tokenizer, model_id: str, revision: str = "unresolved"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.family, self.layers, self.embed, self.final_norm, self.lm_head = self._locate()
        self.components: list[Component] = self._discover_components()
        self.component_by_id: dict[str, Component] = {c.id: c for c in self.components}
        self.tied = self._is_tied()
        try:
            self.model.config.use_cache = False
        except Exception:
            pass
        self.model.eval()

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_pretrained(cls, model_id: str, revision: str = "main", dtype: str | torch.dtype = "auto",
                        device: str = "cpu", device_map: str | None = None,
                        trust_remote_code: bool = True, resolve_revision: bool = True,
                        attn_implementation: str | None = None) -> "ModelAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if isinstance(dtype, str):
            dtype = {"auto": None, "fp32": torch.float32, "float32": torch.float32,
                     "fp16": torch.float16, "float16": torch.float16,
                     "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[dtype]
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        sha = resolve_hub_revision(model_id, "model", revision) if resolve_revision else revision
        log(f"Loading {model_id}@{revision} ({sha}) dtype={dtype} device={device_map or device}")
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        kw: dict[str, Any] = dict(revision=revision, dtype=dtype, trust_remote_code=trust_remote_code)
        if device_map:
            kw["device_map"] = device_map
        # "flash_attention_2" needs the flash-attn wheel and fp16/bf16; "sdpa" is
        # the PyTorch fused path and is already the default on recent versions.
        # Attention is not the bottleneck for short scoring prompts, so a failure
        # here falls back rather than aborting the run.
        if attn_implementation and attn_implementation != "auto":
            kw["attn_implementation"] = attn_implementation
        def _load(kwargs):
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            except TypeError:  # transformers < 5 uses torch_dtype
                kwargs = dict(kwargs)
                kwargs["torch_dtype"] = kwargs.pop("dtype")
                return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        try:
            model = _load(kw)
        except Exception as e:
            if "attn_implementation" not in kw:
                raise
            log(f"  {attn_implementation} unavailable ({type(e).__name__}); falling back to default attention")
            kw.pop("attn_implementation")
            model = _load(kw)
        log(f"  attention: {getattr(model.config, '_attn_implementation', 'unknown')}")
        if not device_map:
            model.to(device)
        model.config.use_cache = False
        return cls(model, tok, model_id, sha)

    def _locate(self):
        """Find the decoder stack, embedding, final norm and head.

        Handles three shapes: GPT-2 (``transformer.h``), the Llama/Mistral/Qwen3
        family (``model.layers``), and multimodal wrappers that nest the decoder
        one level deeper (``model.language_model.layers``), e.g. Qwen3.5's
        ``Qwen3_5ForConditionalGeneration``. Hybrid stacks such as LFM2, whose
        ``layer_types`` mixes conv and attention blocks, use the second shape;
        the per-block component scan below copes with the heterogeneity.
        """
        m = self.model
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            t = m.transformer
            return "gpt2", list(t.h), t.wte, getattr(t, "ln_f", None), m.lm_head
        # decoder body: model.model, or model.model.language_model for VLM wrappers
        body = getattr(m, "model", None)
        if body is not None and not hasattr(body, "layers") and hasattr(body, "language_model"):
            body = body.language_model
        if body is not None and hasattr(body, "layers"):
            fam = type(m).__name__.lower()
            family = ("lfm2" if "lfm2" in fam else
                      "multimodal" if hasattr(getattr(m, "model", None), "language_model") else "llama")
            head = getattr(m, "lm_head", None)
            return (family, list(body.layers), body.embed_tokens,
                    getattr(body, "norm", None) or getattr(body, "final_layernorm", None), head)
        raise ValueError(f"Unknown architecture: {type(m).__name__}; add a case to ModelAdapter._locate")

    def _discover_components(self) -> list[Component]:
        comps: list[Component] = []
        for li, block in enumerate(self.layers):
            found = []
            for name, mod in block.named_modules():
                if isinstance(mod, nn.Linear):
                    out_f, in_f = mod.weight.shape
                    conv = False
                elif isinstance(mod, Conv1D):
                    in_f, out_f = mod.weight.shape
                    conv = True
                else:
                    continue
                leaf = name.split(".")[-1]
                if leaf not in KNOWN_PROJ and not (mod.weight.dim() == 2 and min(mod.weight.shape) >= 64):
                    continue
                if ATTN_PAT.search(name):
                    kind = "attn"
                elif MLP_PAT.search(name):
                    kind = "mlp"
                elif CONV_PAT.search(name):
                    kind = "conv"       # LFM2 / linear-attention short-conv blocks
                else:
                    kind = "other"
                found.append(Component(id=f"L{li}.{name}", layer=li, kind=kind, name=name,
                                       module=mod, is_conv1d=conv, out_features=out_f, in_features=in_f))
            # stable order: attention first, then mlp, then by name
            found.sort(key=lambda c: ({"attn": 0, "conv": 1, "mlp": 2, "other": 3}[c.kind],
                                      KNOWN_PROJ.index(c.name.split(".")[-1]) if c.name.split(".")[-1] in KNOWN_PROJ else 99,
                                      c.name))
            comps.extend(found)
        if not comps:
            raise ValueError("No quantizable components discovered")
        return comps

    def _is_tied(self) -> bool:
        try:
            return self.lm_head.weight.data_ptr() == self.embed.weight.data_ptr()
        except Exception:
            return False

    # --------------------------------------------------------------- weights
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def get_weight(self, comp: Component | str) -> torch.Tensor:
        """Weight as a fresh (out, in) float32 tensor regardless of storage."""
        c = self.component_by_id[comp] if isinstance(comp, str) else comp
        w = c.module.weight.data
        return (w.t() if c.is_conv1d else w).detach().float().clone()

    def set_weight(self, comp: Component | str, w_out_in: torch.Tensor) -> None:
        c = self.component_by_id[comp] if isinstance(comp, str) else comp
        target = c.module.weight
        src = w_out_in.t() if c.is_conv1d else w_out_in
        if src.shape != target.shape:
            raise ValueError(f"{c.id}: shape {tuple(src.shape)} != storage {tuple(target.shape)}")
        target.data.copy_(src.to(target.dtype).to(target.device))

    def components_of(self, layer: int | None = None, kind: str | None = None) -> list[Component]:
        out = self.components
        if layer is not None:
            out = [c for c in out if c.layer == layer]
        if kind is not None:
            out = [c for c in out if c.kind == kind]
        return out

    def excluded_tensors(self) -> list[ExcludedTensor]:
        """Everything that stays at storage precision: embeddings, norms, biases, head."""
        comp_params = {id(c.module.weight) for c in self.components}
        out: list[ExcludedTensor] = []
        seen: set[int] = set()
        embed_ptr = self.embed.weight.data_ptr()
        for name, p in self.model.named_parameters():
            if id(p) in comp_params or id(p) in seen:
                continue
            seen.add(id(p))
            bits = torch.finfo(p.dtype).bits if p.dtype.is_floating_point else p.element_size() * 8
            tied = None
            if p.data_ptr() == embed_ptr and "lm_head" in name:
                tied = "embed"  # counted once
            out.append(ExcludedTensor(name=name, numel=p.numel(), dtype_bits=bits, tied_to=tied))
        return out

    # ----------------------------------------------------------------- hooks
    @contextlib.contextmanager
    def layer_output_capture(self, store: list[list[torch.Tensor]] | None = None):
        """Collect each block's output hidden state (before final norm)."""
        if store is None:
            store = [[] for _ in self.layers]
        handles = []
        for i, blk in enumerate(self.layers):
            def mk(i):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    store[i].append(h.detach())
                return hook
            handles.append(blk.register_forward_hook(mk(i)))
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def module_input_capture(self, modules: Iterable[nn.Module], fn: Callable[[nn.Module, torch.Tensor], None]):
        """Call ``fn(module, input_tensor)`` for every forward of the given modules."""
        handles = [m.register_forward_hook(lambda mod, inp, out, fn=fn: fn(mod, inp[0])) for m in modules]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def catch_layer_inputs(self, batches: list[dict[str, torch.Tensor]], layer: int = 0):
        """Run the model up to ``layer`` and return the block inputs and kwargs.

        This is the standard sequential-GPTQ 'catcher': the block raises after
        recording its inputs, so nothing beyond it is computed.
        """
        blk = self.layers[layer]
        captured: list[tuple[tuple, dict]] = []

        class _Stop(Exception):
            pass

        class Catcher(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                # Drop cache objects: replaying a block with a live cache appends
                # keys on every call and breaks the attention mask.
                kw = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()
                      if k not in _CACHE_KWARGS}
                if "use_cache" in kwargs:
                    kw["use_cache"] = False
                captured.append((tuple(a.detach() if torch.is_tensor(a) else a for a in args), kw))
                raise _Stop

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.inner, name)

        self._swap_layer(layer, Catcher(blk))
        try:
            for b in batches:
                with torch.no_grad():
                    try:
                        self.model(**{k: v.to(self.device) for k, v in b.items()})
                    except _Stop:
                        pass
        finally:
            self._swap_layer(layer, blk)
        return captured

    def _swap_layer(self, idx: int, new: nn.Module) -> None:
        container = self.model.transformer.h if self.family == "gpt2" else self.model.model.layers
        container[idx] = new
        self.layers[idx] = new if not isinstance(new, nn.Module) or not hasattr(new, "inner") else new.inner

    # -------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        n_comp_w = sum(c.n_weights for c in self.components)
        excl = self.excluded_tensors()
        n_excl = sum(e.numel for e in excl if e.tied_to is None)
        return {
            "model_id": self.model_id, "revision": self.revision, "family": self.family,
            "n_layers": self.n_layers, "n_components": len(self.components),
            "component_weights": n_comp_w, "excluded_weights": n_excl,
            "tied_lm_head": self.tied, "dtype": str(self.dtype),
            "components_per_layer": [c.name for c in self.components_of(0)],
            "component_kinds": {k: sum(1 for c in self.components if c.kind == k)
                                for k in ("attn", "conv", "mlp", "other")},
            "layer_types": getattr(self.model.config, "layer_types", None),
        }


def check_weight_roundtrip(adapter: ModelAdapter, text: str = "The quick brown fox", atol: float = 0.0) -> dict:
    """Dense parity: get/set every component and confirm logits are unchanged.

    A wrong Conv1D orientation would change logits here; the test also
    confirms that set_weight(get_weight(c)) is an exact identity.
    """
    tok = adapter.tokenizer(text, return_tensors="pt").to(adapter.device)
    with torch.no_grad():
        ref = adapter.model(**tok).logits.float().clone()
    for c in adapter.components:
        adapter.set_weight(c, adapter.get_weight(c))
    with torch.no_grad():
        new = adapter.model(**tok).logits.float()
    diff = (ref - new).abs().max().item()
    return {"max_abs_logit_diff": diff, "ok": diff <= atol, "n_components": len(adapter.components)}
