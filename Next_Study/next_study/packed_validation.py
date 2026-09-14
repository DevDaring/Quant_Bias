"""F6: packed 4-bit GPTQ validation (plan section 7).

Runs in its own environment (the packed backend pins its own torch/transformers);
this module therefore depends only on torch, transformers, gptqmodel and the
quantbias scorer. Settings are matched to the simulator (4-bit, group 128, asymmetric,
no activation ordering, static groups where exposed, damping 0.01, the same 128 C4
calibration sequences); every unavoidable difference is written to
``packed_backend_manifest.json``. A backend failure is recorded, with the exact
exception and a minimal reproduction command, as BACKEND_UNSUPPORTED.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from . import synthbias as S
from .common import MODELS, PROTOCOL, Manifest, log, read_json, read_jsonl, results_root, sha256_file, write_json, write_jsonl
from .confirmation import QUANT, accuracy_stats, calibration


class _Shim:
    """Adapter-like view of a packed model for the quantbias scorer and perplexity."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _backend_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("gptqmodel", "torch", "transformers", "optimum", "auto_gptq"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def quantize_packed(key: str, out_dir: Path, calib_n_seq: int, smoke: bool) -> dict[str, Any]:
    """Create the packed GPTQ4 checkpoint with GPTQModel."""
    from gptqmodel import GPTQModel
    try:
        from gptqmodel import QuantizeConfig
    except ImportError:                      # renamed in newer releases
        from gptqmodel import GPTQConfig as QuantizeConfig
    rev = read_json(PROTOCOL / "LOCKED_PROTOCOL.json")["models"][key]["revision"] if (PROTOCOL / "LOCKED_PROTOCOL.json").exists() else "main"
    mid = MODELS[key]["id"]
    kw = {"bits": 4, "group_size": QUANT["group_size"], "sym": QUANT["sym"], "desc_act": QUANT["gptq_actorder"],
          "damp_percent": QUANT["gptq_percdamp"]}
    diffs: list[str] = []
    extra = {"static_groups": True, "act_group_aware": False}   # no reordering of any kind, static groups
    while True:
        try:
            qc = QuantizeConfig(**kw, **extra)
            break
        except TypeError as e:
            dropped = next((k for k in list(extra) if k in str(e)), None)
            if dropped is None:
                raise
            extra.pop(dropped)
            diffs.append(f"{dropped} option not exposed by this GPTQModel version")
    if "act_group_aware" in diffs[-1:] and "act_group_aware" not in extra:
        diffs.append("group-aware reordering could not be disabled explicitly; backend default applies")
    model = GPTQModel.load(mid, qc, revision=rev) if "revision" in GPTQModel.load.__code__.co_varnames else GPTQModel.load(mid, qc)
    tok = model.tokenizer if hasattr(model, "tokenizer") and model.tokenizer is not None else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cal = calibration(_Shim(model.model if hasattr(model, "model") else model, tok), calib_n_seq, QUANT["calib"]["max_tokens"])
    import torch
    data = [{"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)} for ids in cal.input_ids]
    t0 = time.time()
    model.quantize(data, batch_size=1)
    q_seconds = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    tok.save_pretrained(str(out_dir))
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    hashes = {p.name: sha256_file(p) for p in sorted(out_dir.glob("*.safetensors"))}
    return {"backend": "gptqmodel", "versions": _backend_versions(), "quantize_config": {**kw, **extra},
            "calibration_hash": cal.hash, "calibration_n_seq": calib_n_seq, "quantize_seconds": round(q_seconds, 1),
            "checkpoint_dir": str(out_dir), "checkpoint_bytes": size, "checkpoint_hashes": hashes, "differences": diffs,
            "reconstruction_command": f"python -m next_study.run packed --model {key}{' --smoke' if smoke else ''}"}


def score_packed(key: str, ckpt: Path, rows: Sequence[S.Row], template: str, batch_size: int, n_final: int | None) -> tuple[list[dict], dict]:
    from gptqmodel import GPTQModel
    from quantbias.evaluate import score_candidates, perplexity
    from quantbias.data import wikitext_test_text
    import torch
    try:
        model = GPTQModel.load(str(ckpt))
    except Exception as e:                   # JIT kernel build unavailable: fall back to the torch kernel
        from gptqmodel import BACKEND
        log(f"  default packed backend failed ({type(e).__name__}); loading with BACKEND.TORCH")
        model = GPTQModel.load(str(ckpt), backend=BACKEND.TORCH)
    hf = model.model if hasattr(model, "model") else model
    tok = model.tokenizer if getattr(model, "tokenizer", None) is not None else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ckpt))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    shim = _Shim(hf, tok)
    from .confirmation import smoke_subset
    fin = S.type2(rows, "final")
    if n_final:
        fin = smoke_subset(fin, n_final)
    ex = S.build_examples(fin, template, key, tok, chat=MODELS[key]["chat"])
    scored = score_candidates(shim, ex, norm="sum", batch_size=batch_size, progress_every=2000)
    by = {r.uid: r for r in fin}
    recs = []
    for s, e in zip(scored, ex):
        r = by[s.uid]; g = e.label; o = 1 - g
        mm = s.logprob_mean[g] - s.logprob_mean[o]
        pred_mean = g if mm > 0 else (o if mm < 0 else s.pred)
        recs.append({"uid": s.uid, "cluster": r.cluster, "split": r.split, "type": r.type, "stereo": r.stereo,
                     "pronoun": r.pronoun.lower(), "template": template, "model": key, "condition": "packed_gptq4",
                     "gold_position": g, "pred": s.pred, "correct": bool(s.pred == g), "pred_other": bool(s.pred == o),
                     "margin_sum": s.logprob_sum[g] - s.logprob_sum[o], "margin_mean": mm,
                     "pred_mean_rule": pred_mean, "correct_mean_rule": bool(pred_mean == g),
                     "logprob_sum": s.logprob_sum, "logprob_mean": s.logprob_mean, "n_tokens": s.n_tokens,
                     "boundary_mismatch": s.boundary_mismatch})
    # perplexity on the completed study's utility corpus
    text = wikitext_test_text()
    ppl = perplexity(shim, text, max_tokens=4096, chunk_size=1024)
    # throughput: warm-up then three timed passes over a fixed batch of prompts
    probe = [tok(e.prompt, return_tensors="pt", padding=True) for e in ex[:1]]
    ids = tok([e.prompt for e in ex[:min(32, len(ex))]], return_tensors="pt", padding=True).to(shim.device)
    with torch.no_grad():
        hf(**ids)
        times = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            hf(**ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    n_tok = int(ids["attention_mask"].sum())
    thr = {"tokens_per_second": [n_tok / t for t in times], "batch_prompts": int(ids["input_ids"].shape[0]), "descriptive_only": True}
    mem = {}
    if torch.cuda.is_available():
        mem["backend_reported_weight_memory_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    return recs, {"perplexity": ppl, "throughput": thr, "memory": mem, "n_rows": len(recs)}


def run(key: str, rows: Sequence[S.Row], smoke: bool, batch_size: int = 16, n_final: int | None = None,
        calib_n_seq: int | None = None) -> dict[str, Any]:
    root = results_root(smoke) / "packed" / key
    root.mkdir(parents=True, exist_ok=True)
    done = root / "packed_COMPLETE.json"
    if done.exists():
        log(f"{key}: packed already complete")
        return read_json(done)
    dense = read_json(results_root(smoke) / "confirmation" / key / "dense_COMPLETE.json")
    template = dense["template"]
    man = Manifest(done, model=key, stage="F6_packed", smoke=smoke, template=template)
    ckpt = root / "checkpoint"
    try:
        qinfo_p = root / "packed_backend_manifest.json"
        if qinfo_p.exists() and (ckpt / "config.json").exists():
            qinfo = read_json(qinfo_p)
        else:
            qinfo = quantize_packed(key, ckpt, calib_n_seq or QUANT["calib"]["n_seq"], smoke)
            write_json(qinfo_p, qinfo)
        recs, extra = score_packed(key, ckpt, rows, template, batch_size, n_final)
        write_jsonl(root / "final_packed_gptq4.jsonl", recs)
        return man.finish(status="VALID_COMPLETE", backend=qinfo, accuracy=accuracy_stats(recs), **extra)
    except Exception as exc:  # the plan's BACKEND_UNSUPPORTED path: one attempt, full diagnosis
        tb = traceback.format_exc()
        rep = {"exception_type": type(exc).__name__, "exception": str(exc)[:2000], "traceback": tb[-6000:],
               "versions": _backend_versions(), "python": sys.version,
               "reproduction_command": f"python -m next_study.run packed --model {key}{' --smoke' if smoke else ''}"}
        write_json(root / "BACKEND_FAILURE.json", rep)
        log(f"{key}: packed backend failed: {type(exc).__name__}: {str(exc)[:200]}")
        return man.finish(status="BACKEND_UNSUPPORTED", failure=rep)
