"""F0 model resolution, F2 prompt selection and dense scoring, F3 simulated quantization.

Records are one JSON line per row and condition. A stage is complete when its
``*_COMPLETE.json`` manifest exists; an interrupted stage is rerun from scratch for
that model/condition cell only (records are rewritten atomically, never appended).
"""
from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from . import synthbias as S
from .common import (CALIB_SEED, MODELS, PROTOCOL, SEED, Manifest, log, read_json, read_jsonl, results_root,
                     stable_hash, write_json, write_jsonl)

QUANT = {"group_size": 128, "sym": False, "gptq_blocksize": 128, "gptq_percdamp": 0.01, "gptq_actorder": False,
         "calib": {"kind": "generic", "seed": CALIB_SEED, "n_seq": 128, "max_tokens": 512}}


# ----------------------------------------------------------------------------- models

def resolve_revisions(keys: Sequence[str]) -> dict[str, dict[str, str]]:
    """Immutable model/tokenizer revisions from the Hub (F0.2)."""
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    for k in keys:
        mid = MODELS[k]["id"]
        info = api.model_info(mid)
        out[k] = {"model_id": mid, "revision": info.sha, "tokenizer_revision": info.sha}
        log(f"{k} {mid} @ {info.sha[:12]}")
    return out


def load_adapter(key: str, device: str = "cuda", attn: str | None = "flash_attention_2", dtype: str = "auto"):
    from quantbias.model_adapters import ModelAdapter
    rev = "main"
    p = PROTOCOL / "LOCKED_PROTOCOL.json"
    if p.exists():
        rev = read_json(p)["models"].get(key, {}).get("revision", "main")
    a = ModelAdapter.from_pretrained(MODELS[key]["id"], revision=rev, dtype=dtype, device=device,
                                     attn_implementation=attn if device != "cpu" else None)
    if a.tokenizer.pad_token_id is None:
        a.tokenizer.pad_token = a.tokenizer.eos_token
    return a


def calibration(adapter, n_seq: int = 128, max_tokens: int = 512):
    from quantbias.data import build_calibration
    c = QUANT["calib"]
    return build_calibration(adapter.tokenizer, c["kind"], c["seed"], n_seq, max_tokens)


def quantizer_for(adapter):
    from quantbias.quantization import Quantizer
    return Quantizer(adapter, group_size=QUANT["group_size"], sym=QUANT["sym"])


# ----------------------------------------------------------------------------- scoring

def score_rows(adapter, rows: Sequence[S.Row], template: str, key: str, condition: str,
               batch_size: int = 16, swap_order: bool = False) -> list[dict[str, Any]]:
    """One record per row. Scores are fp32 log-softmax sums; the gold-minus-other
    margin is stored under both the sum and the mean rule."""
    from quantbias.evaluate import score_candidates
    ex = S.build_examples(rows, template, key, adapter.tokenizer, chat=MODELS[key]["chat"], swap=swap_order)
    scored = score_candidates(adapter, ex, norm="sum", batch_size=batch_size, progress_every=2000)
    by = {r.uid: r for r in rows}
    out = []
    for s, e in zip(scored, ex):
        r = by[s.uid]
        g = e.label
        o = 1 - g
        ms = s.logprob_sum[g] - s.logprob_sum[o]
        mm = s.logprob_mean[g] - s.logprob_mean[o]
        pred_mean = g if mm > 0 else (o if mm < 0 else s.pred)
        out.append({"uid": s.uid, "cluster": r.cluster, "split": r.split, "type": r.type, "stereo": r.stereo,
                    "pronoun": r.pronoun.lower(), "template": template, "model": key, "condition": condition,
                    "gold_position": g, "pred": s.pred, "correct": bool(s.pred == g), "pred_other": bool(s.pred == o),
                    "margin_sum": ms, "margin_mean": mm, "pred_mean_rule": pred_mean, "correct_mean_rule": bool(pred_mean == g),
                    "logprob_sum": s.logprob_sum, "logprob_mean": s.logprob_mean, "n_tokens": s.n_tokens,
                    "boundary_mismatch": s.boundary_mismatch, "swapped_order": swap_order})
    return out


def accuracy_stats(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def acc(rs):
        return sum(r["correct"] for r in rs) / len(rs) if rs else float("nan")
    pro = [r for r in recs if r["stereo"] == "pro"]
    anti = [r for r in recs if r["stereo"] == "anti"]
    p0 = [r for r in recs if r["gold_position"] == 0]
    p1 = [r for r in recs if r["gold_position"] == 1]
    return {"n": len(recs), "accuracy": acc(recs), "acc_pro": acc(pro), "acc_anti": acc(anti), "n_pro": len(pro), "n_anti": len(anti),
            "acc_pos0": acc(p0), "acc_pos1": acc(p1), "position_gap": abs(acc(p0) - acc(p1)),
            "min_pro_anti": min(acc(pro), acc(anti)), "n_anti_correct": sum(r["correct"] for r in anti),
            "boundary_mismatch_rate": sum(r["boundary_mismatch"] for r in recs) / max(1, len(recs))}


def smoke_subset(rows: Sequence[S.Row], n: int) -> list[S.Row]:
    """A balanced pro/anti prefix for smoke runs (the CSV is grouped by label)."""
    pro = [r for r in rows if r.stereo == "pro"][: n // 2]
    anti = [r for r in rows if r.stereo == "anti"][: n - n // 2]
    return pro + anti


# ----------------------------------------------------------------------------- F2

def select_prompt(adapter, key: str, rows_sel: Sequence[S.Row], out: Path, batch_size: int) -> dict[str, Any]:
    """Score P1-P3 on the selection split with the dense model and apply the frozen rule."""
    stats = {}
    for t in sorted(S.TEMPLATES):
        recs = score_rows(adapter, rows_sel, t, key, "dense", batch_size)
        write_jsonl(out / f"selection_dense_{t}.jsonl", recs)
        stats[t] = accuracy_stats(recs)
        log(f"  {key} {t}: acc={stats[t]['accuracy']:.3f} pro={stats[t]['acc_pro']:.3f} anti={stats[t]['acc_anti']:.3f} "
            f"posgap={stats[t]['position_gap']:.3f}")
    def key_of(t):
        st = stats[t]
        f = lambda x: -1.0 if x != x else x          # a NaN (empty stratum) ranks last
        return (-round(f(st["min_pro_anti"]), 6), round(f(st["position_gap"]), 6), -round(f(st["accuracy"]), 6), t)
    ranked = sorted(stats, key=key_of)
    choice = {"model": key, "template": ranked[0], "rule": ["max min(pro,anti) accuracy", "min |position gap|", "max accuracy", "lowest template id"],
              "ranking": ranked, "stats": stats}
    write_json(out / "prompt_selection.json", choice)
    log(f"  {key}: selected {ranked[0]}")
    return choice


def order_swap_audit(adapter, key: str, template: str, rows_final: Sequence[S.Row], out: Path, batch_size: int,
                     n_audit: int = 1000, dense_recs: dict[str, dict] | None = None) -> dict[str, Any]:
    rng = random.Random(SEED)
    pool = sorted(rows_final, key=lambda r: r.uid)
    audit = rng.sample(pool, min(n_audit, len(pool)))
    swapped = score_rows(adapter, audit, template, key, "dense", batch_size, swap_order=True)
    write_jsonl(out / "dense_order_swap_audit.jsonl", swapped)
    changed = 0
    for s in swapped:
        d = dense_recs[s["uid"]]
        # displayed positions are reversed, so the same occupation is predicted iff pred flips index
        same_occupation = (1 - s["pred"]) == d["pred"]
        changed += not same_occupation
    rep = {"n": len(swapped), "changed": changed, "changed_rate": changed / len(swapped), "seed": SEED}
    write_json(out / "dense_order_swap_audit.json", rep)
    log(f"  {key}: order-swap audit changed {changed}/{len(swapped)} ({rep['changed_rate']:.3%})")
    return rep


def run_dense(adapter, key: str, rows: Sequence[S.Row], smoke: bool, batch_size: int = 16, n_sel: int | None = None,
              n_final: int | None = None) -> dict[str, Any]:
    root = results_root(smoke) / "confirmation" / key
    root.mkdir(parents=True, exist_ok=True)
    done = root / "dense_COMPLETE.json"
    if done.exists():
        log(f"{key}: dense already complete")
        return read_json(done)
    man = Manifest(done, model=key, stage="F2_dense", smoke=smoke)
    sel = S.type2(rows, "selection")
    mech = S.type2(rows, "mechanism")
    fin = S.type2(rows, "final")
    if n_sel:
        sel = smoke_subset(sel, n_sel)
    if n_final:
        mech, fin = smoke_subset(mech, n_final), smoke_subset(fin, n_final)
    choice = select_prompt(adapter, key, sel, root, batch_size)
    t = choice["template"]
    recs_m = score_rows(adapter, mech, t, key, "dense", batch_size)
    write_jsonl(root / "mechanism_dense.jsonl", recs_m)
    recs_f = score_rows(adapter, fin, t, key, "dense", batch_size)
    write_jsonl(root / "final_dense.jsonl", recs_f)
    audit = order_swap_audit(adapter, key, t, fin, root, batch_size, n_audit=min(1000, len(fin)),
                             dense_recs={r["uid"]: r for r in recs_f})
    stats = {"selection": choice["stats"][t], "mechanism": accuracy_stats(recs_m), "final": accuracy_stats(recs_f)}
    return man.finish(template=t, stats=stats, order_swap=audit, n_rows={"selection": len(sel), "mechanism": len(mech), "final": len(fin)})


# ----------------------------------------------------------------------------- F3

def run_condition(adapter, key: str, condition: str, rows: Sequence[S.Row], smoke: bool, batch_size: int = 16,
                  n_final: int | None = None, calib_n_seq: int | None = None) -> dict[str, Any]:
    """Quantize the whole model (simulated, exact restore afterwards) and score the final split."""
    from quantbias.quantization import PrecisionMap
    from quantbias.model_adapters import check_weight_roundtrip
    root = results_root(smoke) / "confirmation" / key
    done = root / f"{condition}_COMPLETE.json"
    if done.exists():
        log(f"{key}/{condition}: already complete")
        return read_json(done)
    dense = read_json(root / "dense_COMPLETE.json")
    t = dense["template"]
    fin = S.type2(rows, "final")
    if n_final:
        fin = smoke_subset(fin, n_final)
    man = Manifest(done, model=key, stage="F3_" + condition, smoke=smoke, template=t)
    q = quantizer_for(adapter)
    bits = int(condition[-1])
    pm = PrecisionMap.uniform(adapter, bits)
    calib_hash = None
    t0 = time.time()
    if condition.startswith("gptq"):
        c = QUANT["calib"]
        cal = calibration(adapter, calib_n_seq or c["n_seq"], c["max_tokens"])
        calib_hash = cal.hash
        q.apply_gptq(pm, cal.batches(adapter.tokenizer.pad_token_id, str(adapter.device), batch_size=1),
                     actorder=QUANT["gptq_actorder"], record_stats=True)
    else:
        q.apply_rtn(pm, record_stats=True)
    q_seconds = time.time() - t0
    stats = q.stats_summary()
    try:
        recs = score_rows(adapter, fin, t, key, condition, batch_size)
        write_jsonl(root / f"final_{condition}.jsonl", recs)
    finally:
        restored = q.restore()
    rt = check_weight_roundtrip(adapter)
    restored_exact = all(q.is_restored(c) for c in pm.quantized_ids()) if hasattr(q, "is_restored") else None
    return man.finish(bits=bits, n_rows=len(recs), quantize_seconds=round(q_seconds, 1), calibration_hash=calib_hash,
                      quant_settings={k: v for k, v in QUANT.items() if k != "calib"}, quant_stats=stats,
                      n_restored=len(restored), restored_exact=restored_exact, roundtrip=rt,
                      precision_map_hash=pm.hash(), accuracy=accuracy_stats(recs))
