"""Section 5.7: selective restoration against controls matched on ADDED BYTES,
intervention type and parameter count.

The source E4 restoration compared predicted sites against random/utility
sites without matching cost: `restore_predicted` on Mistral used 5806 MiB
against a 4354 MiB ceiling, and mixed whole layers with single components. A
restoration advantage cannot be established that way. Here every arm restores
the same number of components of the same kind to the same bit-width from the
same uniform-4 start, and random arms are repeated so one lucky draw is not
mistaken for a method.

Sites are selected on the SELECTION split and evaluated on the FINAL split.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from .common import log, write_json, provenance
from quantbias.model_adapters import ModelAdapter
from quantbias.quantization import PrecisionMap, Quantizer, account_bytes, DENSE_BITS, ByteFormat


def matched_pool(adapter: ModelAdapter, kind: str, n_params_tol: float = 0.25) -> dict[str, list[str]]:
    """Components grouped by (kind, parameter-count bucket) so a control can be
    drawn from the same bucket as the predicted site."""
    pool: dict[str, list[str]] = {}
    for c in adapter.components:
        if c.kind != kind:
            continue
        bucket = f"{kind}:{round(__import__('math').log10(max(c.n_weights, 1)) / n_params_tol) * n_params_tol:.2f}"
        pool.setdefault(bucket, []).append(c.id)
    return pool


def bucket_of(adapter: ModelAdapter, cid: str, n_params_tol: float = 0.25) -> str:
    c = adapter.component_by_id[cid]
    return f"{c.kind}:{round(__import__('math').log10(max(c.n_weights, 1)) / n_params_tol) * n_params_tol:.2f}"


def arms(adapter: ModelAdapter, predicted: Sequence[str], utility_ranked: Sequence[str],
         k: int, n_random: int = 5, seed: int = 0, bits_restore: int = DENSE_BITS,
         start_bits: int = 4, fmt: ByteFormat | None = None) -> dict[str, dict[str, Any]]:
    """Build precision maps for each arm, all with the same k restored components
    drawn bucket-matched to the predicted set."""
    rng = random.Random(seed)
    pred = list(predicted)[:k]
    buckets = [bucket_of(adapter, c) for c in pred]
    pool = {}
    for c in adapter.components:
        pool.setdefault(bucket_of(adapter, c.id), []).append(c.id)

    def matched_from(ranked: Sequence[str]) -> list[str]:
        """Take, for each predicted site's bucket, the best-ranked site in that bucket."""
        chosen, used = [], set(pred)
        for b in buckets:
            cand = [c for c in ranked if c in pool.get(b, []) and c not in used and c not in chosen]
            if not cand:
                cand = [c for c in pool.get(b, []) if c not in used and c not in chosen]
            if cand:
                chosen.append(cand[0])
        return chosen

    def random_matched(s: int) -> list[str]:
        r = random.Random(s); chosen = []
        for b in buckets:
            cand = [c for c in pool.get(b, []) if c not in chosen and c not in pred]
            if cand:
                chosen.append(r.choice(cand))
        return chosen

    def pm_for(ids: Sequence[str]) -> PrecisionMap:
        pm = PrecisionMap.uniform(adapter, start_bits)
        for c in ids:
            pm[c] = bits_restore
        return pm

    out = {"uniform_start": {"ids": [], "pm": PrecisionMap.uniform(adapter, start_bits)},
           "predicted": {"ids": pred, "pm": pm_for(pred)},
           "utility_matched": {"ids": (u := matched_from(utility_ranked)), "pm": pm_for(u)}}
    for i in range(n_random):
        ids = random_matched(seed + 100 + i)
        out[f"random_matched_{i}"] = {"ids": ids, "pm": pm_for(ids)}
    for name, a in out.items():
        a["bytes"] = account_bytes(adapter, a["pm"], fmt).total
        a["n_restored"] = len(a["ids"])
        a["buckets"] = [bucket_of(adapter, c) for c in a["ids"]]
    return out


def run(adapter: ModelAdapter, quantizer: Quantizer, evaluate: Callable[[PrecisionMap], dict[str, Any]],
        predicted: Sequence[str], utility_ranked: Sequence[str], k: int = 4, n_random: int = 5,
        seed: int = 0, out_path=None, tag: str = "model") -> dict[str, Any]:
    """``evaluate(pm)`` must score the FINAL split under ``pm`` and return the
    corrected outcomes (harm.all_outcomes-style dict). Returns all arms with
    bytes so a reviewer can see the cost is actually equal."""
    A = arms(adapter, predicted, utility_ranked, k, n_random, seed)
    res: dict[str, Any] = {"tag": tag, "k": k, "n_random": n_random, "arms": {}}
    for name, a in A.items():
        quantizer.apply_rtn(a["pm"], record_stats=False)
        try:
            m = evaluate(a["pm"])
        finally:
            quantizer.restore()
        res["arms"][name] = {"ids": a["ids"], "bytes": a["bytes"], "n_restored": a["n_restored"],
                             "buckets": a["buckets"], "outcomes": m}
        log(f"  {name:20s} bytes={a['bytes']/2**20:.0f}MiB restored={a['n_restored']} "
            f"harmful/dc={m.get('task_damage_bbq_disambig',{}).get('harmful_per_dense_correct')}")
        if out_path:
            write_json(out_path, res)
    # equal-cost check: every restoring arm must land within 1% bytes of 'predicted'
    pb = res["arms"]["predicted"]["bytes"]
    res["equal_cost_ok"] = all(abs(a["bytes"] - pb) <= 0.01 * pb for n, a in res["arms"].items()
                               if n not in ("uniform_start",))
    rnd = [res["arms"][f"random_matched_{i}"]["outcomes"] for i in range(n_random)]
    key = lambda m: (m.get("task_damage_bbq_disambig") or {}).get("harmful_per_dense_correct")
    res["summary"] = {
        "predicted_harmful": key(res["arms"]["predicted"]["outcomes"]),
        "utility_matched_harmful": key(res["arms"]["utility_matched"]["outcomes"]),
        "random_matched_harmful_mean": (sum(key(m) or 0 for m in rnd) / len(rnd)) if rnd else None,
        "random_matched_harmful_min": min((key(m) for m in rnd if key(m) is not None), default=None),
        "uniform_start_harmful": key(res["arms"]["uniform_start"]["outcomes"]),
    }
    res["provenance"] = provenance()
    if out_path:
        write_json(out_path, res)
    return res
