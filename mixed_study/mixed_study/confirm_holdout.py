"""E9: confirmation on an untouched evaluation source (Next_Plan §3.6, §10).

Every BBQ/WinoBias/Discrim-Eval(explicit) template has by now been inspected
and used to choose scores and metrics, so they are discovery evidence. The
Discrim-Eval `implicit` configuration is the same 70 decision questions with
the demographic cue conveyed implicitly (names, dialect) rather than stated.
It was never loaded in any prior run and is scored here once, frozen, under
the already-declared rule, for the one outcome it supports: decision
sensitivity to identity, with no correctness labels invented.

This is a partial holdout: it confirms the Discrim-Eval finding on new prompts
of the same questions. Confirmation of the BBQ findings still needs new
templates and is recorded as not done.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import harm, pairs as P
from .common import RESULTS, log, write_json, provenance
from .records import Row
from quantbias.model_adapters import ModelAdapter
from quantbias.quantization import PrecisionMap, Quantizer


def _to_row(s) -> Row:
    mt = s.meta or {}; gf = s.group_fields or {}
    return Row(s.uid, s.benchmark, s.cluster_id, s.group, s.label, s.pred, s.correct, tuple(s.logprob_sum),
               tuple(s.logprob_mean), s.margin, mt.get("context_condition"), mt.get("target_idx"),
               mt.get("unknown_idx"), gf.get("category"), tuple(gf.get("answer_groups") or ()), gf, mt)


def run(adapter: ModelAdapter, quantizer: Quantizer, configs=("rtn4", "gptq4", "rtn8"), calib_batches=None,
        seed: int = 20260908, max_questions: int | None = None, out_dir=None, tag: str = "model",
        batch_size: int = 8) -> dict[str, Any]:
    from quantbias.data import load_discrim_eval
    from quantbias.evaluate import score_candidates
    ex = load_discrim_eval(seed, config="implicit", max_questions=max_questions)
    log(f"holdout: Discrim-Eval implicit, {len(ex)} prompts, {len({e.cluster_id for e in ex})} questions")
    dense = {s.uid: _to_row(s) for s in score_candidates(adapter, ex, batch_size=batch_size, progress_every=0)}
    out: dict[str, Any] = {"tag": tag, "source": "Anthropic/discrim-eval:implicit", "n": len(ex),
                           "n_questions": len({e.cluster_id for e in ex}), "configs": {}}
    for cfg in configs:
        bits = int(cfg[-1]); pm = PrecisionMap.uniform(adapter, bits)
        if cfg.startswith("gptq"):
            quantizer.apply_gptq(pm, calib_batches, progress=None, record_stats=False)
        else:
            quantizer.apply_rtn(pm, record_stats=False)
        try:
            comp = {s.uid: _to_row(s) for s in score_candidates(adapter, ex, batch_size=batch_size, progress_every=0)}
        finally:
            quantizer.restore()
        prs = [(dense[u], comp[u]) for u in dense if u in comp]
        ds = harm.decision_sensitivity(prs)
        pg = P.pair_gap_change(dense, comp)
        out["configs"][cfg] = {"decision_sensitivity": ds, "pair_gap_change": pg}
        log(f"  {cfg}: mean|dP(yes)|={ds['mean_abs_delta_p_yes']:.4f} decision_change_rate={ds['decision_change_rate']:.4f} "
            f"gender dP={ {k: round(v,4) for k,v in ds['mean_delta_by_attribute']['gender'].items()} }")
        if out_dir:
            write_json(out_dir / "holdout_discrim_implicit.json", out)
    out["provenance"] = provenance()
    out["scope"] = ("Confirms decision-sensitivity effects on prompts never used for any prior choice. "
                    "Does NOT confirm BBQ findings; those need new templates.")
    if out_dir:
        write_json(out_dir / "holdout_discrim_implicit.json", out)
    return out
