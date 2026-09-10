"""Experiment driver for E0-E5 (plan, Section 9).

Run from ``python/``:
    .venv/bin/python -m quantbias.run_experiment --exp e0 --model M1 --device cpu --quick
    .venv/bin/python -m quantbias.run_experiment --exp e1 --model M3 --device cuda
    .venv/bin/python -m quantbias.run_experiment --exp e2 --model M3
    .venv/bin/python -m quantbias.run_experiment --exp e3 --model M2
    .venv/bin/python -m quantbias.run_experiment --exp e4 --model M3
    .venv/bin/python -m quantbias.run_experiment --exp e5 --model M5

``--quick`` limits every dataset so a CPU run on GPT-2 finishes in minutes;
results of quick runs are written under results/bias/<exp>/<tag>-quick/.
"""
from __future__ import annotations

import argparse
import copy
import random
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import allocate as A
from . import baselines as B
from . import bridge as BR
from . import data as D
from . import evaluate as E
from . import statistics as S
from . import trace as T
from . import report as R
from .common import (CONFIG_DIR, RESULTS_DIR, Manifest, load_env, load_yaml, log, set_seed,
                     stable_hash, write_json, append_jsonl, get_device, default_seed)
from .model_adapters import ModelAdapter, check_weight_roundtrip
from .quantization import PrecisionMap, Quantizer, account_bytes, ByteFormat, DENSE_BITS
from .pruning import collect_feature_norms, wanda_per_row


# ----------------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------------

class Ctx:
    def __init__(self, args):
        load_env()
        if getattr(args, 'smoke', False):
            args.quick = True   # smoke reuses every quick-mode size cap, then shrinks further
        self.args = args
        self.cfg = load_yaml(CONFIG_DIR / "experiments.yaml")
        self.models = load_yaml(CONFIG_DIR / "models.yaml")
        self.mcfg = self.models[args.model]
        self.seed = args.seed if args.seed is not None else int(self.cfg.get("seed", default_seed()))
        set_seed(self.seed)
        self.device = get_device(args.device)
        self.tag = self.mcfg["tag"] + ("-smoke" if args.smoke else "-quick" if args.quick else "")
        self.out = RESULTS_DIR / args.exp / self.tag
        self.out.mkdir(parents=True, exist_ok=True)
        self.norm = self.cfg["scoring"]["norm"]
        self.fmt = ByteFormat(group_size=self.cfg["quantization"]["group_size"])
        self.adapter: ModelAdapter | None = None
        self.quantizer: Quantizer | None = None
        self._examples: list[D.Example] | None = None
        self._wiki: str | None = None
        # Sampling profile: caps instantiations per template cell, never clusters.
        samp = self.cfg.get("sampling", {})
        prof = args.profile or ("smoke" if args.smoke else samp.get("profile", "budget"))
        self.profile = prof
        pcfg = samp.get(prof, {}) or {}
        self.limits = {"bbq_per_cluster_per_cell": pcfg.get("bbq_per_cluster_per_cell"),
                       "discrim_ages": pcfg.get("discrim_ages")}
        if args.smoke:
            self.limits.update({"bbq_per_category": 2, "discrim_questions": 1})
        elif args.quick:
            self.limits.update({"bbq_per_category": 40, "discrim_questions": 4})
        self.limits["bbq_answer_format"] = self.cfg["scoring"]["bbq_answer_format"]
        self.limits["discrim_config"] = self.cfg.get("discrim_config", "explicit")

    # ---- lazy resources
    def model(self) -> ModelAdapter:
        if self.adapter is None:
            self.adapter = ModelAdapter.from_pretrained(
                self.mcfg["id"], revision=self.mcfg.get("revision", "main"), dtype=self.mcfg.get("dtype", "auto"),
                device=self.device, device_map=("auto" if self.device == "cuda" and self.args.device_map else None),
                attn_implementation=(self.mcfg.get("attn") or self.args.attn))
            q = self.cfg["quantization"]
            self.quantizer = Quantizer(self.adapter, group_size=q["group_size"], sym=q["symmetric"],
                                       percdamp=q["percdamp"])
        return self.adapter

    def examples(self) -> list[D.Example]:
        if self._examples is None:
            self._examples = D.load_benchmarks(self.seed, self.cfg["benchmarks"],
                                               tuple(self.cfg["split_fractions"]), self.limits)
            if self.args.quick:
                # keep WinoBias small too
                wb = [e for e in self._examples if e.benchmark == "winobias"]
                rng = random.Random(self.seed)
                keep_clusters = set(rng.sample(sorted({e.cluster_id for e in wb}), min(60, len({e.cluster_id for e in wb}))))
                self._examples = [e for e in self._examples if e.benchmark != "winobias" or e.cluster_id in keep_clusters]
        return self._examples

    def split(self, name: str) -> list[D.Example]:
        return D.by_split(self.examples(), name)

    def wiki(self) -> str:
        if self._wiki is None:
            self._wiki = D.wikitext_test_text(max_chars=200_000 if self.args.quick else None)
        return self._wiki

    def calibration(self, kind: str = "generic", seed: int = 0, examples=None) -> D.CalibrationSet:
        c = self.cfg["calibration"]
        n_seq = 8 if self.args.quick else c["n_seq"]
        max_tokens = 128 if self.args.quick else c["max_tokens"]
        return D.build_calibration(self.model().tokenizer, kind, seed + self.seed, n_seq, max_tokens,
                                   examples=examples if examples is not None else self.examples(),
                                   demographic_fraction=c["demographic_fraction"],
                                   c4_docs=300 if self.args.quick else 4000)

    def manifest(self, exp: str, method: str = "dense", pm: PrecisionMap | None = None, **extra) -> Manifest:
        a = self.model()
        m = Manifest(experiment_id=exp, model_id=a.model_id, model_revision=a.revision,
                     tokenizer_revision=a.revision, method=method, seed=self.seed,
                     precision_map_hash=pm.hash() if pm is not None else "",
                     split_hash=D.split_hash(self.examples()) if self._examples else "",
                     extra={"config_hash": stable_hash(self.cfg), "quick": self.args.quick,
                            "sampling_profile": self.profile, "limits": self.limits, **extra})
        return m

    def batch_size(self) -> int:
        return int(self.mcfg.get("batch_size", 8))

    # ---- shared evaluation of the *current* model state
    def subsample(self, split: str, n: int | None) -> list[D.Example]:
        """Deterministic cluster-level subsample, used for allocator search only."""
        ex = self.split(split)
        if not n or len(ex) <= n:
            return ex
        clusters = sorted({e.cluster_id for e in ex})
        rng = random.Random(self.seed)
        rng.shuffle(clusters)
        keep: set[str] = set()
        count = 0
        per = {c: sum(1 for e in ex if e.cluster_id == c) for c in clusters}
        for c in clusters:
            if count >= n:
                break
            keep.add(c)
            count += per[c]
        return [e for e in ex if e.cluster_id in keep]

    def evaluate_state(self, split: str, with_ppl: bool = True, examples=None,
                       ppl_max_tokens: int | None = None) -> dict[str, Any]:
        a = self.model()
        ex = self.split(split) if examples is None else examples
        scored = E.score_candidates(a, ex, norm=self.norm, batch_size=self.batch_size(),
                                    max_len=self.cfg["scoring"]["max_len"])
        out = {"scored": scored, "metrics": E.metrics_for(scored, self.norm)}
        if with_ppl:
            u = self.cfg["utility"]
            out["ppl"] = E.perplexity(a, self.wiki(),
                                      max_tokens=(ppl_max_tokens or (1024 if self.args.quick else u["wikitext_max_tokens"])),
                                      chunk_size=u["wikitext_chunk"])
        return out

    def apply_config(self, qc: dict[str, Any], calib: D.CalibrationSet | None = None) -> PrecisionMap:
        a, q = self.model(), self.quantizer
        if qc["method"] == "dense":
            q.restore()
            return PrecisionMap.dense(a)
        pm = PrecisionMap.uniform(a, int(qc["bits"]))
        if qc["method"] == "rtn":
            q.apply_rtn(pm)
        elif qc["method"].startswith("gptq"):
            calib = calib or self.calibration("generic", 0)
            q.apply_gptq(pm, calib.batches(a.tokenizer.pad_token_id, self.device, batch_size=1),
                         actorder=qc["method"].endswith("actorder"))
        else:
            q.apply(pm, qc["method"])
        return pm


def _records(scored: list[E.ScoredExample], manifest: Manifest, config: str, pm_hash: str) -> list[dict]:
    base = {"experiment_id": manifest.experiment_id, "model_id": manifest.model_id,
            "model_revision": manifest.model_revision, "code_revision": manifest.code_revision,
            "seed": manifest.seed, "config": config, "precision_map_hash": pm_hash, "status": "ok"}
    return [{**base, **s.as_dict()} for s in scored]


def _harm_and_ci(ctx: Ctx, dense: list[E.ScoredExample], quant: list[E.ScoredExample]) -> dict[str, Any]:
    st = ctx.cfg["statistics"]
    out: dict[str, Any] = {"harm": {}, "ci": {}, "flips": E.flip_table(dense, quant)}
    qd = {s.uid: s for s in quant}
    for bench in ("bbq", "winobias"):
        rows = [(d, qd[d.uid]) for d in dense if d.benchmark == bench and d.uid in qd and d.correct is not None
                and (bench != "bbq" or d.meta.get("context_condition") == "disambig")]
        if not rows:
            continue
        out["harm"][bench] = E.harm_summary(dense, quant, bench, st["min_group_n"])
        out["ci"][bench] = S.group_ci_table([d.correct for d, _ in rows], [q.correct for _, q in rows],
                                             [d.group for d, _ in rows], [d.cluster_id for d, _ in rows],
                                             min_n=st["min_group_n"], n_boot=(200 if ctx.args.quick else st["n_boot"]),
                                             seed=ctx.seed, alpha=st["alpha"])
    return out


# ----------------------------------------------------------------------------
# E0  reproducibility pilot
# ----------------------------------------------------------------------------

def run_e0(ctx: Ctx) -> dict[str, Any]:
    a, q = ctx.model(), ctx.quantizer
    man = ctx.manifest("e0")
    res: dict[str, Any] = {"adapter": a.summary(), "roundtrip": check_weight_roundtrip(a)}
    log(f"adapter: {res['adapter']}")
    # quantization grid + bytes
    for bits in (8, 4):
        pm = PrecisionMap.uniform(a, bits)
        res[f"bytes_rtn{bits}"] = account_bytes(a, pm, ctx.fmt).as_dict()
        q.apply_rtn(pm)
        res[f"grid_rtn{bits}"] = q.stats_summary()
        res[f"ppl_rtn{bits}"] = E.perplexity(a, ctx.wiki(), max_tokens=1024 if ctx.args.quick else 4096)
        q.restore()
    res["bytes_dense"] = account_bytes(a, PrecisionMap.dense(a), ctx.fmt).as_dict()
    res["ppl_dense"] = E.perplexity(a, ctx.wiki(), max_tokens=1024 if ctx.args.quick else 4096)
    # GPTQ-4 with generic calibration
    calib = ctx.calibration("generic", 0)
    batches = calib.batches(a.tokenizer.pad_token_id, ctx.device)
    t0 = time.time()
    q.apply_gptq(PrecisionMap.uniform(a, 4), batches)
    res["gptq4"] = {**q.stats_summary(), "seconds": time.time() - t0, "calibration_hash": calib.hash,
                    "ppl": E.perplexity(a, ctx.wiki(), max_tokens=1024 if ctx.args.quick else 4096)}
    q.restore()
    res["restored_exact"] = all(q.is_restored(c.id) for c in a.components)
    # per-row Wanda verification
    fn = collect_feature_norms(a, batches)
    res["wanda"] = wanda_per_row(a, fn, 0.5)
    res["wanda"]["ppl"] = E.perplexity(a, ctx.wiki(), max_tokens=1024 if ctx.args.quick else 4096)
    q.restore()  # Wanda used set_weight without stash -> restore from originals captured by gptq stash
    for c in a.components:  # originals were stashed during gptq, so restore() above covered every component
        assert q.is_restored(c.id), c.id
    # scoring sanity: known answer + label permutation
    ex = ctx.split("selection")[:60]
    scored = E.score_candidates(a, ex, norm=ctx.norm, batch_size=ctx.batch_size())
    res["scoring"] = {"n": len(scored), "boundary_mismatch_rate": sum(s.boundary_mismatch for s in scored) / max(1, len(scored)),
                      "acc": float(np.mean([s.correct for s in scored if s.correct is not None] or [np.nan]))}
    known = D.Example(uid="known", benchmark="sanity", cluster_id="k", split="selection", group="g",
                      prompt="Question: What is the capital of France?\nAnswer:", candidates=[" Paris", " Zebra", " Purple"], label=0)
    res["scoring"]["known_answer_ok"] = E.score_candidates(a, [known], norm=ctx.norm)[0].pred == 0
    bbq = [e for e in ex if e.benchmark == "bbq"][:20]
    if bbq:
        perm_ex = []
        for e in bbq:
            p = copy.deepcopy(e)
            p.candidates = [e.candidates[2], e.candidates[0], e.candidates[1]]
            p.label = [2, 0, 1].index(e.label)
            perm_ex.append(p)
        s1 = E.score_candidates(a, bbq, norm=ctx.norm)
        s2 = E.score_candidates(a, perm_ex, norm=ctx.norm)
        res["scoring"]["permutation_consistent"] = all(
            abs(x.logprob_sum[x.label] - y.logprob_sum[y.label]) < 1e-3 for x, y in zip(s1, s2))
    res["splits"] = D.check_split_disjoint(ctx.examples())
    res["manifest"] = man.finish()
    write_json(ctx.out / "e0_summary.json", res)
    return res


# ----------------------------------------------------------------------------
# E1  quantization and subgroup outcomes
# ----------------------------------------------------------------------------

def run_e1(ctx: Ctx, split: str = "final") -> dict[str, Any]:
    a, q = ctx.model(), ctx.quantizer
    configs = ctx.cfg["quantization"]["configs"]
    if ctx.args.configs:
        configs = [c for c in configs + ctx.cfg["quantization"].get("optional_configs", []) if c["name"] in ctx.args.configs]
    dense_scored = None
    summary_all = {}
    calib = None
    for qc in configs:
        log(f"=== E1 {qc['name']} on {split} ===")
        man = ctx.manifest("e1", method=qc["method"], config=qc)
        try:
            if qc["method"].startswith("gptq") and calib is None:
                calib = ctx.calibration("generic", 0)
            pm = ctx.apply_config(qc, calib)
            man.precision_map_hash = pm.hash()
            if calib is not None and qc["method"].startswith("gptq"):
                man.calibration_hash = calib.hash
            ev = ctx.evaluate_state(split)
            scored = ev["scored"]
            summ = {"config": qc, "metrics": ev["metrics"], "ppl": ev["ppl"],
                    "bytes": account_bytes(a, pm, ctx.fmt).as_dict(), "quant_stats": q.stats_summary()}
            if qc["method"] == "dense":
                dense_scored = scored
            elif dense_scored is not None:
                summ.update(_harm_and_ci(ctx, dense_scored, scored))
                pairs = D.counterfactual_pairs(ctx.split(split))
                pg = E.pair_gap_summary(dense_scored, scored, pairs, ctx.norm)
                pg.pop("rows", None)
                summ["pair_gap"] = pg
            summ["manifest"] = man.finish()
            append_jsonl(ctx.out / f"records_{qc['name']}.jsonl", _records(scored, man, qc["name"], pm.hash()))
            write_json(ctx.out / f"summary_{qc['name']}.json", summ)
            summary_all[qc["name"]] = {k: v for k, v in summ.items() if k in ("ppl", "bytes")} | {
                "bbq": summ["metrics"].get("bbq", {}).get("disambig"),
                "harm_bbq": summ.get("harm", {}).get("bbq", {})}
            if "harm" in summ and "bbq" in summ["harm"]:
                R.plot_subgroup_change(summ["harm"]["bbq"], summ["ci"].get("bbq"), ctx.out / f"fig_subgroup_{qc['name']}.png",
                                       title=f"{ctx.tag} {qc['name']} BBQ")
        except Exception as ex:  # record failures explicitly
            log(f"FAILED {qc['name']}: {ex}")
            man.finish("failed", traceback.format_exc())
            write_json(ctx.out / f"summary_{qc['name']}.json", {"config": qc, "manifest": man})
        finally:
            q.restore()
    write_json(ctx.out / "e1_overview.json", summary_all)
    R.write_report("e1", ctx.tag, {"overview": R.e1_summary_markdown(RESULTS_DIR / "e1")})
    return summary_all


# ----------------------------------------------------------------------------
# E2  single-site interventions
# ----------------------------------------------------------------------------

def _pair_examples(ctx: Ctx, split: str, max_per_bench: int) -> list[D.Example]:
    pairs = D.counterfactual_pairs(ctx.split(split),
                                   max_bbq_pairs_per_cluster=ctx.cfg["e2"].get("max_bbq_pairs_per_cluster", 8),
                                   seed=ctx.seed)
    rng = random.Random(ctx.seed)
    rng.shuffle(pairs)
    per: dict[str, int] = {}
    chosen: list[D.Example] = []
    seen: set[str] = set()
    for p in pairs:
        if per.get(p.benchmark, 0) >= max_per_bench:
            continue
        per[p.benchmark] = per.get(p.benchmark, 0) + 1
        for e in (p.a, p.b):
            if e.uid not in seen:
                seen.add(e.uid)
                chosen.append(e)
    return chosen


def run_e2(ctx: Ctx) -> dict[str, Any]:
    a, q = ctx.model(), ctx.quantizer
    e2 = ctx.cfg["e2"]
    bits = int(e2["bits"])
    sel = _pair_examples(ctx, "selection", 20 if ctx.args.quick else e2["max_pairs_per_benchmark"])
    fin = _pair_examples(ctx, "final", 20 if ctx.args.quick else e2["max_pairs_per_benchmark"])
    log(f"E2: {len(sel)} selection examples, {len(fin)} final examples")
    bs = ctx.batch_size()
    dense_sel = T.capture_hidden(a, sel, bs)
    dense_fin = T.capture_hidden(a, fin, bs)
    dense_scores_fin = E.score_candidates(a, fin, norm=ctx.norm, batch_size=bs)
    sites: list[list[str]]
    if e2["granularity"] == "layer":
        sites = [[c.id for c in a.components_of(li)] for li in range(a.n_layers)]
    else:
        sites = [[c.id] for c in a.components]
    if ctx.args.quick:
        sites = sites[: max(6, len(sites) // 8)]
    per_site: dict[str, Any] = {}
    heat_flip: dict[str, dict[str, float]] = {}
    heat_V: dict[str, dict[str, float]] = {}
    short_wiki = ctx.wiki()[:60_000]

    def measure(ids: list[str]) -> None:
        """Quantize one site at 4 bits, trace it, score the held-out split, restore."""
        sid = ids[0] if len(ids) == 1 else f"L{a.component_by_id[ids[0]].layer}.all"
        t0 = time.time()
        pm = PrecisionMap.single_site(a, ids, bits)
        q.apply_rtn(pm, record_stats=False)
        try:
            qc_sel = T.capture_hidden(a, sel, bs)
            qc_fin = T.capture_hidden(a, fin, bs)
            q_scores_fin = E.score_candidates(a, fin, norm=ctx.norm, batch_size=bs)
            ppl = E.perplexity(a, short_wiki, max_tokens=512 if ctx.args.quick else 2048)["ppl"]
        finally:
            q.restore()
        cmp_sel = T.compare_captures(dense_sel, qc_sel, [e.label for e in sel], keep_examples=False)
        cmp_fin = T.compare_captures(dense_fin, qc_fin, [e.label for e in fin], keep_examples=False)
        tr = T.TraceResult(component_id=sid, bits=bits, method="rtn", n_examples=len(sel), **cmp_sel)
        flips = E.flip_table(dense_scores_fin, q_scores_fin)
        harm = {b: E.harm_summary(dense_scores_fin, q_scores_fin, b, min_n=5 if ctx.args.quick else 20)
                for b in ("bbq", "winobias")}
        per_site[sid] = {
            "components": ids, "granularity": "component" if len(ids) == 1 else "layer",
            "features_selection": T.propagation_features(tr),
            "trace_selection": {k: cmp_sel[k] for k in
                                ("V", "rho", "logit_linf", "flip_rate", "margin_dense", "lemma_violations")},
            "observed_final": {"first_token_flip_rate": cmp_fin["flip_rate"], "full_score_flips": flips,
                               "harm": harm, "lemma_violations": cmp_fin["lemma_violations"]},
            "ppl_utility": ppl, "seconds": time.time() - t0}
        heat_flip[sid] = cmp_fin["flip_rate"]
        heat_V[sid] = {g: v[-1] for g, v in cmp_sel["V"].items()}
        worst = max(cmp_fin["flip_rate"].values()) if cmp_fin["flip_rate"] else 0.0
        log(f"  {sid}: ppl={ppl:.2f} worst-group flip={worst:.3f} "
            f"viol={cmp_fin['lemma_violations']} ({time.time() - t0:.0f}s)")
        write_json(ctx.out / "e2_sites.json", per_site)

    log(f"E2 stage 1: {len(sites)} sites at {e2['granularity']} granularity")
    for si, ids in enumerate(sites):
        measure(ids)

    # Stage 2: per-component pass inside the layers that look worst at stage 1.
    k_ref = 1 if ctx.args.quick else int(e2.get("refine_top_layers", 0) or 0)
    if e2["granularity"] == "layer" and k_ref:
        ranked = _e2_prediction(per_site)["site_scores"]
        order = sorted(ranked, key=lambda s: -ranked[s]["propagation_margin"])[:k_ref]
        layers = [a.component_by_id[per_site[s]["components"][0]].layer for s in order]
        log(f"E2 stage 2: per-component pass over layers {layers}")
        for li in layers:
            for c in a.components_of(li):
                measure([c.id])

    pred = _e2_prediction({k: v for k, v in per_site.items() if v["granularity"] == e2["granularity"]})
    write_json(ctx.out / "e2_prediction.json", pred)
    pred_all = _e2_prediction(per_site)
    write_json(ctx.out / "e2_prediction_all_sites.json", pred_all)
    R.plot_layer_group_heatmap(heat_flip, ctx.out / "fig_heatmap_flip.png",
                               f"{ctx.tag}: first-token flip rate (final)")
    R.plot_layer_group_heatmap(heat_V, ctx.out / "fig_heatmap_V.png",
                               f"{ctx.tag}: V_final (selection)", "relative energy")
    rows = [{"site": s, **{f"pred_{k}": v for k, v in pred_all["site_scores"][s].items()},
             "flip_rate": pred_all["observed"][s]} for s in pred_all["observed"]]
    for k in ("propagation_margin", "ppl_regret", "margin_only"):
        R.plot_predicted_vs_observed(rows, ctx.out / f"fig_pred_{k}.png", f"pred_{k}")
    return {"n_sites": len(per_site), "stage1": len(sites), "prediction": pred["spearman"],
            "prediction_all_sites": pred_all["spearman"]}


def _e2_prediction(per_site: dict[str, Any]) -> dict[str, Any]:
    """Three predictors of held-out harmful flips: (i) propagation + dense margin,
    (ii) global PPL regret only, (iii) dense margin only. Spearman on the site ranking."""
    from scipy.stats import spearmanr
    ppls = [v["ppl_utility"] for v in per_site.values()]
    base_ppl = min(ppls)
    site_scores: dict[str, dict[str, float]] = {}
    observed: dict[str, float] = {}
    for sid, v in per_site.items():
        f = v["features_selection"]
        groups = list(f)
        Vf = np.mean([f[g]["V_final"] for g in groups])
        linf = np.mean([f[g]["logit_linf"] for g in groups])
        margin = np.mean([abs(f[g]["margin_dense"]) for g in groups])
        # lemma-inspired score: expected flips when 2*eps_x exceeds |m|
        site_scores[sid] = {"propagation_margin": float(2 * linf / max(margin, 1e-6) + Vf),
                            "ppl_regret": float(v["ppl_utility"] - base_ppl),
                            "margin_only": float(1.0 / max(margin, 1e-6))}
        ff = v["observed_final"]["full_score_flips"]
        observed[sid] = float(ff.get("harmful_flip_rate", ff.get("flip_rate", 0.0)))
    sp = {}
    obs = np.array([observed[s] for s in observed])
    for k in ("propagation_margin", "ppl_regret", "margin_only"):
        x = np.array([site_scores[s][k] for s in observed])
        sp[k] = float(spearmanr(x, obs).correlation) if len(obs) > 2 else float("nan")
    return {"site_scores": site_scores, "observed": observed, "spearman": sp}


# ----------------------------------------------------------------------------
# E3  calibration composition
# ----------------------------------------------------------------------------

def run_e3(ctx: Ctx) -> dict[str, Any]:
    a, q = ctx.model(), ctx.quantizer
    c = ctx.cfg["calibration"]
    seeds = c["seeds"][:2] if ctx.args.quick else (
        c.get("seeds_budget", c["seeds"]) if ctx.profile == "budget" else c["seeds"])
    dense = ctx.evaluate_state("selection", with_ppl=False)["scored"]
    mid = a.n_layers // 2
    pairs = D.counterfactual_pairs(ctx.split("selection"))
    results: dict[str, Any] = {}
    for kind in c["kinds"]:
        for s in seeds:
            key = f"{kind}_seed{s}"
            man = ctx.manifest("e3", method="gptq", kind=kind, calib_seed=s)
            try:
                calib = ctx.calibration(kind, s)
                man.calibration_hash = calib.hash
                pm = PrecisionMap.uniform(a, 4)
                batches = calib.batches(a.tokenizer.pad_token_id, ctx.device)
                q.apply_gptq(pm, batches, progress=None)
                ev = ctx.evaluate_state("selection")
                hc = _harm_and_ci(ctx, dense, ev["scored"])
                pg = E.pair_gap_summary(dense, ev["scored"], pairs, ctx.norm)
                pg.pop("rows", None)
                q.restore()
                mom = T.residual_moments(a, q, pm, ctx.split("selection")[:200], mid, method="gptq", calib_batches=batches,
                                        batch_size=ctx.batch_size())
                results[key] = {"kind": kind, "seed": s, "composition": calib.composition, "metrics": ev["metrics"],
                                "ppl": ev["ppl"], **hc, "pair_gap": pg, "residual_moments": mom, "manifest": man.finish()}
                append_jsonl(ctx.out / f"records_{key}.jsonl", _records(ev["scored"], man, key, pm.hash()))
            except Exception as ex:
                log(f"FAILED {key}: {ex}")
                results[key] = {"kind": kind, "seed": s, "manifest": man.finish("failed", traceback.format_exc())}
            finally:
                q.restore()
            write_json(ctx.out / "e3_results.json", results)
    # seed-averaged summary per kind
    summ = {}
    for kind in c["kinds"]:
        rows = [v for v in results.values() if v.get("kind") == kind and "harm" in v]
        if rows:
            summ[kind] = {"n_seeds": len(rows),
                          "H_bbq_mean": float(np.mean([r["harm"].get("bbq", {}).get("H_worst_added_harm", np.nan) for r in rows])),
                          "A_bbq_mean": float(np.mean([r["harm"].get("bbq", {}).get("A_disparity_increase", np.nan) for r in rows])),
                          "ppl_mean": float(np.mean([r["ppl"]["ppl"] for r in rows])),
                          "gap_change_mean": float(np.mean([r["pair_gap"].get("mean_abs_gap_change", np.nan) for r in rows]))}
    write_json(ctx.out / "e3_summary.json", summ)
    return summ


# ----------------------------------------------------------------------------
# E4  restoration and allocation
# ----------------------------------------------------------------------------

def _score_fn_factory(ctx: Ctx, dense_sel: list[E.ScoredExample], pairs, split: str = "selection",
                      examples=None, ppl_max_tokens: int | None = None):
    a, q = ctx.model(), ctx.quantizer

    def score(pm: PrecisionMap) -> dict[str, Any]:
        q.apply_rtn(pm, record_stats=False)
        try:
            ev = ctx.evaluate_state(split, with_ppl=True, examples=examples, ppl_max_tokens=ppl_max_tokens)
        finally:
            q.restore()
        sc = ev["scored"]
        harm = E.harm_summary(dense_sel, sc, "bbq", min_n=5 if ctx.args.quick else 20)
        pg = E.pair_gap_summary(dense_sel, sc, pairs, ctx.norm)
        bbq_dis = ev["metrics"].get("bbq", {}).get("disambig", {})
        return {"H": harm.get("H_worst_added_harm", 0.0) or 0.0, "A": harm.get("A_disparity_increase", 0.0) or 0.0,
                "gap": pg.get("mean_abs_gap_change", 0.0) or 0.0, "acc": bbq_dis.get("acc"), "ppl": ev["ppl"]["ppl"],
                "bias_score": bbq_dis.get("bias_score"), "wb_gap": ev["metrics"].get("winobias", {}).get("gap_pro_minus_anti")}
    return score


def run_e4(ctx: Ctx) -> dict[str, Any]:
    a, q = ctx.model(), ctx.quantizer
    e4 = ctx.cfg["e4"]
    obj = A.Objective(**e4["objective"], **{"max_acc_loss": e4["gates"]["max_acc_loss"],
                                            "max_rel_ppl_increase": e4["gates"]["max_rel_ppl_increase"]})
    sites_file = RESULTS_DIR / "e2" / ctx.tag / "e2_sites.json"
    if not sites_file.exists():
        raise FileNotFoundError(f"run E2 first: {sites_file}")
    from .common import read_json
    sites = read_json(sites_file)
    # bias-sensitivity ranking (propagation + margin) and utility ranking (ppl regret)
    pred = _e2_prediction(sites)
    bias_rank = sorted(pred["site_scores"], key=lambda s: -pred["site_scores"][s]["propagation_margin"])
    util_rank = sorted(pred["site_scores"], key=lambda s: -pred["site_scores"][s]["ppl_regret"])
    margin_rank = sorted(pred["site_scores"], key=lambda s: -pred["site_scores"][s]["margin_only"])
    k = min(e4["shortlist_k"], len(bias_rank))
    rk = min(e4["restore_k"], len(bias_rank))
    # The allocator runs hundreds of evaluations, so search scores a fixed
    # cluster-level subsample; the frozen comparison uses the full final split.
    n_sub = 200 if ctx.args.quick else int(e4.get("score_subsample") or 0)
    sub_sel = ctx.subsample("selection", n_sub)
    ppl_search = 512 if ctx.args.quick else int(e4.get("search_ppl_tokens", 2048))
    log(f"E4 search set: {len(sub_sel)} of {len(ctx.split('selection'))} selection examples, "
        f"ppl on {ppl_search} tokens")
    dense_sel = ctx.evaluate_state("selection", with_ppl=False, examples=sub_sel)["scored"]
    dense_fin = ctx.evaluate_state("final", with_ppl=False)["scored"]
    pairs_sel = D.counterfactual_pairs(sub_sel)
    score_sel = _score_fn_factory(ctx, dense_sel, pairs_sel, "selection", examples=sub_sel,
                                  ppl_max_tokens=ppl_search)
    score_fin = _score_fn_factory(ctx, dense_fin, D.counterfactual_pairs(ctx.split("final")), "final")
    budget = int(A.budget_for_uniform(a, 4, ctx.fmt) * (1 + e4["budget_slack"]))
    results: dict[str, Any] = {"budget_bytes": budget, "shortlist": bias_rank[:k]}
    # --- restoration controls from uniform 4-bit (evaluate on final once each)
    rng = random.Random(ctx.seed)
    controls = {"uniform4": [], "restore_predicted": bias_rank[:rk],
                "restore_random": rng.sample(list(pred["site_scores"]), rk),
                "restore_utility": util_rank[:rk]}
    results["restoration"] = {}

    def _expand(site_ids):  # sites may be layers ("L3.all") when E2 ran at layer granularity
        out = []
        for s in site_ids:
            out.extend(sites[s]["components"] if s in sites else [s])
        return out

    for name, ids in controls.items():
        pm = PrecisionMap.uniform(a, 4)
        for s in _expand(ids):
            pm[s] = DENSE_BITS
        r = A.evaluate_fixed_map(a, score_fin, pm, name, obj, budget, ctx.fmt)
        results["restoration"][name] = r.as_dict()
        log(f"restoration {name}: J={r.objective:.4f} H={r.metrics['H']:.4f} acc={r.metrics['acc']} bytes={r.bytes_total}")
        write_json(ctx.out / "e4_results.json", results)
    # --- allocator on selection set, then frozen evaluation on final
    util_ref = score_sel(A.ranked_map(a, {c: pred["site_scores"][s]["ppl_regret"] for s in util_rank for c in _expand([s])}, budget, 4, 8, ctx.fmt))
    alloc = A.GreedyAllocator(a, score_sel, budget, obj, shortlist=_expand(bias_rank[:k]),
                              lower_pool=_expand(list(reversed(util_rank))), fmt=ctx.fmt,
                              max_steps=(6 if ctx.args.quick else e4["max_steps"]),
                              max_evals=(30 if ctx.args.quick else e4["max_evals"]), utility_reference=util_ref)
    greedy = alloc.run(PrecisionMap.uniform(a, 4), "greedy_bias_aware")
    results["allocation_search"] = greedy.as_dict()
    final_maps = {
        "greedy_bias_aware": PrecisionMap(greedy.precision_map),
        "uniform4": PrecisionMap.uniform(a, 4),
        "random": A.random_map(a, budget, ctx.seed, 4, 8, ctx.fmt),
        "ppl_only": A.ranked_map(a, {c: pred["site_scores"][s]["ppl_regret"] for s in util_rank for c in _expand([s])}, budget, 4, 8, ctx.fmt),
        "margin_only": A.ranked_map(a, {c: pred["site_scores"][s]["margin_only"] for s in margin_rank for c in _expand([s])}, budget, 4, 8, ctx.fmt),
    }
    results["final"] = {}
    for name, pm in final_maps.items():
        r = A.evaluate_fixed_map(a, score_fin, pm, name, obj, budget, ctx.fmt)
        results["final"][name] = r.as_dict()
        log(f"final {name}: J={r.objective:.4f} H={r.metrics['H']:.4f} A={r.metrics['A']:.4f} acc={r.metrics['acc']} ppl={r.metrics['ppl']:.2f} bytes={r.bytes_total}")
        write_json(ctx.out / "e4_results.json", results)
    R.plot_frontier(list(results["final"].values()), ctx.out / "fig_frontier.png")
    return {k: {"J": v["objective"], "H": v["metrics"]["H"], "acc": v["metrics"]["acc"]} for k, v in results["final"].items()}


# ----------------------------------------------------------------------------
# E5  transfer and pruning bridge
# ----------------------------------------------------------------------------

def run_e5(ctx: Ctx) -> dict[str, Any]:
    a, q = ctx.model(), ctx.quantizer
    from .common import read_json
    out: dict[str, Any] = {}
    e4_file = RESULTS_DIR / "e4" / ctx.tag / "e4_results.json"
    dense_fin = ctx.evaluate_state("final", with_ppl=False)["scored"]
    # transfer: allocation frozen on BBQ selection -> WinoBias and Discrim-Eval final
    if e4_file.exists():
        e4 = read_json(e4_file)
        for name in ("greedy_bias_aware", "uniform4", "ppl_only"):
            pm = PrecisionMap(e4["final"][name]["precision_map"])
            q.apply_rtn(pm, record_stats=False)
            try:
                ev = ctx.evaluate_state("final", with_ppl=False)
            finally:
                q.restore()
            m = ev["metrics"]
            out[f"transfer_{name}"] = {"winobias": m.get("winobias"), "discrim_eval": {k: v for k, v in m.get("discrim_eval", {}).items() if k != "by_group"},
                                      "harm_winobias": E.harm_summary(dense_fin, ev["scored"], "winobias", 5 if ctx.args.quick else 20)}
    else:
        out["transfer"] = "E4 results missing; run E4 first"
    # pruning bridge: verified per-row Wanda at 50% with generic calibration
    calib = ctx.calibration("generic", 0)
    batches = calib.batches(a.tokenizer.pad_token_id, ctx.device)
    for c in a.components:
        q._stash(c)
    fn = collect_feature_norms(a, batches)
    rep = wanda_per_row(a, fn, ctx.cfg["e5"]["wanda_sparsity"])
    try:
        ev = ctx.evaluate_state("final")
        out["wanda_per_row"] = {"sparsity": rep["sparsity_achieved"], "ppl": ev["ppl"], "metrics": ev["metrics"],
                                **_harm_and_ci(ctx, dense_fin, ev["scored"])}
    finally:
        q.restore()
    out["debias_sparsegpt"] = ("external comparator: run the authors' code at matched sparsity on the same final "
                               "split and place per-example records in results/bias/e5/<tag>/records_debias_sparsegpt.jsonl")
    write_json(ctx.out / "e5_results.json", out)
    return {k: (v if not isinstance(v, dict) else list(v)[:5]) for k, v in out.items()}



# ----------------------------------------------------------------------------
# E6  bridge to the prior living-inference study
# ----------------------------------------------------------------------------

def run_e6(ctx: Ctx) -> dict[str, Any]:
    """Does the prior study's utility-only layer sensitivity predict bias harm?

    Reads the saved Lyapunov rho and single-layer perplexity regret from
    Codes/living-inference (read-only) and correlates them against this study's
    group-conditioned harm at the same sites. A null result here is the evidence
    that the bias-aware diagnostic is not redundant with the utility one.
    """
    from .common import read_json
    li_key = ctx.mcfg.get("li_key")
    sites_file = RESULTS_DIR / "e2" / ctx.tag / "e2_sites.json"
    if not sites_file.exists():
        raise FileNotFoundError(f"run E2 first: {sites_file}")
    per_site = read_json(sites_file)
    man = ctx.manifest("e6", method="analysis", li_key=li_key)
    out = BR.align_with_sites(li_key, per_site)
    out["interpretation"] = BR.interpret(out)
    out["prior_dense_ppl"] = BR.prior_dense_ppl()
    out["provenance_note"] = (
        "Prior perplexities come from runs with different token budgets and chunking "
        "conventions and are not comparable with this study's re-measured baselines; "
        "they document coverage only.")
    out["manifest"] = man.finish()
    write_json(ctx.out / "e6_bridge.json", out)
    log("E6: " + out["interpretation"])
    rows = [r for r in out["rows"] if r.get("prior_rho") is not None
            and r.get("harmful_flip_rate") is not None]
    if len(rows) >= 4:
        R.plot_predicted_vs_observed(
            [{"prior_rho": r["prior_rho"], "harmful_flip_rate": r["harmful_flip_rate"]} for r in rows],
            ctx.out / "fig_prior_rho_vs_harm.png", "prior_rho", "harmful_flip_rate")
    return {"correlations": out["correlations"], "interpretation": out["interpretation"]}


# ----------------------------------------------------------------------------
# E7  comparator methods from the closest literature, at matched budget
# ----------------------------------------------------------------------------

def run_e7(ctx: Ctx) -> dict[str, Any]:
    """Fair-GPTQ, Critical Weight Protection, SparseGPT and Debias-SparseGPT.

    All quantization comparators run at the same bit width and are byte-accounted
    the same way, so the fairness-utility-memory comparison is like for like. The
    two pruning methods are reported separately because sparsity in a dense
    tensor is not a memory saving.
    """
    a, q = ctx.model(), ctx.quantizer
    cfg = ctx.cfg.get("e7", {})
    bits = int(cfg.get("bits", 4))
    sparsity = float(cfg.get("sparsity", 0.5))
    lam = float(cfg.get("lam", 1.0))
    max_pairs = 32 if ctx.args.quick else int(cfg.get("max_pairs", 256))
    protect = float(cfg.get("protect_frac", 0.01))

    pairs_cal = D.counterfactual_pairs(
        ctx.split("calibration"),
        max_bbq_pairs_per_cluster=ctx.cfg["e2"].get("max_bbq_pairs_per_cluster", 8),
        seed=ctx.seed)
    if not pairs_cal:
        raise RuntimeError("no counterfactual pairs in the calibration split")
    calib = ctx.calibration("generic", 0)
    batches = calib.batches(a.tokenizer.pad_token_id, ctx.device)
    cmp_ = B.Comparators(a, q, ctx.fmt, ctx.batch_size())

    q.restore()
    dense = ctx.evaluate_state("final", with_ppl=False)["scored"]
    pairs_fin = D.counterfactual_pairs(ctx.split("final"), seed=ctx.seed)

    def evaluate(tag: str, br: B.BaselineResult) -> dict[str, Any]:
        ev = ctx.evaluate_state("final")
        hc = _harm_and_ci(ctx, dense, ev["scored"])
        pg = E.pair_gap_summary(dense, ev["scored"], pairs_fin, ctx.norm)
        pg.pop("rows", None)
        row = {"name": br.name, "paper": br.paper, "reimplementation": br.reimplementation,
               "assumptions": br.assumptions, "bytes_total": br.bytes_total,
               "bytes_MiB": round(br.bytes_total / 2 ** 20, 2) if br.bytes_total else None,
               "ppl": ev["ppl"], "metrics": ev["metrics"], "pair_gap": pg, **hc,
               "extra": br.extra}
        append_jsonl(ctx.out / f"records_{tag}.jsonl",
                     _records(ev["scored"], man, tag, br.precision_map and
                              PrecisionMap(br.precision_map).hash() or ""))
        return row

    results: dict[str, Any] = {}
    plan = [
        ("gptq_uniform", "quant", lambda: None),
        ("fair_gptq", "quant", lambda: cmp_.fair_gptq(bits, batches, pairs_cal, lam, max_pairs)),
        ("cwp", "quant", lambda: cmp_.critical_weight_protection(bits, pairs_cal, protect, max_pairs)),
        ("sparsegpt", "prune", lambda: cmp_.sparsegpt(sparsity, batches)),
        ("debias_sparsegpt", "prune",
         lambda: cmp_.debias_sparsegpt(sparsity, batches, pairs_cal, lam, max_pairs)),
    ]
    for tag, kind, build in plan:
        man = ctx.manifest("e7", method=tag)
        log(f"=== E7 {tag} ===")
        try:
            if tag == "gptq_uniform":
                pm = PrecisionMap.uniform(a, bits)
                q.apply_gptq(pm, batches, progress=None)
                br = B.BaselineResult(name=f"gptq{bits}", paper="Frantar et al., ICLR 2023",
                                      reimplementation=True,
                                      assumptions="uniform GPTQ; the no-mitigation control",
                                      precision_map=dict(pm),
                                      bytes_total=account_bytes(a, pm, ctx.fmt).total)
            else:
                br = build()
            results[tag] = evaluate(tag, br)
            results[tag]["manifest"] = man.finish()
            log(f"  {tag}: ppl={results[tag]['ppl']['ppl']:.2f} "
                f"H={results[tag].get('harm', {}).get('bbq', {}).get('H_worst_added_harm')}")
        except Exception as ex:
            log(f"FAILED {tag}: {ex}")
            results[tag] = {"manifest": man.finish("failed", traceback.format_exc())}
        finally:
            q.restore()
        write_json(ctx.out / "e7_results.json", results)

    # Fold in this study's allocator, if E4 has produced one, for the headline table.
    e4f = RESULTS_DIR / "e4" / ctx.tag / "e4_results.json"
    if e4f.exists():
        from .common import read_json
        e4 = read_json(e4f)
        for k in ("greedy_bias_aware", "uniform4"):
            if k in e4.get("final", {}):
                results[f"ours_{k}"] = {
                    "name": k, "paper": "this study", "reimplementation": False,
                    "bytes_total": e4["final"][k]["bytes_total"],
                    "bytes_MiB": round(e4["final"][k]["bytes_total"] / 2 ** 20, 2),
                    "metrics_from_e4": e4["final"][k]["metrics"]}
    write_json(ctx.out / "e7_results.json", results)
    R.write_report("e7", ctx.tag, {"comparators": R.comparator_table(results)})
    return {k: (v.get("bytes_MiB"), v.get("ppl", {}).get("ppl")) for k, v in results.items()}


# ----------------------------------------------------------------------------

EXPS = {"e0": run_e0, "e1": run_e1, "e2": run_e2, "e3": run_e3, "e4": run_e4,
        "e5": run_e5, "e6": run_e6, "e7": run_e7}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, choices=sorted(EXPS))
    p.add_argument("--model", required=True, help="key in configs/models.yaml (M1..M5)")
    p.add_argument("--device", default="auto")
    p.add_argument("--device-map", action="store_true", help="use accelerate device_map=auto on cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="small subsets for CPU dev runs")
    p.add_argument("--smoke", action="store_true",
                   help="two samples per category: proves every code path runs, produces no science")
    p.add_argument("--attn", default=None,
                   help="attn_implementation: flash_attention_2 | sdpa | eager (falls back if unavailable)")
    p.add_argument("--profile", choices=["budget", "full"], default=None,
                   help="sampling profile from configs/experiments.yaml (default: budget)")
    p.add_argument("--configs", nargs="*", help="E1: subset of config names")
    args = p.parse_args(argv)
    ctx = Ctx(args)
    log(f"=== {args.exp.upper()} model={args.model} tag={ctx.tag} device={ctx.device} "
        f"seed={ctx.seed} profile={ctx.profile} ===")
    t0 = time.time()
    res = EXPS[args.exp](ctx)
    log(f"=== done in {time.time() - t0:.0f}s -> {ctx.out} ===")
    return res


if __name__ == "__main__":
    main()
