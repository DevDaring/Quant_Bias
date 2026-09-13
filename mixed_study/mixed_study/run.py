"""Driver for the integrated study.

CPU, runs now on the saved records:
    python -m mixed_study.run audit            # P0: corrected reanalysis -> results/v2/audit/AUDIT.md
    python -m mixed_study.run ladder           # §6 nested predictors, leave-layer-out -> LADDER.md
    python -m mixed_study.run legacy --model M1  # §5.1 reproduce the legacy trace (GPT-2 fits on CPU)

GPU, the B1 panel and equal-cost restoration:
    python -m mixed_study.run legacy  --model M3 --device cuda
    python -m mixed_study.run b1      --model M3 --device cuda [--granularity layer|component]
    python -m mixed_study.run restore --model M3 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import CONFIGS, RESULTS, TAGS, load_env, log, read_json, write_json, provenance


def _ctx(model_key: str, device: str, attn: str | None, dtype: str | None = None):
    import yaml
    from quantbias.model_adapters import ModelAdapter
    from quantbias.quantization import Quantizer
    load_env()
    models = yaml.safe_load((Path(__file__).resolve().parents[2] / "quant-bias" / "configs" / "models.yaml").read_text())
    m = models[model_key]
    a = ModelAdapter.from_pretrained(m["id"], revision=m.get("revision", "main"),
                                     dtype=dtype or m.get("dtype", "auto"),
                                     device=device, attn_implementation=attn)
    q = Quantizer(a)
    return m, a, q


def _examples_and_calib(a, cfg, quick: bool):
    import yaml
    from quantbias import data as D
    qcfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "quant-bias" / "configs" / "experiments.yaml").read_text())
    prof = qcfg["sampling"]["smoke" if quick else "budget"]
    lim = {"bbq_per_cluster_per_cell": prof["bbq_per_cluster_per_cell"], "discrim_ages": prof["discrim_ages"],
           "bbq_answer_format": "text", "discrim_config": "explicit"}
    if quick:
        lim.update({"bbq_per_category": 2, "discrim_questions": 1})
    ex = D.load_benchmarks(qcfg["seed"], ("bbq", "winobias", "discrim_eval"), tuple(qcfg["split_fractions"]), lim)
    sel = D.by_split(ex, "selection"); fin = D.by_split(ex, "final")
    n = 16 if quick else int(cfg["b1"]["answer_examples"])
    calib = D.build_calibration(a.tokenizer, "generic", 0, 4 if quick else 32, 128 if quick else 512, examples=ex,
                                c4_docs=100 if quick else 2000)
    return sel[:n], fin[:n], calib.batches(a.tokenizer.pad_token_id, a.device)


def cmd_audit(args):
    from . import audit
    audit.main(quick=args.quick)


def cmd_ladder(args):
    from . import group_prediction
    group_prediction.main()


def cmd_legacy(args):
    from . import legacy_trace as LT
    from quantbias.data import wikitext_test_text
    m, a, q = _ctx(args.model, args.device, args.attn, args.dtype)
    text = wikitext_test_text(max_chars=60_000)
    res = LT.compare_to_saved(a, m["li_key"], text, seq_len=512)
    # The legacy 7B profiles were measured in fp16 (colab_unified_eval.py); the
    # study's default is bf16. Keep dtype in the output path so both conditions
    # are stored and the numerical-precision hypothesis can be tested directly.
    sub = TAGS[args.model] + (f"-{args.dtype}" if args.dtype else "")
    out = RESULTS / "legacy" / sub; out.mkdir(parents=True, exist_ok=True)
    res["dtype"] = args.dtype or m.get("dtype", "auto")
    res["provenance"] = provenance(model=m["id"])
    write_json(out / "legacy_reproduction.json", res)
    log(f"{TAGS[args.model]}: {res['verdict']}  shape_spearman={res.get('shape_spearman')}  "
        f"within3sd={res.get('frac_saved_within_3sd_of_fresh')}  saved n={res.get('n_saved')} fresh n={res.get('n_fresh')}")


def cmd_b1(args):
    import yaml
    from . import matched_residuals as B
    cfg = yaml.safe_load((CONFIGS / "integrated_v2.yaml").read_text())
    m, a, q = _ctx(args.model, args.device, args.attn)
    sel, fin, calib = _examples_and_calib(a, cfg, args.quick)
    out = RESULTS / "b1" / (TAGS[args.model] + ("-quick" if args.quick else "")); out.mkdir(parents=True, exist_ok=True)
    b = cfg["b1"]
    res = B.run_b1(a, q, fin, calib, sources=tuple(b["sources"]) if not args.quick else ("rtn4", "wanda50"),
                   n_random=b["n_random_directions"], granularity=args.granularity,
                   k_layers=b["k_layers"] if not args.quick else 3, batch_size=b["propagation_batch"],
                   out_dir=out, tag=TAGS[args.model])
    log("summary: " + str({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                           for k, v in res["summary"].items()}))


def cmd_restore(args):
    import yaml
    from . import matched_restoration as MR, harm
    from quantbias.evaluate import score_candidates
    from .records import Row
    cfg = yaml.safe_load((CONFIGS / "integrated_v2.yaml").read_text())
    m, a, q = _ctx(args.model, args.device, args.attn)
    sel, fin, _ = _examples_and_calib(a, cfg, args.quick)
    tag = TAGS[args.model]
    # predicted sites from the saved E2 map (selection split); utility ranking from ppl regret
    sites = read_json(Path(__file__).resolve().parents[2] / "quant-bias" / "results" / "e2" / tag / "e2_sites.json")
    comp_sites = {s: v for s, v in sites.items() if v.get("granularity") == "component"}
    import numpy as np
    def vmean(v): f = v["features_selection"]; return float(np.mean([f[g]["V_final"] for g in f]))
    predicted = sorted(comp_sites, key=lambda s: -vmean(comp_sites[s]))
    utility = sorted(comp_sites, key=lambda s: -(comp_sites[s]["ppl_utility"] or 0))
    dense_rows = {s.uid: s for s in score_candidates(a, fin, batch_size=8, progress_every=0)}

    def to_row(s):  # ScoredExample -> records.Row
        mt = s.meta or {}; gf = s.group_fields or {}
        return Row(s.uid, s.benchmark, s.cluster_id, s.group, s.label, s.pred, s.correct, tuple(s.logprob_sum),
                   tuple(s.logprob_mean), s.margin, mt.get("context_condition"), mt.get("target_idx"),
                   mt.get("unknown_idx"), gf.get("category"), tuple(gf.get("answer_groups") or ()), gf, mt)
    D = {u: to_row(s) for u, s in dense_rows.items()}

    def evaluate(pm):
        C = {s.uid: to_row(s) for s in score_candidates(a, fin, batch_size=8, progress_every=0)}
        pairs = [(D[u], C[u]) for u in D if u in C]
        return {"task_damage_bbq_disambig": (lambda t: t.__dict__ | t.rates())(harm.task_damage(
                    [(d, c) for d, c in pairs if d.benchmark == "bbq" and d.context_condition == "disambig"])),
                "stereotype_damage": (lambda t: t.__dict__ | t.rates())(harm.stereotype_damage(pairs)),
                "group_disparity_bbq": harm.group_disparity(pairs)}
    out = RESULTS / "restoration" / (tag + ("-quick" if args.quick else "")); out.mkdir(parents=True, exist_ok=True)
    r = cfg["restoration"]
    res = MR.run(a, q, evaluate, predicted, utility, k=r["k_sites"] if not args.quick else 2,
                 n_random=r["n_random_schedules"] if not args.quick else 2, out_path=out / "restoration.json", tag=tag)
    log(f"equal_cost_ok={res['equal_cost_ok']}  summary={res['summary']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("audit", cmd_audit), ("ladder", cmd_ladder), ("legacy", cmd_legacy), ("b1", cmd_b1), ("restore", cmd_restore)):
        s = sub.add_parser(name); s.set_defaults(fn=fn)
        s.add_argument("--model", default="M1"); s.add_argument("--device", default="cpu")
        s.add_argument("--attn", default=None); s.add_argument("--quick", action="store_true")
        s.add_argument("--granularity", default="layer", choices=["layer", "component"])
        s.add_argument("--dtype", default=None, help="override model dtype, e.g. fp16 to match the legacy runs")
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
