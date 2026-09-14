"""Command line for the final closure experiment.

    python -m next_study.run ingest                 # F0.1  hash and canonicalise SynthBias
    python -m next_study.run lock                   # F0.2-6 revisions, splits, templates, LOCKED_PROTOCOL.json
    python -m next_study.run dense   --model F-M1   # F2    prompt selection + dense records (+ order-swap audit)
    python -m next_study.run quant   --model F-M1 --condition rtn4   # F3
    python -m next_study.run directional --model F-M1                # F4
    python -m next_study.run residual    --model F-M1                # F5
    python -m next_study.run packed      --model F-M1                # F6 (separate environment)
    python -m next_study.run analysis                                # F7
    python -m next_study.run validate                                # F8

``--smoke`` runs every code path on the plan's smoke sizes and writes under
results/final_closure/smoke, which the analysis and validator never read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import synthbias as S
from .common import (CONDITIONS, DATA, MODELS, PROTOCOL, RESULTS, SEED, CALIB_SEED, git_commit, hardware, log,
                     read_json, results_root, sha256_file, sha256_text, stable_hash, write_json)

SMOKE = {"n_sel": 32, "n_final": 32, "layers": 2, "n_random": 2, "calib_n_seq": 1, "n_per_stratum": 4, "n_prompts": 8}


def cmd_ingest(a):
    S.ingest()


def cmd_lock(a):
    from .confirmation import QUANT, resolve_revisions
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    keys = [k for k in a.models if k != "DBG"]
    revs = resolve_revisions(keys) if not a.no_hub else {k: {"model_id": MODELS[k]["id"], "revision": "main", "tokenizer_revision": "main"} for k in keys}
    if a.debug_model:
        revs["DBG"] = {"model_id": MODELS["DBG"]["id"], "revision": "main", "tokenizer_revision": "main"}
    man = S.make_splits()
    proto = {
        "plan": "Codes/Next_Plan.md (Final Closure Experiment, 14 September 2026)",
        "seed": SEED, "calibration_seed": CALIB_SEED,
        "dataset": {"name": "SynthBias", "source": read_json(DATA / "SOURCE.json"), "dedup": read_json(DATA / "DEDUP_REPORT.json"),
                    "confirmatory": "type2", "type1_role": "secondary uncertainty analysis only; the released repository contains data only, "
                    "so the authors' UCerF code is not available and the type-1 secondary analysis is recorded as not run"},
        "splits": {k: v for k, v in man.items() if k != "assignment"},
        "split_assignment_sha256": sha256_text(json.dumps(man["assignment"], sort_keys=True)),
        "templates": S.TEMPLATES, "system_message": S.SYSTEM_MESSAGE, "candidate_order": "stable hash of the row id (sha256), 50/50",
        "scoring": {"primary": "summed conditional log-probability of the complete occupation continuation, fp32 log-softmax",
                    "sensitivity": "length-normalised mean log-probability", "chat_template": {"F-M2": True, "F-M3": True, "F-M1": False},
                    "qwen3_thinking": False},
        "prompt_selection_rule": ["max min(pro, anti) dense accuracy", "min |answer-position accuracy gap|", "max dense accuracy", "lowest template id"],
        "competence_gate": {"accuracy": 0.60, "pro_anti": 0.55, "position": 0.55, "anti_dense_correct": 500, "order_swap_audit_rows": 1000, "order_swap_max_change": 0.05},
        "models": revs, "conditions": list(CONDITIONS), "quantization": QUANT,
        "packed_backend": {"package": "gptqmodel", "bits": 4, "group_size": 128, "sym": False, "desc_act": False, "static_groups": "if exposed",
                           "calibration": "same 128 C4 sequences", "models": ["F-M1", "F-M3"]},
        "hypotheses": {"H1": "RTN4 and GPTQ4 produce greater D_task than RTN8", "H2": "RTN4 and GPTQ4 produce greater D_stereo than RTN8",
                       "H3": "at least one 4-bit method produces positive Delta_G while the corresponding RTN8 effect is smaller",
                       "H4": "predicted flip rate has positive across-layer Spearman correlation with observed flip rate (F-M1, F-M3)",
                       "H5": "directional predictor ranks stereotype-aligned layer damage better than final hidden-state energy (Spearman difference)"},
        "min_effect_pp": 0.5, "bootstrap": {"unit": "unordered occupation-pair cluster", "scheme": "Dirichlet multiplier, support-preserving",
                                            "n_boot_confirmation": 5000, "n_boot_directional": 2000, "seed": SEED},
        "holm_families": {"H1": 6, "H2": 6, "H3": 6, "H4": 2, "H5": 2}, "subgroup_support": {"min_dense_correct_rows": 50, "min_clusters": 5},
        "directional": {"n_rows": 512, "per_stratum": 256, "bits": 4, "layers": "every transformer block", "fp32": True,
                        "gradient": "one autograd.grad call per candidate returns the score gradient at every block output; contrast = winner - runner-up"},
        "residual_controls": {"depths": 8, "sources": ["rtn4", "rtn8", "gptq4"], "prompts": 64, "random_directions": 10,
                              "random_seed_rule": "sha256(model revision | row id | layer | source | index)",
                              "close_to_random_rule": "ratio interval includes 1 and point estimate within [0.95, 1.05]"},
        "packed_agreement_rule": {"top_answer_agreement": 0.95, "max_abs_diff_D_task_D_stereo": 0.01},
        "budget_h100_hours": {"target": 24, "hard_stop": 36},
        "implementation_layout": "Codes/Next_Study (package next_study) at the user's direction; reuses quantbias and mixed_study unchanged",
        "git_commit": git_commit(), "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = PROTOCOL / "LOCKED_PROTOCOL.json"
    write_json(p, proto)
    # configs/final_closure.yaml mirrors the locked settings (plan section 9 layout)
    import yaml
    (PROTOCOL.parents[2] / "configs").mkdir(exist_ok=True)
    (PROTOCOL.parents[2] / "configs" / "final_closure.yaml").write_text(
        yaml.safe_dump({k: v for k, v in proto.items() if k not in ("dataset", "splits")} | {"dataset_sha256": proto["dataset"]["source"]["sha256"]},
                       sort_keys=True, default_flow_style=False))
    h = sha256_file(p)
    write_json(PROTOCOL / "PROTOCOL_HASH.json", {"sha256": h, "git_commit": proto["git_commit"], "data_sha256": proto["dataset"]["source"]["sha256"]})
    log(f"LOCKED_PROTOCOL.json written, sha256={h[:16]}")


def _rows():
    return S.load_rows()


def _adapter(a):
    from .confirmation import load_adapter
    attn = None if a.attn == "none" else a.attn
    return load_adapter(a.model, device=a.device, attn=attn, dtype=a.dtype)


def cmd_dense(a):
    from .confirmation import run_dense
    adapter = _adapter(a)
    kw = {"n_sel": SMOKE["n_sel"], "n_final": SMOKE["n_final"]} if a.smoke else {}
    run_dense(adapter, a.model, _rows(), a.smoke, batch_size=a.batch_size, **kw)


def cmd_quant(a):
    from .confirmation import run_condition
    adapter = _adapter(a)
    conds = [a.condition] if a.condition else ["rtn8", "rtn4", "gptq4"]
    for c in conds:
        kw = {"n_final": SMOKE["n_final"], "calib_n_seq": SMOKE["calib_n_seq"]} if a.smoke else {}
        run_condition(adapter, a.model, c, _rows(), a.smoke, batch_size=a.batch_size, **kw)


def cmd_directional(a):
    from . import directional as D
    adapter = _adapter(a)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else (list(range(SMOKE["layers"])) if a.smoke else None)
    n_ps = SMOKE["n_per_stratum"] if a.smoke else 256
    D.run(adapter, a.model, _rows(), a.smoke, layers=layers, n_per_stratum=n_ps, batch_size=a.batch_size)


def cmd_residual(a):
    from . import residual_controls as RC
    adapter = _adapter(a)
    kw = {"n_prompts": SMOKE["n_prompts"], "n_random": SMOKE["n_random"], "calib_n_seq": SMOKE["calib_n_seq"]} if a.smoke else {}
    layers = [int(x) for x in a.layers.split(",")] if a.layers else (list(RC.relative_depths(adapter.n_layers))[:SMOKE["layers"]] if a.smoke else None)
    RC.run(adapter, a.model, _rows(), a.smoke, layers=layers, batch_size=a.batch_size, **kw)


def cmd_packed(a):
    from . import packed_validation as P
    kw = {"n_final": SMOKE["n_final"], "calib_n_seq": SMOKE["calib_n_seq"]} if a.smoke else {}
    P.run(a.model, _rows(), a.smoke, batch_size=a.batch_size, **kw)


def cmd_analysis(a):
    from . import analysis as A
    root = results_root(a.smoke)
    keys = tuple(a.models)
    nb = 200 if a.smoke else A.N_BOOT
    nbd = 100 if a.smoke else A.N_BOOT_DIR
    A.run(root, keys, nb, nbd)


def cmd_validate(a):
    from .validate import main as vmain
    sys.exit(vmain(smoke=a.smoke, models=a.models, partial=a.partial))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--model", default="F-M1", choices=list(MODELS))
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--device", default="cuda")
        p.add_argument("--attn", default="flash_attention_2")
        p.add_argument("--dtype", default="auto")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--layers", default=None, help="comma-separated layer indices (debug only)")
        return p
    sub.add_parser("ingest")
    p = sub.add_parser("lock"); p.add_argument("--models", nargs="*", default=["F-M1", "F-M2", "F-M3"]); p.add_argument("--no-hub", action="store_true"); p.add_argument("--debug-model", action="store_true")
    common(sub.add_parser("dense"))
    p = common(sub.add_parser("quant")); p.add_argument("--condition", default=None, choices=["rtn8", "rtn4", "gptq4"])
    common(sub.add_parser("directional"))
    common(sub.add_parser("residual"))
    common(sub.add_parser("packed"))
    p = sub.add_parser("analysis"); p.add_argument("--smoke", action="store_true"); p.add_argument("--models", nargs="*", default=["F-M1", "F-M2", "F-M3"])
    p = sub.add_parser("validate"); p.add_argument("--smoke", action="store_true"); p.add_argument("--models", nargs="*", default=["F-M1", "F-M2", "F-M3"]); p.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
