"""F8: validation and closure (plan section 10, F8; section 15.2).

Every check is applicable-or-passing before FINAL_COMPLETE is written. A check that
fails prints its reason; the file is not written and the exit code is non-zero.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import synthbias as S
from .common import DATA, MODELS, PROTOCOL, RESULTS, SMOKE, log, read_json, read_jsonl, results_root, sha256_file, sha256_text, write_json

VALIDATOR_VERSION = "1.0"
QUANT_CONDS = ("rtn8", "rtn4", "gptq4")


def _finite(recs: list[dict]) -> bool:
    for r in recs:
        vals = [r["margin_sum"], r["margin_mean"]] + list(r["logprob_sum"]) + list(r["logprob_mean"])
        if any(not math.isfinite(float(v)) for v in vals):
            return False
    return True


PRESENCE = ("_records", "_complete", "_status", "analysis_present", "stage_status_present", "every_cell_terminal",
            "_all_layers", "_all_cells", "_covers_final_split")


def main(smoke: bool = False, models: Sequence[str] = ("F-M1", "F-M2", "F-M3"), partial: bool = False) -> int:
    """``partial``: while the run is in progress, only content checks on existing
    artefacts count; presence/completeness checks are reported but not failed."""
    root = results_root(smoke)
    checks: dict[str, Any] = {}
    fails: list[str] = []
    def check(name: str, ok: bool, detail: Any = None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok and not (partial and (any(name.endswith(x) or x in name for x in PRESENCE) or detail == "missing")):
            fails.append(name)
    # --- protocol and data hashes
    proto_p = PROTOCOL / "LOCKED_PROTOCOL.json"
    check("protocol_present", proto_p.exists())
    if proto_p.exists():
        proto = read_json(proto_p)
        ph = read_json(PROTOCOL / "PROTOCOL_HASH.json")
        check("protocol_hash_matches", sha256_file(proto_p) == ph["sha256"], ph["sha256"][:16])
        check("data_hash_matches", sha256_file(DATA / "synthbias_data.csv") == proto["dataset"]["source"]["sha256"])
        man = read_json(DATA / "split_manifest.json")
        check("split_assignment_hash_matches", sha256_text(json.dumps(man["assignment"], sort_keys=True)) == proto["split_assignment_sha256"])
        check("no_cluster_leakage", man["disjoint"] and man["counterparts_together"], man["cluster_overlap"])
        rows = S.load_rows()
        check("unique_row_ids", len({r.uid for r in rows}) == len(rows))
        rev = proto["models"]
    else:
        proto, rev, rows = {}, {}, S.load_rows()
    # --- confirmation records
    dense_uids: dict[str, set] = {}
    for k in models:
        conf = root / "confirmation" / k
        dm = conf / "dense_COMPLETE.json"
        check(f"{k}/dense_records", dm.exists() and (conf / "final_dense.jsonl").exists())
        if not dm.exists():
            continue
        d = read_jsonl(conf / "final_dense.jsonl")
        dense_uids[k] = {r["uid"] for r in d}
        check(f"{k}/dense_unique_rows", len(dense_uids[k]) == len(d))
        check(f"{k}/dense_finite", _finite(d))
        check(f"{k}/dense_template_locked", read_json(dm).get("template") in S.TEMPLATES)
        check(f"{k}/manifest_hardware", all(x in read_json(dm) for x in ("hardware", "wall_seconds", "started_utc", "finished_utc")))
        if not smoke:
            final_uids = {r.uid for r in S.type2(rows, "final")}
            check(f"{k}/dense_covers_final_split", dense_uids[k] == final_uids, {"dense": len(dense_uids[k]), "final": len(final_uids)})
        for c in QUANT_CONDS:
            cm = conf / f"{c}_COMPLETE.json"
            if not cm.exists():
                check(f"{k}/{c}_records", False, "missing"); continue
            m = read_json(cm)
            q = read_jsonl(conf / f"final_{c}.jsonl")
            check(f"{k}/{c}_row_alignment", {r["uid"] for r in q} == dense_uids[k])
            check(f"{k}/{c}_finite", _finite(q))
            check(f"{k}/{c}_grid_valid", bool((m.get("quant_stats") or {}).get("grid_ok_all", False)), m.get("quant_stats"))
            check(f"{k}/{c}_restored_exact", m.get("restored_exact") in (True, None) and (m.get("roundtrip") or {}).get("ok", True), m.get("roundtrip"))
            if c == "gptq4" and not smoke:
                check(f"{k}/gptq4_calibration_hash_recorded", bool(m.get("calibration_hash")))
    # --- directional and residual controls for mechanistic, eligible models
    audit = root / "audit"
    status = read_json(audit / "STAGE_STATUS.json") if (audit / "STAGE_STATUS.json").exists() else {}
    check("stage_status_present", bool(status))
    for k in [m for m in models if MODELS[m]["mechanistic"]]:
        st = status.get(f"{k}/dense")
        if st == "COMPETENCE_INELIGIBLE":
            continue
        d = root / "directional" / k
        dm = d / "directional_COMPLETE.json"
        check(f"{k}/directional_complete", dm.exists())
        if dm.exists():
            m = read_json(dm)
            layers = m["layers"]
            files = sorted(int(p.stem.split("_")[1]) for p in d.glob("layer_*.json"))
            check(f"{k}/directional_all_layers", files == sorted(layers), {"expected": len(layers), "found": len(files)})
            if files:
                first = read_json(d / f"layer_{files[0]:02d}.json")
                n = len(first["rows"])
                check(f"{k}/directional_sample_size", n == (m["n_examples"]) and (smoke or n == 512), n)
                check(f"{k}/directional_restored_exact", all(read_json(d / f"layer_{li:02d}.json")["summary"]["restored_exact"] for li in files))
                check(f"{k}/directional_finite", all(math.isfinite(r["pred_delta"]) and math.isfinite(r["actual_delta"]) for li in files for r in read_json(d / f"layer_{li:02d}.json")["rows"]))
        r = root / "residual_controls" / k
        rm = r / "residual_COMPLETE.json"
        check(f"{k}/residual_complete", rm.exists())
        if rm.exists():
            m = read_json(rm)
            cells = sorted(p.name for p in r.glob("cell_L*_*.json"))
            check(f"{k}/residual_all_cells", len(cells) == len(m["layers"]) * len(m["sources"]), {"expected": len(m["layers"]) * len(m["sources"]), "found": len(cells)})
            if not smoke:
                check(f"{k}/residual_panel_size", len(m["layers"]) == 8 and m["n_prompts"] == 64 and m["n_random"] == 10, {"layers": len(m["layers"]), "prompts": m["n_prompts"], "random": m["n_random"]})
        pk = root / "packed" / k / "packed_COMPLETE.json"
        check(f"{k}/packed_status", pk.exists() and read_json(pk).get("status") in ("VALID_COMPLETE", "BACKEND_UNSUPPORTED"), read_json(pk).get("status") if pk.exists() else "missing")
        if pk.exists() and read_json(pk).get("status") == "VALID_COMPLETE":
            q = read_jsonl(root / "packed" / k / "final_packed_gptq4.jsonl")
            check(f"{k}/packed_row_alignment", {x["uid"] for x in q} == dense_uids.get(k, set()))
            check(f"{k}/packed_backend_manifest", (root / "packed" / k / "packed_backend_manifest.json").exists())
    # --- analysis reproducibility: bootstraps retained, numbers reproducible from records
    ca = audit / "confirmation_analysis.json"
    check("analysis_present", ca.exists() and (root / "FINAL_FINDINGS.md").exists())
    if ca.exists():
        conf = read_json(ca)
        n_boot = conf["hypotheses"]["n_boot"]
        allok = True
        for k, m in conf["models"].items():
            for c, v in m.get("conditions", {}).items():
                for name in ("D_task", "D_stereo", "Delta_G"):
                    allok &= v[name]["n_boot_valid"] == n_boot == v[name]["n_boot"]
        check("all_bootstrap_draws_retained", allok, n_boot)
        # recompute one point estimate per model/condition directly from the records
        from .analysis import align, _outcome_fns, load_records
        repro = True
        for k in models:
            rec = load_records(root, k)
            m = conf["models"].get(k, {})
            for c in [x for x in QUANT_CONDS if x in rec and x in m.get("conditions", {})]:
                d, q = align(rec["dense"], rec[c])
                fns, _ = _outcome_fns(d, q)
                for name, fn in fns.items():
                    repro &= abs(fn(np.ones(len(d))) - m["conditions"][c][name]["estimate"]) < 1e-12
        check("estimates_reproduce_from_records", repro)
        check("no_smoke_records_in_analysis", "smoke" not in str(root) or smoke)
        check("every_cell_terminal", all(v != "INVALID_RETRY_REQUIRED" for v in status.values()), [k for k, v in status.items() if v == "INVALID_RETRY_REQUIRED"])
    verdict = "PASS" if not fails else "FAIL"
    out = {"validator_version": VALIDATOR_VERSION, "verdict": verdict, "smoke": smoke, "checks": checks, "failed": fails,
           "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    audit.mkdir(exist_ok=True)
    write_json(audit / "VALIDATION.json", out)
    log(f"VALIDATE_FINAL_CLOSURE {verdict}: {sum(v['ok'] for v in checks.values())} ok, {len(fails)} failed"
        + (f", {sum(not v['ok'] for v in checks.values()) - len(fails)} pending (partial)" if partial else ""))
    for f in fails:
        log(f"  FAIL {f}: {checks[f]['detail']}")
    if verdict == "PASS" and not smoke and not partial:
        ph = read_json(PROTOCOL / "PROTOCOL_HASH.json")
        index = {}
        for p in sorted(root.rglob("*.json")):
            if "checkpoint" in p.parts or p.name == "FINAL_COMPLETE":
                continue
            index[str(p.relative_to(root))] = sha256_file(p)
        for p in sorted(root.rglob("*.jsonl")):
            index[str(p.relative_to(root))] = sha256_file(p)
        write_json(audit / "RESULT_INDEX.json", index)
        text = ("protocol_sha256=%s\nresult_index_sha256=%s\nvalidator_version=%s\ncompleted_utc=%s\n\n"
                "The experimental programme is closed. No further empirical results are required or authorised by this plan.\n"
                % (ph["sha256"], sha256_file(audit / "RESULT_INDEX.json"), VALIDATOR_VERSION, out["checked_utc"]))
        (root / "FINAL_COMPLETE").write_text(text)
        log("FINAL_COMPLETE written")
    return 0 if verdict == "PASS" else 1
