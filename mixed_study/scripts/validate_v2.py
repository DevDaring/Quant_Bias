#!/usr/bin/env python3
"""Content-correctness check for mixed_study results/v2 (runs on the GPU host).

Opens every artifact a completed stage should have produced and checks it is
complete, parses, and is numerically sane. Consistency checks cross-reference
stages: a B1 layer-panel must cover the prespecified layers; equal-cost
restoration must actually be equal-cost; legacy reproductions must record dtype.
Exit 0 unless a FAIL is found. Never modifies anything.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
STATE = V2 / "_logs" / "state.tsv"
TAGS = {"M1": "gpt2_small", "M3": "mistral_7b_v0_1", "M5": "qwen3_8b", "M2": "qwen3_5_2b",
        "M4": "llama_2_7b", "M6": "gpt2_medium", "M7": "lfm2_2.6b"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(p: Path):
    if not p.exists(): return None, f"missing {p.relative_to(ROOT)}"
    if p.stat().st_size == 0: return None, f"empty {p.relative_to(ROOT)}"
    try: return json.loads(p.read_text()), None
    except json.JSONDecodeError as e: return None, f"invalid JSON {p.relative_to(ROOT)}: {e}"


def fin(x, lo=-math.inf, hi=math.inf):
    return isinstance(x, (int, float)) and math.isfinite(x) and lo <= x <= hi


def parse_state():
    rows = {}
    if STATE.exists():
        for ln in STATE.read_text().splitlines():
            f = ln.split("\t")
            if len(f) >= 2: rows[f[0]] = f[1]
    return rows


def check_legacy(sub, out):
    d, err = load(V2 / "legacy" / sub / "legacy_reproduction.json")
    if err: out["fail"].append(f"legacy/{sub}: {err}"); return
    if "dtype" not in d: out["warn"].append(f"legacy/{sub}: dtype not recorded")
    if d.get("verdict", "").startswith("NOT"): out["info"].append(f"legacy/{sub}: {d['verdict']} (a finding, not an error)")
    for seed_rho in d.get("fresh_rho", []):
        if not all(fin(r, 0, 1e3) for r in seed_rho): out["fail"].append(f"legacy/{sub}: non-finite rho")
    out["ok"].append(f"legacy/{sub}")


def check_b1(sub, out, k_layers=8, granularity="layer"):
    d, err = load(V2 / "b1" / sub / f"b1_cells_{granularity}.json")
    if err: out["fail"].append(f"b1/{sub}: {err}"); return
    cells = d.get("cells", {})
    if not cells: out["fail"].append(f"b1/{sub}: no cells"); return
    n_bad = 0
    for k, c in cells.items():
        p = c.get("propagation", {})
        if "actual" not in p or "sign_reversed" not in p or not any(s.startswith("random_") for s in p):
            out["fail"].append(f"b1/{sub}/{k}: missing a residual source"); n_bad += 1; continue
        for s, v in p.items():
            if not (fin(v.get("local_abs"), 0) and fin(v.get("final_abs"), 0) and fin(v.get("amplification"), 0)):
                out["fail"].append(f"b1/{sub}/{k}/{s}: non-finite propagation"); n_bad += 1
            if s.startswith("random_") and v.get("cos_final_vs_actual") is not None and not fin(v["cos_final_vs_actual"], -1.0001, 1.0001):
                out["fail"].append(f"b1/{sub}/{k}/{s}: cosine outside [-1,1]"); n_bad += 1
        # sign-reversed must match actual's local magnitude (same residual, flipped)
        a, r = p["actual"], p["sign_reversed"]
        if a["local_abs"] > 0 and abs(a["local_abs"] - r["local_abs"]) > 1e-3 * a["local_abs"]:
            out["fail"].append(f"b1/{sub}/{k}: sign_reversed local norm != actual"); n_bad += 1
        # random sources must be norm-matched to actual
        for s, v in p.items():
            if s.startswith("random_") and a["local_abs"] > 0 and abs(v["local_abs"] - a["local_abs"]) > 1e-2 * a["local_abs"]:
                out["fail"].append(f"b1/{sub}/{k}/{s}: random not norm-matched ({v['local_abs']:.3e} vs {a['local_abs']:.3e})"); n_bad += 1
        b = c.get("behaviour", {})
        if not (fin(b.get("flip_rate"), 0, 1) and fin(b.get("harmful_rate"), 0, 1)):
            out["fail"].append(f"b1/{sub}/{k}: behaviour rates out of range"); n_bad += 1
    # coverage: full (non-quick) panel must have the prespecified layer count x sources
    if not sub.endswith("-quick"):
        layers = d.get("layers", []); srcs = d.get("sources", [])
        if d.get("granularity") == "layer":
            exp = len(layers) * len(srcs)
            if len(cells) != exp: out["fail"].append(f"b1/{sub}: {len(cells)} cells, expected {exp} (layers x sources)")
        if len(layers) < min(k_layers, 3): out["fail"].append(f"b1/{sub}: only {len(layers)} layers")
    if "summary" not in d: out["warn"].append(f"b1/{sub}: no summary (run may be mid-flight)")
    out["ok"].append(f"b1/{sub} ({len(cells)} cells, {n_bad} bad)")


def check_restore(sub, out):
    d, err = load(V2 / "restoration" / sub / "restoration.json")
    if err: out["fail"].append(f"restoration/{sub}: {err}"); return
    if d.get("equal_cost_ok") is False: out["fail"].append(f"restoration/{sub}: arms are NOT equal-cost")
    arms = d.get("arms", {})
    need = {"uniform_start", "predicted", "utility_matched"}
    if not need <= set(arms): out["fail"].append(f"restoration/{sub}: missing arms {need - set(arms)}")
    nr = sum(1 for a in arms if a.startswith("random_matched_"))
    if nr < 2: out["warn"].append(f"restoration/{sub}: only {nr} random schedules")
    for a, v in arms.items():
        h = (v.get("outcomes") or {}).get("task_damage_bbq_disambig", {}).get("harmful_per_dense_correct")
        if h is not None and not fin(h, 0, 1): out["fail"].append(f"restoration/{sub}/{a}: harmful rate out of range")
    out["ok"].append(f"restoration/{sub} ({len(arms)} arms, equal_cost={d.get('equal_cost_ok')})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    out = {"ok": [], "warn": [], "fail": [], "info": []}
    rows = parse_state()
    n_failed = sum(1 for v in rows.values() if v == "FAILED")
    for key, st in rows.items():
        if st != "ok": continue
        parts = key.split("/")            # mode/stage/model[/--flag/value...]
        if len(parts) < 3: continue
        mode, stage, model = parts[0], parts[1], parts[2]
        if mode == "smoke":
            continue   # two-sample wiring checks; not results
        tag = TAGS.get(model, model)
        extras = parts[3:]
        if stage == "legacy":
            sub = tag + (f"-{extras[1]}" if "--dtype" in extras else "")
            check_legacy(sub, out)
        elif stage == "b1":
            gran = extras[extras.index("--granularity") + 1] if "--granularity" in extras else "layer"
            sub = tag + ("-quick" if mode == "smoke" else "")
            check_b1(sub, out, granularity=gran)
        elif stage == "restore":
            check_restore(tag + ("-quick" if mode == "smoke" else ""), out)
    verdict = "FAIL" if out["fail"] else ("WARN" if out["warn"] else "PASS")
    if a.json:
        print(json.dumps({"verdict": verdict, "n_failed_in_state": n_failed, **out}, indent=2))
    else:
        log(f"VALIDATE_V2 {verdict}: {len(out['ok'])} ok, {len(out['warn'])} warn, {len(out['fail'])} fail, {n_failed} FAILED-in-state")
        for m in out["fail"]: log(f"  FAIL {m}")
        for m in out["warn"]: log(f"  WARN {m}")
        for m in out["info"]: log(f"  INFO {m}")
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
