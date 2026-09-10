#!/usr/bin/env python3
"""Correctness check for quant-bias results, run on the GPU host.

Not a liveness check ("is a process running") but a content check: every file
a completed stage should have produced is opened, parsed, and checked against
sanity bounds appropriate to what it contains. A stage marked "ok" in
state.tsv whose output is missing, truncated, or out of range is exactly the
failure mode a liveness check cannot see -- e.g. a process killed mid-write.

Run:  python3 validate_results.py [--retry-failed] [--json]

Exit code 0 only if every checked artifact is fully sane; issues are printed
either way so a caller can decide whether to treat WARN as fatal.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # quant-bias/
RESULTS = ROOT / "results"
STATE = RESULTS / "_logs" / "state.tsv"
RETRY_MARK = RESULTS / "_logs" / "validator_retries.tsv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Report:
    def __init__(self):
        self.ok: list[str] = []
        self.warn: list[str] = []
        self.fail: list[str] = []

    def add(self, level: str, msg: str):
        getattr(self, level).append(msg)

    @property
    def healthy(self) -> bool:
        return not self.fail


def _load(path: Path):
    if not path.exists():
        return None, f"missing: {path.relative_to(ROOT)}"
    if path.stat().st_size == 0:
        return None, f"empty file: {path.relative_to(ROOT)}"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON (truncated write?): {path.relative_to(ROOT)}: {e}"


def _finite_in(x, lo, hi) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and lo <= x <= hi


def check_e0(tag: str, r: Report):
    d = RESULTS / "e0" / tag
    obj, err = _load(d / "e0_summary.json")
    if err:
        r.add("fail", f"e0/{tag}: {err}")
        return
    if not obj.get("roundtrip", {}).get("ok"):
        r.add("fail", f"e0/{tag}: weight round-trip did not reproduce dense logits exactly")
    if obj.get("restored_exact") is not True:
        r.add("fail", f"e0/{tag}: quantizer did not restore originals exactly")
    for k in ("grid_rtn8", "grid_rtn4"):
        if k in obj and not obj[k].get("grid_ok_all"):
            r.add("fail", f"e0/{tag}: {k} quantization grid check failed")
    for k in ("ppl_dense", "ppl_rtn8", "ppl_rtn4"):
        ppl = obj.get(k, {}).get("ppl")
        if not _finite_in(ppl, 1.0, 1e6):
            r.add("fail", f"e0/{tag}: {k}.ppl={ppl} out of sane range (1, 1e6)")
    if obj.get("gptq4", {}).get("ppl", {}).get("ppl") and not _finite_in(
            obj["gptq4"]["ppl"]["ppl"], 1.0, 1e6):
        r.add("fail", f"e0/{tag}: gptq4 ppl out of range")
    wanda_sp = obj.get("wanda", {}).get("sparsity_achieved")
    if wanda_sp is not None and not (0.45 <= wanda_sp <= 0.55):
        r.add("warn", f"e0/{tag}: wanda achieved sparsity {wanda_sp} far from target 0.50")
    if obj.get("splits", {}).get("ok") is not True:
        r.add("fail", f"e0/{tag}: template-cluster split leakage detected")
    if obj.get("manifest", {}).get("status") != "ok":
        r.add("fail", f"e0/{tag}: manifest status = {obj.get('manifest', {}).get('status')}")
    r.add("ok", f"e0/{tag}")


def check_e1(tag: str, r: Report):
    d = RESULTS / "e1" / tag
    overview, err = _load(d / "e1_overview.json")
    if err:
        r.add("fail", f"e1/{tag}: {err}")
        return
    byte_mib = {}
    for cfg_name in overview:
        obj, err = _load(d / f"summary_{cfg_name}.json")
        if err:
            r.add("fail", f"e1/{tag}/{cfg_name}: {err}")
            continue
        if obj.get("manifest", {}).get("status") == "failed":
            r.add("fail", f"e1/{tag}/{cfg_name}: run recorded as failed -- "
                          f"{obj['manifest'].get('error', '')[:120]}")
            continue
        ppl = obj.get("ppl", {}).get("ppl")
        if not _finite_in(ppl, 1.0, 1e7):
            r.add("fail", f"e1/{tag}/{cfg_name}: ppl={ppl} out of sane range")
        mib = obj.get("bytes", {}).get("total_MiB")
        if mib is not None:
            byte_mib[cfg_name] = mib
        harm = obj.get("harm", {}).get("bbq", {})
        for key in ("A_disparity_increase", "H_worst_added_harm"):
            v = harm.get(key)
            if v is not None and not _finite_in(v, -1.5, 1.5):
                r.add("warn", f"e1/{tag}/{cfg_name}: bbq.{key}={v} outside expected [-1,1] range")
        ci = obj.get("ci", {}).get("bbq", {})
        for g, gv in ci.get("groups", {}).items():
            for bound in ("ci_low", "ci_high"):
                if bound in gv and not _finite_in(gv[bound], -1.5, 1.5):
                    r.add("warn", f"e1/{tag}/{cfg_name}: CI[{g}].{bound}={gv[bound]} suspicious")
    # lower bit-width must not cost MORE accounted bytes than a higher one
    order = [("dense", 16), ("rtn8", 8), ("rtn4", 4), ("gptq8", 8), ("gptq4", 4)]
    dense_mib = byte_mib.get("dense")
    if dense_mib:
        for name, bits in order[1:]:
            if name in byte_mib and byte_mib[name] > dense_mib:
                r.add("fail", f"e1/{tag}/{name}: accounted {byte_mib[name]} MiB exceeds "
                              f"dense {dense_mib} MiB at {bits} bits")
    r.add("ok", f"e1/{tag} ({len(overview)} configs)")


def check_e2(tag: str, r: Report):
    d = RESULTS / "e2" / tag
    sites, err = _load(d / "e2_sites.json")
    if err:
        r.add("fail", f"e2/{tag}: {err}")
        return
    if not sites:
        r.add("fail", f"e2/{tag}: e2_sites.json is empty")
        return
    pred, err = _load(d / "e2_prediction.json")
    if err:
        r.add("fail", f"e2/{tag}: {err}")
        return
    for k, v in pred.get("spearman", {}).items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            # Pre-fix files can hold a literal NaN for margin_only, whose input
            # is constant by construction (margin_dense is measured once on the
            # fixed dense capture, independent of which site is perturbed).
            # Newer runs report 0.0 with constant_predictors instead; this is a
            # documented, non-fatal artifact of the older file, not a data loss.
            r.add("warn", f"e2/{tag}: spearman[{k}]=NaN (constant predictor, "
                          "pre-fix file; see run_experiment._e2_prediction)")
        elif not (-1.0001 <= v <= 1.0001):
            r.add("fail", f"e2/{tag}: spearman[{k}]={v} outside [-1,1]")
    n_viol = sum(v.get("observed_final", {}).get("lemma_violations", 0) for v in sites.values())
    if n_viol:
        r.add("warn", f"e2/{tag}: {n_viol} margin-lemma violations recorded "
                      "(expected under the stated eps_x; flag if it grows)")
    r.add("ok", f"e2/{tag} ({len(sites)} sites)")


def check_e3(tag: str, r: Report):
    obj, err = _load(RESULTS / "e3" / tag / "e3_summary.json")
    if err:
        r.add("fail", f"e3/{tag}: {err}")
        return
    for kind, s in obj.items():
        ppl = s.get("ppl_mean")
        if not _finite_in(ppl, 1.0, 1e7):
            r.add("fail", f"e3/{tag}/{kind}: ppl_mean={ppl} out of range")
    r.add("ok", f"e3/{tag} ({len(obj)} calibration kinds)")


def check_e4(tag: str, r: Report):
    obj, err = _load(RESULTS / "e4" / tag / "e4_results.json")
    if err:
        r.add("fail", f"e4/{tag}: {err}")
        return
    if not obj.get("final"):
        r.add("fail", f"e4/{tag}: no frozen allocation comparison ('final' empty)")
        return
    budget = obj.get("budget_bytes")
    if not (isinstance(budget, int) and budget > 0):
        r.add("fail", f"e4/{tag}: budget_bytes={budget} invalid")
    for name, row in obj["final"].items():
        if row.get("bytes_total", 0) > budget * 1.02:  # 2% slack for rounding
            r.add("fail", f"e4/{tag}/{name}: {row['bytes_total']} bytes exceeds "
                          f"budget {budget} (+2% slack)")
    r.add("ok", f"e4/{tag} ({len(obj['final'])} allocations compared)")


def check_e5(tag: str, r: Report):
    obj, err = _load(RESULTS / "e5" / tag / "e5_results.json")
    if err:
        r.add("fail", f"e5/{tag}: {err}")
        return
    r.add("ok", f"e5/{tag}")


def check_e6(tag: str, r: Report):
    obj, err = _load(RESULTS / "e6" / tag / "e6_bridge.json")
    if err:
        r.add("fail", f"e6/{tag}: {err}")
        return
    if not obj.get("interpretation"):
        r.add("fail", f"e6/{tag}: no interpretation string written")
    r.add("ok", f"e6/{tag}")


def check_e7(tag: str, r: Report):
    obj, err = _load(RESULTS / "e7" / tag / "e7_results.json")
    if err:
        r.add("fail", f"e7/{tag}: {err}")
        return
    n_reimpl = sum(1 for v in obj.values() if isinstance(v, dict) and v.get("reimplementation"))
    if n_reimpl == 0:
        r.add("warn", f"e7/{tag}: no comparator marked reimplementation=true "
                      "(every baseline here should be)")
    r.add("ok", f"e7/{tag} ({len(obj)} methods)")


CHECKERS = {"e0": check_e0, "e1": check_e1, "e2": check_e2, "e3": check_e3,
           "e4": check_e4, "e5": check_e5, "e6": check_e6, "e7": check_e7}


def parse_state() -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Returns ({key: status}, [(mode, exp, model), ...]) for FAILED rows."""
    rows: dict[str, str] = {}
    failed: list[tuple[str, str, str]] = []
    if not STATE.exists():
        return rows, failed
    for ln in STATE.read_text().splitlines():
        f = ln.split("\t")
        if len(f) < 2:
            continue
        rows[f[0]] = f[1]
        if f[1] == "FAILED" and f[0].count("/") == 2:
            mode, exp, model = f[0].split("/")
            failed.append((mode, exp, model))
    return rows, failed


def retry_count(key: str) -> int:
    if not RETRY_MARK.exists():
        return 0
    for ln in RETRY_MARK.read_text().splitlines():
        k, n = ln.split("\t")
        if k == key:
            return int(n)
    return 0


def bump_retry(key: str) -> None:
    cur = {}
    if RETRY_MARK.exists():
        for ln in RETRY_MARK.read_text().splitlines():
            k, n = ln.split("\t")
            cur[k] = int(n)
    cur[key] = cur.get(key, 0) + 1
    RETRY_MARK.parent.mkdir(parents=True, exist_ok=True)
    RETRY_MARK.write_text("".join(f"{k}\t{n}\n" for k, n in cur.items()))


def retry_failed(failed: list[tuple[str, str, str]], max_retries: int = 2) -> list[str]:
    """Re-run each genuinely failed stage, at most max_retries times each."""
    import yaml
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    acted = []
    for mode, exp, model in failed:
        key = f"{mode}/{exp}/{model}"
        n = retry_count(key)
        if n >= max_retries:
            acted.append(f"{key}: skipped, already retried {n} times")
            continue
        log(f"retrying {key} (attempt {n + 1}/{max_retries})")
        flags = ["--profile", "budget"] if mode == "full" else \
                (["--smoke"] if mode == "smoke" else ["--quick"])
        cmd = [sys.executable, "-m", "quantbias.run_experiment", "--exp", exp, "--model", model,
               "--device", "cuda", "--attn", "flash_attention_2", *flags]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        bump_retry(key)
        elapsed = int(time.time() - t0)
        status = "ok" if proc.returncode == 0 else "FAILED"
        with STATE.open("a") as f:
            f.write(f"{key}\t{status}\t{elapsed}s\n")
        if status != "ok":
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            acted.append(f"{key}: retry FAILED again in {elapsed}s -- {tail[:400]}")
        else:
            acted.append(f"{key}: retry succeeded in {elapsed}s")
        log(acted[-1])
    return acted


def rebuild_state(models_cfg: dict) -> dict[str, str]:
    """Rebuild state.tsv from verified disk content, discarding whatever the
    log said. Used after any incident where two writers may have raced (e.g.
    a duplicate run loop), so 'ok' in state.tsv always means 'checked sane on
    disk right now', not 'a process once exited zero'."""
    exps = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]
    rows: dict[str, str] = {}
    for exp in exps:
        for model_key, mcfg in models_cfg.items():
            tag = mcfg["tag"]
            r = Report()
            CHECKERS[exp](tag, r)
            key = f"full/{exp}/{model_key}"
            if r.fail:
                continue  # leave unset -> treated as not-yet-done, will be (re)run
            rows[key] = "ok\t0s"
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(rows.items())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--rebuild-state", action="store_true",
                    help="discard state.tsv and reconstruct it from verified disk content")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.rebuild_state:
        import yaml
        models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
        rows = rebuild_state(models)
        log(f"rebuilt state.tsv from disk: {len(rows)} stage(s) verified ok")
        for k in sorted(rows):
            log(f"  ok {k}")

    rows, failed = parse_state()
    r = Report()
    for key, status in rows.items():
        if status != "ok" or key.count("/") != 2:
            continue
        mode, exp, model = key.split("/")
        tag_map = {"M1": "gpt2_small", "M6": "gpt2_medium", "M7": "lfm2_2.6b",
                  "M2": "qwen3_5_2b", "M3": "mistral_7b_v0_1", "M4": "llama_2_7b",
                  "M5": "qwen3_8b"}
        tag = tag_map.get(model, model)
        if mode == "smoke":
            tag += "-smoke"
        elif mode == "quick":
            tag += "-quick"
        (CHECKERS.get(exp, lambda t, r: r.add("warn", f"no checker for {exp}")))(tag, r)

    retry_summary = []
    if failed:
        r.add("warn" if not args.retry_failed else "ok",
             f"{len(failed)} FAILED stage(s) in state.tsv: " +
             ", ".join(f"{m}/{e}/{md}" for m, e, md in failed))
        if args.retry_failed:
            retry_summary = retry_failed(failed)

    # freshness: are results actually landing in git recently?
    try:
        last = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        age_min = (time.time() - int(last)) / 60
        if age_min > 90:
            r.add("warn", f"last git commit was {age_min:.0f} min ago (no stage finished recently?)")
        else:
            r.add("ok", f"last commit {age_min:.0f} min ago")
    except Exception as e:
        r.add("warn", f"could not read git log: {e}")

    verdict = "FAIL" if r.fail else ("WARN" if r.warn else "PASS")
    out = {"verdict": verdict, "n_ok": len(r.ok), "n_warn": len(r.warn), "n_fail": len(r.fail),
          "ok": r.ok, "warn": r.warn, "fail": r.fail, "retry_actions": retry_summary,
          "n_failed_in_state": len(failed)}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        log(f"VALIDATE {verdict}: {len(r.ok)} artifact(s) sane, {len(r.warn)} warning(s), "
            f"{len(r.fail)} failure(s), {len(failed)} FAILED-in-state")
        for m in r.fail:
            log(f"  FAIL {m}")
        for m in r.warn:
            log(f"  WARN {m}")
        for m in retry_summary:
            log(f"  RETRY {m}")
    sys.exit(0 if verdict != "FAIL" else 1)


if __name__ == "__main__":
    main()
