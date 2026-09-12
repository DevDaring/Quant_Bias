"""Paths, provenance and versioned output for the integrated study."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                                  # Codes/mixed_study
CODES = PROJECT.parent                                 # Codes
QUANT_BIAS = CODES / "quant-bias"
LIVING = CODES / "living-inference"
QB_RESULTS = QUANT_BIAS / "results"                    # source records: READ ONLY
LI_RESULTS = LIVING / "python" / "results"
LI_VENDORED = QUANT_BIAS / "provenance" / "living_inference_results"
RESULTS = PROJECT / "results" / "v2"                   # every corrected output lands here
CONFIGS = PROJECT / "configs"

# make the sibling package importable without installing it
if str(QUANT_BIAS) not in sys.path:
    sys.path.insert(0, str(QUANT_BIAS))

TAGS = {"M1": "gpt2_small", "M6": "gpt2_medium", "M7": "lfm2_2.6b", "M2": "qwen3_5_2b",
        "M3": "mistral_7b_v0_1", "M4": "llama_2_7b", "M5": "qwen3_8b"}
KEYS = {v: k for k, v in TAGS.items()}
PRIMARY = ("M3", "M5")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_env() -> dict[str, str]:
    from quantbias.common import load_env as _le
    return _le()


def li_results_root() -> Path:
    return LI_RESULTS if LI_RESULTS.exists() else LI_VENDORED


def read_json(p: Path | str) -> Any:
    return json.loads(Path(p).read_text())


def write_json(p: Path | str, obj: Any) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default))
    return p


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_jsonl(p: Path | str) -> list[dict]:
    out = []
    with Path(p).open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def source_tags(exp: str, include_dev: bool = False) -> list[str]:
    """Model tags that have a real (non-quick/smoke) result dir for ``exp``."""
    d = QB_RESULTS / exp
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if not p.is_dir():
            continue
        if not include_dev and (p.name.endswith("-quick") or p.name.endswith("-smoke")):
            continue
        out.append(p.name)
    return out


def provenance(**extra) -> dict[str, Any]:
    """Stamp every v2 output with where its inputs came from."""
    import subprocess
    def sha(cwd):
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "codes_git": sha(CODES), "living_inference_git": sha(LIVING),
            "source_results": str(QB_RESULTS), "li_results": str(li_results_root()),
            "note": "source result files are read only; this is a corrected reanalysis", **extra}
