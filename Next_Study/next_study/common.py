"""Paths, hashing, atomic output and provenance for the final closure experiment.

The package lives beside ``quant-bias`` and ``mixed_study`` and imports both without
installation. Every scientific output goes under ``results/final_closure``; nothing
under ``quant-bias/results`` or ``mixed_study/results`` is ever written.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                                   # Codes/Next_Study
CODES = PROJECT.parent                                  # Codes
QUANT_BIAS = CODES / "quant-bias"
MIXED = CODES / "mixed_study"
DATA = PROJECT / "data" / "synthbias"
CONFIGS = PROJECT / "configs"
RESULTS = PROJECT / "results" / "final_closure"
PROTOCOL = RESULTS / "protocol"
SMOKE = RESULTS / "smoke"

for p in (QUANT_BIAS, MIXED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEED = 20260914                 # every new random choice in this campaign
CALIB_SEED = 20260908           # the completed study's calibration seed (frozen)

MODELS = {
    "F-M1": {"id": "mistralai/Mistral-7B-v0.1", "tag": "mistral_7b_v0_1", "chat": False,
             "role": "direct connection to the completed mechanism study", "mechanistic": True},
    "F-M2": {"id": "mistralai/Mistral-7B-Instruct-v0.3", "tag": "mistral_7b_instruct_v0_3", "chat": True,
             "role": "instruction-tuned deployment boundary", "mechanistic": False},
    "F-M3": {"id": "Qwen/Qwen3-8B", "tag": "qwen3_8b", "chat": True,
             "role": "architecture-family and transfer test", "mechanistic": True},
    # debug alias, never part of the frozen scope: exercises every code path on a CPU
    "DBG": {"id": "openai-community/gpt2", "tag": "gpt2_small_debug", "chat": False,
            "role": "code-path check only", "mechanistic": True},
}
CONDITIONS = ("dense", "rtn8", "rtn4", "gptq4")
SYSTEM_MESSAGE = "Answer with exactly one of the supplied occupations."


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_hash(obj: Any, n: int = 16) -> str:
    """SHA256 of a canonical JSON rendering; never Python's process-salted hash()."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))[:n]


def seed_from(*parts: Any) -> int:
    """Counter-based seed: a 63-bit integer derived from the named parts."""
    return int(sha256_text("|".join(str(p) for p in parts))[:15], 16)


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(p: Path | str) -> Any:
    return json.loads(Path(p).read_text())


def write_json(p: Path | str, obj: Any) -> Path:
    """Atomic: write to a temporary file in the same directory, then rename."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default))
    os.replace(tmp, p)
    return p


def write_jsonl(p: Path | str, rows: list[dict]) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=_default) + "\n")
    os.replace(tmp, p)
    return p


def read_jsonl(p: Path | str) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def git_commit(path: Path = PROJECT) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def hardware() -> dict[str, Any]:
    out: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                           "hostname_class": os.environ.get("HOSTNAME_CLASS", platform.node()[:12])}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            try:
                out["driver"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().split()[0]
            except Exception:
                out["driver"] = "unknown"
    except Exception:
        pass
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    return out


def provenance(**extra: Any) -> dict[str, Any]:
    return {"git_commit": git_commit(), "hardware": hardware(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra}


class Manifest:
    """Start/end time, peak memory and wall time for one stage, written next to its output."""

    def __init__(self, path: Path | str, **fields: Any):
        self.path = Path(path)
        self.fields = dict(fields)
        self.t0 = time.time()
        self.fields["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def finish(self, status: str = "complete", **more: Any) -> Path:
        self.fields.update(more)
        self.fields["status"] = status
        self.fields["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.fields["wall_seconds"] = round(time.time() - self.t0, 1)
        try:
            import torch
            if torch.cuda.is_available():
                self.fields["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception:
            pass
        self.fields.update(provenance())
        return write_json(self.path, self.fields)


def results_root(smoke: bool) -> Path:
    return SMOKE if smoke else RESULTS
