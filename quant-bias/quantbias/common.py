"""Shared plumbing: secrets, logging, seeds, hashing, manifests, result I/O.

All experiment outputs go under RESULTS_DIR (python/results/bias). Existing
result files elsewhere in the repository are never touched.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PKG_DIR = Path(__file__).resolve().parent           # Codes/quant-bias/quantbias
PROJECT_DIR = PKG_DIR.parent                        # Codes/quant-bias
CODES_DIR = PROJECT_DIR.parent                      # Codes
RESULTS_DIR = Path(os.environ.get("QUANTBIAS_RESULTS", PROJECT_DIR / "results"))
CONFIG_DIR = PROJECT_DIR / "configs"
ENV_FILE = CODES_DIR / ".env"
LIVING_INFERENCE_DIR = CODES_DIR / "living-inference"   # read-only: prior results


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Secrets
# ----------------------------------------------------------------------------

def load_env(env_file: Path = ENV_FILE) -> dict[str, str]:
    """Load Codes/.env and export the keys this pipeline uses.

    Only the Hugging Face token is required (gated checkpoints such as
    Mistral-7B). Other keys in the file are left in os.environ untouched for
    optional judges, but nothing here calls a paid API.
    """
    values: dict[str, str] = {}
    if env_file.exists():
        try:
            from dotenv import dotenv_values
            values = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
        except ImportError:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in values.items():
        os.environ.setdefault(k, v)
    hf = values.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf)
    seed = values.get("RANDOM_SEED")
    if seed and "BIAS_SEED" not in os.environ:
        os.environ["BIAS_SEED"] = seed
    return values


def default_seed() -> int:
    raw = os.environ.get("BIAS_SEED", "20260908")
    try:
        return int(str(raw).strip()) % (2**31 - 1)
    except ValueError:
        return 20260908


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Hashing and JSON
# ----------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def to_json(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, default=_json_default)


def stable_hash(obj: Any, n: int = 16) -> str:
    """Deterministic hash of any JSON-serialisable object."""
    return hashlib.sha256(to_json(obj, indent=None).encode()).hexdigest()[:n]


def write_json(path: Path | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(to_json(obj))
    tmp.replace(path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path | str, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as f:
        for r in rows:
            f.write(to_json(r, indent=None) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[dict]:
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------

def code_revision() -> str:
    """Hash of the package sources, plus git SHA if the repo has one."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.rglob("*.py")):
        h.update(p.relative_to(PKG_DIR).as_posix().encode())
        h.update(p.read_bytes())
    src = h.hexdigest()[:12]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL, text=True).strip()
        return f"git:{sha}+src:{src}"
    except Exception:
        return f"src:{src}"


def resolve_hub_revision(repo_id: str, repo_type: str = "model", revision: str = "main") -> str:
    """Return the commit SHA a Hub revision currently points to."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if repo_type == "dataset":
            info = api.dataset_info(repo_id, revision=revision)
        else:
            info = api.model_info(repo_id, revision=revision)
        return str(info.sha)
    except Exception as e:  # offline or gated without token
        return f"unresolved({revision}):{type(e).__name__}"


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


@dataclass
class Manifest:
    """Provenance record attached to every result file (plan, Section 10)."""
    experiment_id: str
    model_id: str
    model_revision: str = "unresolved"
    tokenizer_revision: str = "unresolved"
    method: str = "dense"
    backend: str = "simulated"
    seed: int = 0
    precision_map_hash: str = ""
    data_revisions: dict[str, str] = field(default_factory=dict)
    split_hash: str = ""
    calibration_hash: str = ""
    code_revision: str = field(default_factory=code_revision)
    hardware: dict[str, Any] = field(default_factory=hardware_info)
    started: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    finished: str = ""
    status: str = "running"
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "ok", error: str = "") -> "Manifest":
        self.finished = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.status = status
        self.error = error
        return self


def result_path(experiment_id: str, model_tag: str, name: str, ext: str = "json") -> Path:
    d = RESULTS_DIR / experiment_id / model_tag
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.{ext}"


def load_yaml(path: Path | str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_device(preferred: str | None = None) -> str:
    import torch
    if preferred and preferred != "auto":
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"
