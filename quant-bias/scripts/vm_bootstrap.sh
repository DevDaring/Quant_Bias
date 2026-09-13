#!/usr/bin/env bash
# Provision a fresh GPU host for the quant-bias study.
# Usage:  bash vm_bootstrap.sh
# Expects HF_TOKEN and GH_TOKEN in the environment (never written to disk).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
WORK=${WORK:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/DevDaring/Quant_Bias.git}

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# The pytorch images ship torch under conda, not /usr/bin/python3.
pick_python(){
  for c in /opt/conda/bin/python python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c "import torch" >/dev/null 2>&1 && { echo "$c"; return; }
  done
  echo "${PY_FALLBACK:-python3}"
}
PY=$(pick_python)
export PY

log "=== host ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
log "python: $PY  ($($PY -c 'import sys;print(sys.version.split()[0])'))"
df -h "$WORK" 2>/dev/null | tail -1 || df -h / | tail -1

log "=== system packages ==="
apt-get update -qq && apt-get install -yqq git curl jq tmux ca-certificates >/dev/null

log "=== clone ==="
mkdir -p "$WORK" && cd "$WORK"
# Access is via a repo-scoped deploy key, not the account-wide token, so a
# compromise of this host cannot reach any other repository.
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${DEPLOY_KEY:-}" ]; then
  printf '%s\n' "$DEPLOY_KEY" > /root/.ssh/quantbias_deploy
  chmod 600 /root/.ssh/quantbias_deploy
fi
ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
SSH_URL="git@github.com:DevDaring/Quant_Bias.git"
if [ -d Quant_Bias/.git ]; then
  cd Quant_Bias && git remote set-url origin "$SSH_URL" && git fetch -q origin && git reset -q --hard origin/main
else
  git clone -q "$SSH_URL" Quant_Bias
  cd Quant_Bias
fi
git config user.email "koushikdeb2009@gmail.com"
git config user.name "DevDaring"
log "at commit $(git rev-parse --short HEAD)"

log "=== python deps ==="
cd "$WORK/Quant_Bias/quant-bias"
$PY -m pip install -q --upgrade pip wheel setuptools
# torch first: keep whatever CUDA build the image ships if it already works
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "installing torch cu124"
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
fi
$PY -m pip install -q -r requirements.txt

log "=== flash-attention (prebuilt wheel, optional) ==="
# Attention is NOT the bottleneck here: scoring prompts are 70-250 tokens and the
# cost is dominated by the LM-head matmul and weight bandwidth. FA2 is installed
# because it is free when a matching wheel exists, and skipped otherwise --
# model_adapters falls back to SDPA automatically.
$PY - <<'PY' || echo "  flash-attn skipped; SDPA will be used"
import subprocess, sys, torch
tv = ".".join(torch.__version__.split(".")[:2])
cu = "cu12" if (torch.version.cuda or "12").startswith("12") else "cu11"
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
maj = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
if maj < 8:
    print(f"  compute capability {maj}.x < 8.0: flash-attn 2 unsupported"); sys.exit(1)
base = "https://github.com/Dao-AILab/flash-attention/releases/download"
for ver in ["2.8.3", "2.7.4.post1", "2.6.3"]:
    whl = f"{base}/v{ver}/flash_attn-{ver}+{cu}torch{tv}cxx11abiFALSE-{cp}-{cp}-linux_x86_64.whl"
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl]).returncode == 0:
        import flash_attn; print(f"  flash-attn {flash_attn.__version__} installed"); sys.exit(0)
print("  no matching prebuilt wheel"); sys.exit(1)
PY

log "=== sanity ==="
$PY - <<'PY'
import torch, transformers, datasets
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"  transformers {transformers.__version__}  datasets {datasets.__version__}")
try:
    import flash_attn; print(f"  flash_attn {flash_attn.__version__}")
except ImportError: print("  flash_attn absent (SDPA fallback)")
PY
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
mkdir -p "$HF_HOME"
$PY -c "
import os
from huggingface_hub import HfApi
print('  hf user:', HfApi().whoami(token=os.environ['HF_TOKEN'])['name'])"

log "=== offline tests (no downloads) ==="
$PY -m pytest tests -q -m "not network" 2>&1 | tail -3
echo "$PY" > /workspace/PYTHON_BIN
log "=== mixed_study offline tests ==="
( cd /workspace/Quant_Bias/mixed_study && $PY -m pytest tests -q 2>&1 | tail -3 )
log "bootstrap complete (python: $PY)"
