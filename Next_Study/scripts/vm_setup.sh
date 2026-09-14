#!/usr/bin/env bash
# Idempotent per-boot setup for the final closure experiment, run after the
# quant-bias bootstrap (which installs torch/transformers/flash-attn and clones the repo).
#   * SynthBias CSV re-downloaded and SHA256-verified against the locked SOURCE.json
#   * canonical rows and splits regenerated; the split hash must match LOCKED_PROTOCOL.json
#   * a separate environment for the packed backend (gptqmodel pins its own torch)
set -uo pipefail
ROOT=/workspace/Quant_Bias; NS=$ROOT/Next_Study
PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo /opt/conda/bin/python)
log(){ echo "[$(date -u +%F_%T)] setup: $*"; }
cd "$NS"
$PY -m pip install -q scipy pyyaml pytest huggingface_hub 2>/dev/null || true

log "=== SynthBias ==="
mkdir -p data/synthbias
WANT=$($PY -c "import json;print(json.load(open('data/synthbias/SOURCE.json'))['sha256'])")
if [ ! -f data/synthbias/synthbias_data.csv ] || [ "$(sha256sum data/synthbias/synthbias_data.csv | cut -d' ' -f1)" != "$WANT" ]; then
  curl -sL -A "Mozilla/5.0" -o data/synthbias/synthbias_data.csv "https://raw.githubusercontent.com/apple-aiml-research/ml-synthbias/main/data/synthbias_data.csv"
fi
GOT=$(sha256sum data/synthbias/synthbias_data.csv | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] && log "csv sha256 verified ($GOT)" || { log "!! csv sha256 MISMATCH got=$GOT want=$WANT"; exit 1; }
$PY - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from next_study import synthbias as S
from next_study.common import DATA, PROTOCOL, sha256_text
src = json.load(open(DATA / "SOURCE.json"))
rows = S.load_csv(); S.link_counterparts(rows)
man = S.assign_splits(rows)
S.write_jsonl(DATA / "synthbias_canonical.jsonl", [r.as_dict() for r in rows])
proto = json.load(open(PROTOCOL / "LOCKED_PROTOCOL.json"))
h = sha256_text(json.dumps(man["assignment"], sort_keys=True))
assert h == proto["split_assignment_sha256"], f"split hash mismatch {h} != {proto['split_assignment_sha256']}"
print("  split assignment hash verified", h[:16], "| type2 final rows:", sum(r.type=='type2' and r.split=='final' for r in rows))
PYEOF
[ $? -eq 0 ] || { log "!! split verification failed"; exit 1; }

log "=== tests ==="
$PY -m pytest tests -q 2>&1 | tail -2

log "=== packed backend environment ==="
VP=/workspace/venv_packed
if [ ! -x "$VP/bin/python" ]; then
  $PY -m venv "$VP" && "$VP/bin/pip" install -q --upgrade pip wheel setuptools ninja
fi
if ! "$VP/bin/python" -c "import gptqmodel" 2>/dev/null; then
  "$VP/bin/pip" install -q -v gptqmodel > /workspace/gptqmodel_install.log 2>&1 || log "!! gptqmodel install failed (see /workspace/gptqmodel_install.log); packed stage will record BACKEND_UNSUPPORTED"
  "$VP/bin/pip" install -q torchvision datasets scipy pyyaml numpy huggingface_hub accelerate >> /workspace/gptqmodel_install.log 2>&1 || true
fi
"$VP/bin/python" -c "import gptqmodel, torch, transformers; print('  gptqmodel', gptqmodel.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)" 2>&1 | tail -1
echo "$VP/bin/python" > /workspace/PYTHON_PACKED
log "setup complete"
