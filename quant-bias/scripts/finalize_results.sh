#!/usr/bin/env bash
# Run ON the GPU host when the study finishes, BEFORE the VM is destroyed.
# Commits and pushes every artifact, including logs and anything the per-stage
# push loop did not cover, then verifies the push actually landed rather than
# trusting a quiet exit code. Exits non-zero if anything is still unsaved, so a
# caller can refuse to destroy the VM.
set -uo pipefail
cd /workspace/Quant_Bias/quant-bias
export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo python3)
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== final validator pass ==="
$PY scripts/validate_results.py 2>&1 | tail -20

log "=== copying stray host-side logs into the repo so they are preserved ==="
mkdir -p results/_logs/host
for f in /workspace/run.log /workspace/bootstrap.log /workspace/full_console.log; do
  [ -f "$f" ] && cp -f "$f" results/_logs/host/"$(basename "$f")"
done

log "=== committing everything under results/ ==="
git add -A results
if git diff --cached --quiet; then
  log "  nothing new to commit"
else
  git commit -q -m "final: complete study results, logs and artifacts

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

log "=== pushing (retry up to 3 times; a swallowed failure here loses data) ==="
PUSHED=0
for i in 1 2 3; do
  git pull -q --rebase origin main 2>/dev/null
  if git push origin main 2>&1 | tail -2; then PUSHED=1; break; fi
  log "  push attempt $i failed, retrying"; sleep 10
done

log "=== verification ==="
AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "?")
DIRTY=$(git status --porcelain results | wc -l)
TRACKED=$(git ls-files results | wc -l)
ONDISK=$(find results -type f | wc -l)
log "  tracked=$TRACKED  on_disk=$ONDISK  uncommitted=$DIRTY  unpushed_commits=$AHEAD"
if [ "$AHEAD" != "0" ] || [ "$DIRTY" != "0" ]; then
  log "  NOT SAFE TO DESTROY: unsaved work remains"
  exit 1
fi
log "  ALL RESULTS SAVED AND PUSHED"
echo "FINALIZE_OK tracked=$TRACKED on_disk=$ONDISK"
