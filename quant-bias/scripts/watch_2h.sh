#!/bin/bash
# Runs in THIS chat session (not on the VM). Every 2 hours: SSH to the GPU
# host, confirm exactly one run process is alive, run the content-correctness
# validator, and print one summary block. Never exits on its own -- the loop
# continues even when every check is clean, per the user's instruction.
set -u
H="root@provider.h100.ams2.val.akash.pub"
P=31810
KEY="$HOME/.ssh/quantbias_ed25519"
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o LogLevel=ERROR"
INTERVAL=${INTERVAL:-7200}
N=0

log(){ echo "[$(date -u +%F_%H:%M:%S)] $*"; }

check_once(){
  N=$((N+1))
  log "===== 2-HOUR CHECK #$N ====="
  out=$(ssh $SSHOPT -p $P $H '
    set -o pipefail
    echo "--- process count ---"
    NRUN=$(ps aux | grep -c "[b]ash /workspace/Quant_Bias/quant-bias/scripts/vm_run.sh")
    NPY=$(ps aux | grep -c "[r]un_experiment")
    echo "vm_run.sh instances: $NRUN   run_experiment instances: $NPY"
    if [ "$NRUN" -gt 1 ]; then echo "ALERT: duplicate vm_run.sh detected"; fi
    if [ "$NRUN" -eq 0 ]; then echo "ALERT: no vm_run.sh running -- container may have restarted and not resumed; needs a manual relaunch (do NOT also rely on autorun if it is mid-relaunch, check bootstrap.log timestamp first)"; fi
    if [ "$NPY" -gt 1 ]; then echo "ALERT: duplicate run_experiment detected"; fi
    echo "--- gpu ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "--- disk ---"
    df -h /workspace | tail -1
    echo "--- current stage ---"
    tail -3 /workspace/full_console.log 2>/dev/null
    echo "--- content-correctness validator ---"
    cd /workspace/Quant_Bias/quant-bias
    export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
    git pull -q --rebase origin main 2>/dev/null
    PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo python3)
    $PY scripts/validate_results.py --retry-failed 2>&1
    echo "--- stage counts ---"
    awk -F"\t" "{c[\$2]++} END{for(k in c) printf \"  %s: %d\n\", k, c[k]}" results/_logs/state.tsv 2>/dev/null
    git add -A results 2>/dev/null
    git diff --cached --quiet || { git commit -q -m "watchdog: validator + retry pass"; git pull -q --rebase origin main 2>/dev/null; git push -q origin main 2>/dev/null; echo "pushed retry/validator results"; }
  ' 2>&1)
  echo "$out"
  if echo "$out" | grep -qE "ALERT|VALIDATE FAIL|Permission denied|Connection refused|lost connection"; then
    echo "  >>> ATTENTION NEEDED (see ALERT/FAIL/connection lines above)"
  else
    echo "  >>> all clear"
  fi
  log "===== END CHECK #$N (next in ${INTERVAL}s) ====="
}

# First check fires soon so problems surface quickly, then every $INTERVAL.
sleep 120
while true; do
  check_once
  sleep "$INTERVAL"
done
