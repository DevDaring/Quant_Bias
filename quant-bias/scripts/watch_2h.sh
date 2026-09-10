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

ssh_retry(){
  # Transient SSH blips (a provider-side sshd reload, brief network flap)
  # should not be reported as an incident; a real outage should. Try a few
  # times before concluding the host is actually down.
  local tries=0 out
  while [ $tries -lt 4 ]; do
    if out=$(ssh $SSHOPT -p $P $H "$1" 2>&1); then echo "$out"; return 0; fi
    tries=$((tries+1))
    sleep 15
  done
  echo "$out"
  return 1
}

check_once(){
  N=$((N+1))
  log "===== 2-HOUR CHECK #$N ====="
  out=$(ssh_retry '
    set -o pipefail
    echo "--- process count ---"
    NRUN=$(ps -eo cmd= | grep -c "^bash /workspace/Quant_Bias/quant-bias/scripts/vm_run\.sh full$")
    NPY=$(ps -eo cmd= | grep -c "^/opt/conda/bin/python -m quantbias\.run_experiment ")
    echo "vm_run_count=$NRUN run_experiment_count=$NPY"
    if [ "$NRUN" -gt 1 ]; then echo "PROC_ALERT: duplicate vm_run.sh, count=$NRUN"; fi
    if [ "$NRUN" -eq 0 ]; then echo "PROC_ALERT: zero vm_run.sh -- container may have restarted; needs a manual relaunch (check bootstrap.log timestamp first, do not double-launch if autorun is already mid-boot)"; fi
    if [ "$NPY" -gt 1 ]; then echo "PROC_ALERT: duplicate run_experiment, count=$NPY"; fi
    echo "--- gpu ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "--- disk ---"
    df -h /workspace | tail -1
    echo "--- current stage ---"
    # AUTORUN writes run.log; a manually-launched copy writes full_console.log.
    # Whichever was modified most recently is the one actually driving the live
    # process, so tail that one rather than an assumed fixed path.
    LATEST=$(ls -t /workspace/run.log /workspace/full_console.log 2>/dev/null | head -1)
    echo "reading: $LATEST"
    tail -3 "$LATEST" 2>/dev/null
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
  ')
  echo "$out"
  if echo "$out" | grep -qE "PROC_ALERT|VALIDATE FAIL|\] +FAIL |Permission denied|Connection refused|lost connection"; then
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
