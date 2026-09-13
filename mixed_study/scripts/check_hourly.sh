#!/usr/bin/env bash
# Hourly, from the operator session: SSH to the GPU host, verify exactly one
# run process, run the content validator on the VM, report progress, flag issues.
# Complements (does not replace) the VM-resident watchdog, which pushes to git.
set -u
H="root@provider.h100.siamaidol.com"; P=${PORT:-31029}
SSHOPT="-i $HOME/.ssh/quantbias_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o LogLevel=ERROR"
INTERVAL=${INTERVAL:-3600}; N=0
check(){
  N=$((N+1)); echo "===== HOURLY CHECK #$N $(date -u +%F_%T) UTC ====="
  local out="" t
  for t in 1 2 3 4; do out=$(ssh $SSHOPT -p $P $H '
    S=/workspace/Quant_Bias/mixed_study/results/v2/_logs
    NRUN=$(ps -eo cmd= | grep -c "^bash /workspace/Quant_Bias/mixed_study/scripts/vm_run\.sh ")
    NPY=$(ps -eo cmd= | grep -c "^/opt/conda/bin/python -m mixed_study\.run ")
    echo "procs: vm_run=$NRUN python=$NPY"
    [ "$NRUN" -gt 1 ] && echo "PROC_ALERT duplicate vm_run.sh"
    [ "$NPY" -gt 1 ] && echo "PROC_ALERT duplicate python run"
    C=""; [ "$NRUN" -eq 0 ] && [ -f $S/full_COMPLETE ] && [ "$(cat /workspace/MODE_OVERRIDE 2>/dev/null)" = full ] && C=$(cat $S/full_COMPLETE)
    [ -z "$C" ] && [ "$NRUN" -eq 0 ] && [ -f $S/smoke_COMPLETE ] && C="smoke phase ended"
    [ "$NRUN" -eq 0 ] && [ -z "$C" ] && echo "PROC_ALERT no run process and no completion marker"
    NEW=$(ls -t $S/*.log 2>/dev/null | head -1); AGE=$(( $(date -u +%s) - $(stat -c %Y "$NEW" 2>/dev/null || echo 0) ))
    echo "freshest log: $(basename "$NEW") ${AGE}s old"; [ -z "$C" ] && [ "$AGE" -gt 1500 ] && echo "PROC_ALERT no log written in ${AGE}s"
    echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    echo "stage: $(grep -E ">>>|<<<" $S/run.log 2>/dev/null | tail -1)"
    echo "state: full_ok=$(grep -c "^full/.*	ok	" $S/state.tsv) smoke_ok=$(grep -c "^smoke/.*	ok	" $S/state.tsv) failed=$(grep -c FAILED $S/state.tsv)"
    [ -n "$C" ] && echo "COMPLETE: $C"
    echo "--- validator (content correctness, on the VM) ---"
    cd /workspace/Quant_Bias/mixed_study && /opt/conda/bin/python scripts/validate_v2.py 2>&1 | grep -vE "^\[.*INFO" | head -12
    echo "git: $(cd /workspace/Quant_Bias && git rev-parse --short HEAD) | vm_watchdog: $(grep -E "OK \||ATTENTION \|" $S/watchdog.log 2>/dev/null | tail -1 | cut -c1-70)"' 2>&1) && break; sleep 20; done
  echo "$out"
  if echo "$out" | grep -qE "PROC_ALERT|VALIDATE_V2 FAIL|FAIL |failed=[1-9]|Connection refused|Permission denied"; then echo ">>> ATTENTION NEEDED"; else echo ">>> all clear"; fi
  echo "$out" | grep -q "COMPLETE: full" && echo ">>> RUN_COMPLETE"
}
sleep 60; while true; do check; sleep "$INTERVAL"; done
