#!/usr/bin/env bash
# Local hourly relay. The VM-resident watchdog does the real checking and
# pushing; this only READS its status (SSH, read-only) and reports here.
set -u
H="root@provider.h100.siamaidol.com"; P=31934
SSHOPT="-i $HOME/.ssh/quantbias_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o LogLevel=ERROR"
INTERVAL=${INTERVAL:-3600}; N=0
while true; do
  N=$((N+1)); echo "===== RELAY #$N $(date -u +%F_%T) ====="
  out=""; for t in 1 2 3; do out=$(ssh $SSHOPT -p $P $H '
    S=/workspace/Quant_Bias/mixed_study/results/v2/_logs
    echo "vm_watchdog: $(grep -E "OK \||ATTENTION \|" $S/watchdog.log 2>/dev/null | tail -1 | cut -c1-200)"
    echo "stage: $(grep -E ">>>|<<<" $S/run.log 2>/dev/null | tail -1)"
    echo "state: ok=$(grep -c "^full/.*	ok	" $S/state.tsv 2>/dev/null) failed=$(grep -c FAILED $S/state.tsv 2>/dev/null)"
    [ -f $S/full_COMPLETE ] && echo "RUN_COMPLETE: $(cat $S/full_COMPLETE)"
    echo "github_head: $(cd /workspace/Quant_Bias && git rev-parse --short HEAD 2>/dev/null)"' 2>&1) && break; sleep 20; done
  echo "$out"
  echo "$out" | grep -qE "ATTENTION|RUN_COMPLETE|FAILED [1-9]|Connection refused|Permission denied" && echo ">>> ATTENTION" || echo ">>> all clear"
  sleep "$INTERVAL"
done
