#!/usr/bin/env bash
# Hourly, ON THE GPU HOST, independent of any operator session:
#   * exactly one run process (0 = stalled/needs relaunch, >1 = duplicate writer)
#   * real progress: freshest per-stage log written within the last 20 minutes
#     (stage-boundary log alone can look frozen for an hour on a long stage)
#   * content correctness + cross-stage consistency via validate_v2.py
#   * commit + push results so GitHub always has the latest state
# Writes a one-line heartbeat to results/final_closure/_logs/watchdog.log and a JSON
# status file so a remote reader can see the verdict without SSH-ing.
set -uo pipefail
ROOT=/workspace/Quant_Bias; MS=$ROOT/Next_Study
INTERVAL=${WATCHDOG_INTERVAL:-3600}
PIDFILE=/workspace/final_watchdog.pid
echo $$ > "$PIDFILE"
export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo python3)
LOGS=$MS/results/final_closure/_logs; mkdir -p "$LOGS"
log(){ echo "[$(date -u +%F_%T)] $*" | tee -a "$LOGS/watchdog.log"; }

check(){
  local now; now=$(date -u +%s)
  local nrun npy alerts=() verdict
  nrun=$(ps -eo cmd= | grep -c "^bash /workspace/Quant_Bias/Next_Study/scripts/run_final_closure\.sh ")
  npy=$(ps -eo cmd= | grep -cE "^/(opt/conda|workspace/venv_packed)/bin/python -m next_study\.run ")
  [ "$nrun" -gt 1 ] && alerts+=("duplicate vm_run.sh ($nrun)")
  [ "$npy" -gt 1 ] && alerts+=("duplicate run_experiment ($npy)")
  local complete=""; [ -f "$LOGS/full_COMPLETE" ] && complete=$(cat "$LOGS/full_COMPLETE")
  if [ -z "$complete" ] && [ "$nrun" -eq 0 ]; then alerts+=("no run process and no full_COMPLETE: STALLED"); fi
  local newest age=-1
  newest=$(ls -t "$LOGS"/*.log 2>/dev/null | head -1)
  [ -n "$newest" ] && age=$(( now - $(stat -c %Y "$newest") ))
  if [ -z "$complete" ] && [ "$age" -gt 1200 ]; then alerts+=("no log written in ${age}s (>20min)"); fi
  local gpu; gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  local nfail; nfail=$(grep -c FAILED "$LOGS/state.tsv" 2>/dev/null); nfail=${nfail:-0}   # grep -c prints 0 AND exits 1 on no match
  [ "$nfail" -gt 0 ] && alerts+=("$nfail FAILED stage(s) in state.tsv")
  # content + consistency
  local vout; vout=$(cd "$MS" && $PY -m next_study.run validate $( [ -f /workspace/MODE_OVERRIDE ] && [ "$(cat /workspace/MODE_OVERRIDE)" = smoke ] && echo --smoke ) --models F-M1 F-M2 F-M3 --partial 2>&1); verdict=$(echo "$vout" | grep -oE "VALIDATE_FINAL_CLOSURE (PASS|WARN|FAIL)" | awk '{print $2}')
  [ "$verdict" = "FAIL" ] && alerts+=("validator FAIL")
  # push
  local pushed="no-change"
  ( cd "$ROOT" && git add -A Next_Study/results 2>/dev/null
    if ! git diff --cached --quiet; then
      git commit -q -m "watchdog(final_closure): hourly validate + push" 2>/dev/null
      git pull -q --rebase origin main 2>/dev/null || true
      if git push -q origin main 2>/dev/null; then echo pushed > /tmp/wd_push; else echo PUSH-FAILED > /tmp/wd_push; fi
    fi )
  [ -f /tmp/wd_push ] && { pushed=$(cat /tmp/wd_push); rm -f /tmp/wd_push; }
  [ "$pushed" = "PUSH-FAILED" ] && alerts+=("git push failed")
  local status="OK"; [ ${#alerts[@]} -gt 0 ] && status="ATTENTION"
  log "$status | procs=$nrun/$npy | gpu=[$gpu] | log_age=${age}s | validator=$verdict | failed=$nfail | push=$pushed | complete=${complete:-no} | ${alerts[*]:-}"
  echo "$vout" | grep -E "FAIL|WARN|INFO" | sed 's/^/    /' | tee -a "$LOGS/watchdog.log" >/dev/null
  cat > "$LOGS/watchdog_status.json" <<EOF
{"time":"$(date -u +%FT%TZ)","status":"$status","run_procs":$nrun,"py_procs":$npy,"gpu":"$gpu","log_age_s":$age,
 "validator":"$verdict","failed_stages":$nfail,"push":"$pushed","complete":"${complete}","alerts":$(printf '%s\n' "${alerts[@]:-}" | python3 -c 'import sys,json;print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))')}
EOF
  if [ -n "$complete" ]; then log "run complete; watchdog exiting after final push"; rm -f "$PIDFILE"; exit 0; fi
}

log "watchdog started (interval ${INTERVAL}s, pid $$)"
while true; do check; sleep "$INTERVAL"; done
