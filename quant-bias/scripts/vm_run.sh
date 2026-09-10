#!/usr/bin/env bash
# Orchestrate the study on the GPU host. Results are committed and pushed after
# every experiment so an interruption loses at most one stage.
#
#   bash vm_run.sh smoke     two samples per category, every model, every experiment
#   bash vm_run.sh full      the real run
set -uo pipefail
MODE=${1:-full}
WORK=${WORK:-/workspace}
cd "$WORK/Quant_Bias/quant-bias"
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo python3)
export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
LOGDIR=results/_logs; mkdir -p "$LOGDIR"
STATE="$LOGDIR/state.tsv"; touch "$STATE"
log(){ echo "[$(date -u +%F_%H:%M:%S)] $*" | tee -a "$LOGDIR/run.log"; }

if [ "$MODE" = smoke ]; then FLAGS="--smoke"; else FLAGS="--profile budget"; fi
FLAGS="$FLAGS --device cuda --attn ${ATTN:-sdpa}"

push(){
  git add -A results 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "results($MODE): $1" || true
    # rebase onto anything pushed meanwhile so concurrent stages do not conflict
    git pull -q --rebase origin main 2>/dev/null || true
    git push -q origin main 2>&1 | tail -2 || log "  push failed (will retry next stage)"
  fi
}

# Restore completed-stage state from the remote so a container restart resumes
# instead of redoing finished work.
git pull -q --rebase origin main 2>/dev/null || true

run(){  # run <exp> <model>
  local exp=$1 model=$2 key="$MODE/$1/$2"
  # State rows are "<mode>/<exp>/<model>\tok\t<seconds>". Matching a prefix
  # (not an anchored whole line) is what makes resume actually work, and keying
  # by mode stops a smoke row from being mistaken for a completed full stage.
  grep -qF "${key}	ok	" "$STATE" 2>/dev/null && { log "skip $key (already ok)"; return 0; }
  log ">>> $key"
  local t0=$SECONDS
  if $PY -m quantbias.run_experiment --exp "$exp" --model "$model" $FLAGS \
        >"$LOGDIR/${exp}_${model}.log" 2>&1; then
    printf "%s\tok\t%ss\n" "$key" "$((SECONDS-t0))" >> "$STATE"
    log "<<< $key ok in $((SECONDS-t0))s"
  else
    printf "%s\tFAILED\t%ss\n" "$key" "$((SECONDS-t0))" >> "$STATE"
    log "<<< $key FAILED in $((SECONDS-t0))s -- see $LOGDIR/${exp}_${model}.log"
    tail -15 "$LOGDIR/${exp}_${model}.log" | sed 's/^/    /' | tee -a "$LOGDIR/run.log"
  fi
  push "$key"
}

ALL="M1 M6 M7 M2 M3 M4 M5"
PRIMARY="M3 M5"

log "===== MODE=$MODE  flags:$FLAGS ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOGDIR/run.log"

log "--- Stage A: adapter and pipeline checks (all models) ---"
for m in $ALL; do run e0 "$m"; done

log "--- Stage B: quantization x subgroup outcomes (all models) ---"
for m in $ALL; do run e1 "$m"; done

log "--- Stage C: layer attribution (all models) ---"
for m in $ALL; do run e2 "$m"; done

log "--- Stage D: bridge to living-inference (all models) ---"
for m in $ALL; do run e6 "$m"; done

log "--- Stage E: calibration composition ---"
for m in M2 M3; do run e3 "$m"; done

log "--- Stage F: restoration and allocation (primary models) ---"
for m in $PRIMARY; do run e4 "$m"; done

log "--- Stage G: comparators from the literature (primary models) ---"
for m in $PRIMARY; do run e7 "$m"; done

log "--- Stage H: transfer and pruning bridge ---"
for m in M4 M5; do run e5 "$m"; done

log "===== summary ====="
column -t "$STATE" 2>/dev/null | tee -a "$LOGDIR/run.log" || cat "$STATE"
FAILED=$(grep -c FAILED "$STATE" 2>/dev/null || echo 0)
log "completed with $FAILED failed stage(s)"
push "final $MODE summary"
echo "$MODE DONE failed=$FAILED" > "$LOGDIR/${MODE}_COMPLETE"
push "$MODE complete marker"
