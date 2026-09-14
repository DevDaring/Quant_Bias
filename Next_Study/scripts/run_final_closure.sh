#!/usr/bin/env bash
# Frozen execution order F1-F8 with per-cell resume and per-stage push.
# Usage: run_final_closure.sh smoke|full   (the supervisor passes AUTORUN; MODE_OVERRIDE wins)
set -uo pipefail
MODE=${1:-full}
[ -f /workspace/MODE_OVERRIDE ] && MODE=$(cat /workspace/MODE_OVERRIDE)
ROOT=/workspace/Quant_Bias; NS=$ROOT/Next_Study
cd "$NS"
PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo /opt/conda/bin/python)
export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
export HF_HOME=${HF_HOME:-/workspace/hf_cache} TOKENIZERS_PARALLELISM=false HOSTNAME_CLASS=akash-h100
ATTN=${ATTN:-flash_attention_2}
LOGS=$NS/results/final_closure/_logs; mkdir -p "$LOGS"; STATE=$LOGS/state.tsv; touch "$STATE"
log(){ echo "[$(date -u +%F_%T)] $*" | tee -a "$LOGS/run.log"; }
SMOKE=""; [ "$MODE" = "smoke" ] && SMOKE="--smoke"
FLAGS="--device cuda --attn $ATTN --batch-size 16"

push(){
  cd "$ROOT"
  git add -A Next_Study/results Next_Study/data/synthbias/SOURCE.json Next_Study/data/synthbias/DEDUP_REPORT.json Next_Study/data/synthbias/split_manifest.json 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "results(final_closure/$MODE): $1" 2>/dev/null
    git pull -q --rebase origin main 2>/dev/null || true
    git push -q origin main 2>/dev/null && log "  pushed $1" || log "  !! PUSH FAILED for $1 (will retry next stage)"
  fi
  cd "$NS"
}

run(){  # run <python> <stage> <model-or-"-"> [extra args]
  local py=$1 stage=$2 model=$3; shift 3
  local key="$MODE/$stage/$model$( [ $# -gt 0 ] && printf '/%s' "$@" )"
  grep -qF "${key}	ok	" "$STATE" 2>/dev/null && { log "skip $key (already ok)"; return 0; }
  log ">>> $key"
  local t0=$SECONDS lf="$LOGS/${MODE}_${stage}_${model}$( [ $# -gt 0 ] && printf '_%s' "$@" ).log"
  local margs=(); [ "$model" != "-" ] && margs=(--model "$model")
  if $py -m next_study.run "$stage" "${margs[@]}" $SMOKE "$@" > "$lf" 2>&1; then
    printf '%s\tok\t%ss\n' "$key" "$((SECONDS-t0))" >> "$STATE"; log "<<< $key ok in $((SECONDS-t0))s"
  else
    printf '%s\tFAILED\t%ss\n' "$key" "$((SECONDS-t0))" >> "$STATE"; log "<<< $key FAILED in $((SECONDS-t0))s"
    tail -25 "$lf" | sed 's/^/    /' | tee -a "$LOGS/run.log"
  fi
  push "$key"
}

log "===== MODE=$MODE ====="
bash "$NS/scripts/vm_setup.sh" >> "$LOGS/setup.log" 2>&1 || { log "!! setup failed (see setup.log)"; tail -5 "$LOGS/setup.log"; }
PYP=$(cat /workspace/PYTHON_PACKED 2>/dev/null || echo "$PY")
if ! { [ -f /workspace/final_watchdog.pid ] && kill -0 "$(cat /workspace/final_watchdog.pid)" 2>/dev/null; }; then
  setsid nohup bash "$NS/scripts/vm_watchdog.sh" > /workspace/final_watchdog.out 2>&1 < /dev/null &
  log "spawned watchdog (pid $!)"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOGS/run.log"

log "--- F2: dense prompt selection, dense records, order-swap audit ---"
for M in F-M1 F-M3 F-M2; do run "$PY" dense "$M" $FLAGS; done
log "--- F3: simulated RTN8, RTN4, GPTQ4 on the final split ---"
for M in F-M1 F-M3 F-M2; do for C in rtn8 rtn4 gptq4; do run "$PY" quant "$M" $FLAGS --condition "$C"; done; done
log "--- F4: expanded directional ladder ---"
for M in F-M1 F-M3; do run "$PY" directional "$M" $FLAGS; done
log "--- F5: residual-direction controls ---"
for M in F-M1 F-M3; do run "$PY" residual "$M" $FLAGS; done
log "--- F6: packed GPTQ4 (separate environment) ---"
for M in F-M1 F-M3; do run "$PYP" packed "$M" --batch-size 16; done
log "--- F7: locked analysis ---"
run "$PY" analysis - --models F-M1 F-M2 F-M3
log "--- F8: validation ---"
run "$PY" validate - --models F-M1 F-M2 F-M3

NF=$(grep -c FAILED "$STATE" 2>/dev/null); NF=${NF:-0}
log "===== COMPLETE: $(grep -c "^$MODE/.*	ok	" "$STATE") ok, $NF failed ====="
echo "$MODE DONE failed=$NF" > "$LOGS/${MODE}_COMPLETE"
push "final ($MODE)"
