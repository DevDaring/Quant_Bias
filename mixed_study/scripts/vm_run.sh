#!/usr/bin/env bash
# GPU stages of the integrated study, in dependency order, with per-stage
# resume and push. Usage: vm_run.sh smoke|full   (run from anywhere)
#
# Stage state is keyed "<mode>/<stage>/<model>" in results/v2/_logs/state.tsv,
# so a container restart resumes at the next unfinished stage and a smoke row
# can never satisfy a full stage. Results are committed and pushed after every
# stage; a swallowed push failure is logged loudly rather than hidden.
set -uo pipefail
MODE=${1:-full}
# The container supervisor relaunches with the AUTORUN value fixed at deploy
# time. A mode-override file lets a restart resume FULL even if the SDL says
# smoke, without redeploying.
[ -f /workspace/MODE_OVERRIDE ] && MODE=$(cat /workspace/MODE_OVERRIDE)
ROOT=/workspace/Quant_Bias
MS=$ROOT/mixed_study
cd "$MS"
PY=$(cat /workspace/PYTHON_BIN 2>/dev/null || echo python3)
export GIT_SSH_COMMAND="ssh -i /root/.ssh/quantbias_deploy -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts"
export HF_HOME=${HF_HOME:-/workspace/hf_cache} TOKENIZERS_PARALLELISM=false
ATTN=${ATTN:-flash_attention_2}
LOGS=results/v2/_logs; mkdir -p "$LOGS"; STATE=$LOGS/state.tsv; touch "$STATE"
log(){ echo "[$(date -u +%F_%T)] $*" | tee -a "$LOGS/run.log"; }

FLAGS="--device cuda --attn $ATTN"
[ "$MODE" = "smoke" ] && FLAGS="$FLAGS --quick"

push(){
  cd "$ROOT"
  git add -A mixed_study/results 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "results(mixed/$MODE): $1" 2>/dev/null
    git pull -q --rebase origin main 2>/dev/null || true
    git push -q origin main 2>/dev/null && log "  pushed $1" || log "  !! PUSH FAILED for $1 (will retry next stage)"
  fi
  cd "$MS"
}

run(){  # run <stage> <model> [extra args]
  local stage=$1 model=$2; shift 2
  local key="$MODE/$stage/$model$( [ $# -gt 0 ] && printf '/%s' "$@" )"
  grep -qF "${key}	ok	" "$STATE" 2>/dev/null && { log "skip $key (already ok)"; return 0; }
  log ">>> $key"
  local t0=$SECONDS lf="$LOGS/${stage}_${model}$( [ $# -gt 0 ] && printf '_%s' "$@" ).log"
  if $PY -m mixed_study.run "$stage" --model "$model" $FLAGS "$@" > "$lf" 2>&1; then
    printf '%s\tok\t%ss\n' "$key" "$((SECONDS-t0))" >> "$STATE"; log "<<< $key ok in $((SECONDS-t0))s"
  else
    printf '%s\tFAILED\t%ss\n' "$key" "$((SECONDS-t0))" >> "$STATE"; log "<<< $key FAILED in $((SECONDS-t0))s"
    tail -25 "$lf" | sed 's/^/    /' | tee -a "$LOGS/run.log"
  fi
  push "$key"
}

log "===== MODE=$MODE flags:$FLAGS ====="
# hourly VM-resident watchdog: validate + push, independent of any operator session
if ! { [ -f /workspace/mixed_watchdog.pid ] && kill -0 "$(cat /workspace/mixed_watchdog.pid)" 2>/dev/null; }; then
  setsid nohup bash "$MS/scripts/vm_watchdog.sh" > /workspace/mixed_watchdog.out 2>&1 < /dev/null &
  log "spawned watchdog (pid $!)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOGS/run.log"

log "--- Stage 1: instrumentation pilot (§5.1): legacy trace on GPT-2 + Mistral ---"
run legacy M1
run legacy M3
run legacy M5
# the saved 7B profiles were measured in fp16; test the precision hypothesis directly
run legacy M3 --dtype fp16
run legacy M5 --dtype fp16

log "--- Stage 2: B1 pilot on GPT-2 Small (cheap; proves the panel end to end) ---"
run b1 M1 --granularity layer
run b1 M1 --granularity component

log "--- Stage 3: B1 primary panel, Mistral-7B ---"
run b1 M3 --granularity layer
run b1 M3 --granularity component

log "--- Stage 4: B1 primary panel, Qwen3-8B ---"
run b1 M5 --granularity layer
run b1 M5 --granularity component

log "--- Stage 5: equal-cost selective restoration (§5.7) ---"
run restore M3
run restore M5

log "--- Stage 6: directional predictor on the ladder protocol (§6) ---"
run dladder M1 --n-examples 48
run dladder M3 --n-examples 48
run dladder M5 --n-examples 48

log "--- Stage 7: restoration sized by power analysis: full final split, two intervention sizes (§8) ---"
run restore M3 --k 8  --n-examples 8000
run restore M3 --k 16 --n-examples 8000
run restore M5 --k 8  --n-examples 8000
run restore M5 --k 16 --n-examples 8000

log "--- Stage 8: held-out confirmation on an untouched source (§3.6) ---"
run confirm M3
run confirm M5

NF=$(grep -c FAILED "$STATE" 2>/dev/null); NF=${NF:-0}
log "===== COMPLETE: $(grep -c "^$MODE/.*	ok	" "$STATE") ok, $NF failed ====="
echo "$MODE DONE failed=$NF" > "$LOGS/${MODE}_COMPLETE"
push "final ($MODE)"
