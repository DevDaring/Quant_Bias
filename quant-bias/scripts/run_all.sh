#!/bin/bash
# Full study on one GPU. Usage: bash scripts/run_all.sh [budget|full]
# Ordering matters: E4 needs E2 output, E5 needs E4 output.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python
PROF=${1:-budget}
R="$PY -m quantbias.run_experiment --device cuda --profile $PROF"

echo "### Stage 1  adapter + pipeline checks on every model"
for M in M1 M2 M3 M4 M5; do $R --exp e0 --model $M; done

echo "### Stage 2  quantization x subgroup outcomes (the headline table)"
for M in M2 M3 M4 M5; do $R --exp e1 --model $M; done

echo "### Stage 3  layer attribution on the two primary models"
$R --exp e2 --model M3
$R --exp e2 --model M5

echo "### Stage 4  calibration composition"
$R --exp e3 --model M2
$R --exp e3 --model M3

echo "### Stage 5  restoration + allocation, then transfer"
$R --exp e4 --model M3
$R --exp e4 --model M5
$R --exp e5 --model M5
$R --exp e5 --model M4
echo "done: results in results/"
