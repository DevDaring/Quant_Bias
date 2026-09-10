#!/bin/bash
# Minimum publishable pilot on one GPU (~6 h H100). Usage: bash scripts/run_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python
R="$PY -m quantbias.run_experiment --device cuda --profile budget"
$R --exp e0 --model M2
$R --exp e0 --model M3
$R --exp e1 --model M2
$R --exp e1 --model M3
$R --exp e2 --model M3
$R --exp e3 --model M2
$R --exp e4 --model M3
$R --exp e5 --model M3
echo "pilot done: results in results/"
