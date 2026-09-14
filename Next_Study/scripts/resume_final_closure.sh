#!/usr/bin/env bash
# Resume after an interruption: every stage checks its own completion marker, and
# the directional ladder resumes per layer, so re-running the frozen order is safe.
exec bash "$(dirname "$0")/run_final_closure.sh" "${1:-full}"
