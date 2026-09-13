# Quant_Bias

Code and experiment results for a study of how weight-only quantization of language
models changes socially relevant decisions, and where inside the network the change
starts.

## Layout

- `quant-bias/` — package `quantbias`: seven-model measurement study (E0–E7). Weight-only
  RTN and GPTQ at 8 and 4 bits with exact byte accounting; BBQ, WinoBias and Discrim-Eval
  scoring by candidate log-likelihood; bias-aware comparators; allocator. Results under
  `quant-bias/results/` (per-example records, per-experiment JSON).
- `mixed_study/` — package `mixed_study`: the integrated study. Corrected measurement
  protocol (four outcomes, margin check, support-preserving bootstrap, template
  eligibility), single-site residual injection with sign-reversed and norm-matched random
  controls, directional predictor on the layer ladder, power-sized restoration, held-out
  confirmation, and the fp16 reproduction of the earlier propagation profiles. Results
  under `mixed_study/results/v2/`.
- `quant-bias/provenance/living_inference_results/` — vendored result files of the earlier
  compression-error study that the propagation reproduction is checked against.
- `quant-bias/scripts/`, `mixed_study/scripts/` — GPU provisioning (Akash), VM run
  supervisors with per-stage git push, watchdogs, and result validators.

## Running

```bash
cd quant-bias && python -m venv .venv && .venv/bin/pip install -e .
cd ../mixed_study && ln -s ../quant-bias/.venv .venv
.venv/bin/python -m mixed_study.run audit      # CPU: rebuilds the reanalysis from stored records
.venv/bin/python scripts/validate_v2.py        # checks every result artifact
bash scripts/vm_run.sh full                    # GPU stages (one 80 GB GPU, ~2.5 h)
```

Keys are read from `.env` (never committed); see `quant-bias/configs/experiments.yaml` for
the run configuration (seed 20260908, group size 128, GPTQ damping 0.01, budget sampling
profile).
