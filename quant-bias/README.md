# quant-bias — social bias propagation under weight-only LLM quantization

Code for the study described in [`../Future_Plan.md`](../Future_Plan.md).
Self-contained: the earlier compression work in `../living-inference/` is read-only
provenance and is never modified by anything here.

## Layout

| Path | Responsibility |
|---|---|
| `quantbias/common.py` | `.env` loading (HF token), logging, seeds, hashing, `Manifest` provenance, result paths |
| `quantbias/model_adapters.py` | One interface over GPT-2 (Conv1D) and Llama/Mistral/Qwen; weights exposed in `(out,in)` orientation; tied-head handling; layer-input catcher |
| `quantbias/quantization.py` | RTN and reference GPTQ (simulated); `PrecisionMap`; byte accounting `C(b)`; exact restore |
| `quantbias/pruning.py` | Verified per-row activation Wanda (E0, E5) |
| `quantbias/data.py` | BBQ, WinoBias, Discrim-Eval loaders; cluster-level splits; counterfactual pairs; C4 calibration mixtures |
| `quantbias/evaluate.py` | Complete-continuation scoring; official BBQ/WinoBias/Discrim-Eval metrics; ΔE_g, A, H; flip tables; pair gaps |
| `quantbias/trace.py` | Single-site residual injection; V_l,g, ρ_l,g; systematic fraction; first-token margins and the lemma check |
| `quantbias/statistics.py` | Paired cluster bootstrap, Holm, TOST equivalence, MDE, design effect, ICC |
| `quantbias/allocate.py` | Greedy exchange allocator under a byte budget; matched baselines |
| `quantbias/report.py` | Tables and figures |
| `quantbias/run_experiment.py` | E0–E5 drivers |
| `configs/` | Pinned models, dataset ids, frozen protocol and sampling profile |
| `results/` | All output. Nothing is written outside this directory. |

## Setup

```bash
cd Codes/quant-bias
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # CPU: see the note at the top of the file
```
The Hugging Face token is read from `../.env` (`HUGGINGFACE_TOKEN`) by `common.load_env()`.

## Tests

```bash
cd Codes/quant-bias
.venv/bin/pytest tests -q                 # 24 tests, no downloads (tiny random models)
.venv/bin/pytest tests -q -m network      # real dataset loaders
```

## Running

```bash
cd Codes/quant-bias
.venv/bin/python -m quantbias.run_experiment --exp e0 --model M1 --quick --device cpu   # 2 min, CPU
bash scripts/run_pilot.sh                 # minimum publishable pilot, one GPU
bash scripts/run_all.sh budget            # full study
bash scripts/run_all.sh full              # full study, no sampling caps
```

Ordering: E4 reads E2's site map, E5 reads E4's allocation. `scripts/` encodes this.
Outputs land in `results/<exp>/<model_tag>/` as `records_*.jsonl` (one row per example),
`summary_*.json`, figures, and `REPORT.md`. `--quick` writes to `<model_tag>-quick/`.

## Sampling profiles

`configs/experiments.yaml: sampling.profile` selects `budget` (default) or `full`.

The bootstrap resamples **template clusters**, so interval width is governed by the
number of clusters, not the number of examples. The `budget` profile therefore keeps
every cluster and every design cell, and caps only how many instantiations are drawn
inside each cell:

| Benchmark | Full | Budget | Clusters kept |
|---|---|---|---|
| BBQ | 58,492 | ~5,500 | 343 / 343 |
| WinoBias | 3,168 | 3,168 | 1,584 / 1,584 |
| Discrim-Eval | 9,450 | ~5,250 | 70 / 70 |

`statistics.subsample_cost` quantifies the trade: at an intra-cluster correlation of
0.3 the budget profile retains 88% of the effective sample size for 9% of the compute.
Run `statistics.icc_from_records` on the pilot records to replace the assumed ICC with a
measured one before fixing the final sample size.

## Declared choices

* Scoring: sum of log-probabilities of the complete candidate continuation (`configs/experiments.yaml: scoring.norm`).
* WinoBias is scored as "who does `<pronoun>` refer to?" with the two occupations as candidates. This is a causal-LM adaptation, not the original coreference evaluator.
* Discrim-Eval appends `Please answer the above question with either "yes" or "no".` and compares ` yes` / ` no`.
* Quantization is simulated (weights quantized then dequantized). Bytes are *accounted* with fp16 scales, packed zero-points and 32-bit row padding; throughput must be measured separately on a packed backend.
* AWQ is not re-implemented. Load an externally quantized checkpoint instead; a silent fallback to RTN is deliberately an error.
