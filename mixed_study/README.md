# mixed_study — the integrated compression-to-outcome study

Implements `../Next_Plan.md`. Joins `../quant-bias` (quantization interventions,
demographic tasks, E0–E7 records) and `../living-inference` (local error
accounting, legacy perturbation profiles, read-only provenance) into one
testable chain:

```
compression operation → local residual → propagation → answer-score change
                      → task damage / stereotype-aligned error / group disparity
```

Reuses `quantbias` as a library (via `.venv -> ../quant-bias/.venv`); never
rewrites a source result. Every corrected output lands under `results/v2/`.

## What runs now (CPU, from the saved records)

```bash
.venv/bin/python -m mixed_study.run audit     # P0 → results/v2/audit/AUDIT.md
.venv/bin/python -m mixed_study.run ladder    # §6  → results/v2/group_prediction/LADDER.md
.venv/bin/python -m mixed_study.run legacy --model M1   # §5.1 legacy-trace reproduction (GPT-2 fits on CPU)
.venv/bin/pytest tests -q                     # 18 tests; B1 and the directional predictor live-fire on a tiny model
```

## What needs a GPU (B1 panel, equal-cost restoration)

```bash
.venv/bin/python -m mixed_study.run legacy  --model M3 --device cuda
.venv/bin/python -m mixed_study.run b1      --model M3 --device cuda --granularity layer
.venv/bin/python -m mixed_study.run b1      --model M3 --device cuda --granularity component
.venv/bin/python -m mixed_study.run restore --model M3 --device cuda
```
Then the same for `M5`. `--quick` runs any of these on two samples first.

## Module map → plan section

| Module | Next_Plan | What it corrects or adds |
|---|---|---|
| `harm.py` | §3.1 | Four *separate* outcomes with declared denominators: task damage, stereotype-aligned damage, group disparity (with independent-cluster counts and an eligibility rule), decision sensitivity. No pooled "harmful flip". |
| `margins.py` | §3.2 | Argmax stability certified by the **dense top-two margin**; gold preservation by a **positive gold margin**. The plan's counter-example is a test. |
| `pairs.py` | §3.3 | Each example scored at **its own** label; Discrim-Eval / WinoBias pairs are matched by construction, BBQ template pairs are flagged **unaudited** and excluded from confirmatory claims. |
| `competence.py` | §3.4 | Dense competence flags per model; sum-vs-mean rule sensitivity; candidate-length confound. |
| `uncertainty.py` | §3.5 | Dirichlet-multiplier bootstrap (every cluster and group present in every draw); **paired** method differences on the same rows; finite-sample coverage simulation on this design. |
| `audit.py` | P0 | Runs all of the above on E1/E2/E7 records; fixed-granularity correlations with the intervention-size confound made explicit. |
| `legacy_trace.py` | §5.1 | Numerically reproduces the legacy random-perturbation trace with the new capture; pins the `rho[i] = v[i+1]/v[i]` convention; exposes both layer→ρ mappings. |
| `residuals.py` | §5.4–5.6 | Injects actual / sign-reversed / norm-matched-random residuals at one boundary through a dense downstream network, full sequence shape, per-token norms. |
| `matched_residuals.py` | §5 (B1) | Prespecified sites × sources × directions, with per-example behaviour. |
| `directional.py` | §6 | `Δs ≈ ∇_h s · residual` on the full-continuation contrast, validated against the finite-difference actual change. |
| `group_prediction.py` | §6 | Nested predictor ladder at fixed granularity, leave-layer-out, so group conditioning has to earn its keep. |
| `matched_restoration.py` | §5.7 | Restoration arms matched on added bytes, component kind and size bucket; repeated random schedules; an equal-cost assertion. |
| `configs/integrated_v2.yaml` | P0 | Frozen estimands, gates, pair kinds, bootstrap scheme, B1 panel. |

## What the reanalysis established (see `results/v2/`)

* **Margin "violations" were artefacts.** Under the corrected test there are **0** argmax and **0** gold violations across 28 model/config pairs; the source run's 17 came from testing |gold margin| against argmax change.
* **The interval was the problem, not the data.** The multiplier bootstrap keeps **300/300** draws where the source kept 293/2000. Only **18 of 43** BBQ groups have ≥3 independent templates.
* **No comparator separates from uniform GPTQ on H** once compared as paired differences; several do on accuracy, in both directions across models.
* **Every dense model fails BBQ-ambiguous** (1.5–15.7% vs 33% chance): it almost never selects "unknown". Ambiguous-context metrics are not fairness outcomes under this adaptation.
* **The legacy trace reproduces** on GPT-2 Small (shape Spearman 1.0). The pipelines measure the same quantity, so the E6 alignment failure is a finding about random-perturbation ρ, not plumbing.
* **Direction matters (pilot).** On GPT-2, actual residuals reach the logits 2.1–2.4× harder than norm-matched random directions at equal per-token norm. This is the B1 hypothesis; it needs the M3/M5 panel to be a result.
* **No predictor is reliable out-of-layer** at fixed granularity. Group conditioning does not show consistent added value.

## Not yet done

The B1 panel and equal-cost restoration on Mistral-7B and Qwen3-8B (GPU),
the held-out confirmation set (`holdout.status: not_yet_collected`), and any
comparison against the published authors' code rather than re-implementations.
