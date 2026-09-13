# Findings v2: the integrated study (mixed_study, full run)

Run completed 2026-09-13 04:26 UTC on one H100 (13/13 stages, 0 failures,
~30 min GPU). Two-sample smoke run first, which caught two GPU-only bugs.
Every claim below points at a file under `results/v2/`; source records under
`quant-bias/results/` were never modified. P0 reanalysis findings are in
`audit/AUDIT.md` and `group_prediction/LADDER.md`.

---

## 1. The legacy trace reproduces — once dtype is matched (§5.1)

`legacy/*/legacy_reproduction.json`. Fresh random-perturbation profiles (3
seeds, eps=0.01, seq 512) vs the saved `living-inference` profile.

| model | dtype | shape Spearman | saved within 3σ of fresh | verdict |
|---|---|---|---|---|
| GPT-2 Small | fp32 (both) | **1.00** | 100% | reproduced |
| Mistral-7B | bf16 (study default) | 0.32 | 61% | not reproduced |
| Mistral-7B | **fp16 (legacy's dtype)** | **0.93** | **100%** | **reproduced** |
| Qwen3-8B | bf16 | 0.31 | 49% | not reproduced |
| Qwen3-8B | fp16 | 0.78 | 49% | partial |

**What this establishes.** The saved 7B profiles encode fp16 rounding: their
relative energies (1e-5–1e-6) sit at bf16's precision floor. With dtype
matched, the two pipelines measure the same quantity on Mistral. The E6
"alignment failure" in `quant-bias/results/FINDINGS.md` was therefore, on
Mistral, a dtype artefact — not evidence that the pipelines differ. Qwen3-8B is
only partly explained by dtype; its saved profile came from a different script
(`colab_qwen3_lyapunov.py`) and something else contributes. Any transfer claim
must state dtype, and the rho index convention is now pinned
(`rho[i] = v[i+1]/v[i]` over block outputs).

## 2. B1: residual direction matters, modestly, and less on larger models (§5)

`b1/*/b1_cells_{layer,component}.json`. Same model, same inputs, same boundary:
the actual compression residual vs its sign-reversal vs 3 norm-matched random
directions, propagated through the dense downstream network.

Ratio of actual to norm-matched-random (>1 means direction carries something
magnitude does not):

| model | panel | source | drift amplification | logit ℓ∞ at output | cos(random drift, actual drift) |
|---|---|---|---|---|---|
| GPT-2 | layer | rtn4 / gptq4 / wanda50 | 1.10 / 1.18 / 1.10 | **1.52 / 0.97 / 1.35** | ~0.00 |
| GPT-2 | component | rtn4 / gptq4 / wanda50 | 1.11 / 1.26 / 1.12 | **1.60 / 1.48 / 1.15** | ~0.00 |
| Mistral-7B | layer | rtn4 / gptq4 / wanda50 | 1.13 / 1.10 / 1.24 | 1.16 / 1.18 / 1.28 | +0.10 / +0.10 / +0.05 |
| Mistral-7B | component | rtn4 / gptq4 / wanda50 | 1.05 / 1.05 / 1.11 | 1.10 / 1.09 / 1.17 | +0.29 / +0.30 / +0.19 |
| Qwen3-8B | layer | rtn4 / gptq4 / wanda50 | 1.10 / 1.15 / 1.15 | 1.14 / 1.10 / 1.29 | +0.08 / +0.10 / +0.04 |
| Qwen3-8B | component | rtn4 / gptq4 / wanda50 | 1.06 / 1.05 / 1.10 | 1.08 / 1.10 / 1.12 | +0.24 / +0.28 / +0.17 |

Three consistent observations across 6 panels × 4 sources (192 cells):

1. **Direction is real but modest.** Actual residuals reach the output 1.0–1.6×
   harder than random directions of identical per-token norm. The effect is
   largest on GPT-2 and for pruning (Wanda), smallest on the 7B models.
2. **8-bit residuals propagate like noise.** `rtn8` gives ratios of 1.00–1.05
   everywhere: at that magnitude the residual has no exploitable structure.
   This is the boundary the plan asked for.
3. **Large models funnel perturbations.** On the 7B models even random
   directions arrive at the output partly aligned with the actual drift
   (cos +0.1 to +0.4; ~0 on GPT-2). The downstream map has low effective rank,
   which is *why* direction matters less there: magnitude is what survives.

Sign-reversed residuals amplify like the actual ones (ratio 0.95–1.04):
propagation is close to odd-symmetric at these magnitudes.

### 2.1 Hidden-state amplification is not behavioural harm

Mistral-7B, whole-layer rtn4, `b1_cells_layer.json`:

| layer | local rel. error | drift amplification (actual) | logit ℓ∞ | answer flips /200 |
|---|---|---|---|---|
| 0 | 1.3e-1 | **32.3** | 0.26 | 4 |
| 4 | 8.1e-2 | 9.6 | 0.22 | 6 |
| 9 | 7.4e-2 | 4.1 | 0.27 | 4 |
| 13 | 7.0e-2 | 3.4 | 0.20 | 1 |
| 18 | 6.2e-2 | 2.2 | 0.26 | 2 |
| 22 | 3.9e-2 | 1.8 | 0.19 | 1 |
| 27 | 3.3e-2 | 1.5 | 0.21 | 2 |
| 31 | 5.2e-2 | **1.0** | **0.36** | **5** |

Energy amplification is a monotone function of depth (early layers amplify
30×, the last layer 1×), but **flips do not follow it**: layer 31 amplifies
nothing and flips most; layer 0 amplifies most and flips less. What tracks
flips is the direct logit sensitivity (ℓ∞), not the hidden-state drift. This is
the mechanistic reason `V_final` (a drift-energy diagnostic) was an
inconsistent predictor in `LADDER.md`, and it is the concrete argument for the
directional score-gradient predictor in `directional.py`.

## 3. Equal-cost restoration: below the resolution of this evaluation (§5.7)

`restoration/*/restoration.json`. From a uniform-4 start, restore 4 components
to 16-bit — predicted, utility-matched, or 5 random schedules — all
bucket-matched on kind and parameter count, bytes equal within 1%
(`equal_cost_ok: true`).

| model | uniform4 | predicted | utility-matched | random ×5 (mean / min) |
|---|---|---|---|---|
| Mistral-7B | 0.0366 | 0.0366 | 0.0366 | 0.0366 / 0.0366 |
| Qwen3-8B | 0.0000 | 0.0000 | 0.0000 | 0.0000 / 0.0000 |

(harmful transitions per dense-correct, BBQ disambiguated, 200 final examples)

Every arm is identical. Restoring 4 of ~224 components moves nothing that 200
examples can detect; on Qwen3-8B uniform-4 already produces zero harmful
flips on this set, so nothing can improve. **This is not "predicted sites don't
help" — it is "the experiment cannot resolve single-site restoration at this
sample size and intervention scale."** A detectable version needs either many
more examples or a larger restored set, chosen and frozen in advance.

## 4. What the integrated study can now claim

**Established.**
- A dtype-controlled reproduction of the legacy propagation profile, and a
  pinned index convention, so the two projects are demonstrably one framework.
- Residual *direction* contributes beyond magnitude, modestly, with a clean
  boundary (8-bit ≈ noise) and a size-dependence explained by output funnelling.
- Hidden-state drift is the wrong target: logit sensitivity, not energy
  amplification, tracks answer flips across depth.
- The corrected measurement protocol (four separate outcomes, valid margin
  test, support-preserving bootstrap, eligibility by independent templates).

**Not established, and why.**
- Group-conditioned prediction: no consistent gain over global diagnostics
  out-of-layer (`LADDER.md`).
- Selective restoration at fixed cost: unresolved at n=200 / 4 sites.
- Any comparator superiority: paired ΔH intervals all include zero
  (`AUDIT.md` §6).
- Qwen3-8B legacy profile: only partly reproduced.

**Next decisive experiment.** Replace `V_final` with the validated
score-gradient predictor (`directional.py`, already verified against finite
differences on the pilot), evaluate it on the same fixed-granularity,
leave-layer-out protocol, and size the restoration test from the pilot's
variance before running it.
