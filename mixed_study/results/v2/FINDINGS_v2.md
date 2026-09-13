# Findings v2: the integrated study (mixed_study, full run)

Two H100 runs on 2026-09-13. Run 1 (04:26 UTC, 13 stages, ~30 min GPU) covers
sections 1–3. Run 2 (09:56 UTC, 9 further stages, ~1h45 GPU) closes the three
P0/P1 gaps that Next_Plan.md left open: sections 5–7. 22/22 stages, 0
failures, validator (`scripts/validate_v2.py`) passes on every output file.
Each run was preceded by a two-sample smoke run of the same code path. Every
claim below points at a file under `results/v2/`; source records under
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
That version was sized from a power simulation and run; see section 6.

## 4. Power: what a restoration test on this data can and cannot see (§8)

`power/mistral_7b_v0_1_rtn4.json`. Simulation on the actual final-split
cluster structure (156 template clusters, 1,766 disambiguated BBQ rows, 1,479
dense-correct, 77 harmful events at rtn4). The null is not "no change" but
*net-zero churn*: two equal-cost 4-bit methods (rtn4 vs gptq4) disagree on 62%
of the answers they break, so a restoration that merely reshuffles which items
flip must not read as an improvement. Under that null the full final split
detects a ≥35% reduction in harmful transitions at 94% power (25%: 3%; 15%:
0%). This threshold was fixed before section 6 was run.

## 5. The directional predictor on the ladder protocol (§6)

`directional/{gpt2_small,mistral_7b_v0_1,qwen3_8b}/directional_sites.json`.
At every whole-layer site (12 / 32 / 36), quantise that layer alone to 4-bit
and, for 48 held-out BBQ examples, compare the first-order prediction
⟨∇s, Δh⟩ of the answer-score change against the rescored actual change. No
fitted parameters.

| model | sites | first-order validity (mean / median Pearson over sites) | predicted flip-rate vs observed any-flip (Spearman across sites) | vs stereotype-aligned flip |
|---|---|---|---|---|
| GPT-2 Small | 12 | 0.79 / 0.88 | ρ=0.58, p=0.05 | ρ=0.34, p=0.29 (4 events) |
| Mistral-7B | 32 | 0.53 / 0.61 | ρ=0.66, p<0.001 | ρ=0.26, p=0.15 (14 events) |
| Qwen3-8B | 36 | 0.69 / 0.79 | ρ=0.28, p=0.10 | undefined (0 events) |

Three things this establishes. (i) The first-order term is a valid local
model of what a single quantised layer does to the answer score: on every
model most sites have r>0.5 (92% / 59% / 72% of sites), and the sign of the
predicted change agrees with the actual sign on 98–99% of examples
(`flip_agreement`). (ii) It carries information the energy proxies do not:
on the same sites `V_final_proxy` is uncorrelated with observed flips on all
three models (|ρ|≤0.39, all p>0.2), and `pred_mean_margin_loss` is likewise
flat. (iii) Its site ranking predicts *where answers change* on the 7B model
that flips most (Mistral, ρ=0.66), weakly on Qwen3-8B (ρ=0.28).

What it does not establish, and why: the correlation with *stereotype-aligned*
flips is not resolved. With 48 examples per site a single layer at 4-bit
produces 0–3 flips per site (16 / 31 / 16 flips in total; 4 / 14 / 0
stereotype-aligned), so the harmful-flip outcome is a count of a handful of
events. On Qwen3-8B there are none at all, and the correlation is undefined by
construction rather than reported as zero. The ladder shows the predictor is
valid and directional; whether the sites it ranks highest are the sites that
matter for *harm* is answered, with adequate power, by the restoration test.

## 6. Power-sized restoration: the predictor is diagnostic, not prescriptive (§5.7, §8)

`restoration/{mistral_7b_v0_1,qwen3_8b}-k{8,16}-n7491/restoration.json`.
Same design as section 3 with the two changes the power analysis asked for:
the whole final split (7,491 examples; 1,766 disambiguated BBQ rows scored per
arm) and k=8 or k=16 restored components instead of 4. All eight arms per cell
are equal-cost (`equal_cost_ok: true`, bytes identical).

New stereotype-aligned errors among dense-correct items (lower is better):

| model, k | uniform-4 start | predicted sites | utility-matched | random ×5 |
|---|---|---|---|---|
| Mistral-7B, k=8 | 36 | 35 (−3%) | 32 | 28, 32, 34, 39, 40 |
| Mistral-7B, k=16 | 36 | 27 (−25%) | 35 | 30, 36, 37, 37, 39 |
| Qwen3-8B, k=8 | 34 | 26 (−24%) | 34 | 29, 32, 33, 34, 38 |
| Qwen3-8B, k=16 | 34 | 33 (−3%) | 31 | 29, 30, 32, 35, 35 |

Rates per dense-correct: Mistral uniform 0.0521 → predicted 0.0467 (k=8),
0.0406 (k=16); Qwen uniform 0.0543 → 0.0448 (k=8), 0.0455 (k=16).

Read against section 4: the predicted arm reduces harmful transitions by 3–25%
(median 14%), below the 35% the design can detect and inside the spread that
five *random* equal-cost schedules produce on the same items (−22% to +11% on
Mistral k=8). In two of four cells (Mistral k=16, Qwen k=8) the predicted arm
is the best of all eight arms, below every random schedule; in the other two
it is indistinguishable from the uniform start while random schedules and the
utility-matched arm (chosen by task loss, no bias signal) do as well or
better. Which two cells are which does not follow k or model, as a real
effect would. **So the
conclusion is now a resolved negative rather than an unresolved one: at 8–16
restored components on a 7B model, choosing the components by the bias
predictor does not buy a detectable reduction in stereotype-aligned errors
over choosing them by utility or at random.** Combined with section 5, the
predictor tells you where a quantised layer will move answers; it does not
give a repair recipe at this intervention scale, and any paper claim must be
phrased that way.

## 7. Held-out confirmation on prompts never used before (§3.6)

`holdout/{mistral_7b_v0_1,qwen3_8b}/holdout_discrim_implicit.json`. The
Discrim-Eval *implicit* split (9,450 prompts, 70 decision questions) was never
loaded in quant-bias or in any earlier mixed_study stage; it was scored once,
under the frozen rule, for the three configurations named in advance.

| model | config | decision change rate: explicit (E1, n=2,325) → implicit (held-out, n=9,450) | mean abs gap change (implicit) |
|---|---|---|---|
| Mistral-7B | rtn4 | 0.100 → 0.098 | −0.002 |
| Mistral-7B | gptq4 | 0.098 → 0.100 | −0.001 |
| Mistral-7B | rtn8 | 0.026 → 0.026 | +0.000 |
| Qwen3-8B | rtn4 | 0.099 → 0.083 | +0.007 |
| Qwen3-8B | gptq4 | 0.067 → 0.097 | −0.007 |
| Qwen3-8B | rtn8 | 0.020 → 0.010 | −0.000 |

Both findings that the explicit split produced replicate on prompts that had
no role in any earlier choice: (a) 4-bit weight-only quantisation changes
roughly one hiring/lending-style decision in ten on 7B models while 8-bit
changes one to three in a hundred, and (b) the demographic *gap* between
matched prompts moves by less than one percentage point in every
configuration — the decisions move, but not systematically against a group.
The per-attribute mean shift in P(yes) is identical across age, gender and
race within each configuration (e.g. −0.026 for all three on Mistral gptq4),
which is what "sensitivity without disparity" looks like in the raw numbers.
Scope: this confirms the Discrim-Eval decision-sensitivity results only. The
BBQ findings were selected and evaluated on the same benchmark and still
need a fresh template set to be called confirmed.

## 8. What the integrated study can now claim

**Established.**
- A dtype-controlled reproduction of the legacy propagation profile, and a
  pinned index convention, so the two projects are demonstrably one framework
  (§1).
- Residual *direction* contributes beyond magnitude, modestly, with a clean
  boundary (8-bit ≈ noise) and a size-dependence explained by output
  funnelling (§2).
- Hidden-state drift is the wrong target: logit sensitivity, not energy
  amplification, tracks answer flips across depth (§2.1); on the ladder the
  first-order score-gradient predictor is valid per example and per site while
  the energy proxies are uncorrelated with flips (§5).
- Selective restoration at fixed cost does **not** reduce stereotype-aligned
  errors detectably at k=8–16 on 7B models, in a test powered to see a 35%
  reduction (§4, §6). A resolved negative.
- Decision sensitivity without group disparity under 4-bit quantisation,
  confirmed on a held-out split (§7).
- The corrected measurement protocol (four separate outcomes, valid margin
  test, support-preserving bootstrap, eligibility by independent templates,
  net-zero-churn null for interventions).

**Not established, and why.**
- Group-conditioned prediction: no consistent gain over global diagnostics
  out-of-layer (`LADDER.md`).
- Predictor ↔ *harmful* flips on the ladder: too few events at 48 examples
  per site (§5); the powered answer is the negative in §6.
- Any comparator superiority: paired ΔH intervals all include zero
  (`AUDIT.md` §6).
- Qwen3-8B legacy profile: only partly reproduced.
- BBQ results on a held-out template set: not run (needs new templates).

**Still open by plan design (P2).** The allocator frontier sweep, the authors'
own comparator implementations, and a packed-backend (real int4 kernel) audit
of the simulated-quantisation assumption.
