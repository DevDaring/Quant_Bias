# Findings: bias propagation under weight-only quantization

Run completed 2026-09-11 05:58:56 UTC. 36/36 stages, 0 failures, 7 models
(124M–8B, four architecture families), budget sampling profile.

This document states what the data supports and what it does not. Several
intended contributions are **not** supported; those are recorded here rather
than presented as successes.

---

## 1. What did not work

### 1.1 The bridge to `living-inference` (E6) fails its own sanity check

The bridge was meant to show that the prior study's utility-only layer
sensitivity (Lyapunov `rho`) does not predict group-conditioned harm — the
evidence for contribution axes A3/A4.

Before that claim can be made, `rho` must be shown to track *this* study's own
utility measure at the same layers. It does not:

| model | align `rho_s` (prior rho vs this study's PPL utility) | verdict |
|---|---|---|
| gpt2_small | −0.26 | align failed |
| gpt2_medium | −0.00 | align failed |
| lfm2_2.6b | −0.39 | align failed (negative) |
| qwen3_5_2b | −0.06 | align failed |
| mistral_7b_v0_1 | +0.00 | align failed |
| qwen3_8b | −0.07 | align failed |
| llama_2_7b | no prior rho available | no claim |

**Consequence.** The observed null relation between prior `rho` and harmful
flips is uninterpretable: it cannot be distinguished from the two pipelines
never having been comparable. **No A3/A4 claim can be made from this data.**

Possible causes to investigate before any re-run: the layer-index mapping
(`rho[t]` is a ratio between layers `t` and `t+1`, mapped here as layer `L` →
index `L−1`), and the fact that the prior `rho` was measured under *random
embedding perturbation* while this study perturbs via *actual quantization
residuals* — these may simply not be the same quantity.

### 1.2 The allocator does not beat uniform quantization (E4)

Mistral-7B, matched byte budget, frozen objective `J = H + A + gap`:

| method | J (lower better) | H | acc | PPL | MiB |
|---|---|---|---|---|---|
| `uniform4` (baseline) | **0.0902** | 0.1389 | 0.819 | 6.36 | 3959 |
| `greedy_bias_aware` (ours) | 0.1048 | 0.1111 | 0.820 | 6.33 | 3989 |
| `margin_only` | 0.1062 | 0.1111 | 0.826 | 6.30 | 4354 |
| `random` | 0.1193 | 0.1667 | 0.817 | 6.33 | 4354 |
| `ppl_only` | 0.1349 | 0.1389 | 0.828 | 6.29 | 4354 |

Uniform 4-bit scores **better** on the frozen objective than the allocator.
The allocator is better on worst-added-harm `H` alone, but that was not the
declared objective. Per `Future_Plan.md` §13, this is a negative result for
the A8/A9 contribution and must be reported as one.

### 1.3 The comparator differences are not statistically distinguishable (E7)

Mistral-7B, `H` with paired cluster-bootstrap CIs:

| method | H | 95% CI |
|---|---|---|
| `fair_gptq4` | 0.0556 | [0.064, 0.205] |
| `cwp4_p0.01` | 0.1389 | [0.103, 0.333] |
| `gptq4` | 0.1667 | [0.083, 0.333] |
| `debias_sparsegpt` | 0.2083 | [0.167, 0.358] |
| `sparsegpt` | 0.3472 | [0.279, 0.500] |

Every quantization method's CI overlaps every other's. On Qwen3-8B the point
estimates are **identical** (`gptq4` = `fair_gptq4` = `cwp` = `sparsegpt` =
0.1667). The apparent "Fair-GPTQ wins" on Mistral is within noise.

**Why `H` is unreliable here.** `H = max_g ΔE_g` is a maximum over groups, and
the smallest BBQ groups in the budget profile have n = 24–36. One flipped
example moves `ΔE_g` by ~0.03–0.04, so `H` is dominated by whichever tiny group
happens to move. Two symptoms confirm the metric is straining: the values are
coarse ratios with small denominators (1/18, 5/36, 1/6, 25/72), and
`fair_gptq4`'s point estimate (0.0556) falls **outside** its own bootstrap CI —
expected behaviour for a max-statistic, whose bootstrap distribution is biased
upward.

**Any future claim on `H` needs larger per-group samples** (the `full` sampling
profile, or BBQ categories pooled to raise the minimum group size).

---

## 2. What the data does support

### 2.1 The group-conditioned residual diagnostic tracks harm, on all 7 models

`V_final` (group-conditioned relative residual energy at the output) vs
observed harmful flips, Spearman, measured independently of the prior study:

| model | `rho_s` | n sites |
|---|---|---|
| gpt2_small | 0.76 | 27 |
| gpt2_medium | 0.64 | 39 |
| qwen3_8b | 0.57 | 63 |
| qwen3_5_2b | 0.56 | 42 |
| llama_2_7b | 0.55 | 32 |
| lfm2_2.6b | 0.47 | 57 |
| mistral_7b_v0_1 | 0.37 | 59 |

**Positive on 7/7 models, 0.37–0.76**, across 124M–8B parameters and four
architecture families (GPT-2, Llama/Mistral, Qwen3 hybrid linear-attention,
LFM2 conv+attention). This stands on this study's own measurements and does
not depend on the prior study, so §1.1's failure does not touch it.

This is the strongest result in the study.

### 2.2 Propagation beats perplexity-regret as a site predictor, but weakly

E2 held-out site ranking (layer granularity), Spearman vs harmful flips:

| model | `propagation_margin` | `ppl_regret` | `margin_only` |
|---|---|---|---|
| qwen3_5_2b | 0.54 | 0.10 | 0.00 |
| lfm2_2.6b | 0.46 | 0.15 | 0.00 |
| llama_2_7b | 0.36 | 0.12 | 0.00 |
| gpt2_medium | 0.29 | −0.15 | 0.00 |
| mistral_7b_v0_1 | 0.18 | 0.15 | 0.00 |
| qwen3_8b | 0.02 | −0.01 | 0.00 |
| gpt2_small | −0.31 | −0.19 | 0.00 |

Mean 0.22 vs 0.02 — propagation is the better predictor on average, but it is
**inconsistent** (negative on gpt2_small, ~0 on qwen3_8b). Not a reliable
predictor as currently formulated.

`margin_only` is 0.00 everywhere because it is constant by construction:
`margin_dense` is measured on the fixed dense capture and does not vary by
site. It is a null baseline, not a competitive one, and should either be
redefined per-example or dropped.

### 2.3 Infrastructure results that hold

- **Exact weight round-trip and restore** verified on all 7 models (E0).
- **Byte accounting** never charges a lower bit-width more than a higher one;
  CWP's sparse high-precision overlay is honestly charged 48 bits/weight, which
  is why it shows 4358 MiB against everything else's 3959 MiB.
- **Margin-lemma violations** are rare and bounded: 2 (qwen3_5_2b), 5
  (mistral_7b_v0_1), 10 (llama_2_7b) across thousands of comparisons —
  consistent with bf16 rounding rather than a broken bound.

---

## 3. Honest summary

The study as executed does **not** support an allocator-superiority paper.
It supports a **measurement paper**: a group-conditioned residual diagnostic
that tracks quantization-induced harm consistently across seven models and four
architecture families, together with a clear negative result that neither the
allocator nor any published comparator separates from uniform quantization once
confidence intervals are computed at these sample sizes.

The most useful next step is not more models but **more examples per group** —
the `full` sampling profile — so that `H` stops being dominated by n=24 groups.
