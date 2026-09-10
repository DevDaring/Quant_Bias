# Research plan: Who Loses Precision? Tracing and Controlling Social Bias in Quantized Language Models

Prepared 8 September 2026 (revision 2). Status: proposed research. The bias experiments below have not been run. Every number quoted from the repository is read from a saved file, not reproduced during this revision.

Revision 2 fixed the model set to five checkpoints with saved results, fixed the evaluation data to three benchmarks with verified Hugging Face identifiers, added seven 2024–2026 comparators, and defined the contribution axes on which this study must differ from them.

Revision 3 (8 September 2026) adds the sampling budget of Section 5.5, records the measured cost of that budget, and points the implementation section at the new `Codes/quant-bias/` package, where the code now lives.

---

## 1. Research question

Post-training quantization is now the default way to deploy a language model on limited hardware. Several 2025–2026 studies report that quantization changes social-bias outcomes while aggregate accuracy and perplexity stay flat. None of them explains *where inside the network* the change is produced, and none allocates precision to control it under a stated memory budget.

**Central question.** Can group-conditioned error propagation identify the layers and components at which weight quantization changes socially relevant decisions, and can a fixed memory budget be allocated across components to limit those changes?

Here “bias” means social stereotyping and disparities in task outcomes across demographic groups. Numerical rounding error is a candidate mechanism, not the harm itself. An affine layer's additive bias parameter is a different concept and is never called “bias” in this document.

The main study uses weight-only quantization. Pruning enters only as a bridge to the existing repository and to one 2026 pruning comparator. Low-rank approximation, activation quantization, layer removal, and semantic caching are optional extensions and are not part of the minimum study.

---

## 2. Models: five checkpoints with saved results

The selection rule is simple: a model enters the study only if the repository already holds a dense baseline and at least one compression result for it. The five below satisfy that rule and span 124 M to 8 B parameters across four architecture families.

| # | Checkpoint (pinned HF id) | Params | Dense PPL on WikiText-103 test (saved file) | Existing compression evidence | Role in this study |
|---|---|---|---|---|---|
| M1 | `openai-community/gpt2` | 124 M | 22.98 in `results/gpt2_ablation.json`; 26.76 in `results/gpt2_activation_wanda.json` (different token budget) | Single-layer and component ablations over 468 matrices, 0 bound violations; activation-Wanda vs simplified weighting (MLP fc: 32.19 vs 464.81) | Adapter and instrumentation checks; Conv1D orientation test; cheap seed sweeps |
| M2 | `Qwen/Qwen3.5-2B` | 2 B | 13.54 in `results/qwen3_5_2b_sota_comparison.json` | Per-component regret map (`results/qwen35/qwen3_5_2b_per_group.json`); Wanda 50% → 25.99, OWL → 31.62 | Smallest capable model; full E1–E4 at low cost |
| M3 | `mistralai/Mistral-7B-v0.1` | 7.2 B | 6.15 in `results/mistral_micro/…_baseline.json`; 6.31 in `results/h100/unified_summary.json` | 15 saved studies: single-layer regret (L0 0.61, L1 0.60, then ≤ 0.09), per-group, cumulative, threshold, Lyapunov, Wanda 50% → 8.44 | Primary model for E2–E4; direct comparison to Rath and Maliakkal (2026), who report BBQ on Mistral-7B |
| M4 | `NousResearch/Llama-2-7b-hf` | 6.7 B | 6.59 in `results/llama_2_7b_hf_llama_comparison.json` | Global vs per-row Wanda (18.36 vs 10.37 at 50%); ATP allocation 8.81; layer-removal with HellaSwag/ARC-E/WinoGrande/PIQA in `results/llama_real_rm.json` | Second 7 B family for architecture transfer; only model with saved downstream scores |
| M5 | `Qwen/Qwen3-8B` | 8 B | 9.27 in `results/h100/qwen3_8b_results.json`; 10.38 in `unified_summary.json` | Lyapunov trajectory over 36 layers; in-place Wanda 50% is catastrophic (PPL 6.4 × 10⁴) | Largest model; shared with Debias-SparseGPT (2026) for a like-for-like pruning comparison |

Two checkpoints of the same model report different dense perplexities because the saved runs used different token budgets, chunk sizes, and label-shift conventions. The new pipeline fixes one evaluation setting and re-measures every dense baseline before any quantized number is reported. Old numbers are cited as provenance only.

**Excluded and why.** `openai-community/gpt2-medium` has results but duplicates M1's architecture at a scale no reviewer will care about. `LiquidAI/LFM2-2.6B` is a hybrid convolution–attention model; its layer map does not fit the per-layer intervention design and would need a separate adapter. `openai/gpt-oss-20b` ran out of memory on an 80 GB H100 (`unified_summary.json`). `microsoft/phi-2` and `TinyLlama-1.1B-Chat` appear only as configuration entries with no saved results.

**Quantizer support to verify in week 1.** GPTQ and AWQ reference implementations support M2–M5. M1 stores projections as `Conv1D`; the plan uses the in-house round-to-nearest quantizer for M1 and treats GPTQ on M1 as optional. If a quantizer fails on a model, the failure is recorded and that cell is reported as missing, not as zero harm.

---

## 3. Data: three evaluation benchmarks, one calibration corpus

### 3.1 Primary and secondary evaluation

| Tier | Benchmark | Verified HF id | Task format | Groups | Why it is here |
|---|---|---|---|---|---|
| 1 (primary) | *BBQ* (Parrish et al., 2022) | `oskarvanderwal/bbq` for the `All` configuration; `Elfsong/BBQ` for the bias-target label; official JSONL at github.com/nyu-mll/BBQ for scoring validation | 3-way multiple-choice QA, ambiguous vs disambiguated contexts | 9 U.S.-English social dimensions plus 2 intersections | Standard in every 2026 comparator; separates “unknown” behaviour from stereotype-aligned errors |
| 1 (primary) | *WinoBias* (Zhao et al., 2018) | `uclanlp/wino_bias` (type1/type2 × pro/anti) | Coreference, scored as a two-candidate causal-LM comparison | Binary gender × 40 occupations | Independent gender check with a clean pro/anti gap; the causal-LM adaptation is declared, not called the original evaluator |
| 2 (transfer) | *Discrim-Eval* (Tamkin et al., 2023) | `Anthropic/discrim-eval` | Binary yes/no decision, 70 scenarios | Age 20–100 (ordinal), gender (3), race (5–9) by construction | Different task format from BBQ; one-token answer makes the margin bound of Section 7 apply exactly; ordinal age supports dose–response analysis |

`heegyu/bbq` is the most downloaded mirror but omits the bias-target field, which the stereotype-aligned error rate requires. The `Elfsong/BBQ` fork adds it. Both mirrors are checked against the official JSONL before use.

*StereoSet* (`McGill-NLP/stereoset`) and *CrowS-Pairs* (`nyu-mll/crows_pairs`) are retained only as robustness checks in an appendix. Pikuliak et al. (2023) document validity problems in their scoring protocols, and CrowS-Pairs was built for masked models. Neither carries a main claim.

### 3.2 Audited counterfactual pairs

For the mechanism study (E2) each benchmark contributes prompt pairs in which only a task-irrelevant identity cue changes while evidence and the correct answer stay fixed. Discrim-Eval provides these by construction. For BBQ and WinoBias the pairs are built from template metadata, then audited for grammaticality, tokenized length, answer order, and name balance. Group labels come from benchmark metadata, never from model output.

### 3.3 Calibration corpus

Every saved calibration run in the repository uses *WikiText*, and `python/calibration.py` reads the test split. The new pipeline replaces this with *C4* (`allenai/c4`, `en`, validation, streaming), which is the standard calibration source for GPTQ, AWQ, and Wanda. Using it makes the utility-only baselines comparable to published numbers and removes the leakage. The repository already contains a C4 loader in `python/scale_validation.py`.

Calibration mixtures for E3: 128 sequences of up to 512 tokens per seed; three compositions (generic C4, demographically balanced, deliberately imbalanced) at equal token budgets; five seeds for every final configuration. Demographic text for the balanced and imbalanced mixtures is drawn from benchmark templates held out from evaluation.

### 3.4 Utility and splits

Utility is measured on *WikiText-103* test perplexity and on the four downstream tasks already in the repository (HellaSwag, ARC-Easy, WinoGrande, PIQA), so that new numbers connect to `results/llama_real_rm.json` and `results/downstream_ablation.json`. WinoGrande is a commonsense task and is never reported as a bias measure.

Three disjoint sets are maintained throughout: **calibration**, **selection** (method and prompt choices), and **final evaluation**. Related templates, counterfactual pairs, and paraphrases stay in the same split. Split membership is fixed by a deterministic template-level partition and its hash is recorded in every result file.

---

## 4. Closest studies and the contribution axes

Compression and social bias is an active topic. Seven studies from 2024–2026 overlap with this plan. Six evaluate or mitigate quantization; one is the closest pruning method and enters through the bridge experiment. All identifiers were resolved on 8 September 2026. Where only the abstract was available, the table says so.

| Study | Venue | Models and settings | Bias evaluation | Main finding or method | Verified from |
|---|---|---|---|---|---|
| Hua, Lotfi, Chen. *Investigating Social Bias Changes in Quantized Language Models* | COLM 2026; arXiv 2602.06181 | 50 quantized models, 4- and 8-bit | 13 datasets combined as PostTrainingBiasBench, includes BBQ | Bias flips on up to 21 % of items; high-uncertainty items 3–11× more likely to flip; group-level shifts of +18.6 % and −14.1 % hidden by aggregates | Abstract |
| Marcuzzi, Ning, Schwartz, Gurevych. *How Quantization Shapes Bias in Large Language Models* | EACL 2026; arXiv 2508.18088 | Weight and activation quantization, several architectures | 13 benchmarks across stereotypes, fairness, toxicity, sentiment | Quantization lowers toxicity but can raise stereotyping under aggressive compression; effects fairly uniform across groups | Abstract |
| Rath, Maliakkal. *Quantization Undoes Alignment* | IEEE Cloud Summit 2026; arXiv 2605.15208 | Qwen2.5-7B, Mistral-7B, Phi-3.5-mini; BF16 to 3-bit | BBQ, 12,148 items × 5 seeds | 6–21 % of unbiased items become stereotyped at 3-bit; “unknown” selection falls 17.4 %; perplexity stays flat | Abstract |
| Wu, Li, Semenova, Zhong. *The Asymmetric Harms of LLM Compression* | arXiv 2608.19670 (Aug 2026) | 3 models × 11 compression methods | Social-bias subgroup analysis | Opposing subgroup shifts are cancelled in aggregate scores; compressed models stay confident when wrong | Abstract only; model list not yet checked |
| Proskurina, Metzler, Velcin. *Fair-GPTQ* | arXiv 2509.15206 (v3, Jul 2026) | GPTQ, 4-bit | Occupational stereotype generation; gender, race, religion | Group-fairness term added to the GPTQ rounding objective; ≥ 90 % accuracy retained | Abstract |
| Al Hakim, Wicaksono, Koto. *Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection* | arXiv 2601.12033 (rev. Jun 2026) | Static and dynamic quantization, multilingual | Intrinsic and extrinsic bias in 5 languages; safety in 3 | Identifies fairness- and safety-critical weights and keeps them at higher precision | Abstract; selection rule and budget accounting not yet checked |
| Proskurina, Metzler, Gourru, Velcin. *Debias-SparseGPT* | EMNLP 2026; arXiv 2609.02496 | 9 models incl. Qwen3-8B; 25 %, 50 %, 1:4, 2:4 sparsity | UnQover, BBQ, CrowS-Pairs; StereoSet dev as calibration | Demographically contrasting inputs enter the pruning Hessian | Full text |

Three earlier papers frame the field and are cited as background rather than compared on the axes: Gonçalves and Strubell (EMNLP 2023) on quantization and distillation; Hong et al. (ICML 2024, *Decoding Compressed Trust*), who find 4-bit preserves and 3-bit degrades trustworthiness; and Xu et al. (Findings of EMNLP 2024, *Beyond Perplexity*), who evaluate pruning and quantization on representational and generative harms.

### 4.1 Contribution axes

The next table is the argument the paper must win. A cell is marked only when the corresponding paper reports that element. Cells for abstract-only papers are provisional and are re-checked against full texts in week 1. If any comparator turns out to cover an axis, that axis is dropped from the claimed contributions.

| Axis | Hua 2026 | Marcuzzi 2026 | Rath 2026 | Wu 2026 | Fair-GPTQ | CWP 2026 | Debias-SparseGPT | **This study** |
|---|---|---|---|---|---|---|---|---|
| A1 Weight-only quantization at 8/4-bit | ✓ | ✓ | ✓ | ✓ | ✓ (4) | ✓ | ✗ (pruning) | ✓ (8, 4; 3 optional) |
| A2 Per-group outcomes with paired uncertainty | ✓ | partial | ✓ | ✓ | ✗ | ✗ | partial | ✓ cluster bootstrap, Holm |
| A3 Layer/component attribution of bias change | ✗ | ✗ | ✗ | ✗ | ✗ | weight-level | ✗ | **✓ single-site quantization map** |
| A4 Group-conditioned hidden-state error propagation | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ e_l, V_l,g, ρ_l,g diagnostics** |
| A5 Causal check by restoring precision at predicted sites | ✗ | ✗ | ✗ | ✗ | ✗ | partial | ✗ | **✓ vs random and utility-matched controls** |
| A6 Calibration-composition control | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (contrastive) | ✓ balanced vs imbalanced at equal tokens |
| A7 Mitigation method | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| A8 Byte-accounted memory budget (scales, zero points, padding) | ✗ | ✗ | ✗ | ✗ | ✗ | not stated | n/a | **✓ C(b) ≤ M** |
| A9 Mixed-precision allocation across components | ✗ | ✗ | ✗ | ✗ | ✗ | not stated | ✗ | **✓ greedy exchange under budget** |
| A10 Cross-format transfer (QA → binary decision) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ BBQ → Discrim-Eval** |
| A11 Formal stability statement tied to logit margins | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ margin lemma, Lean 4** |
| A12 Scale range within one protocol | many models | several | 3 × 7 B | 3 | ? | ? | 9 × 7–9 B | 124 M to 8 B, 4 families |

Rows A3–A5, A8–A11 are where the study is currently alone. Rows A1, A2, A6, A7 are shared and must be matched, not claimed. The strongest single claim available is A3 + A4 + A5 together: a *predicted* site, a *measured* propagation signal, and a *causal* restoration test. The strongest engineering claim is A8 + A9: a schedule that a real backend can load, with bytes counted.

Critical Weight Protection is the nearest neighbour to the restoration idea. The full text must be read before any claim of difference is written. The provisional distinction is that it protects individual weights identified by a fairness signal, while this study allocates precision at the component level under a stated byte budget and validates the sites by propagation diagnostics first.

---

## 5. Hypotheses

Each hypothesis has an outcome that would falsify it, and a negative result is publishable under the same protocol.

| Question | Hypothesis | Evidence that confirms or refutes |
|---|---|---|
| H1 Does flat average utility hide uneven group effects? | Some quantized configurations change group error rates unevenly. | Paired dense/quantized group tables with intervals; uniform change refutes H1 for that model |
| H2 Which layers explain harmful changes? | Group-conditioned propagation plus dense answer margin predicts harmful flips better than global perplexity sensitivity. | Out-of-sample prediction on held-out templates; then restoration at predicted vs random vs utility-ranked sites |
| H3 Does calibration composition matter at fixed budget? | Imbalanced calibration changes which activations survive and shifts disparity at the same bit budget. | Token-matched mixtures, five seeds; both amplification and reduction are admissible outcomes |
| H4 Can allocation help at fixed cost? | Protecting components chosen by bias diagnostics lowers added disparity at equal bytes with bounded utility loss. | Held-out fairness–utility–memory frontier against equally tuned baselines |

The study does not assume that early MLP layers encode bias, that more precision is always fairer, or that lower representation error implies better social outcomes.

---

### 5.5 Sampling budget

Interval width in this design is governed by the number of template clusters, not the number of examples, because the bootstrap of Section 6 resamples clusters. BBQ instantiates 343 templates about 170 times each, and the extra instantiations inside a template behave close to replicates, so they add little independent information.

The default `budget` profile therefore keeps every template cluster and every design cell, and caps only how many instantiations are drawn inside each cell.

| Benchmark | Full | Budget | Clusters kept | Rule |
|---|---|---|---|---|
| BBQ | 58,492 | 7,868 | 343 of 343 | 6 instantiations per (context condition × question polarity) cell |
| WinoBias | 3,168 | 3,168 | 1,584 of 1,584 | kept whole; already small |
| Discrim-Eval | 9,450 | 5,250 | 70 of 70 | ages 20, 40, 60, 80, 100; full gender × race grid |

All 11 BBQ categories survive, the smallest retaining 278 disambiguated examples. Groups with at least 20 disambiguated examples fall from 50 to 46 of 54; the four lost are sparse intersections that Section 6 already confines to exploratory reporting. The ordinal age axis keeps five levels, which is enough for the dose-response check but not for a fine age curve; state that limit rather than fitting one.

Under the Kish design effect `DEFF = 1 + (m − 1)·ICC`, cutting from about 170 to 24 instantiations per template retains 92% of the effective sample size at an intra-cluster correlation of 0.3 and 77% at 0.1, widening every interval by 4% and 14% respectively. The pilot must replace this assumed correlation with the measured one before the final sample size is frozen. If the measured correlation falls below about 0.1, raise the cap rather than report intervals inflated by more than roughly 15%.

Metrics, group definitions, splits and scoring are identical under both profiles. The `full` profile disables the caps and is the setting for any final confirmatory run the budget allows. The profile in force is recorded in every result manifest, and any figure produced under `budget` says so.

---

## 6. Evaluation protocol and metrics

Scoring is frozen before final evaluation. Multiple-choice answers are scored as complete candidate continuations under a declared length-normalization rule. Answer-order permutations are tested and kept within the same cluster. For instruction models an additional generated-answer check uses fixed decoding and reports invalid answers and abstentions separately.

For BBQ the report includes the official signed bias scores for ambiguous and disambiguated contexts, unknown-answer accuracy in ambiguous contexts, disambiguated accuracy, stereotype-aligned error rate, and group-level accuracy. A model answering “unknown” everywhere fails the utility criterion. For WinoBias the report includes pro- and anti-stereotypical accuracy and their gap. For Discrim-Eval it includes the positive-decision rate per group and the log-odds gap against a fixed reference group, as in the original paper.

For a group `g` let `E_g` be the disambiguated error rate. The three headline quantities are

```text
compression harm:        ΔE_g = E_g(quantized) − E_g(dense)
disparity increase:      A    = [max_g E_g(q) − min_g E_g(q)] − [max_g E_g(dense) − min_g E_g(dense)]
worst added group harm:  H    = max_g ΔE_g
```

For stereotype scores the report keeps the official sign and also reports the change in distance from the neutral value, because a sign reversal replaces one preference with another. For counterfactual pairs the report gives the dense answer-probability gap, the quantized gap, the signed change, and the absolute-gap change. Stability from dense to quantized measures preservation of behaviour, not fairness of the dense model.

Uncertainty uses paired cluster bootstrap over template or pair clusters, with calibration-seed variation nested inside. Confirmatory subgroup comparisons are Holm-corrected. Sparse intersections are reported as exploratory. A minimum detectable effect is estimated from the pilot before the final runs. A claim of “no meaningful harm” requires a declared equivalence margin, not a non-significant difference.

---

## 7. Mechanism and a limited theoretical statement

Let `h_l(x)` be the dense hidden state and `h_l^q(x)` the quantized state at layer `l`. The diagnostics are

```text
e_l(x) = h_l^q(x) − h_l(x)
V_l,g  = mean over x in group g of ||e_l(x)||² / (||h_l(x)||² + ε)
ρ_l,g  = V_(l+1),g / V_l,g
```

with a declared floor `ε` and ratios marked undefined when input energy is too small. Hidden states are aligned at the final prompt token or at audited semantic spans; unrelated token positions are never compared. The identity difference `h_l(x_g) − h_l(x_g')` is not compression error and is not treated as evidence of bias on its own.

To separate new error injection from propagation, one component is quantized at a time and its actual residual is passed through the dense downstream network. Combined quantization is then tested for interaction. Because a fall in relative energy can come from growth of the clean norm, absolute error and answer margins are inspected alongside `ρ`.

The analytical starting point is the conditional recurrence

```text
||e_(l+1)(x)|| ≤ K_l(x) ||e_l(x)|| + η_l(x)
||e_L(x)||     ≤ Σ_l η_l(x) Π_(j>l) K_j(x)      when e_0 = 0,
```

where `η_l` bounds local approximation error and `K_l` bounds downstream sensitivity on the relevant states. Empirical energy ratios are diagnostics, not certified values of `K_l`.

**Margin lemma (to be proved).** For a single next-token comparison with logits `z_a, z_b`, let `m(x) = z_a(x) − z_b(x)`. If every logit change on input `x` is bounded by `ε_x` in infinity norm, then `|m_q(x) − m(x)| ≤ 2ε_x`, and the preference cannot flip when `|m(x)| > 2ε_x`. For two prompts the change in their margin difference is at most `2ε_x + 2ε_x'`. This is a stability statement, not a fairness statement. It applies exactly to Discrim-Eval's one-token yes/no answer; for multi-token BBQ answers a sequence-score version must be derived separately.

The Lean 4 file `lean4/LivingInference.lean` holds twelve theorem declarations with no `sorry`, but `multi_layer_error_bound` assumes total error equals a supplied sum and `lyapunov_contraction_bound` bounds a supplied geometric expression. Neither establishes an actual transformer recurrence. The lemma above is added only after stating how hidden-state bounds reach logits through final normalization and a possibly quantized output head. If the formal bound is vacuous on real models, the paper says so and treats the mechanism as empirical.

---

## 8. Method: bias-aware precision allocation

One global allocation is chosen offline and applied to every user. Group labels are used only for calibration and evaluation; nothing infers a user's identity at inference time.

Let `b_j ∈ {4, 8, 16}` be the precision of component `j` with `n_j` weights. The budget is

```text
C(b) = Σ_j n_j b_j / 8 + quantization metadata + uncompressed storage ≤ M
```

where metadata counts scales, zero points, packing and padding, shared tensors, and excluded layers. Runtime memory (activations, KV cache) is reported separately. A schedule is deployable only if the chosen backend can load its mixed configuration; every reported schedule is loaded and timed.

The search is a transparent greedy exchange on the selection set:

1. Start from a feasible uniform low-bit allocation; record utility, group errors, and pair gaps.
2. Shortlist candidate restorations using the propagation diagnostics of Section 7, then score their actual output effect.
3. Rank feasible changes by improvement in the frozen bias objective per added byte; offset each restoration by lowering precision elsewhere so the budget stays fixed.
4. Re-score after each accepted change; effects are not assumed additive. Stop when no feasible improving exchange remains.
5. Freeze the allocation and evaluate once on the final sets, reporting search time and calibration cost.

Proposed engineering gates, frozen after the pilot: at most 1 percentage point loss in aggregate disambiguated accuracy and at most 5 % relative perplexity increase against an equally budgeted utility-only allocation. These are targets, not results.

Baselines at matched bytes and matched search evaluations: uniform quantization; random allocation; perplexity-only allocation; dense-margin-only allocation; balanced calibration without allocation; and, on the shared 4-bit GPTQ setting, Fair-GPTQ. Ablations remove group conditioning, propagation features, and answer margins one at a time. If balanced calibration alone matches the allocator, the paper reports that and does not claim the allocator.

---

## 9. Experiments

| ID | Design | Models | Data | Deliverable |
|---|---|---|---|---|
| E0 Reproducibility pilot | Dense parity against reference forward pass; verified per-row activation Wanda; RTN 8/4-bit; GPTQ and AWQ 4-bit load test | M1, M3 | WikiText-103, C4 calibration | Adapter coverage report, quantization-grid and byte-accounting checks, reproducible manifests |
| E1 Quantization and subgroup outcomes | Dense, RTN-8, RTN-4, GPTQ-4, AWQ-4; 3-bit only if utility gate passes | M2–M5 | BBQ, WinoBias, Discrim-Eval | Paired group-outcome table with intervals; `ΔE_g`, `A`, `H`; utility; measured bytes |
| E2 Single-site interventions | Quantize one layer at 4-bit with all others dense; expand to attention/MLP components at selected layers | M3 first, then M2, M5 | Counterfactual pairs from all three benchmarks | Utility-sensitivity vs bias-sensitivity map; held-out prediction of harmful flips vs global-sensitivity and margin baselines |
| E3 Calibration composition | Generic, balanced, imbalanced C4 mixtures at equal tokens; five seeds under `full`, three under `budget`; residual mean and covariance per group | M2, M3 | Same | Controlled estimate of calibration effect on disparity |
| E4 Restoration and allocation | Restore predicted sites vs random vs utility-ranked; run the allocator and ablations at matched bytes | M3, M5 | Same | Causal evidence for E2 predictions; fairness–utility–memory frontier |
| E5 Transfer and pruning bridge | Freeze on BBQ, test on Discrim-Eval and WinoBias; replicate selected effects on M4; per-row Wanda and Debias-SparseGPT at 50 % on M5 | M4, M5 | Same | Scope of generalization; link to repository pruning results and to the closest pruning comparator |

Measured cost of the staged plan under the Section 5.5 budget profile, on one 80 GB accelerator: E1 across M2–M5 about 1.2 GPU-hours, E2 at layer level plus a per-component pass over the four worst layers on M3 and M5 about 3.1, E3 with three calibration seeds on M2 and M3 about 0.6, E4 on M3 and M5 about 10.0, E5 on M4 and M5 about 0.6. The total is about 19 GPU-hours on an A100-80 and 8 on an H100, including a 20% allowance for downloads, statistics and figures. A pilot limited to M2 and M3 costs about 8 and 3.5 hours respectively.

Peak accelerator memory is 21.7 GB for Qwen3-8B, whose 151,936-token vocabulary makes the scoring logits comparable in size to the weights; 40 GB is the practical minimum for the five models and 24 GB requires a reduced scoring batch. Host memory holds one dense copy of the weights so that any component can be restored exactly, which puts a floor of 32 GB on system RAM. These figures are estimates from measured token counts at an assumed 90 TFLOP/s effective throughput, and must be recalibrated from the E0 pilot before the full sequence is committed.

The minimum main study is: four capable models (M2–M5), three evaluations, two nonzero quantization levels, the E2 intervention map on one model, and the E4 held-out comparison. A one-model result supports a pilot report only. Resource measurements use fixed hardware, batch size, context, and output length, with warm-ups, synchronization, repeated timing, and intervals. Simulated quantization is labelled as such; efficiency is measured on packed kernels only.

---

## 10. What the repository provides and what must change first

Paths are relative to `Codes/living-inference`.

| Asset | Reusable for | Required change |
|---|---|---|
| `python/gpt2_ablation.py`, `python/gpt2_medium_validate.py` | Layer and component intervention loop (E2) | Replace low-rank/sparse approximation with the quantizer; add group-conditioned outputs |
| `python/lyapunov_analysis.py`, `python/lyapunov_deep.py`, `python/cfi_validation.py` | Clean/perturbed forward passes and hooks (Section 7) | Inject actual quantization residuals instead of embedding noise; align tokens; split by group |
| `python/calibration.py`, `python/wanda.py`, `python/gpt2_act_wanda_cpu.py` | Activation collection, pruning baseline (E5) | Read C4 not WikiText test; enforce per-row sparsity; raise an error instead of silently falling back to weight magnitudes |
| `python/greedy_calibrated.py` | Sequential candidate evaluation (Section 8) | Real bit widths, byte cost, bias objective, re-scoring after each change |
| `python/h100_downstream_eval.py`, `results/llama_real_rm.json` | Utility harness (E1) | Add bias tasks and per-example records |
| `lean4/LivingInference.lean` | Norm lemmas (Section 7) | Pinned Lean/Mathlib; add and prove the margin lemma |

Six corrections precede any new experiment. (1) `colab_unified_eval.py`, `h100_downstream_v2.py`, and `cfi_validation.py` use weight-column magnitudes in code labelled Wanda; these are relabelled as weight-statistic heuristics. (2) GPT-2 `Conv1D` orientation, tied weights, and output heads are handled by an adapter, not a hard-coded path. (3) Calibration, selection, and evaluation are disjoint. (4) No quantization pipeline exists; every bit-width number is new. (5) Theoretical wording in `CLAUDE.md` and `docs/MATH_FRAMEWORK.md` is audited against what the Lean file proves. (6) Zero weights in dense tensors are not counted as memory savings; the deployed representation is measured.

New code lives in the self-contained package `Codes/quant-bias/` (`quantbias/model_adapters.py`, `data.py`, `quantization.py`, `pruning.py`, `evaluate.py`, `trace.py`, `allocate.py`, `statistics.py`, `report.py`, `run_experiment.py`, plus `configs/`, `tests/` and `scripts/`), with results under `Codes/quant-bias/results/`. The `living-inference` tree is read-only provenance: no file in it is modified or overwritten.

Every result record carries: experiment id, code revision, model and tokenizer revisions, data revision, split and calibration hashes, method and backend version, seed, precision map, group/template/example ids, raw answer scores, prediction, correctness, bias-label mapping, resource measurements, and status. Failures are recorded as failures.

---

## 11. Journal readiness

The following items are the ones a journal referee checks that a workshop referee does not.

**Pre-registration.** Hypotheses H1–H4, the three headline metrics, the equivalence margin, the utility gates, and the split hashes are frozen in `configs/` and time-stamped before final runs. The paper states which analyses were confirmatory and which exploratory.

**Reproducibility.** One command reproduces every table from the raw per-example records. Checkpoint revisions, quantizer versions, and hardware are pinned. Calibration seeds and split hashes are published. Failed runs and out-of-memory entries appear in the artefact.

**Ethics and scope.** Group labels are used only offline for calibration and evaluation; the deployed model carries no demographic inference. BBQ and WinoBias encode U.S.-English social categories, and Discrim-Eval encodes a U.S. decision context; conclusions are restricted to that setting. Dense-model bias is reported but not claimed to be corrected.

**Error analysis.** For each model, the paper shows the items whose answer flips under quantization, grouped by predicted site and by dense margin, with examples. This is the place where a referee sees the mechanism rather than reads about it.

**Deployment framing.** Every reported schedule states parameter count, serialized bytes, peak accelerator memory, prefill latency, and decode throughput on named hardware, so that the fairness–utility trade-off is also a fairness–utility–cost trade-off.

**Limitations to state in the paper.** Only weight-only post-training quantization is studied. Only three benchmarks in one language. The margin lemma covers one-token answers. Propagation diagnostics are empirical, not certified bounds. Five models is enough for a cross-family pattern, not for a universal claim.

**Venue fit.** The protocol suits *Transactions of the Association for Computational Linguistics*, *Transactions on Machine Learning Research*, and *Computational Linguistics*. Each accepts negative and mechanistic results when the protocol is pre-specified. The final target is chosen after the pilot shows which contribution axis carries the strongest evidence.

---

## 12. Six-week sequence

| Week | Work | Exit condition |
|---|---|---|
| 1 | Read full texts of the seven comparators and fill the provisional cells of Section 4.1; freeze scope; implement adapters, quantizer, scoring; re-measure five dense baselines under one setting | Contribution axes confirmed or cut; dense parity on all five models |
| 2 | E0 and E1 pilot on M2 and M3; sample-size estimate; byte accounting validated against a packed backend | Utility-competent 8/4-bit configurations; primary metric fixed |
| 3 | E2 single-site map on M3; held-out prediction test | Propagation-plus-margin predictor compared with global-sensitivity baseline |
| 4 | E3 calibration controls; E4 restoration and allocator on M3 | Feasible fixed-budget schedules with search cost recorded |
| 5 | Locked final runs on M2–M5; E5 transfer and pruning bridge; uncertainty analysis | Reproducible artefact including negative results and failed cells |
| 6 | Figures, margin lemma if supported, manuscript | Every claim maps to a table cell, a figure, or a conditional proof |

Pilot GPU-hours are measured in week 2 and extrapolated before week 5. If hardware runs short, M4 and 3-bit are cut before any evaluation set or uncertainty analysis is cut.

---

## 13. Presentation

The paper opens with the fairness question and the new evidence. The repository's sensitivity and proof machinery is introduced as the instrument, with its original configurations and provenance cited. No existing perplexity plot is re-labelled as a bias result.

Contribution statements, each used only after its experiment succeeds:

1. A controlled account of how weight quantization changes task outcomes across demographic groups on five models from 124 M to 8 B parameters, under fixed byte budgets.
2. A group-conditioned residual-propagation and answer-margin analysis that predicts harmful changes on held-out templates better than aggregate sensitivity, validated by restoring precision at the predicted sites.
3. A byte-accounted precision allocator that improves the measured fairness–utility trade-off on held-out tasks and a held-out task format, with calibration and search costs reported.

Main figures: paired dense/quantized subgroup-change plot; layer-by-group sensitivity heat-map; predicted versus observed harmful flips with baselines; restoration controls; fairness–utility frontier annotated with measured bytes. One table separates reused technical evidence, newly reproduced baselines, and new bias findings.

If the propagation predictor fails, the paper reports its limits. If the allocator does not beat matched baselines, the paper keeps the controlled measurement and restoration results only if they add evidence beyond the seven comparators. If an apparent bias reduction disappears after controlling for competence or abstention, it is not reported as mitigation.

---

## References resolved during this revision

- Al Hakim, Wicaksono, Koto. Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection. arXiv:2601.12033.
- Gonçalves, Strubell. Understanding the Effect of Model Compression on Social Bias in Large Language Models. EMNLP 2023.
- Hong et al. Decoding Compressed Trust. ICML 2024. arXiv:2403.15447.
- Hua, Lotfi, Chen. Investigating Social Bias Changes in Quantized Language Models. COLM 2026. arXiv:2602.06181.
- Marcuzzi, Ning, Schwartz, Gurevych. How Quantization Shapes Bias in Large Language Models. EACL 2026. arXiv:2508.18088.
- Parrish et al. BBQ: A Hand-Built Bias Benchmark for Question Answering. Findings of ACL 2022.
- Pikuliak et al. In-Depth Look at Word Filling Societal Bias Measures. EACL 2023.
- Proskurina, Metzler, Velcin. Fair-GPTQ. arXiv:2509.15206.
- Proskurina, Metzler, Gourru, Velcin. Debias-SparseGPT. EMNLP 2026. arXiv:2609.02496.
- Rath, Maliakkal. Quantization Undoes Alignment. IEEE Cloud Summit 2026. arXiv:2605.15208.
- Tamkin et al. Evaluating and Mitigating Discrimination in Language Model Decisions. arXiv:2312.03689.
- Wu, Li, Semenova, Zhong. The Asymmetric Harms of LLM Compression. arXiv:2608.19670.
- Xu et al. Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression. Findings of EMNLP 2024. arXiv:2407.04965.
- Zhao et al. Gender Bias in Coreference Resolution. NAACL 2018.
