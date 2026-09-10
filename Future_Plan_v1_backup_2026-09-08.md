# Future research plan: Bias propagation under language-model compression

Prepared 8 September 2026. Status: proposed research; the bias experiments and methods below have not been implemented or evaluated.

## 1. Recommended direction

**Working title: “Who Loses Precision? Tracing and Controlling Social Bias in Quantized Language Models.”**

Make the central question: **Can group-conditioned error propagation identify where quantization changes socially relevant decisions, and can a fixed memory budget be allocated to limit those changes?**

Here, “bias” primarily means social stereotyping and disparities in task performance across demographic groups. Numerical quantization bias, such as a nonzero mean rounding error, is a possible mechanism to investigate. An affine layer's additive bias parameter is a different concept. Neither numerical error nor a changed hidden representation alone establishes social harm.

Build a distinct study through new questions, demographic evaluation, controlled interventions, and a precision-allocation method. Reuse the existing compression infrastructure and validated technical observations as foundations, with their provenance clearly identified. Existing perplexity plots cannot become new bias results by changing their labels. The main figures and conclusions should come from the new experiments.

Prioritize weight-only quantization for the main study, with pruning as a bridge to the existing work. Keep low-rank approximation, activation quantization, layer removal, and semantic caching as optional extensions. This gives the project a focused identity while using what is already available.

## 2. What the repository provides, and how to reuse it

Paths in this section are relative to `Codes/living-inference`. Numbers are observations read from saved files, not results reproduced during this planning review. Different files use different evaluation settings; do not pool their numbers into a common leaderboard.

| Existing asset | Evidence or functionality available | Role in the bias study | New work required |
|---|---|---|---|
| `python/gpt2_ablation.py`, `python/results/gpt2_ablation.json` | Single-layer and component interventions. The saved run has baseline PPL 22.98 and layer-0-only approximation PPL about 1.54 million, with zero recorded local bound violations. | Starting implementation for layer interventions; illustrates that local norm bounds do not ensure usable predictions. | Repeat at non-catastrophic compression levels on demographic tasks, measuring bias and accuracy together. |
| `python/gpt2_medium_validate.py`, `python/results/gpt2_medium_validate.json` | Layer and component sensitivity beyond GPT-2 Small. | Small-model debugging and a check against assuming a universal sensitive-layer ranking. | New paired demographic inputs and matched precision settings. |
| `python/lyapunov_analysis.py`, `python/lyapunov_deep.py`, `python/cfi_validation.py` | Clean/perturbed forward passes, relative error energy, and layer hooks. | Infrastructure for group-conditioned propagation measurements. | Replace embedding noise with actual quantization residuals, separate groups, and measure output decisions. |
| `python/calibration.py`, `python/wanda.py`, `python/gpt2_act_wanda_cpu.py` | Activation collection and activation-weighted pruning components. | Starting point for controlled calibration-distribution experiments and pruning baselines. | Separate calibration from evaluation; validate module coverage, per-row masking, and activation shapes. |
| `python/results/gpt2_activation_wanda.json` | Saved baseline PPL 26.76; MLP expansion compression gives PPL 32.19 with activation statistics versus 464.81 with the simplified weighting. | Motivation to distinguish true activation calibration from weight-only proxies. | Reproduce with verified GPT-2 adapters before using as a baseline; evaluate group disparities. |
| `python/greedy_calibrated.py`, `python/results/greedy_calibrated_balanced.json` | Component ranking and sequential compression evaluation. A saved run stays near baseline through seven groups, then rises to PPL 37.2 at eight. | Scaffold for evaluating allocation candidates and interaction effects. | Introduce actual bit widths, parameter-weighted cost, a bias objective, and re-scoring after each accepted change. |
| `python/results/llama_real_rm.json`, `python/results/downstream_ablation.json` | Utility and speed observations for layer-removal configurations. In the first file, dense PPL is 5.47; Rho-4 has lower PPL than BI-4, but lower mean task accuracy. | Motivation to evaluate several outcomes rather than optimize PPL alone; optional depth-pruning comparator. | Re-run common workloads and add demographic outcomes. WinoGrande scores here are not WinoBias evaluations. |
| `python/results/h100/unified_summary.json` | Cross-model results plus explicit architecture-loading failures and an out-of-memory entry. | Model-adapter and resource planning. | Treat failed entries as missing experiments; audit the pruning implementation before interpreting successful entries. |
| `lean4/LivingInference.lean` | Twelve theorem declarations, including local operator-norm and conditional composition inequalities. | Mathematical building blocks for a limited statement about score changes. | Rebuild in a pinned Lean/Mathlib environment and prove the new score-margin connection. |

### Required corrections before reusing experiments

1. **Validate method identities.** `colab_unified_eval.py`, `h100_downstream_v2.py`, and `cfi_validation.py` use weight-column magnitudes in code labeled Wanda. Treat these as weight-statistic heuristics. `wanda.py` uses activation norms but a global mask, so match the reference method's per-output-row sparsity when claiming a standard Wanda baseline. Silent fallback from activations to weights must become an explicit error or separately named method.
2. **Handle model structure explicitly.** GPT-2 projections use Conv1D storage and require correct weight orientation; hooks restricted to `torch.nn.Linear` can miss them. Validate all intended modules, tied weights, and output heads. Use architecture adapters instead of one hard-coded layer path.
3. **Remove evaluation leakage in the new pipeline.** `calibration.py` reads WikiText test data, and `greedy_calibrated.py` uses test text to rank candidates and report performance. New calibration, selection, and final evaluation sets must be disjoint.
4. **Implement quantization.** The inspected Python sources do not provide a quantization evaluation pipeline. Sparsity/rank settings described as compression levels are not bit widths. All quantization findings will require new runs.
5. **Audit theoretical language.** The Lean source currently has no `sorry` token, despite stale guidance in `CLAUDE.md`. However, `multi_layer_error_bound` assumes total error equals a supplied sum, and `lyapunov_contraction_bound` proves an inequality on a supplied geometric expression. Those statements do not independently establish an actual transformer recurrence or fairness guarantee. This review did not run `lake build`.
6. **Reconcile resource claims.** Saved speed measurements differ across implementations and workloads. Zero weights held in dense tensors do not establish memory or latency savings. Measure the deployed representation and kernels.

## 3. Position the new contribution against existing research

Compression and social bias are already an established combination. The proposed contribution must be more specific than “quantization affects bias” or “aggregate scores hide subgroup effects.”

| Closest work | Overlap to acknowledge | Target distinction to test |
|---|---|---|
| [Gonçalves and Strubell, Understanding the Effect of Model Compression on Social Bias in Large Language Models (2023)](https://aclanthology.org/2023.emnlp-main.161/) | Quantization and distillation evaluated for social bias. | Explain particular changes using actual residual propagation and controlled layer interventions. |
| [Marcuzzi et al., How Quantization Shapes Bias in Large Language Models (2026)](https://aclanthology.org/2026.eacl-long.17/) | Broad quantization evaluation across bias types, demographic subgroups, and 13 benchmarks. | A focused mechanistic predictor and resource-constrained intervention, rather than another broad benchmark sweep. |
| [Wu et al., The Asymmetric Harms of LLM Compression (2026)](https://arxiv.org/abs/2608.19670) | Opposing subgroup shifts can be hidden by aggregate bias metrics. | Predict which compression sites cause those shifts and validate the prediction by restoring precision. |
| [Proskurina et al., Debias-SparseGPT: Bias-Aware Pruning for Large Language Models (2026)](https://arxiv.org/abs/2609.02496) | Demographically contrasting calibration inputs and a bias-aware pruning intervention. | Mixed-precision allocation across layers/components, with propagation and answer-margin analysis. Include this comparator if making pruning-mitigation claims. |

This is a proposed distinction, not a verified priority claim. Before locking the contribution, inspect the closest papers' methods and code, search specifically for fairness-aware mixed-precision allocation and calibration, and record the overlap. If a comparable method exists, evaluate against it and narrow the claim to the new mechanism or evidence.

## 4. Research questions and falsifiable hypotheses

| Question | Hypothesis | Evidence needed, including a possible negative result |
|---|---|---|
| Does similar average utility hide different compression effects across groups? | At least some compressed configurations change group outcomes unevenly. | Paired dense/compressed comparisons with confidence intervals. Uniform or negligible changes remain valid findings. |
| Which layers explain harmful changes? | Group-conditioned propagation plus answer margins predicts harmful flips better than global PPL sensitivity alone. | Out-of-sample prediction, followed by layer restoration against random and utility-based controls. |
| Does calibration composition affect bias? | Calibration imbalance changes which activations are preserved and can alter disparity at the same bit budget. | Token- and content-matched calibration interventions; test both amplification and reduction. |
| Can allocation help at fixed cost? | Protecting components selected by bias-sensitive diagnostics reduces compression-induced disparity without unacceptable utility loss. | A held-out fairness–utility–memory frontier against equally tuned baselines. |

Avoid assuming that early MLP layers always encode bias, that higher precision is always fairer, or that lower representation error necessarily improves social outcomes.

## 5. Data and evaluation protocol

Use **BBQ as the primary task benchmark**. It distinguishes under-informative contexts from contexts with sufficient evidence and includes nine social dimensions relevant to U.S. English. Report these conditions separately using its documented scoring and label mappings. Restrict conclusions to the evaluated setting. [BBQ paper](https://aclanthology.org/2022.findings-acl.165/), [official data and documentation](https://github.com/nyu-mll/BBQ).

Use **WinoBias as an independent gender/coreference check**, reporting pro-/anti-stereotypical accuracy and their gap. Specify the causal-LM scoring adaptation; do not describe it as the original coreference evaluator. [Original dataset repository](https://github.com/uclanlp/corefBias).

Add **manually audited counterfactual prompt pairs** for the mechanism study. Change only a task-irrelevant identity cue while keeping evidence and the correct answer fixed; include identity-relevant controls where invariance is inappropriate. Label group membership from benchmark metadata, not from model guesses. Balance occupations, answer order, names, cue length, and prompt templates. Audit grammatical and semantic validity, especially when tokenization lengths differ.

StereoSet and CrowS-Pairs can be secondary robustness checks, but cannot carry the main claim: published analysis identifies validity problems in their measurement protocols. [Pikuliak et al., In-Depth Look at Word Filling Societal Bias Measures](https://aclanthology.org/2023.eacl-main.265/).

### Splits, scoring, and uncertainty

- Maintain three independent sets: **compression calibration**, **method/prompt selection**, and **final evaluation**. Group related template variants, counterfactual pairs, and paraphrases in the same split. Where a benchmark lacks suitable splits, document a deterministic template-level partition and use an untouched benchmark as an additional transfer test.
- Start calibration with 128 sequences of up to 512 tokens per seed; compare generic, balanced, and deliberately imbalanced mixtures with equal total token budgets. Use five independent calibration seeds for the final selected configurations. Keep the source content and length distribution matched as closely as possible.
- Freeze scoring before final evaluation. For multiple-choice tasks, score complete candidate continuations under a declared sum/length-normalization rule. Test answer-order permutations and keep them within the same statistical cluster. Check multi-token labels and prompt-boundary tokenization. For capable instruction models, add a small generated-answer check with fixed decoding and report invalid answers and abstentions separately.
- Report BBQ's official signed bias scores, ambiguous-context unknown-answer accuracy, disambiguated accuracy, stereotype-aligned error rate, and group-level accuracy. A model answering “unknown” to everything must fail the utility criteria. Do not combine different bias metrics into an unexplained average.
- Use paired cluster bootstrap intervals over template/pair clusters, incorporating calibration-seed variation hierarchically. Report group sample counts and intervals; treat sparse intersections as exploratory. Correct confirmatory subgroup comparisons for multiplicity, for example with Holm correction. Avoid treating every token as an independent demographic observation.
- Estimate the minimum detectable effect from the pilot; expand data where available rather than equating a nonsignificant difference with equivalence. For a claimed absence of meaningful harm, use a declared equivalence/noninferiority margin.

For a predefined group `g`, let `E_g` be the disambiguated task error rate. Report:

```text
compression harm:       ΔE_g = E_g(compressed) − E_g(dense)
disparity increase:     A = [max_g E_g(compressed) − min_g E_g(compressed)]
                          − [max_g E_g(dense) − min_g E_g(dense)]
worst added group harm: H = max_g ΔE_g
```

For stereotype scores, retain the official sign and also report change in distance from the metric's neutral value. A sign reversal may replace one preference with another. For audited counterfactual pairs, define a task-relevant answer-probability gap and report its dense value, compressed value, signed change, and absolute-gap change. Dense-to-compressed stability measures preservation of behavior, not fairness of the dense model itself.

## 6. Mechanistic analysis and a limited theoretical contribution

Let `h_l(x)` be the dense hidden state and `h_l^c(x)` the compressed state at layer `l`. Record both absolute drift and relative energy:

```text
e_l(x) = h_l^c(x) − h_l(x)
V_l,g  = mean over x in group g of ||e_l(x)||² / (||h_l(x)||² + ε)
ρ_l,g  = V_(l+1),g / V_l,g
```

Use a declared numerical floor and mark ratios undefined when input energy is too small. Align the last prompt token or audited semantic spans across counterfactual prompts; avoid comparing unrelated token indices. Report group distributions and tails, not only ratios of averages. The identity difference `h_l(x_g) − h_l(x_g')` is not itself compression error or proof of bias.

When multiple layers are compressed, `ρ` reflects both new error injection and propagation. To separate them, compress **one component at a time**, pass its actual residual through a dense downstream network, and measure how the residual and answer scores change. Then test combined compression to measure interactions. A drop in relative energy can come from growth of the clean hidden-state norm, so inspect absolute error and answer margins too.

A useful analytical starting point is a conditional recurrence:

```text
||e_(l+1)(x)|| ≤ K_l(x) ||e_l(x)|| + η_l(x)
||e_L(x)|| ≤ Σ_l η_l(x) Π_(j>l) K_j(x), when e_0 = 0.
```

Here `η_l` bounds local approximation error and `K_l` must bound downstream sensitivity on the relevant states. Empirical energy ratios are diagnostics, not certified `K_l` values. Squared-energy ratios cannot be inserted into this norm recurrence without conversion and justification. Group-conditioned residual means and covariance can additionally test whether systematic numerical error predicts harm better than error magnitude alone.

**Proposed score-margin lemma:** for a single next-token comparison with logits `z_a,z_b`, define `m(x)=z_a(x)−z_b(x)`. If all logit changes on this input are bounded by `ε_x` in infinity norm, then `|m_c(x)−m(x)| ≤ 2ε_x`; the preference cannot flip when `|m(x)| > 2ε_x`. For two prompts, the change in their margin difference is at most `2ε_x + 2ε_x'` by the triangle inequality. This is a stability result, not a claim that either preference is fair. For full multi-token answers, derive bounds on the actual sequence score; the single-token formula is insufficient.

Extend the existing Lean norm lemmas only after stating how hidden-state bounds reach logits, including final normalization and any compressed output head. Keep empirical estimates separate from certified bounds. If formal bounds are vacuous, report their looseness and treat the mechanism as empirical. The strongest practical result would be that residual size, group-conditioned propagation, and dense answer margin jointly predict harmful output changes on unseen templates and models.

## 7. Proposed method: bias-aware precision allocation

Start with weight-only quantization and one global allocation applied to every user. Group annotations are used offline for calibration and evaluation; the method does not need to infer a user's demographic identity at inference time.

Let `b_j ∈ {4,8,16}` be the precision of component `j`, with `n_j` weights. Use an actual resource constraint:

```text
C(b) = Σ_j n_j b_j / 8 + quantization metadata + uncompressed storage ≤ M
```

Count scales, zero points, packing/padding, shared tensors, and excluded layers. Report runtime memory including activations and KV cache separately. A count of compressed groups is not a memory budget. A precision schedule is deployable only if the selected backend supports its mixed configuration.

On the selection set, search for allocations minimizing a prespecified combination of worst added group error and increased counterfactual gap, subject to memory and utility constraints. Set tolerances using the pilot and freeze them before final tests. One initial engineering gate is at most **1 percentage point** loss in aggregate disambiguated accuracy and at most **5% relative** PPL increase versus an equally budgeted utility-only baseline. These are proposed tolerances, not achieved outcomes or universal standards.

Implement a transparent greedy search first:

1. Begin from a feasible uniform low-bit allocation and measure utility, group errors, and pair gaps.
2. Evaluate candidate precision restorations on the selection set. Use propagation diagnostics to shortlist candidates, then score actual output effects.
3. Rank feasible changes by bias-objective improvement per added byte. Offset restorations by lowering precision elsewhere when necessary to keep the budget fixed.
4. Re-score after each accepted change. Component effects are not assumed additive. Stop when no feasible improving exchange remains.
5. Freeze the allocation and evaluate once on the final sets. Include search time and calibration cost.

Compare against uniform quantization, random allocation, PPL/utility-only allocation, dense-margin-only allocation, and balanced calibration without allocation. Match search evaluations and total memory. Ablate group conditioning, propagation features, and answer margins independently. If balanced calibration performs as well, the more complicated method has not justified itself.

## 8. Experiments and minimum viable scope

Use GPT-2 Small/Medium for adapter and instrumentation checks. For substantive bias claims, start with two adequately capable model families already familiar to the repository, such as Mistral-7B and Qwen3-8B, subject to available hardware and validated quantizer support. Freeze exact checkpoint revisions, tokenizer revisions, base/instruction status, and chat/reasoning settings. Do not attribute differences between unrelated checkpoints to compression.

| Experiment | Design | Deliverable |
|---|---|---|
| E0: Reproducibility pilot | Dense forward-pass parity; verified activation pruning; weight-only 8-bit and 4-bit reference quantization; one small model plus one capable model. | Adapter coverage report, score sanity checks, correct label shifting and padding, reproducible artifact metadata. |
| E1: Quantization and subgroup outcomes | Two capable model families; dense, round-to-nearest 8/4-bit, and validated GPTQ/AWQ 4-bit configurations; optionally 3-bit only after a utility check. | Paired group outcome table with uncertainty, absolute bias, added harm, utility, and memory. |
| E2: Layer/component interventions | On one capable model first, quantize each layer at 4 bits with all other layers dense; expand to attention/MLP components for selected sites. | Utility-sensitivity versus bias-sensitivity map; predictive comparison on held-out templates. |
| E3: Calibration composition | Generic, balanced, and imbalanced calibration at fixed size/bit budget, five seeds; quantify residual mean/covariance and propagation. | Controlled estimate of calibration effects on output disparities. |
| E4: Restoration and allocation | Restore suspected components; compare random and utility-matched restorations; evaluate the proposed allocator and its ablations. | Evidence that interventions change the predicted outcomes, plus the held-out trade-off frontier. |
| E5: Transfer and pruning bridge | Freeze settings on one benchmark/category and test another; replicate selected effects on a second architecture. Add verified Wanda and, if relevant, Debias-SparseGPT at matched sparsity. | Scope of generalization and a direct connection to the reusable pruning infrastructure. |

The full table is staged, not a mandatory Cartesian product. A minimum main study is **two capable model families, two independent task evaluations, two nonzero quantization levels, a layer-intervention study, and a held-out mitigation comparison**. A one-model result can justify a pilot report, but not an architecture-general claim.

Benchmark serialized bytes, peak accelerator memory, prefill latency, and decode tokens/second using fixed hardware, batch size, context length, and output length. Use warmups, synchronization, repeated timing, and uncertainty. Simulated quantization is useful for isolating numerical behavior; label it explicitly and measure efficiency on packed kernels separately. Load models sequentially and stream summaries of activations to control memory.

## 9. Implementation roadmap

The following are planned files, not files created by this document. Add them under `Codes/living-inference/python/bias/` with outputs under `python/results/bias/`.

| Planned module | Responsibility | Existing code to adapt carefully |
|---|---|---|
| `model_adapters.py` | Checkpoint loading, layer/module mapping, Conv1D orientation, tied-weight handling. | `cfi_validation.py`, `gpt2_living.py` |
| `data.py` | Benchmark metadata, template-level splits, audited pairs, calibration mixtures. | `calibration.py` |
| `quantization.py` | Reference quantizer, external GPTQ/AWQ adapters, exact per-component precision and byte accounting. | New implementation; no existing quantization pipeline to relabel. |
| `evaluate.py` | Batched candidate scoring, official task metrics, invalid-answer accounting, per-example records. | Perplexity/downstream harness patterns |
| `trace.py` | Single-site residual injection, aligned hidden states, group diagnostics, margins. | `lyapunov_analysis.py`, `lyapunov_deep.py` |
| `allocate.py` | Resource-constrained candidate search and utility/bias ablations. | `greedy_calibrated.py` |
| `statistics.py`, `report.py` | Paired cluster intervals, multiplicity correction, figures and tables. | Existing JSON result convention |
| `configs/` | Pinned model/data revisions, seeds, budgets, split hashes, excluded modules, frozen tolerances. | New manifests |

Every result should include an experiment ID, code revision, exact model/tokenizer/data revisions, split and calibration hashes, method/backend version, seed, precision map, group/template/example IDs, raw answer scores, prediction, correctness, bias-label mapping, resource measurements, and status. Record failures explicitly; do not treat missing results as zero harm. Preserve existing result files and write new results to a separate directory.

Add meaningful checks when implementing: dense adapter parity against the reference model, expected quantization grid and memory accounting, exact pruning cardinality, per-row activation shape checks, no split overlap by template, known-answer scoring, counterfactual label permutation, unchanged-weight controls, and correct restoration of a component. These validate the scientific experiment rather than just the shape of its output.

### Suggested six-week sequence

| Week | Work | Exit condition |
|---|---|---|
| 1 | Audit reusable runs, freeze scope, implement adapters and scoring, inspect closest methods. | Reproducible dense baseline and a precise remaining contribution. |
| 2 | E0/E1 pilot, reference and packed quantization, sample-size planning. | Utility-competent configurations and an interpretable primary metric. |
| 3 | E2 traces and single-site interventions. | Held-out prediction comparison against global sensitivity and margin baselines. |
| 4 | E3 calibration controls and E4 restoration/allocation. | Feasible fixed-budget allocations with documented search cost. |
| 5 | Locked final runs, second-family and second-benchmark transfer, uncertainty analysis. | Reproducible results including negative effects and failed runs. |
| 6 | Figures, limited theorem if supported, research narrative and artifact review. | Every claim maps to an experiment or an explicitly conditional proof. |

Timing is an estimate contingent on hardware and model access. Measure pilot GPU-hours and extrapolate before expanding; cut optional models and compression families before cutting independent evaluation or uncertainty analysis.

## 10. How to present this as a distinct research project

Lead the narrative with the fairness question and the new evidence. Introduce the existing sensitivity/proof machinery as the instrument that makes the investigation possible. Keep its original configurations and provenance clear, and cite relevant public antecedents wherever the new work relies on them.

Suggested contribution statements, to use only after the corresponding experiments succeed:

1. A controlled account of how quantization changes task outcomes across demographic groups under fixed resource budgets.
2. A group-conditioned residual-propagation and answer-margin analysis that predicts harmful changes beyond aggregate sensitivity baselines, validated by precision restoration.
3. A precision allocator that improves the measured fairness–utility trade-off on held-out tasks and models, with transparent calibration and resource costs.

Suggested main figures are: a paired dense/compressed subgroup-change plot; a layer-by-group sensitivity heatmap; predicted versus observed harmful flips with baseline comparisons; restoration controls; and a fairness–utility frontier annotated with actual memory. A table should separate reused technical evidence, newly reproduced baselines, and new bias findings. Do not transplant existing aggregate-quality plots as the main social-bias evidence.

**Proposal abstract:** “This project investigates how weight quantization changes social-bias outcomes in language models. It combines paired demographic evaluation with layer-level compression interventions to distinguish numerical error, its downstream propagation, and changes in task-relevant answer preferences. We will test whether group-conditioned residual propagation and answer margins explain harmful changes beyond aggregate utility measures, and whether precision can be allocated to reduce these changes under a fixed memory budget. Evaluation will jointly measure task competence, subgroup outcomes, resource use, and transfer across benchmarks and model families.”

If the propagation predictor fails, report its limits rather than claiming a mechanism. If mitigation does not beat matched baselines, retain the controlled measurement and restoration results only if they add evidence beyond the closest literature. If apparent bias reduction disappears after controlling for competence or abstention, it is not a successful mitigation result. A distinct research identity comes from the new evidence and contribution; its title should reflect what the experiments actually establish.
