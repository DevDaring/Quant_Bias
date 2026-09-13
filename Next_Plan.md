# Next plan: integrate compression mechanics with social-bias measurement

**Date:** 11 September 2026. **Status:** evidence review and proposed next experiments; no new model runs were performed for this document.

## 1. Recommendation

**Develop one integrated mechanistic study: “From Compression Error to Unequal Outcomes: Propagation, Decision Margins, and Social Bias in Language Models.”**

The two projects have complementary roles. `living-inference` supplies local compression-error analysis, perturbation instrumentation, and pruning/structural-sensitivity experiments. `quant-bias` supplies actual quantization interventions, demographic tasks, group metrics, calibration experiments, and allocation controls. The scientific connection is:

```text
Compression operation
    → local residual magnitude and direction
    → propagation through the remaining network
    → change in task-relevant answer scores
    → errors, stereotypes, and disparities across groups
```

The best route is a **measurement and mechanism contribution**, with mitigation as a conditional extension. Successful execution is already demonstrated; the current data do not yet establish a reliable bias predictor, a successful quantitative transfer of the earlier rho profile, or an allocator that beats uniform quantization. Integration should establish which parts of the compression analysis transfer, and under which conditions, rather than require every existing method to win.

**Immediate priority:** reanalyse existing records and correct the measurement issues below, then run a small matched pruning–quantization experiment on Mistral-7B and Qwen3-8B. Expand examples and independent templates after that pilot. Another seven-model allocator sweep is premature.

## 2. Evidence reviewed and current results

I read [FINDINGS.md](quant-bias/results/FINDINGS.md), E0–E7 result summaries and selected per-example records, the experiment configurations, and the relevant measurement, bridge, allocation, and statistical code. I compared these with the surviving `living-inference` result files and propagation implementation.

The result validator was run without retry actions: **36 checked stages, zero artifact failures, three margin-check warnings, and one repository-freshness warning**. This confirms artifact completeness under the validator's checks, not scientific validity of every metric. Although stage names contain `full/`, this completed run used the **budget sampling profile**. Quick-run directories are excluded from the tables below.

### 2.1 Residual energy is promising, but the strongest summary mixes intervention types

I recomputed Spearman correlations from each model's [E6 bridge rows](quant-bias/results/e6), using average ranks for ties. These compare selection-set `V_final_mean` with final-set `harmful_flip_rate`. The group-conditioned energies are first averaged across groups; the outcome is an aggregate correct-to-incorrect flip rate.

| Model | All sites: n / correlation | Whole layers only: n / correlation | Components only: n / correlation |
|---|---:|---:|---:|
| GPT-2 Small | 28 / **0.760** | 12 / **−0.208** | 16 / 0.809 |
| GPT-2 Medium | 40 / **0.635** | 24 / **0.475** | 16 / 0.361 |
| LFM2-2.6B | 58 / **0.466** | 30 / **0.422** | 28 / 0.364 |
| Qwen3.5-2B | 49 / **0.560** | 24 / **0.487** | 25 / 0.346 |
| Mistral-7B-v0.1 | 60 / **0.366** | 32 / **0.146** | 28 / 0.423 |
| Qwen3-8B | 64 / **0.568** | 36 / **0.126** | 28 / 0.684 |
| Llama-2-7B | 60 / **0.548** | 32 / **0.318** | 28 / 0.431 |

These are exploratory point estimates, without new uncertainty intervals. The site counts in `FINDINGS.md` §2.1 do not consistently match the corresponding `V_final_vs_harmful_flips.n` fields; use the source-specific counts above when reporting this calculation.

**Interpretation:** a positive pooled association is present in all seven models. It is not uniform evidence of predictive success at a fixed intervention granularity. Whole-layer quantization changes many more weights than individual-component quantization. Components are additionally sampled from four layers shortlisted using the selection-set predictor. Pooling these intervention types can inflate a correlation through intervention size and selection effects. Layers and their subcomponents are also dependent observations.

Component-only correlations remain promising, but concern selected components, not an unbiased sample of all components. The layer-only results identify a real next question: **does residual direction and answer-score sensitivity explain what residual magnitude misses?**

### 2.2 Allocation and restoration results

The original objective is `J = H + A + gap`, with signed disparity/gap changes. Preserve its results as originally defined.

| Model / final allocation | J, lower better | H | Disambiguated BBQ accuracy | Accounted MiB |
|---|---:|---:|---:|---:|
| Mistral uniform4 | **0.0902** | 0.1389 | 0.8194 | 3958.51 |
| Mistral greedy | 0.1048 | 0.1111 | 0.8199 | 3988.74 |
| Qwen3-8B uniform4 | **0.2731** | 0.1667 | 0.8641 | 5815.96 |
| Qwen3-8B greedy | 0.3344 | 0.1875 | 0.8641 | 5819.99 |

Sources: [Mistral E4](quant-bias/results/e4/mistral_7b_v0_1/e4_results.json), [Qwen3-8B E4](quant-bias/results/e4/qwen3_8b/e4_results.json).

The greedy allocator loses to uniform4 on the declared objective on **both** models. Mistral's lower H alone does not establish superiority. Both searches used 200 evaluations and accepted two steps; their wall times were approximately 99 and 127 minutes. This documents the behavior of the current search, not an optimum over all allocations.

The allocations obey a common budget ceiling, but do not consume identical bytes. The restoration controls are a different experiment: for example, Mistral `restore_predicted` uses **5806.13 MiB**, exceeding the **4354.36 MiB** allocation ceiling. Predicted, random, and utility restoration also change different numbers of weights because whole layers and components are mixed. They cannot establish a fixed-cost restoration advantage.

E3 does not support “balanced calibration always helps.” On Mistral, mean H is 0.0972 for generic, 0.1111 for balanced, and 0.0833 for imbalanced calibration across three saved seeds; Qwen3.5-2B also has a higher H for balanced than generic calibration. These are descriptive values under the current metric and protocol, not evidence favoring deliberate imbalance. [E3 results](quant-bias/results/e3).

### 2.3 Evidence that remains reusable

- E0 records exact weight round-trip and restoration checks across seven models. This is valuable experimental infrastructure.
- E1 provides dense/RTN/GPTQ records with candidate scores and demographic metadata. These support substantial reanalysis without GPU inference.
- E2 measures actual quantization residuals on selection inputs and evaluates answer flips on final inputs. This separation is useful even though the current predictor and target need refinement.
- `living-inference` contributes local operator-norm inequalities and a substantial library of compression interventions. Its random-noise profiles, pruning results, and local proofs retain value within their actual assumptions.
- E5 provides pruning outcomes, but transfer coverage is incomplete: the Llama-2-7B result explicitly says its E4 allocation is missing. Stage completion should not be described as completion of every intended scientific comparison.

## 3. Measurement issues to resolve before spending on larger runs

### 3.1 Define social harm separately from generic answer damage

In [evaluate.py](quant-bias/quantbias/evaluate.py), `flip_table` labels a correct-to-incorrect transition a `harmful_flip`. This is a utility-loss event. It does not require a stereotype-aligned error, unequal treatment, or a group disparity. Its denominator includes all scored rows, including Discrim-Eval rows without correctness labels; those cannot contribute a correct-to-incorrect event.

For the next protocol, report separate outcomes:

1. **Task damage:** correct-to-incorrect transitions among labeled examples, with both total-labeled and dense-correct denominators declared.
2. **Stereotype-aligned damage:** newly incorrect stereotype-aligned answers on disambiguated BBQ; report unconditional counts/rates and the official bias metrics.
3. **Disparity:** group-specific changes in error and their contrasts, adjusting for benchmark/category and task composition.
4. **Decision sensitivity to identity:** changed answer probabilities on genuinely matched Discrim-Eval or audited counterfactual inputs, without inventing correctness labels.

Retain beneficial flips and absolute dense/compressed performance. A smaller gap achieved by damaging the better-performing group is not sufficient evidence of mitigation. BBQ groups in this implementation describe benchmark stereotype targets; they should not be presented as observed outcomes for real demographic populations.

The present E6 macro-average of group energies versus aggregate task damage does **not** show that group conditioning adds predictive value. Compare a global diagnostic, a group-macro diagnostic, and a model-by-site-by-group diagnostic explicitly.

### 3.2 Correct the margin check before attributing violations to rounding

[trace.py](quant-bias/quantbias/trace.py) computes a margin relative to the gold answer, takes its absolute value, and checks stability of the predicted answer. With three candidates, these are different statements when the dense answer is already wrong.

An exact-arithmetic counterexample, with candidate 0 the gold answer:

```text
dense logits:      [-10, 1, 0]       prediction = 1
compressed logits: [-10, 0, 1]       prediction = 2
epsilon = 1; absolute gold margin = 11 > 2 × epsilon
```

The predicted wrong answer changes, while the gold answer remains wrong. This satisfies the implemented violation condition without violating a correct margin-stability theorem. It provides a concrete alternative explanation for the recorded warnings; the individual events must be examined before assigning their cause.

Use either the **dense winner's top-two margin** to certify argmax stability, or a **positive gold margin** to certify preservation of a correct answer. Treat ties and duplicate first-token candidate IDs explicitly. Apply the theorem to the same score and candidate set used in evaluation. Full-continuation answers require sequence-score analysis; the first-token diagnostic is only a proxy. Unify BOS insertion, truncation, and boundary handling between tracing and candidate scoring.

### 3.3 Audit counterfactual pairs and answer correspondence

[data.py](quant-bias/quantbias/data.py) constructs BBQ pairs from rows sharing a template/polarity but having different correct-answer groups. This does not ensure that only an irrelevant identity cue changed. These pairs remain unaudited. Meanwhile, `pair_gap_summary` in [evaluate.py](quant-bias/quantbias/evaluate.py) uses the first example's label index to score both examples, even when labels/candidate ordering differ.

Fix semantic answer correspondence first. If the estimand compares probabilities of the correct answer, use each example's own correct-answer mapping. If it compares a shared decision, map that decision explicitly. Neither fix alone makes arbitrary template-matched examples valid counterfactuals. Exclude unaudited pairs from confirmatory counterfactual claims and retain them as template-matched comparisons if useful.

This affects the interpretation of the E4 `gap` term, calibration/comparator contrasts, and any claim that identity changes caused an observed effect. Preserve the original results; create a corrected protocol version with fresh validation.

### 3.4 Establish competence under the scoring adaptation

The dense results need attention before treating all seven models as equally informative fairness evaluations:

- Mistral's disambiguated BBQ accuracy is **83.75%**, but its ambiguous-context accuracy is **1.47%**. Qwen3-8B has **89.75%** and **3.00%**, respectively.
- Mistral's WinoBias pro-/anti-stereotypical accuracies are approximately **25.6% / 25.0%** in a two-candidate task; Llama-2-7B is similarly below the 50% chance reference.

These observations do not prove a parser or scoring bug, but they require label, prompt, and candidate-length checks. The saved records include both summed and mean log-probabilities: compare their behavior as a declared sensitivity analysis. Audit example labels manually, inspect answer lengths and identity words, and validate a balanced answer-letter format on the selection set. Do not choose a format because it improves the desired fairness outcome. Preserve the original sum-score results as one evaluation condition.

Use GPT-2 variants principally for mechanistic instrumentation until task competence is established. Limit claims for other low-performing model/task combinations accordingly. BBQ ambiguous and disambiguated conditions serve different purposes and should remain separate. [BBQ protocol](https://aclanthology.org/2022.findings-acl.165/), [WinoBias source](https://github.com/uclanlp/corefBias).

### 3.5 Repair uncertainty estimation; full sampling alone is insufficient

In the Mistral and Qwen3-8B E7 results, the H/A bootstrap retains only **293 of 2000 draws (14.65%)**. [statistics.py](quant-bias/quantbias/statistics.py) returns NaN whenever any included group is absent from a bootstrap draw, then discards that draw. The reported interval is therefore conditional on a restrictive group-presence event, in addition to the difficulties of bootstrapping a maximum.

The final disambiguated BBQ records contain 43 groups, of which 29 meet the current minimum of 20 examples. The smallest included group has 24 examples, and some included groups have only **two distinct template clusters**. More variants of those two templates do not create new independent templates. A full sampling profile can improve within-template estimation but cannot guarantee reliable generalization across templates.

Reanalyse group-wise paired differences, record examples **and supporting clusters per group**, and construct a joint interval using a resampling scheme appropriate to the shared-template structure. A cluster-weight or multiplier approach can preserve group support, but its finite-sample coverage must be checked in simulations matching this design. Groups with too few independent templates need a restricted descriptive claim or more independent data. Do not merely suppress NaNs, pool unlike identities to obtain significance, or apply the current bootstrap more times.

Overlapping method-specific confidence intervals do **not** establish that methods are statistically indistinguishable. Compare paired differences between methods on the same examples/clusters. Failure to establish superiority is also not equivalence. Report the existing E7 ranking as unresolved under the current analysis, and recompute direct comparisons with valid uncertainty.

### 3.6 Preserve clean validation and comparator identities

- The existing final results have now informed this revised plan. They are discovery evidence for new scores, metrics, and methods. Reserve new independent templates/data or a genuinely untouched evaluation source for confirmation; merely adding instantiations of inspected templates is weaker evidence.
- E5 describes transfer from BBQ, but E4 constructs its search examples from the available selection split across benchmarks, and its pair-gap term uses those pairs. WinoBias/Discrim-Eval are therefore not necessarily unseen task formats. Audit actual benchmark membership and enforce BBQ-only selection for any new task-transfer claim.
- [baselines.py](quant-bias/quantbias/baselines.py) explicitly implements literature-inspired methods with stated assumptions, in some cases from abstracts. Their results are evidence about those implementations. Validate the authors' code or full-method reproduction before making claims against Fair-GPTQ, Debias-SparseGPT, or Critical Weight Protection.
- The present quantization is simulated; MiB values are accounted storage, not packed-runtime measurements. Pruning comparator `bytes_total=0` is a missing-value convention, not zero storage. Mark it missing in comparison tables.

## 4. The integrated research question and contribution

**Question:** Which properties of a compression residual determine whether it damages general performance, changes a socially relevant decision, or creates a disparity between groups?

Use three parts, each traceable to both projects:

| Part | Existing foundation | Required integrated evidence |
|---|---|---|
| Local error | `living-inference` norm inequalities, pruning, and component interventions | Quantization and pruning measured at the same component interfaces with the same input tensors. |
| Propagation | Existing random-perturbation traces; `quant-bias` actual-residual traces | Transfer tests for residual magnitude **and direction**, controlling site, perturbation size, depth, and architecture. |
| Behavioral consequence | `quant-bias` candidate scores and demographic metadata | Output-score changes connected to task damage, stereotype errors, and group contrasts separately. |

Do not define success as showing that the earlier rho “fails fairness.” A local contraction ratio from random embedding noise is not automatically a ranking of the PPL cost of quantizing a layer. `bridge.interpret`'s positive-correlation threshold of 0.30 is neither necessary nor sufficient to establish matching pipelines.

**A successful bridge can show transfer, limited transfer, or a well-controlled boundary.** For example, if norm-matched random residuals propagate differently from quantization residuals, that is a useful mechanistic result when demonstrated on the same model, tensors, and tasks. It would explain why the earlier diagnostic requires an extension, while preserving its role in the research.

## 5. Minimal experiment that makes the integration substantive

Call this **B1: matched residual-source and direction experiment**. Implement it within `quant-bias` using a common adapter and read-only provenance from `living-inference`.

1. **Instrumentation pilot:** GPT-2 Small plus Mistral-7B. Match checkpoint, dtype, tokenizer, sequence, attention settings, and hidden-state boundaries. Reproduce a selected legacy random-perturbation trace numerically before testing its transfer. Record whether an array includes embeddings or only post-block states; never infer the mapping solely from its length.
2. **Primary models:** Mistral-7B and Qwen3-8B. Retain the seven-model results as breadth evidence; no new model family is necessary now.
3. **Prespecified sites:** eight layers spread across depth per primary model, including endpoints. Quantize complete layers in one analysis; analyse an equal, prespecified attention/MLP component panel separately. Avoid selecting all sites using the outcome or new predictor.
4. **Residual sources:** actual RTN quantization, validated GPTQ, and verified activation-based Wanda pruning. Use 8/4-bit quantization where useful and pruning settings chosen on calibration data to span overlapping local residual magnitudes. Equal sparsity and bit width are not equal cost or equal error.
5. **Direction controls:** at the same boundary, inject the actual residual, its sign-reversed version, and several norm-matched random directions through an otherwise dense downstream network. For clean mechanistic comparisons, preserve the full affected sequence-state shape and per-token norm convention. Match residual magnitudes on calibration data or analyse them as a covariate.
6. **Outcomes:** local absolute/relative error, downstream absolute/relative drift, change in each full answer score, correctness transitions, stereotype-aligned transitions, and category-conditioned group effects. Store per-example records, not only group means.
7. **Restoration:** on fresh inputs, restore single sites selected using development data and compare with utility-selected and random sites matched by added bytes, intervention type, and parameter count. Repeat random selections. Test a few two-site combinations to identify interaction effects.

Random and sign-reversed interventions are mechanism probes, not deployment methods. Restoring all compressed weights reproduces the dense model by construction; only selective restoration against appropriate controls tests the diagnostic's usefulness.

**Exit criteria:** reproducible legacy/new tensor correspondence; a valid outcome metric; evidence on independent inputs that the propagation description transfers or that residual direction explains a transfer boundary. A positive scalar-rho/PPL correlation is not an exit criterion.

## 6. Improve prediction using task-relevant direction

The current `2 × mean(logit_linf) / mean(abs(group_mean_margin)) + mean(V_final)` mixes a loose global-vocabulary bound with averaged margins. It need not estimate flip probability. `margin_only` is constant across sites; its Spearman correlation is undefined, despite being stored as 0. Its allocator baseline becomes an ordering/tie-break control, not a competitive margin-based method.

For a task-relevant answer contrast `s(x)`, define a new hypothesis:

```text
delta_s(j,x) ≈ gradient_hj s(x) · residual_j(x)
```

This combines the local residual from compression with downstream directional sensitivity. Use a differentiable full-continuation contrast and account for all prompt/candidate states required by that score. A gradient at only the last prompt token is a restricted approximation, not automatically a model of the complete answer. Compare the approximation with finite differences and actual score changes.

For audited pairs, compare **aligned decision-score changes** across the two inputs. For gold-answer preservation, compare predicted score decreases with the dense positive gold margin, per example, before aggregation. Do not divide group-mean error by group-mean margin and call the result an event probability.

Evaluate a nested set of baselines:

- Parameter count, intervention type, and layer depth.
- Local residual norm/energy alone.
- Dense full-answer margin alone at the **example** level.
- General PPL/NLL sensitivity under the same intervention, with a genuine dense reference.
- Global final residual energy and global output-score drift.
- Group-conditioned energy and propagation summaries.
- Directional score change plus margin; then any group-conditioned extension.

Report both explanatory diagnostics requiring compressed forward passes and predictors that estimate new sites without those passes. `V_final` already requires a compressed forward pass; its useful claim may be transfer from a small calibration sample, not avoiding quantization altogether. Gradient-based features have a compute cost that must be measured.

Use fixed granularity, development-only fitting, template-held-out validation, and leave-layer-out tests. If features are selected using these runs, confirm them on new inputs. Compare dependent correlations with paired resampling; report uncertainty for differences, not only separate p-values. For rare harmful events, include precision–recall and calibration against prevalence. Demonstrate added value beyond intervention size and generic task damage before calling a predictor bias-specific.

## 7. Theory: a useful, limited connection

For a compressed linear map, reuse the local inequality:

```text
||(W_compressed − W_dense)x|| ≤ ||W_compressed − W_dense||op ||x||.
```

Combine this with an explicitly assumed downstream Lipschitz/sensitivity bound for the **same states and score**. If candidate-score changes satisfy `|S_c(k) − S_d(k)| ≤ epsilon` for every candidate, the dense winner remains the winner whenever its top-two score margin exceeds `2 epsilon`. This applies to full sequence scores if their bounds have actually been established; it does not obtain those bounds for free.

The extension worth formalizing is the chain from a local compression perturbation to stability of a specified decision contrast. These norm/margin inequalities are standard; their value here is correct composition and empirical validation, not a claim that elementary algebra is new or that it certifies fairness.

Do not use measured rho values as certified constants. A product of ratios measured along the same trace telescopes to its endpoint ratio; that is an identity, not independent evidence of predictive accuracy. Estimate any transferable propagation model on separate perturbations/inputs and test it out of sample. Keep a source-level proof audit and a fresh Lean build separate from numerical validation of its premises.

## 8. Sampling and a defensible primary outcome

Use BBQ disambiguated stereotype-aligned errors and group error changes as the primary social-bias outcomes after scoring validation. Retain ambiguous BBQ as a separately audited condition. Use WinoBias only where its adapter and dense competence pass checks. Discrim-Eval supplies a different, decision-probability outcome with no accuracy ground truth. [Discrim-Eval study](https://arxiv.org/abs/2312.03689).

For each group report dense error, compressed error, harmful and beneficial transitions, sample count, independent cluster count, and uncertainty. Standardize or stratify across comparable task categories before interpreting group contrasts; differences in task difficulty are not evidence of an identity effect.

Retain H as a secondary worst-observed-group statistic with its limitations. Consider a prespecified category-macro average of positive added group error, with group/category eligibility frozen in advance, alongside individual group estimates. Changing the metric defines a new protocol: do not retrospectively relabel the current allocator as a success under it. A smoother training/search objective is acceptable only if assessed against independent social outcomes and utility gates.

Use the existing paired records to estimate variance and within-template dependence. Choose the smallest effect of practical interest before new evaluation, then perform simulation-based power/coverage analysis. Expand full examples for the primary models where useful, but add independent templates or an independent task when the number of supporting clusters is the limiting factor. Treat repeated calibration seeds as algorithm variability, not additional independent people or prompts.

## 9. Mitigation as a second-stage contribution

After B1 and prediction validation, test **selective precision protection** with fixed byte ceilings and measured frontier points. Start with a small candidate set and full-answer loss/group constraints that avoid the current pair-mapping and maximum-statistic issues. Include uniform4, a size-matched random schedule, utility-only protection, local-error-only protection, balanced calibration, and the validated directional predictor.

Prespecify a utility gate on both the primary task and an independent utility workload; for example, a maximum one-percentage-point loss against the utility baseline is an engineering tolerance to justify and freeze, not an achieved result. Protect worst-group absolute performance as well as average disparity so the method cannot look fairer by damaging everyone.

Use identical candidate coverage and search-evaluation budgets. Deduplicate overlapping layer/component expansions. Report actual bytes consumed and a frontier across budgets; label common-ceiling comparisons accurately. Compare multiple random schedules rather than one lucky or unlucky order. Diagnose selection-to-evaluation generalization before spending another 200 evaluations per model.

For claims against published methods, validate the authors' algorithms and record revisions/configurations. Existing literature already includes [Fair-GPTQ](https://arxiv.org/abs/2509.15206), [Critical Weight Protection](https://arxiv.org/abs/2601.12033), and [Debias-SparseGPT](https://arxiv.org/abs/2609.02496). The provisional distinction is the matched cross-compression mechanism and validated directional explanation, not merely bias-aware protection.

If selective protection does not outperform validated controls, retain it as a boundary on the diagnostic's intervention value. The measurement study can still succeed if its mechanism and transfer evidence are strong. Do not make an allocator win a prerequisite for completing the research.

## 10. Concrete work packages and cost control

All paths below are proposed additions or future edits under `quant-bias`; only this plan was written during the current review. Keep source result files unchanged and place corrected analyses/runs under a versioned results directory.

| Priority | Work package / likely files | Reuse | Completion evidence |
|---|---|---|---|
| P0 | `analysis/audit_results.py`; fixes to `evaluate.py`, `trace.py`, `statistics.py`, `data.py` | E1/E2/E7 records and existing tests | Counts reconciled; no mixed-granularity headline; explicit harm definitions; semantic pair mapping; correct margin tests; uncertainty coverage checks. |
| P0 | `configs/integrated_v2.yaml`, data audit and holdout manifest | Existing checkpoint SHAs and split metadata | Scoring, estimands, site sampling, and untouched confirmation set frozen. |
| P1 | `bridge.py` plus `experiments/matched_residuals.py` | Legacy perturbation routines and new adapters | Same-model, same-input trace correspondence; B1 source/direction controls. |
| P1 | `analysis/group_prediction.py` and richer trace records | Existing per-group V and site maps | Fixed-granularity predictive evaluation, global/local baselines, per-example/group outcomes. |
| P1 | `experiments/matched_restoration.py` | E4 restoration machinery | Equal-cost controls, repeated random schedules, independent outcome evaluation. |
| P2 | `allocate.py` and official comparator adapters | Existing byte accounting and search logging | Held-out frontier with paired method comparisons; complete utility gates. |
| P2 | Packed-backend audit | Simulated quantization maps | Numerical agreement and measured serialized size/latency on a small representative panel. |

**Suggested order:** approximately 2–3 working days for CPU reanalysis and protocol checks; 2–4 days for the matched instrumentation pilot; about a week for the focused primary-model experiments and analysis; then expand only the comparisons supported by those results. These are scheduling estimates, not measured GPU requirements.

Estimate GPU time from a small B1 pilot. The initial core panel is two models × eight layers × three compression sources = **48 source/site cells**, before direction controls, seeds, and scoring variants. Control runs can reuse dense captures and residuals. Run a short single-seed development panel first, then freeze a confirmatory panel with multiple independent calibration/noise seeds. Do not extrapolate uncertainty from repeated timings or repeated evaluation of identical deterministic weights.

Increasing `sampling.profile` to `full` alone will not solve all coverage problems: E2 separately caps counterfactual pairs per benchmark/cluster, and E4 caps search examples. Budget these explicitly. The easiest money to save is avoiding broad reruns before the scoring, pairing, and inference issues are fixed.

## 11. How to present the integrated study

Use a single problem statement: **compression quality depends on how numerical perturbations interact with the decisions a model makes for different groups.** Introduce both repositories as parts of the same experimental framework, with traceable provenance for reused results and clear labels for new measurements.

Suggested manuscript structure:

1. Task and harm definitions, with dense competence and scoring checks.
2. Local compression residuals and the established error-accounting framework.
3. Matched pruning/quantization propagation experiments and directional controls.
4. Group-conditioned answer-score and behavioral analysis, including where global sensitivity succeeds or fails.
5. Selective restoration and, if supported, precision protection under resource constraints.
6. Scope: English benchmarks, task adaptations, model/architecture limits, empirical versus certified quantities, negative transfer/allocation results.

| Main figure/table | What it would establish |
|---|---|
| Shared error-to-decision diagram and evidence ledger | Both projects contribute to one testable chain. |
| Legacy/new matched traces plus residual-direction controls | The technical integration is measured rather than inferred from model names. |
| Fixed-granularity correlation table | Existing pooled effects and their limits are visible. |
| Group prediction versus global prediction | Whether demographic conditioning adds information beyond generic damage. |
| Equal-cost selective restoration | Whether the diagnostic supports a useful intervention. |
| Utility–social-outcome–memory frontier | Mitigation success or its limits under feasible budgets. |

Broad quantization-bias surveys and subgroup-change analyses already exist: [How Quantization Shapes Bias](https://aclanthology.org/2026.eacl-long.17/) and [The Asymmetric Harms of LLM Compression](https://arxiv.org/abs/2608.19670). The proposed niche is **controlled transfer from compression mechanics to task- and group-specific behavior**, supported by matched perturbations and selective interventions. Verify this distinction against the full methods before making a priority claim.

### Claims available now versus claims to earn

**Available now:** seven-model simulated-quantization measurements; reusable exact-restoration infrastructure; a positive pooled residual-energy association with task-damaging flips; substantial variation when intervention granularity is controlled; negative results for the current allocator objective.

**Require new evidence:** a faithful quantitative bridge to the legacy propagation profiles; bias-specific predictive value beyond global error and task difficulty; valid counterfactual conclusions; superiority over published methods; improved deployed speed or memory; formal fairness guarantees.

**Draft positioning paragraph:** “We investigate how compression perturbations become changes in task outcomes across demographic groups. We connect local error accounting and pruning diagnostics with quantization measurements under a shared intervention protocol. Initial experiments reveal positive pooled associations between residual energy and answer damage, while stratification by intervention granularity exposes important limits. Our next experiments test whether residual direction and task-relevant margins explain these limits and support selective precision protection.”

The project can become a strong integrated study through a reproducible explanation, even if uniform quantization remains difficult to beat. The next decisive result is a controlled connection between residual source, downstream behavior, and valid group outcomes—not another favorable aggregate score.
