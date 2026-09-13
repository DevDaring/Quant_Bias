# Completion plan for the integrated compression–bias study

**Updated:** 13 September 2026. **Purpose:** finish the scientific work needed for a defensible TACL-targeted manuscript. This document specifies remaining work; it does not report new experimental results or authorize infrastructure spending.

## 1. Decision and finite scope

Complete **three focused experiments**, preceded by a measurement/code audit. Keep the seven-model exploratory results; use **Mistral-7B-v0.1 and Qwen3-8B** for the main confirmation experiments. Use GPT-2 Small for numerical and implementation checks. No additional model family, broad allocator sweep, or new compression algorithm is required.

The final research question is:

> How do compression residuals propagate into task-relevant score changes, and when do those changes create stereotype-aligned errors or unequal outcomes across groups?

The three experiments are:

1. **C1 — Independent behavioral confirmation and predictor evaluation:** new templates, validated scoring, and out-of-sample comparison of directional, energy, and utility diagnostics.
2. **C2 — Direction and dtype controls:** establish the scope of residual-direction effects and the fp16/bf16 reproduction result, using matched inputs and valid uncertainty.
3. **C3 — Selective restoration:** test whether the diagnostic identifies useful interventions at matched added cost, with a sample size justified before evaluation.

Successful completion means these questions have defensible answers, including negative answers. It does not require a mitigation win. If C3 establishes no practical benefit, the manuscript remains a measurement-and-mechanism paper. TACL readiness additionally requires a clear contribution beyond existing compression–bias studies and claims supported by the final evidence; completion cannot guarantee acceptance. [TACL criteria](https://transacl.org/ojs/index.php/tacl/about/submissions).

## 2. What is already complete and should be reused

| Existing evidence | Status | Use in the final paper |
|---|---|---|
| `quant-bias` E0–E7 across seven models | Completed exploratory run | Breadth, baseline outcomes, calibration/allocation findings, and records for reanalysis. |
| Corrected behavioral outcome definitions, pair handling, and margin checks | Implemented in `mixed_study`; validation caveats below | Measurement methods, with explicit scope and tests. |
| Legacy reproduction | GPT-2 reproduced; Mistral improves from approximately 0.32 in bf16 to 0.93 in fp16; Qwen remains partial | Controlled reproducibility finding with model-specific conclusions. |
| B1 layer/component/source panels | Completed discovery panels | Hypothesis generation and planning; strengthen before confirmatory mechanism claims. |
| Fixed-granularity predictor ladder | Completed exploratory analysis; no consistent group-conditioning advantage | Negative evidence, after correcting baseline labels and targets. |
| Equal-cost restoration pilot | Completed, with identical reported harm rates across arms at n=200 | Pilot for selecting effect size, intervention size, and sample design. |
| Local error accounting in `living-inference` | Existing Lean source and numerical evidence | Technical foundation; verify exact theorem assumptions, build status, and provenance before manuscript claims. |

Sources: [current findings](quant-bias/results/FINDINGS.md), [integrated findings](mixed_study/results/v2/FINDINGS_v2.md), [audit](mixed_study/results/v2/audit/AUDIT.md), [predictor ladder](mixed_study/results/v2/group_prediction/LADDER.md).

Preserve E0–E7 and `results/v2/`. Write corrected analyses and new runs under **`Codes/mixed_study/results/v3/`**, with independent manifests and source hashes. Update the final findings only after the relevant evidence exists.

## 3. P0: prerequisites before new scientific runs

### 3.1 Correct specific reporting claims

- The Mistral B1 RTN4 table has **6 answer flips at layer 4 and 5 at layer 31**. Remove the claim that the last layer flips the most. The weaker observation that maximum amplification does not coincide with maximum flips is supported.
- Distinguish absolute amplification, final absolute drift, and relative energy. They are different variables; “internal drift” should not stand for all three.
- Treat “8-bit behaves like random noise” as an untested equivalence hypothesis. Ratios near one are insufficient to establish equivalence.
- Treat “larger networks funnel errors because their downstream maps have low effective rank” as a hypothesis. Cosine alignment alone does not identify rank or isolate model size from architecture and dtype.
- Describe new stereotype-aligned errors alongside stereotype errors removed and total error changes. New stereotype errors alone do not establish net bias amplification.
- Verify the source for **14,040 checks over 30 configurations**. `gpt2_ablation.json` is not automatically the correct source for a Pareto-sweep count. Trace the figure to its records and generating code, or remove the unsupported count.
- Attribute dramatic legacy “Wanda” PPL changes to the implementation actually used. The previously inspected `colab_unified_eval.py` uses weight statistics in its pruning heuristic. Do not present those numbers as standard activation-aware Wanda results without a validated reproduction.
- Separate numerical correctness checks and corrected project bugs from externally novel methodological contributions.

These are reporting corrections, not reasons to discard the work.

### 3.2 Fix the predictor baseline and score definition

The current [group_prediction.py](mixed_study/mixed_study/group_prediction.py) labels `V_macro` as `local_energy`, but `_rows` constructs it from `features_selection[g]["V_final"]`. It is an endpoint-energy baseline. Add actual injected energy from `V_inject` and keep the final-energy baseline separate. The current size feature counts components, not parameters; add exact parameter counts and compressed bytes. Whole-layer sizes can differ, particularly in hybrid models.

The ladder still uses the original pooled `harmful_flip_rate`. New social-outcome prediction must use the corrected stereotype-aligned and group-specific targets, with declared denominators. Retain the original target only as a separately named task-damage analysis.

The current [directional.py](mixed_study/mixed_study/directional.py) evaluates full-continuation scores but differentiates and injects the residual only at **prompt positions**, padding candidate-position residuals with zeros. This is a well-defined prompt-boundary intervention. It is not automatically the same as weight quantization, which also changes candidate-token computation. Establish both:

1. Approximation accuracy for the exact prompt-only intervention it models.
2. Predictive accuracy for actual quantization, using candidate-specific residuals over the relevant teacher-forced sequence where needed.

Unify joint prompt/continuation tokenization, BOS handling, truncation, attention masks, and score normalization with the main evaluator. A finite-difference test using the same incomplete representation cannot establish parity with the actual compression experiment.

For three or more candidates, evaluate contrasts against **every competing candidate**. A dense runner-up crossing is only one route to an argmax change. Use a positive gold margin for preservation of correctness, and the dense winner's margins for preservation of its decision. Report ties and duplicate candidate tokens separately.

### 3.3 Put propagation and behavior on the same examples

In [matched_residuals.py](mixed_study/mixed_study/matched_residuals.py), propagation uses `examples[:batch_size]`, ordinarily **eight prompts**, while actual-compression behavior uses the full example set, ordinarily 200. Also, behavior is scored under real compression; random and reversed injections are currently evaluated for propagation without the corresponding behavioral comparison.

For v3, collect per-example propagation and candidate-score changes on the same audited inputs for actual, reversed, and random interventions. Batch and stream these records to avoid retaining all hidden states. Record separate sample counts for each analysis. The existing eight-prompt measurements remain discovery observations, not a 200-example matched mechanism test.

Replace `hash(key)` for direction seeds with a deterministic digest of model revision, site, source, example/cluster, and seed. Python's default string hash can change between processes. Verify that changing batching or rerunning the same manifest preserves inputs, interventions, and recorded outputs within declared numerical tolerances.

### 3.4 Repair the coverage validation

The current [uncertainty.py](mixed_study/mixed_study/uncertainty.py) simulation defines `truth` from the realized sample difference between `comp_err` and `dense_err`. It therefore measures whether an interval contains its own sample estimate, not coverage of a fixed population estimand. The reported 100% coverage does **not** validate nominal 95% coverage.

Replace this with a data-generating process whose true marginal effect is known analytically or from an independent, very large reference population. For example, specify dense error probability `p`, loss probability `a` conditional on a correct answer, and recovery probability `b` conditional on an incorrect answer. With homogeneous probabilities the true added error is `(1 − p)a − pb`; with cluster effects integrate over their declared distribution rather than substitute the observed sample effect.

Simulate the actual unbalanced cluster/group structure, rare events, null effects, heterogeneous effects, and 2/3/5/10/20 supporting-template regimes. Examine coverage, interval width, type-I error, and power for group differences and paired method differences; assess the maximum statistic separately. Report Monte Carlo uncertainty. Start with 500 simulation replicates and increase only if simulation precision is inadequate. Keeping all resamples is useful but is not itself proof of valid inference.

Retain H as secondary. Three independent templates are an eligibility rule, not a guarantee of adequate inference. Freeze a common eligible group set across methods, disclose restricted groups, and do not merge identities merely to obtain narrow intervals.

### 3.5 Audit restoration controls

The existing restoration code checks a 1% tolerance on **total model bytes**. For a small intervention that can hide a substantial mismatch in the **added bytes**. Match `bytes(arm) − bytes(uniform4)` directly, record exact component counts/kinds, and assert the intended number of restorations. Choose an explicit added-cost tolerance before runs, preferably exact matching where tensor shapes permit.

Allow a utility baseline to select the same component as the proposed method when it ranks it highest. The present control construction excludes predicted components; that is an exclusion-constrained control rather than an unrestricted utility baseline. Retain it only as an additional labeled control. Independent random schedules may also overlap naturally with the predicted set.

**P0 completion:** tests demonstrate the corrected score semantics, true local/final features, same-example tracing, stable seeds, restoration costs, and statistically meaningful coverage. No GPU sweep starts before this gate passes.

## 4. Shared confirmation protocol

### Models and tasks

- **Primary:** the exact Mistral-7B-v0.1 and Qwen3-8B checkpoints already used, with resolved revisions and explicit dtype/attention/tokenizer settings.
- **Instrumentation:** GPT-2 Small. Other completed models remain exploratory breadth evidence.
- **Primary social task:** independently authored and audited disambiguated QA with a stereotype annotation, plus the existing BBQ results as discovery context. Label new data as a new evaluation set, not official BBQ.
- **Independent task format:** new matched decision scenarios following a Discrim-Eval-style yes/no design. State their provenance and validation; never invent correctness labels for decisions without ground truth.
- **Utility:** disambiguated task accuracy and a separate language-modeling evaluation passage set. Calibration, site selection, and final utility passages must be disjoint.
- **Conditional:** WinoBias only for model/adapter combinations that demonstrate competence. Ambiguous BBQ remains a separately reported scoring-limitation analysis unless its adapter is repaired and independently validated.

### New templates and audit

All inspected templates are discovery data for this protocol. Prepare a versioned confirmation set with new underlying situations, not simple name substitutions or paraphrases of inspected templates. A development starting design is **four social dimensions × 24 independent templates × eight balanced variants = 768 questions**. The number of independent templates, their group coverage, and the power analysis determine the final size; 768 is not a promised adequate sample.

Balance stereotype-aligned and counter-stereotypical correct answers, answer positions, name/cue lengths, and task difficulty. Make identity changes task-irrelevant only where the intended counterfactual requires that. Store explicit semantic answer mappings. Have annotators independently verify labels and stereotype/counterfactual validity, record agreement and adjudication, and keep validators blind to model outcomes. Generated drafts require this validation too.

Freeze dataset hashes, group taxonomy, primary metrics, hypothesis directions, exclusion rules, model settings, and analysis code before scoring the confirmation set. Developers can inspect the data for validity but must not adapt models or methods to its outcomes. Use separate development templates for selecting prompts, thresholds, sites, and restoration budgets. New scenarios must also be checked for overlap with existing calibration and evaluation inputs.

### Outcomes and uncertainty

The primary social endpoint is the rate of **newly incorrect stereotype-aligned answers**, with both all-labeled and dense-correct denominators reported. Also report removed stereotype errors, net stereotype-error change, generic harmful/beneficial transitions, accuracy, and category-conditioned group contrasts. Decision-probability sensitivity is a separate endpoint for matched yes/no cases.

Use paired intervals over the same template clusters for method comparisons. Shared templates, sites, and calibration seeds are not independent observations; the analysis must reflect their crossed or nested dependence. Freeze multiplicity handling, for example Holm correction for the small confirmatory hypothesis family. For rare-event prediction use precision–recall, calibration, and prevalence baselines, not agreement dominated by unchanged answers.

Choose a minimum practically relevant effect and desired interval precision **before** final evaluation. Use development data and the corrected simulations to choose sample size with approximately 80% power for that effect where feasible. Report which conclusions remain descriptive when power cannot be achieved. Do not increase samples repeatedly until a desired p-value appears.

## 5. C1 — Does directional sensitivity predict new social outcomes?

**Primary question:** does task-relevant directional information improve prediction beyond local error, depth, intervention size, and general utility sensitivity?

### Design

1. Begin with both primary models, RTN4, and eight prespecified layers spread across depth. Treat whole layers and individual components as separate panels. Use GPTQ4 on a smaller prespecified replication panel after RTN instrumentation passes.
2. Development pilot: four sites × 64 development examples per model. Measure gradient cost, finite-difference accuracy, and harmful-event prevalence. Choose the final panel size from these measurements rather than assuming the earlier 30-minute runtime applies.
3. Fit/tune only on development templates. Test on independent confirmation templates, with held-out layers for any learned site predictor. Model-wide normalization and feature selection must not use held-out targets. Site selection must not use confirmation outcomes.
4. Capture per-example local residuals, absolute/relative final drift, task-relevant score changes, directional estimates, group/category, template ID, and all candidate predictions. Candidate-score gradients should be computed once per reusable dense input/site where possible.

### Required baselines

| Baseline | Purpose |
|---|---|
| Parameter count + depth + intervention type | Control structural/size confounds. |
| True injected residual energy | Test the local-error explanation inherited from `living-inference`. |
| Final residual energy | Separate local error from empirical downstream propagation. |
| Dense answer margins | Control pre-existing decision fragility at the example level. |
| General NLL/PPL sensitivity | Test a utility-only explanation under the same intervention. |
| Global task-score sensitivity | Test whether demographic conditioning adds anything. |
| Directional contrast estimate + margins | Proposed mechanistic predictor. |
| Group-conditioned extension of that predictor | Test the additional social-group hypothesis explicitly. |

Observed final score changes are an explanatory upper-reference diagnostic, not a cheap predictor if their computation already performs the intervention. Disclose which methods require backward passes, compressed passes, or labels. A group-macro feature versus an aggregate error rate does not establish group-specific prediction.

### Completion and interpretation

Report confidence intervals for **differences in predictive performance**, separately by model, granularity, and endpoint. A general predictive claim needs an independently validated improvement over the strongest relevant baseline; if improvement occurs only on one model or task, narrow the claim. If no improvement is supported, retain a bounded negative result with precision sufficient to assess a declared meaningful effect, or clearly state the unresolved range. Do not require the predictor to win to finish the experiment.

**Outputs proposed:** `results/v3/c1/predictions.jsonl`, `comparisons.json`, `REPORT.md`, input/site split manifests, and compute logs.

## 6. C2 — Which direction effects survive dtype and magnitude controls?

**Primary question:** does residual direction affect task-relevant scores beyond magnitude, and is the apparent 8-bit boundary distinguishable from a numerical floor?

### Design

Use the same primary models and a small panel of early/middle/late layers plus a prespecified component panel. Include actual RTN4, RTN8, GPTQ4, and verified Wanda residuals. Reuse C1 inputs/captures where compatible. For each example/site compare actual residuals with sign-reversed and at least five deterministically seeded, per-token-norm-matched random directions.

Measure both full-sequence prompt-boundary interventions and actual compression, labeling their distinct scopes. Evaluate candidate-score changes and social outcomes under the injected controls too, not only hidden-state statistics. Report per-example effects and paired template-level intervals rather than ratios of unrelated averages.

On a smaller development panel, rescale the **same** residual directions over a fixed amplitude grid. Compare 4-bit residual directions rescaled to 8-bit magnitude and vice versa. Include no-injection repeatability checks, fp16 and bf16, and fp32 reference computations where feasible. Measure the residual actually realized **after** casting and addition; a nominal matched norm is insufficient if many entries round away.

For an equivalence claim, freeze a justified practical margin—for example, ±10% for a specified actual/random score-drift ratio is a candidate tolerance, not an established standard. Use log ratios where well-defined, absolute differences near the numerical floor, and require the appropriate equivalence interval to lie inside the declared bounds. A nonsignificant difference means unresolved, not equivalent.

Finish the Qwen legacy audit by matching the saved script, dtype, checkpoint, input tokens, perturbation scaling, and attention implementation as far as provenance permits. Keep the already successful GPT-2/Mistral cases. If missing provenance prevents exact Qwen reproduction, document that boundary and omit a universal reproduction claim; this should not trigger endless reruns.

### Claims to retain or omit

“Low effective rank explains output alignment” is optional. To keep it, estimate the relevant downstream Jacobian spectrum or perturbation covariance spectrum and use architecture/size controls. Otherwise report observed alignment without a rank or scaling explanation. A controlled direction effect across two primary models is sufficient scope; a general law of model size is not required.

**Completion:** supported direction effects, equivalence within a specified margin, or quantified inconclusive/negative effects, each with dtype and magnitude scope. Correct interpretation is the deliverable.

**Outputs proposed:** `results/v3/c2/cells.jsonl`, `equivalence.json`, `legacy_audit.json`, and `REPORT.md`.

## 7. C3 — Does selective restoration improve outcomes at equal added cost?

**Primary question:** can sites selected on development data reduce stereotype-aligned damage or added group error on independent cases, beyond utility and random controls?

Start from uniform4. Compare the validated diagnostic, unrestricted utility-only selection, local-error-only selection, and at least five independently sampled matched random schedules. Uniform4 is the zero-added-cost anchor. Restore the same tensor kinds and match **added bytes**, using the P0 corrections. Let controls overlap naturally.

Pilot a small predefined intervention-size grid on development data, such as 4/8/16 components or equivalent added-byte budgets. Choose one primary and, if affordable, one secondary budget before the final run. Increase independent template support according to power, not merely the number of repeated instances. If the development task has essentially zero damage under uniform4, change the development design or declare the mitigation endpoint uninformative before opening confirmation results.

Use C1's frozen site-selection method, but keep C3 confirmation outcomes hidden until its allocations are fixed. Report recovery of previously correct answers, stereotype errors created and removed, group error changes, and net utility. A smaller disparity obtained by lowering both groups' competence is not mitigation.

Specify utility noninferiority criteria in advance. An initial candidate is no more than one percentage point loss in disambiguated accuracy and 5% relative PPL increase against the utility-selected schedule, subject to scientific justification and adequate uncertainty. Do not accept schedules based only on their point estimates if the claim is noninferiority.

If hypothesis-test resolution remains inadequate, report a bound on the detectable/relevant effect rather than asserting that the method cannot help. Identical outcomes at n=200 do not alone prove underpower or equivalence; inspect whether the restored sites change scores at all, whether the arms differ as intended, and whether the endpoint has room to improve.

**Completion:** a valid, independent comparison with uncertainty and recorded costs. If the diagnostic loses, end the mitigation branch and retain the mechanism/measurement contribution. No additional allocator tuning on the final set.

**Outputs proposed:** `results/v3/c3/arms.json`, `records.jsonl`, `paired_comparisons.json`, and `REPORT.md`.

## 8. Necessary checks that do not require another broad study

### Comparator fidelity

Before claiming a result against Fair-GPTQ, CWP, or Debias-SparseGPT, validate the authors' code or a faithful full-method reproduction at matched settings. For a limited completion scope, reproduce **one closest comparator**, such as Fair-GPTQ, on the two primary models and final tasks if making comparative mitigation claims. Preserve other variants as explicitly named literature-inspired implementations. If no mitigation-superiority claim is made, author-code reproduction can be omitted, but the paper must then remove claims that the published methods themselves fail.

### Numerical and deployment scope

The main work can be complete as a **simulated weight-quantization study**. A small packed-backend parity experiment is needed if claiming deployment behavior or practical savings; measured speedup is not required for the mechanism paper. Do not present accounted storage as measured runtime memory. Record dtype, packing assumptions, scales, zero points, and excluded tensors.

### Proofs and provenance

Rebuild the Lean project in a pinned environment, inspect theorem assumptions and unproved declarations, and link each quantitative claim to an exact artifact. Local norm inequalities and standard margin arguments are foundations, not fairness guarantees. Reusing their empirical checks does not make them new bias results.

### Final integrity audit

Regenerate tables directly from immutable records. Check denominators, template counts, overlap, method names, granularity, and actual source paths. Reconcile 300-draw versus 2000-draw statements in the current reports. Update stale READMEs that still describe completed B1/restoration work as unfinished. Keep exploratory and confirmatory results clearly distinguishable.

## 9. Implementation order and proposed file changes

| Order | Proposed work | Files to extend/create | Exit gate |
|---|---|---|---|
| 1 | P0 correctness and reporting audit | Existing `directional.py`, `group_prediction.py`, `uncertainty.py`, `matched_residuals.py`, `matched_restoration.py`; focused regression tests | Correct estimands, parity, coverage evaluation, seed reproducibility, and cost checks. |
| 2 | New data and power planning | `data/confirmation_v3/`, `configs/integrated_v3.yaml`, `analysis/power_v3.py` | Audited templates; independent splits; frozen hypotheses and sample-size decision. |
| 3 | Small primary-model pilot | Proposed `experiments/confirm_predictor.py`, `experiments/direction_controls.py` | Gradients and actual interventions agree within characterized approximation error; runtime/memory measured. |
| 4 | C1/C2 confirmation | Same runners with frozen manifests | Complete independent records and paired uncertainty for the principal claims. |
| 5 | C3 confirmation | Restoration runner with frozen schedules | Equal-added-cost comparison, utility checks, and final decision on mitigation scope. |
| 6 | Comparator/proof audit and manuscript evidence freeze | Optional official comparator adapter; report generator; proof build record | Every headline has direct support; unresolved extensions removed from main claims. |

These v3 files and commands are **proposed**, not implemented by writing this document. The current CLI does not yet provide the C1–C3 runners; do not treat existing `b1` or `restore` commands as execution of this new protocol.

Estimate runtime from the pilot's per-example/per-site forward and backward costs. Gradient studies on full continuations can cost much more than the approximately 30-minute v2 GPU panel. Stream sufficient statistics, cache only reusable dense data, load models sequentially, and record actual GPU-hours. Choose the affordable final design before confirmation; do not launch a new seven-model sweep by default.

## 10. Definition of a scientifically complete paper

The study is ready for its final manuscript when all of the following hold:

- [ ] Scoring and labels are validated for the tasks/models carrying the main claims.
- [ ] The P0 fixes pass meaningful tests; same-example propagation/behavior records and stable seeds are used.
- [ ] At least one genuinely independent confirmation set is evaluated under a frozen protocol.
- [ ] C1 reports task damage and social outcomes separately, with valid baselines and uncertainty.
- [ ] C2 states exactly which direction/dtype/magnitude effects are supported; equivalence claims have equivalence evidence.
- [ ] C3 is completed with adequate design and uncertainty, or the final scope explicitly omits intervention-effectiveness claims and reports the pilot as inconclusive.
- [ ] Statistical coverage is assessed against known population quantities; sparse-template limitations remain visible.
- [ ] Legacy integration is documented model by model; incomplete Qwen reproduction is bounded and disclosed.
- [ ] Published-method comparisons use validated implementations or are explicitly restricted to the implemented variants.
- [ ] The strongest contribution is clearly distinguished from existing compression–bias work and ordinary implementation corrections.
- [ ] All headline numbers, formal claims, tables, and figures match their source artifacts.
- [ ] Negative results and exploratory analyses remain in the evidence record; no final-set retuning is used to manufacture a positive result.

**Writing can begin now** for the problem, related work, methods, and verified discovery results. Finalize the abstract and contributions after C1/C2, and finalize any mitigation claim after C3. If directional prediction and restoration both remain negative, write the narrower study of measurement validity and controlled limits of compression diagnostics only to the extent its independent evidence demonstrates a substantive new insight. Do not keep adding experiments solely to force a success narrative.
