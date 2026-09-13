# What this research found, in plain terms

*Study: how making a language model smaller changes the answers it gives to
different social groups — and where inside the model that change begins.*
Three linked projects, seven models, three benchmarks. Last updated
13 September 2026. Every number below points to a file in this repository;
the original result files were never edited.

---

## 1. The problem, in one paragraph

Large language models are expensive to run. The usual fix is **quantization**:
storing each weight with fewer bits (4 or 8 instead of 16) so the model needs
less memory. It is well known that this costs a little accuracy. What is far
less understood is *whether that cost falls evenly*. A model that drops one
point of average accuracy might do so by getting slightly worse on everyone,
or by getting much worse on questions about one particular group of people.
Average scores cannot tell these apart. This research asks: **when a
compressed model changes an answer, where inside the network did the change
start, how did it travel to the output, and does it land unevenly across
groups?**

## 2. What was built

Three pieces of software, each answering one part of the question, designed
to be used together.

**`living-inference/`** (the earliest project) asks: *when a weight matrix is
approximated — pruned, low-rank, or cached — how large is the error it
introduces, and can that be proven rather than hoped?* It contains

- twelve machine-checked theorems (Lean 4) bounding the error of each
  approximation step and of their composition, including a condition under
  which a residual network *shrinks* an error as it passes through a layer
  (`lean4/LivingInference.lean`);
- a check of those bounds on a real model: across 468 weight matrices of
  GPT-2 Small and 30 compression settings, **0 of 14,040** bound checks were
  violated (`python/results/gpt2_ablation.json`);
- per-layer measurements of how a small random perturbation at the input
  grows or shrinks through each of seven models — the "propagation profile"
  — and pruning experiments (Wanda, OWL, layer removal) on the same models
  (`python/results/h100/`, `python/results/mistral_micro/`).

Its bounds are local: they say how much one layer can move a signal. They do
not by themselves say what a changed signal does to a *decision*, and they
say nothing about groups of people. That gap is what the next two projects
fill.

**`quant-bias/`** compresses a model and measures the social consequences. It
runs seven models (124 million to 8 billion parameters, four different
architectures) through weight-only quantization at 8 and 4 bits, using both
simple rounding (RTN) and an error-correcting method (GPTQ). It then scores
each model on three tests of social behaviour:

| benchmark | what it measures | how the model answers |
|---|---|---|
| BBQ | does the model pick a stereotype when the context does not justify it? | three-way choice: two people, or "unknown" |
| WinoBias | does the model resolve "she"/"he" by occupation stereotype? | which of two occupations the pronoun refers to |
| Discrim-Eval | does the model say yes/no differently for identical cases that differ only in age, gender or race? | probability of "yes" |

It records, for every single question and every compression setting, what the
model answered and how confident it was. That is 343 result files, so any
later analysis can be redone without touching a GPU. It also re-implements
three published bias-aware compression methods and a memory-budgeted
allocator, so the new measurements can be compared against them.

**`mixed_study/`** joins the two. It reproduces `living-inference`'s
propagation profiles with the new instrumentation (so the projects are
demonstrably measuring the same thing), corrects four measurement flaws found
in `quant-bias`, and adds the experiment that ties everything together: inject
a compression error of controlled size and *direction* at one layer, let it
travel through an otherwise-uncompressed network, and record what reaches
the decision and which answers flip.

The whole pipeline was first run on two samples per category to prove every
code path, then in full. Every stage pushes its output to GitHub the moment
it finishes, so nothing depends on a rented machine staying alive.

## 3. What was found

### 3.0 The error bounds hold on a real model, and errors mostly shrink with depth

`living-inference` established the ground the rest stands on. Its formal
bounds — proven, not fitted — held without exception on every one of 14,040
checks on GPT-2 Small. Its propagation profiles show that in most layers of
most models a perturbation *shrinks* as it passes through (contraction ratio
below 1 in 11 of 11 GPT-2 Small layers, 21 of 31 Mistral-7B layers, 18 of 35
Qwen3-8B layers), with amplification concentrated in a few late layers.
Pruning half the weights in place costs Mistral-7B an 11× rise in
perplexity but destroys Qwen3-8B (a 42,922× rise), a difference the
profiles anticipate.
*Evidence:* `../../living-inference/python/results/h100/unified_summary.json`,
`gpt2_ablation.json`; vendored copies under `../provenance/living_inference_results/`.

**Why this matters.** These are utility results: they say how much a compressed
model's *overall* predictions degrade. The natural next question — do the
layers that amplify error also cause the socially relevant flips? — turned
out to have a surprising answer (§3.2).

### 3.1 Four-bit compression does measurably change socially relevant answers

On the BBQ questions where the context *does* contain enough information to
answer correctly, compressing to 4 bits made the model abandon a correct
answer on **5–13 %** of the questions it previously got right, depending on
the model. At 8 bits this was **0.1–2 %**. Of those newly wrong answers at
4 bits, **18 to 63 per model** were the stereotype-aligned choice
specifically — the model did not just become uncertain, it moved toward the
stereotype.
*Evidence:* `../../mixed_study/results/v2/audit/outcomes_e1.json`; table in
`../../mixed_study/results/v2/audit/AUDIT.md` §2.

### 3.2 The change starts at specific layers, and the layer that amplifies the most is not the one that flips the most answers

Injecting the *actual* compression error at one layer and letting it travel
through an otherwise-uncompressed network shows a clear pattern on Mistral-7B:
an error at layer 0 grows **32×** by the output, an error at the last layer
does not grow at all. Yet the last layer flipped **the most answers** and layer
0 fewer. What tracks answer flips is not how much the internal signal drifts
but how directly the error reaches the final decision scores.
*Evidence:* `../../mixed_study/results/v2/b1/mistral_7b_v0_1/b1_cells_layer.json`;
table in `../../mixed_study/results/v2/FINDINGS_v2.md` §2.1.

**Why this matters.** `living-inference` measured how much internal signals
drift, and its profiles are correct (§3.4). This result shows drift is the
wrong quantity to watch for *behaviour*: the layers it flags as dangerous are
not the layers that flip answers. Decision-score sensitivity is the right
quantity. That redirects the whole line of work without invalidating the
earlier measurements.

### 3.3 The *direction* of the compression error carries information, but only a little

The experiment compared the real compression error with random errors of
exactly the same size at the same layer. The real error reached the output
**1.0–1.6× harder** than a random one. So the structure of the error matters,
modestly. Two boundaries were found: at 8 bits the error behaves exactly like
random noise (ratio ≈ 1.00), and on the 7-billion-parameter models even random
errors arrive at the output partly aligned with the real one — the large
networks funnel any perturbation toward a common direction, which is why
direction matters less at scale.
*Evidence:* `../../mixed_study/results/v2/b1/*/b1_cells_{layer,component}.json`;
192 experimental cells summarised in `FINDINGS_v2.md` §2.

### 3.4 The two projects measure the same thing — once numerical precision is matched

The earlier project's per-layer sensitivity profiles were recorded in 16-bit
floating point (fp16). This study's default is a different 16-bit format
(bf16). Re-measuring the same profile in bf16 did not reproduce the saved one
(rank correlation 0.32 on Mistral-7B); re-measuring in fp16 **did** (0.93,
every saved value within three standard deviations of the fresh ones). The
saved profiles encode rounding at the edge of what fp16 can represent.
*Evidence:* `../../mixed_study/results/v2/legacy/*/legacy_reproduction.json`.

**Why this matters.** An earlier draft of this document reported that the two
projects' measurements did not align and drew a conclusion from that. The
conclusion was wrong: the mismatch was a precision artefact. Anyone comparing
propagation profiles across studies must state the floating-point format.

### 3.5 Correct measurement changes the picture more than any method does

Reanalysing the same records with corrected definitions:

- A theorem-based check for "impossible" answer flips reported 17 violations
  in the original run. Under the correctly stated test there are **0**. The
  original test asked the wrong question when the model was already wrong.
- Confidence intervals for the worst-affected group had been built from
  **293 of 2000** bootstrap samples, the rest discarded because a small group
  vanished from the sample. A resampling scheme that keeps every group present
  uses **2000 of 2000**.
- Only **18 of 43** demographic groups in the test set are backed by three or
  more independent question templates. The rest have many *instances* of one
  or two templates, which is not the same as independent evidence.
- **All seven** models nearly never choose "unknown" when the context is
  ambiguous (1.5–16 % correct against a 33 % chance rate). Ambiguous-context
  results cannot be read as fairness outcomes under this scoring.
*Evidence:* `../../mixed_study/results/v2/audit/AUDIT.md` §3, §5, §6, §7.

## 4. What was tried and did not work

Stating these is part of the contribution: a reader should not repeat them.

| attempt | result | where |
|---|---|---|
| A search that spends a fixed memory budget unevenly across layers to reduce group harm | Lost to plain uniform 4-bit on its own objective, on both primary models (J = 0.105 vs 0.090 on Mistral) | `e4/*/e4_results.json` |
| Three published bias-aware compression methods, re-implemented from their papers | No method's worst-group harm is distinguishable from uniform GPTQ once compared as paired differences on the same questions | `../../mixed_study/results/v2/audit/AUDIT.md` §6 |
| Predicting which layer will cause harm from internal-signal energy, tested on layers the predictor never saw | No consistent gain over a size-and-depth baseline | `../../mixed_study/results/v2/group_prediction/LADDER.md` |
| Restoring four "predicted" components to full precision at fixed cost | Every arm — predicted, utility-chosen, five random — gave an identical result; 200 test questions cannot resolve an intervention this small | `../../mixed_study/results/v2/restoration/*/restoration.json` |

## 5. The contribution, stated directly

0. **A proven and verified error-accounting foundation** (`living-inference`):
   machine-checked local bounds, confirmed on a real model with zero
   violations, and propagation profiles for seven models.
1. **A measurement protocol that separates four things previous work pooled:**
   losing a correct answer, losing it *to the stereotype*, the gap between
   groups, and sensitivity to identity on matched cases — each with a declared
   denominator, an eligibility rule based on independent templates, and an
   interval that keeps every group in every resample.
2. **A mechanistic result:** decision-score sensitivity, not internal drift,
   is what predicts answer flips across depth; compression-error direction
   contributes modestly beyond magnitude; 8-bit error is indistinguishable
   from noise.
3. **A verified link between two studies** (dtype-controlled reproduction of
   the earlier propagation profile), so their results can be combined
   without inferring compatibility from model names.
4. **Reproducible negative results** for a mixed-precision allocator and for
   three re-implemented published methods, with the sample-size reason stated.

This is a measurement-and-mechanism study. It does not deliver a method that
beats uniform quantization, and says so.

## 6. What is still open

- A held-out confirmation set of *new* question templates. Everything above
  is discovery evidence; the templates have been inspected.
- The decision-score-gradient predictor (`mixed_study/mixed_study/directional.py`,
  already validated against finite differences) evaluated on the same
  held-out-layer protocol that the energy predictor failed.
- A restoration test sized from this pilot's variance rather than guessed.
- Qwen3-8B's legacy profile is only partly reproduced in fp16; the remainder
  is unexplained.
- Comparisons against the published authors' own code, not re-implementations.

## 7. How to check any claim here

```bash
cd Codes/mixed_study
.venv/bin/python -m mixed_study.run audit    # rebuilds AUDIT.md from the E1/E2/E7 records, CPU only
.venv/bin/python -m mixed_study.run ladder   # rebuilds LADDER.md
.venv/bin/python scripts/validate_v2.py      # checks every GPU-run artifact is complete and sane
```

The GPU stages (`legacy`, `b1`, `restore`) are recorded with their full
per-cell output; re-running them needs one 80 GB GPU for about 30 minutes.

## 8. Provenance

| run | date (UTC) | hardware | stages | wall time | cost |
|---|---|---|---|---|---|
| living-inference GPT-2 ablations, Lean proofs | 2026-02 → 2026-03 | CPU + Colab | — | — | — |
| living-inference 7-model profiles + pruning | 2026-03 → 2026-09 | Colab A100 / H100 | — | — | — |
| quant-bias E0–E7 | 2026-09-10 → 09-11 | 1× H100 80 GB (Akash) | 36/36 ok | ~20 h | ~$55 |
| mixed_study B1 + restoration | 2026-09-13 | 1× H100 80 GB (Akash) | 13/13 ok | ~30 min | ~$1 |
| mixed_study P0 reanalysis | 2026-09-12 | CPU | — | ~1 min | — |

Sampling used the `budget` profile: every BBQ question template is kept, and
only the number of name/group fills inside each template is capped (6 per
cell). At an intra-template correlation of 0.3 this retains about 92 % of the
effective sample for 13 % of the compute; the calculation is in
`quantbias/statistics.py::subsample_cost`.

Related: `../../Future_Plan.md` (original design), `../../Next_Plan.md`
(critical review that produced `mixed_study`),
`../../mixed_study/results/v2/FINDINGS_v2.md` (full detail of §3.2–3.4).
