# Vendored results from `living-inference`

Read-only copies of the result JSONs produced by the prior compression study
(<https://github.com/nabaos/living-inference>). Only `.json` files are copied —
no code, no git history.

They are vendored so the bridge experiment (E6) reproduces on a fresh clone.
E6 correlates the prior study's utility-only layer sensitivity (Lyapunov `rho`,
single-layer perplexity regret) against this study's group-conditioned harm at
the same layers.

Perplexities in these files come from runs with different token budgets and
chunking conventions. They document which checkpoints the prior work covered;
they are **not** comparable with this study's re-measured baselines and must not
be pooled with them.
