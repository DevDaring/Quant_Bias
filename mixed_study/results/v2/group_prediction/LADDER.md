# Nested predictor ladder, leave-layer-out, fixed granularity

Target: harmful flip rate on the held-out FINAL split. Spearman of out-of-layer predictions vs observed.

| model | granularity | n | size_depth | local_energy | ppl_regret | global_energy_linf | group_energy | group_energy_linf |
|---|---|---|---|---|---|---|---|---|
| gpt2_medium | layer | 24 | -0.97 | +0.23 | -0.96 | +0.07 | +0.26 | +0.08 |
| gpt2_medium | component | 16 | -0.14 | +0.29 | -0.45 | +0.30 | +0.26 | +0.19 |
| gpt2_small | layer | 12 | +0.28 | -0.27 | -0.61 | -0.50 | -0.52 | -0.45 |
| gpt2_small | component | 16 | -0.41 | +0.68 | +0.27 | +0.78 | +0.78 | +0.84 |
| lfm2_2.6b | layer | 30 | +0.60 | +0.35 | -0.24 | +0.35 | +0.63 | +0.63 |
| lfm2_2.6b | component | 28 | +0.28 | +0.29 | -0.37 | +0.24 | +0.35 | +0.21 |
| llama_2_7b | layer | 32 | -0.03 | +0.15 | -0.21 | +0.20 | +0.04 | +0.37 |
| llama_2_7b | component | 28 | -0.31 | +0.35 | +0.04 | +0.41 | +0.36 | +0.36 |
| mistral_7b_v0_1 | layer | 32 | +0.12 | -0.06 | -0.04 | +0.03 | -0.31 | -0.19 |
| mistral_7b_v0_1 | component | 28 | +0.26 | +0.11 | +0.23 | +0.32 | -0.15 | +0.19 |
| qwen3_5_2b | layer | 24 | +0.37 | +0.20 | -0.74 | +0.35 | +0.18 | +0.19 |
| qwen3_5_2b | component | 25 | +0.14 | +0.20 | -0.30 | +0.27 | +0.37 | +0.35 |
| qwen3_8b | layer | 36 | +0.22 | +0.00 | -0.95 | +0.05 | +0.35 | +0.29 |
| qwen3_8b | component | 28 | -0.30 | -0.38 | -0.10 | -0.08 | -0.41 | +0.03 |

Read across a row: if `group_energy` does not beat `local_energy`/`global_energy_linf` out of layer, group conditioning has not shown added predictive value on that model.
