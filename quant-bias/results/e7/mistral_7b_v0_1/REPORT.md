# e7 — mistral_7b_v0_1

## comparators

| method | source | reimpl | MiB | ppl | bbq_acc | bbq_bias | H | A | gap_change |
|---|---|---|---|---|---|---|---|---|---|
| gptq4 | Frantar et al. | yes | 3959 | 6.347 | 0.8182 | -0.05647 | 0.1667 | -0.04167 | -0.001066 |
| fair_gptq4 | Proskurina et al. | yes | 3959 | 6.329 | 0.8262 | -0.07587 | 0.05556 | -0.04167 | 0.003464 |
| cwp4_p0.01 | Al Hakim et al. | yes | 4358 | 6.367 | 0.8137 | -0.06606 | 0.1389 | -0.04167 | -0.007979 |
| sparsegpt_sp0.5 | Frantar and Alistarh | yes |  | 7.797 | 0.7571 | -0.05461 | 0.3472 | -0.04167 | -0.001744 |
| debias_sparsegpt_sp0.5 | Proskurina et al. | yes |  | 7.802 | 0.7837 | -0.02958 | 0.2083 | 0 | 0.001862 |
| greedy_bias_aware | this study | no | 3989 |  |  |  |  |  |  |
| uniform4 | this study | no | 3959 |  |  |  |  |  |  |


Every row marked reimpl=yes is this project's re-implementation from the published description, not the original authors' code; see each method's `assumptions` field in e7_results.json.
