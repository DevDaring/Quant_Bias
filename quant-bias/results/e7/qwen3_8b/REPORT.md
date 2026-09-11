# e7 — qwen3_8b

## comparators

| method | source | reimpl | MiB | ppl | bbq_acc | bbq_bias | H | A | gap_change |
|---|---|---|---|---|---|---|---|---|---|
| gptq4 | Frantar et al. | yes | 5816 | 11.81 | 0.8562 | -0.04877 | 0.1667 | 0.125 | -0.02575 |
| fair_gptq4 | Proskurina et al. | yes | 5816 | 11.76 | 0.8788 | -0.04943 | 0.1667 | 0.04167 | -0.007203 |
| cwp4_p0.01 | Al Hakim et al. | yes | 6213 | 11.99 | 0.8692 | -0.06859 | 0.1667 | 0.1042 | -0.01236 |
| sparsegpt_sp0.5 | Frantar and Alistarh | yes |  | 13.04 | 0.8867 | -0.03129 | 0.1667 | 0.02083 | -0.01509 |
| debias_sparsegpt_sp0.5 | Proskurina et al. | yes |  | 13.28 | 0.8884 | -0.0394 | 0.125 | 0.04167 | -0.02498 |
| greedy_bias_aware | this study | no | 5820 |  |  |  |  |  |  |
| uniform4 | this study | no | 5816 |  |  |  |  |  |  |


Every row marked reimpl=yes is this project's re-implementation from the published description, not the original authors' code; see each method's `assumptions` field in e7_results.json.
