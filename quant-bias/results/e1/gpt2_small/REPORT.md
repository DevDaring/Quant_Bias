# e1 — gpt2_small

## overview

| model | config | bytes_MiB | ppl | bbq_dis_acc | bbq_dis_bias | bbq_amb_bias | wb_gap | H | A |
|---|---|---|---|---|---|---|---|---|---|
| gpt2_medium | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 677.2 | 20.56 | 0.4819 | -0.1207 | -0.0923 | 0.02815 |  |  |
| gpt2_small | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 237.5 | 26.76 | 0.4881 | -0.1017 | -0.08437 | 0.03182 |  |  |
| gpt2_small | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 117.6 | 28.12 | 0.4898 | -0.1161 | -0.08154 | 0.02326 | 0.09211 | -0.05556 |
| gpt2_small | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 158.4 | 26.77 | 0.4858 | -0.1026 | -0.07758 | 0.03182 | 0.04167 | 0 |
| gpt2_small | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 117.6 | 29.38 | 0.4943 | -0.1093 | -0.08437 | 0.003672 | 0.09211 | -0.1412 |
| gpt2_small | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 158.4 | 26.79 | 0.4853 | -0.1051 | -0.07928 | 0.03427 | 0.02778 | 0 |
