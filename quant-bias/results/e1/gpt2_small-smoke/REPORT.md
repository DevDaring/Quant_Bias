# e1 — gpt2_small-smoke

## overview

| model | config | bytes_MiB | ppl | bbq_dis_acc | bbq_dis_bias | bbq_amb_bias | wb_gap | H | A |
|---|---|---|---|---|---|---|---|---|---|
| gpt2_small-smoke | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 237.5 | 15.79 | 0.4 | 0.2 | 0.4286 | 0 |  |  |
| gpt2_small-smoke | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 117.6 | 18.12 | 0.6 | -0.2 | 0.2857 | 0.03125 |  |  |
| gpt2_small-smoke | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 158.4 | 15.8 | 0.4 | 0.2 | 0.4286 | 0 |  |  |
| gpt2_small-smoke | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 117.6 | 17.24 | 0.6 | -0.2 | 0.2857 | 0 |  |  |
| gpt2_small-smoke | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 158.4 | 15.79 | 0.4 | 0.2 | 0.4286 | 0 |  |  |
