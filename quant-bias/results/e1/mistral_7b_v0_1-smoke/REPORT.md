# e1 — mistral_7b_v0_1-smoke

## overview

| model | config | bytes_MiB | ppl | bbq_dis_acc | bbq_dis_bias | bbq_amb_bias | wb_gap | H | A |
|---|---|---|---|---|---|---|---|---|---|
| gpt2_medium-smoke | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 677.2 | 12.5 | 0.2 | 0.2 | 0.4286 | -0.03125 |  |  |
| gpt2_medium-smoke | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 250.8 | 13.36 | 0.6 | -0.2 | 0.4286 | 0 |  |  |
| gpt2_medium-smoke | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 395.9 | 12.47 | 0.2 | 0.2 | 0.4286 | -0.03125 |  |  |
| gpt2_medium-smoke | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 250.8 | 13.05 | 0.4 | 0.2 | 0.4286 | 0.0625 |  |  |
| gpt2_medium-smoke | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 395.9 | 12.53 | 0.2 | 0.2 | 0.4286 | 0 |  |  |
| gpt2_small-smoke | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 237.5 | 15.79 | 0.4 | 0.2 | 0.4286 | 0 |  |  |
| gpt2_small-smoke | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 117.6 | 18.12 | 0.6 | -0.2 | 0.2857 | 0.03125 |  |  |
| gpt2_small-smoke | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 158.4 | 15.8 | 0.4 | 0.2 | 0.4286 | 0 |  |  |
| gpt2_small-smoke | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 117.6 | 17.24 | 0.6 | -0.2 | 0.2857 | 0 |  |  |
| gpt2_small-smoke | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 158.4 | 15.79 | 0.4 | 0.2 | 0.4286 | 0 |  |  |
| lfm2_2.6b-smoke | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 4900 | 7.788 | 0.4 | 0.2 | 0 | -0.03125 |  |  |
| lfm2_2.6b-smoke | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 1463 | 8.382 | 0.4 | 0.2 | -0.1429 | 0.0625 |  |  |
| lfm2_2.6b-smoke | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 2633 | 7.797 | 0.4 | 0.2 | 0.1429 | 0 |  |  |
| lfm2_2.6b-smoke | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 1463 | 8.113 | 0.4 | 0.2 | -0.2857 | -0.03125 |  |  |
| lfm2_2.6b-smoke | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 2633 | 7.8 | 0.4 | 0.2 | 0 | 0 |  |  |
| mistral_7b_v0_1-smoke | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 1.381e+04 | 3.862 | 0.6 | -0.2 | 0.1429 | 0 |  |  |
| mistral_7b_v0_1-smoke | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 3959 | 3.969 | 0.4 | 0.2 | 0.4286 | 0.03125 |  |  |
| mistral_7b_v0_1-smoke | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 7313 | 3.865 | 0.6 | -0.2 | 0.1429 | 0 |  |  |
| mistral_7b_v0_1-smoke | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 3959 | 3.949 | 0.6 | -0.2 | 0.1429 | 0 |  |  |
| mistral_7b_v0_1-smoke | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 7313 | 3.864 | 0.6 | -0.2 | 0.1429 | 0 |  |  |
| qwen3_5_2b-smoke | {'bits': 16, 'method': 'dense', 'name': 'dense'} | 3589 | 8.128 | 0.4 | 0.2 | -0.4286 | 0.0625 |  |  |
| qwen3_5_2b-smoke | {'bits': 4, 'method': 'gptq', 'name': 'gptq4'} | 1653 | 8.647 | 0.4 | 0.2 | -0.4286 | 0.09375 |  |  |
| qwen3_5_2b-smoke | {'bits': 8, 'method': 'gptq', 'name': 'gptq8'} | 2312 | 8.158 | 0.4 | 0.2 | -0.4286 | 0.09375 |  |  |
| qwen3_5_2b-smoke | {'bits': 4, 'method': 'rtn', 'name': 'rtn4'} | 1653 | 9.358 | 0.4 | 0.2 | -0.4286 | 0.03125 |  |  |
| qwen3_5_2b-smoke | {'bits': 8, 'method': 'rtn', 'name': 'rtn8'} | 2312 | 8.138 | 0.4 | 0.2 | -0.4286 | 0.0625 |  |  |
