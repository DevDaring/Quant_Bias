# P0 audit: corrected reanalysis of the completed run

Generated 2026-09-12T15:44:26. Source records under `quant-bias/results/` are unchanged.

## 1. Fixed-granularity correlations (V_final macro vs harmful flip rate)

| model | all n/rho | layer n/rho | component n/rho | ppl vs harm (layer) | #comp vs harm (all) |
|---|---|---|---|---|---|
| gpt2_medium | 40 / +0.635 | 24 / +0.475 | 16 / +0.361 | -0.152 | +0.696 |
| gpt2_small | 28 / +0.760 | 12 / -0.208 | 16 / +0.809 | -0.191 | +0.663 |
| lfm2_2.6b | 58 / +0.466 | 30 / +0.422 | 28 / +0.364 | +0.152 | +0.329 |
| llama_2_7b | 60 / +0.548 | 32 / +0.318 | 28 / +0.431 | +0.117 | +0.288 |
| mistral_7b_v0_1 | 60 / +0.366 | 32 / +0.146 | 28 / +0.423 | +0.148 | +0.293 |
| qwen3_5_2b | 49 / +0.560 | 24 / +0.487 | 25 / +0.347 | +0.095 | +0.459 |
| qwen3_8b | 64 / +0.568 | 36 / +0.126 | 28 / +0.684 | -0.007 | +0.604 |


The last column is the intervention-size confound: whole layers touch many more weights than one component. Only the layer column is an unselected, fixed-size sample.

## 2. Separate outcomes, BBQ disambiguated, sum rule (E1)

| model | config | dense acc | harmful/dense-correct | beneficial/dense-wrong | new stereo errs | new other errs | H (elig groups) | H group n/clusters | macro +dE | #elig/#restricted |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2_medium | gptq4 | 0.482 | 0.085 | 0.093 | 29 | 43 | 0.062 | 48/4 | 0.009 | 18/25 |
| gpt2_medium | gptq8 | 0.482 | 0.001 | 0.007 | 1 | 0 | 0.000 | 96/8 | 0.000 | 18/25 |
| gpt2_medium | rtn4 | 0.482 | 0.114 | 0.121 | 40 | 57 | 0.067 | 30/3 | 0.018 | 18/25 |
| gpt2_medium | rtn8 | 0.482 | 0.005 | 0.007 | 1 | 3 | 0.028 | 36/3 | 0.003 | 18/25 |
| gpt2_small | gptq4 | 0.488 | 0.094 | 0.093 | 30 | 51 | 0.092 | 76/7 | 0.018 | 18/25 |
| gpt2_small | gptq8 | 0.488 | 0.008 | 0.003 | 3 | 4 | 0.022 | 92/11 | 0.003 | 18/25 |
| gpt2_small | rtn4 | 0.488 | 0.131 | 0.137 | 37 | 76 | 0.092 | 76/7 | 0.024 | 18/25 |
| gpt2_small | rtn8 | 0.488 | 0.013 | 0.007 | 3 | 8 | 0.028 | 36/3 | 0.006 | 18/25 |
| lfm2_2.6b | gptq4 | 0.506 | 0.059 | 0.073 | 18 | 35 | 0.083 | 36/3 | 0.015 | 18/25 |
| lfm2_2.6b | gptq8 | 0.506 | 0.011 | 0.013 | 4 | 6 | 0.042 | 48/4 | 0.004 | 18/25 |
| lfm2_2.6b | rtn4 | 0.506 | 0.091 | 0.108 | 28 | 53 | 0.062 | 48/7 | 0.013 | 18/25 |
| lfm2_2.6b | rtn8 | 0.506 | 0.010 | 0.009 | 2 | 7 | 0.047 | 64/6 | 0.006 | 18/25 |
| llama_2_7b | gptq4 | 0.698 | 0.089 | 0.114 | 46 | 64 | 0.208 | 48/7 | 0.029 | 18/25 |
| llama_2_7b | gptq8 | 0.698 | 0.011 | 0.036 | 9 | 5 | 0.047 | 64/6 | 0.006 | 18/25 |
| llama_2_7b | rtn4 | 0.698 | 0.104 | 0.135 | 51 | 77 | 0.104 | 48/4 | 0.038 | 18/25 |
| llama_2_7b | rtn8 | 0.698 | 0.022 | 0.036 | 15 | 12 | 0.042 | 48/4 | 0.005 | 18/25 |
| mistral_7b_v0_1 | gptq4 | 0.837 | 0.046 | 0.118 | 34 | 34 | 0.167 | 36/3 | 0.035 | 18/25 |
| mistral_7b_v0_1 | gptq8 | 0.837 | 0.004 | 0.007 | 3 | 3 | 0.028 | 36/3 | 0.005 | 18/25 |
| mistral_7b_v0_1 | rtn4 | 0.837 | 0.052 | 0.157 | 36 | 41 | 0.139 | 36/3 | 0.032 | 18/25 |
| mistral_7b_v0_1 | rtn8 | 0.837 | 0.005 | 0.014 | 3 | 5 | 0.021 | 48/4 | 0.004 | 18/25 |
| qwen3_5_2b | gptq4 | 0.669 | 0.068 | 0.236 | 34 | 46 | 0.048 | 84/7 | 0.006 | 18/25 |
| qwen3_5_2b | gptq8 | 0.669 | 0.014 | 0.024 | 6 | 10 | 0.033 | 30/3 | 0.009 | 18/25 |
| qwen3_5_2b | rtn4 | 0.669 | 0.105 | 0.185 | 63 | 61 | 0.167 | 30/3 | 0.038 | 18/25 |
| qwen3_5_2b | rtn8 | 0.669 | 0.011 | 0.014 | 4 | 9 | 0.028 | 36/3 | 0.007 | 18/25 |
| qwen3_8b | gptq4 | 0.898 | 0.057 | 0.099 | 45 | 46 | 0.076 | 92/11 | 0.037 | 18/25 |
| qwen3_8b | gptq8 | 0.898 | 0.002 | 0.022 | 1 | 2 | 0.014 | 72/6 | 0.001 | 18/25 |
| qwen3_8b | rtn4 | 0.898 | 0.054 | 0.144 | 34 | 51 | 0.167 | 48/4 | 0.031 | 18/25 |
| qwen3_8b | rtn8 | 0.898 | 0.001 | 0.028 | 0 | 2 | 0.028 | 36/3 | 0.002 | 18/25 |


## 3. Corrected margin checks (E1, sum rule)

| model | config | n | argmax certified | argmax changed | ARGMAX VIOLATIONS | gold certified | gold lost | GOLD VIOLATIONS | ties | mean eps |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2_medium | gptq4 | 7491 | 5275 | 457 | 0 | 786 | 114 | 0 | 0 | 0.551 |
| gpt2_medium | gptq8 | 7491 | 7374 | 15 | 0 | 1356 | 1 | 0 | 0 | 0.037 |
| gpt2_medium | rtn4 | 7491 | 4868 | 583 | 0 | 693 | 152 | 0 | 0 | 0.699 |
| gpt2_medium | rtn8 | 7491 | 7251 | 29 | 0 | 1321 | 4 | 0 | 0 | 0.052 |
| gpt2_small | gptq4 | 7491 | 3157 | 467 | 0 | 789 | 156 | 0 | 0 | 0.725 |
| gpt2_small | gptq8 | 7491 | 7301 | 31 | 0 | 1510 | 11 | 0 | 0 | 0.041 |
| gpt2_small | rtn4 | 7491 | 3357 | 749 | 0 | 475 | 332 | 0 | 0 | 0.958 |
| gpt2_small | rtn8 | 7491 | 7254 | 40 | 0 | 1499 | 14 | 0 | 0 | 0.063 |
| lfm2_2.6b | gptq4 | 7491 | 3445 | 831 | 0 | 956 | 130 | 0 | 74 | 1.378 |
| lfm2_2.6b | gptq8 | 7491 | 6968 | 100 | 0 | 1531 | 17 | 0 | 74 | 0.152 |
| lfm2_2.6b | rtn4 | 7491 | 3303 | 1231 | 0 | 775 | 193 | 0 | 74 | 1.707 |
| lfm2_2.6b | rtn8 | 7491 | 6882 | 104 | 0 | 1517 | 18 | 0 | 74 | 0.180 |
| llama_2_7b | gptq4 | 7491 | 4404 | 434 | 0 | 1001 | 113 | 0 | 11 | 0.407 |
| llama_2_7b | gptq8 | 7491 | 7043 | 78 | 0 | 1549 | 15 | 0 | 11 | 0.061 |
| llama_2_7b | rtn4 | 7491 | 3824 | 423 | 0 | 917 | 130 | 0 | 11 | 0.481 |
| llama_2_7b | rtn8 | 7491 | 6974 | 91 | 0 | 1530 | 27 | 0 | 11 | 0.073 |
| mistral_7b_v0_1 | gptq4 | 7491 | 4145 | 498 | 0 | 1430 | 69 | 0 | 128 | 0.403 |
| mistral_7b_v0_1 | gptq8 | 7491 | 6855 | 101 | 0 | 1859 | 8 | 0 | 128 | 0.062 |
| mistral_7b_v0_1 | rtn4 | 7491 | 3634 | 542 | 0 | 1348 | 83 | 0 | 128 | 0.493 |
| mistral_7b_v0_1 | rtn8 | 7491 | 6738 | 106 | 0 | 1861 | 9 | 0 | 128 | 0.065 |
| qwen3_5_2b | gptq4 | 7491 | 2070 | 745 | 0 | 671 | 144 | 0 | 86 | 1.199 |
| qwen3_5_2b | gptq8 | 7491 | 6781 | 115 | 0 | 1595 | 29 | 0 | 86 | 0.128 |
| qwen3_5_2b | rtn4 | 7491 | 1423 | 1430 | 0 | 456 | 195 | 0 | 86 | 1.464 |
| qwen3_5_2b | rtn8 | 7491 | 6589 | 107 | 0 | 1563 | 24 | 0 | 86 | 0.166 |
| qwen3_8b | gptq4 | 7491 | 4068 | 888 | 0 | 1525 | 428 | 0 | 44 | 1.778 |
| qwen3_8b | gptq8 | 7491 | 6944 | 96 | 0 | 2467 | 15 | 0 | 44 | 0.230 |
| qwen3_8b | rtn4 | 7491 | 2919 | 1057 | 0 | 1200 | 441 | 0 | 44 | 2.638 |
| qwen3_8b | rtn8 | 7491 | 6850 | 122 | 0 | 2456 | 14 | 0 | 44 | 0.251 |


A violation column of 0 everywhere means the recorded 'margin-lemma violations' in the source run were artefacts of testing |gold margin| against argmax change, exactly the counter-example in Next_Plan §3.2.

## 4. Pair gaps with semantic answer correspondence (E1, sum rule)

| model | config | pair type | n | confirmatory | |gap| dense | |gap| comp | d|gap| |
|---|---|---|---|---|---|---|---|
| gpt2_medium | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.008 | 0.007 | -0.001 |
| gpt2_medium | gptq4 | winobias:matched_by_construction | 817 | True | 0.062 | 0.092 | 0.030 |
| gpt2_medium | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.280 | 0.291 | 0.011 |
| gpt2_medium | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.008 | 0.008 | 0.000 |
| gpt2_medium | gptq8 | winobias:matched_by_construction | 817 | True | 0.062 | 0.061 | -0.001 |
| gpt2_medium | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.280 | 0.280 | -0.000 |
| gpt2_medium | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.008 | 0.006 | -0.001 |
| gpt2_medium | rtn4 | winobias:matched_by_construction | 817 | True | 0.062 | 0.086 | 0.024 |
| gpt2_medium | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.280 | 0.284 | 0.004 |
| gpt2_medium | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.008 | 0.008 | -0.000 |
| gpt2_medium | rtn8 | winobias:matched_by_construction | 817 | True | 0.062 | 0.064 | 0.002 |
| gpt2_medium | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.280 | 0.281 | 0.001 |
| gpt2_small | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.011 | 0.014 | 0.003 |
| gpt2_small | gptq4 | winobias:matched_by_construction | 817 | True | 0.043 | 0.037 | -0.005 |
| gpt2_small | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.310 | 0.313 | 0.004 |
| gpt2_small | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.011 | 0.011 | -0.000 |
| gpt2_small | gptq8 | winobias:matched_by_construction | 817 | True | 0.043 | 0.042 | -0.000 |
| gpt2_small | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.310 | 0.310 | 0.001 |
| gpt2_small | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.011 | 0.015 | 0.003 |
| gpt2_small | rtn4 | winobias:matched_by_construction | 817 | True | 0.043 | 0.013 | -0.030 |
| gpt2_small | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.310 | 0.307 | -0.002 |
| gpt2_small | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.011 | 0.012 | 0.000 |
| gpt2_small | rtn8 | winobias:matched_by_construction | 817 | True | 0.043 | 0.043 | -0.000 |
| gpt2_small | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.310 | 0.310 | 0.001 |
| lfm2_2.6b | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.036 | 0.035 | -0.001 |
| lfm2_2.6b | gptq4 | winobias:matched_by_construction | 817 | True | 0.030 | 0.050 | 0.019 |
| lfm2_2.6b | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.190 | 0.195 | 0.004 |
| lfm2_2.6b | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.036 | 0.036 | 0.000 |
| lfm2_2.6b | gptq8 | winobias:matched_by_construction | 817 | True | 0.030 | 0.031 | 0.001 |
| lfm2_2.6b | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.190 | 0.191 | 0.000 |
| lfm2_2.6b | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.036 | 0.048 | 0.012 |
| lfm2_2.6b | rtn4 | winobias:matched_by_construction | 817 | True | 0.030 | 0.033 | 0.002 |
| lfm2_2.6b | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.190 | 0.189 | -0.001 |
| lfm2_2.6b | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.036 | 0.037 | 0.001 |
| lfm2_2.6b | rtn8 | winobias:matched_by_construction | 817 | True | 0.030 | 0.032 | 0.002 |
| lfm2_2.6b | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.190 | 0.191 | 0.001 |
| llama_2_7b | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.023 | 0.021 | -0.002 |
| llama_2_7b | gptq4 | winobias:matched_by_construction | 817 | True | 0.043 | 0.042 | -0.001 |
| llama_2_7b | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.203 | 0.208 | 0.005 |
| llama_2_7b | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.023 | 0.022 | -0.001 |
| llama_2_7b | gptq8 | winobias:matched_by_construction | 817 | True | 0.043 | 0.043 | 0.000 |
| llama_2_7b | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.203 | 0.203 | 0.000 |
| llama_2_7b | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.023 | 0.018 | -0.004 |
| llama_2_7b | rtn4 | winobias:matched_by_construction | 817 | True | 0.043 | 0.040 | -0.003 |
| llama_2_7b | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.203 | 0.219 | 0.016 |
| llama_2_7b | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.023 | 0.022 | -0.001 |
| llama_2_7b | rtn8 | winobias:matched_by_construction | 817 | True | 0.043 | 0.043 | -0.000 |
| llama_2_7b | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.203 | 0.204 | 0.001 |
| mistral_7b_v0_1 | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.021 | 0.024 | 0.003 |
| mistral_7b_v0_1 | gptq4 | winobias:matched_by_construction | 817 | True | 0.069 | 0.082 | 0.013 |
| mistral_7b_v0_1 | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.163 | 0.169 | 0.006 |
| mistral_7b_v0_1 | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.021 | 0.019 | -0.002 |
| mistral_7b_v0_1 | gptq8 | winobias:matched_by_construction | 817 | True | 0.069 | 0.070 | 0.001 |
| mistral_7b_v0_1 | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.163 | 0.163 | 0.000 |
| mistral_7b_v0_1 | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.021 | 0.020 | -0.000 |
| mistral_7b_v0_1 | rtn4 | winobias:matched_by_construction | 817 | True | 0.069 | 0.050 | -0.019 |
| mistral_7b_v0_1 | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.163 | 0.184 | 0.021 |
| mistral_7b_v0_1 | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.021 | 0.020 | -0.001 |
| mistral_7b_v0_1 | rtn8 | winobias:matched_by_construction | 817 | True | 0.069 | 0.070 | 0.000 |
| mistral_7b_v0_1 | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.163 | 0.163 | -0.000 |
| qwen3_5_2b | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.020 | 0.022 | 0.002 |
| qwen3_5_2b | gptq4 | winobias:matched_by_construction | 817 | True | 0.114 | 0.164 | 0.050 |
| qwen3_5_2b | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.207 | 0.220 | 0.013 |
| qwen3_5_2b | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.020 | 0.020 | -0.000 |
| qwen3_5_2b | gptq8 | winobias:matched_by_construction | 817 | True | 0.114 | 0.109 | -0.005 |
| qwen3_5_2b | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.207 | 0.209 | 0.002 |
| qwen3_5_2b | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.020 | 0.018 | -0.002 |
| qwen3_5_2b | rtn4 | winobias:matched_by_construction | 817 | True | 0.114 | 0.108 | -0.005 |
| qwen3_5_2b | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.207 | 0.209 | 0.001 |
| qwen3_5_2b | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.020 | 0.020 | -0.000 |
| qwen3_5_2b | rtn8 | winobias:matched_by_construction | 817 | True | 0.114 | 0.116 | 0.002 |
| qwen3_5_2b | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.207 | 0.210 | 0.002 |
| qwen3_8b | gptq4 | discrim_eval:matched_by_construction | 2294 | True | 0.068 | 0.057 | -0.011 |
| qwen3_8b | gptq4 | winobias:matched_by_construction | 817 | True | 0.252 | 0.181 | -0.071 |
| qwen3_8b | gptq4 | bbq:template_matched_unaudited | 1610 | False | 0.126 | 0.161 | 0.035 |
| qwen3_8b | gptq8 | discrim_eval:matched_by_construction | 2294 | True | 0.068 | 0.069 | 0.001 |
| qwen3_8b | gptq8 | winobias:matched_by_construction | 817 | True | 0.252 | 0.250 | -0.002 |
| qwen3_8b | gptq8 | bbq:template_matched_unaudited | 1610 | False | 0.126 | 0.127 | 0.001 |
| qwen3_8b | rtn4 | discrim_eval:matched_by_construction | 2294 | True | 0.068 | 0.062 | -0.007 |
| qwen3_8b | rtn4 | winobias:matched_by_construction | 817 | True | 0.252 | 0.211 | -0.041 |
| qwen3_8b | rtn4 | bbq:template_matched_unaudited | 1610 | False | 0.126 | 0.153 | 0.026 |
| qwen3_8b | rtn8 | discrim_eval:matched_by_construction | 2294 | True | 0.068 | 0.065 | -0.004 |
| qwen3_8b | rtn8 | winobias:matched_by_construction | 817 | True | 0.252 | 0.254 | 0.002 |
| qwen3_8b | rtn8 | bbq:template_matched_unaudited | 1610 | False | 0.126 | 0.125 | -0.001 |


## 5. Dense competence under the scoring adaptation

| model | rule | bbq disambig | bbq ambig | ambig unknown-rate | wb pro | wb anti | picks longest |
|---|---|---|---|---|---|---|---|
| gpt2_medium | sum | 0.482 | 0.053 | 0.053 | 0.285 | 0.257 | 0.406 |
| gpt2_medium | mean | 0.507 | 0.014 | 0.014 | 0.317 | 0.274 | 0.624 |
| gpt2_medium | verdict | BBQ ambiguous accuracy 0.053 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias pro |  |  |  |  |  |
| gpt2_small | sum | 0.488 | 0.081 | 0.081 | 0.362 | 0.330 | 0.413 |
| gpt2_small | mean | 0.506 | 0.014 | 0.014 | 0.382 | 0.337 | 0.646 |
| gpt2_small | verdict | BBQ ambiguous accuracy 0.081 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias pro |  |  |  |  |  |
| lfm2_2.6b | sum | 0.506 | 0.157 | 0.157 | 0.267 | 0.256 | 0.404 |
| lfm2_2.6b | mean | 0.506 | 0.079 | 0.079 | 0.299 | 0.268 | 0.575 |
| lfm2_2.6b | verdict | BBQ ambiguous accuracy 0.157 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias pro |  |  |  |  |  |
| llama_2_7b | sum | 0.698 | 0.016 | 0.016 | 0.245 | 0.246 | 0.444 |
| llama_2_7b | mean | 0.661 | 0.005 | 0.005 | 0.305 | 0.262 | 0.670 |
| llama_2_7b | verdict | BBQ ambiguous accuracy 0.016 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias pro |  |  |  |  |  |
| mistral_7b_v0_1 | sum | 0.837 | 0.015 | 0.015 | 0.256 | 0.250 | 0.424 |
| mistral_7b_v0_1 | mean | 0.810 | 0.009 | 0.009 | 0.344 | 0.274 | 0.596 |
| mistral_7b_v0_1 | verdict | BBQ ambiguous accuracy 0.015 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias pro |  |  |  |  |  |
| qwen3_5_2b | sum | 0.669 | 0.032 | 0.032 | 0.338 | 0.277 | 0.443 |
| qwen3_5_2b | mean | 0.642 | 0.012 | 0.012 | 0.387 | 0.290 | 0.602 |
| qwen3_5_2b | verdict | BBQ ambiguous accuracy 0.032 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias pro |  |  |  |  |  |
| qwen3_8b | sum | 0.898 | 0.030 | 0.030 | 0.666 | 0.469 | 0.458 |
| qwen3_8b | mean | 0.905 | 0.017 | 0.017 | 0.652 | 0.469 | 0.515 |
| qwen3_8b | verdict | BBQ ambiguous accuracy 0.030 is near/below chance: the model almost never selects 'unknown'; ambiguous-context metrics are not interpretable as fairness outcomes until this is understood; WinoBias ant |  |  |  |  |  |


## 6. E7 comparators: PAIRED differences vs uniform GPTQ (support-preserving bootstrap)

| model | method | dAcc | dAcc CI | verdict | dH | dH CI | draws kept |
|---|---|---|---|---|---|---|---|
| mistral_7b_v0_1 | cwp | -0.005 | [-0.0195, +0.0120] | not distinguishable on these examples (NOT equivalence) | -0.028 | [-0.095, +0.073] | 300/300 |
| mistral_7b_v0_1 | debias_sparsegpt | -0.035 | [-0.0562, -0.0083] | B better than A | 0.042 | [-0.122, +0.179] | 300/300 |
| mistral_7b_v0_1 | fair_gptq | 0.008 | [-0.0048, +0.0212] | not distinguishable on these examples (NOT equivalence) | -0.111 | [-0.165, +0.038] | 300/300 |
| mistral_7b_v0_1 | sparsegpt | -0.061 | [-0.0902, -0.0334] | B better than A | 0.181 | [-0.033, +0.329] | 300/300 |
| qwen3_8b | cwp | 0.013 | [-0.0051, +0.0299] | not distinguishable on these examples (NOT equivalence) | 0.091 | [-0.086, +0.163] | 300/300 |
| qwen3_8b | debias_sparsegpt | 0.032 | [+0.0128, +0.0546] | A better than B | 0.028 | [-0.123, +0.127] | 300/300 |
| qwen3_8b | fair_gptq | 0.023 | [+0.0071, +0.0362] | A better than B | -0.011 | [-0.134, +0.019] | 300/300 |
| qwen3_8b | sparsegpt | 0.031 | [+0.0103, +0.0545] | A better than B | 0.033 | [-0.114, +0.111] | 300/300 |


Eligible groups (n>=20 AND >=3 independent templates): mistral_7b_v0_1: 18 eligible, 25 restricted; qwen3_8b: 18 eligible, 25 restricted


## 7. Interval coverage on this design (simulation)

Nominal 0.95, empirical 1.000 over 150 simulations (target group backed by 3 clusters): **adequate**
