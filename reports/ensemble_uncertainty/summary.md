# Ensemble epistemic uncertainty experiment

- Members: 2; seeds: [72000, 72001]
- Cumulative member training: 5226.33 seconds
- Latest training invocation wall time: 2595.9039064590006 seconds with 2 concurrent members
- Inference benchmark: 0.066 ms/model step; 0.162 ms/ensemble step (2.47x)
- Disagreement: maximum pairwise L2 distance between deterministic MDN mixture means.
- Aleatoric scale is recorded separately from epistemic disagreement.

## Disagreement by trajectory category

| Category | Trajectories | Mean (95% bootstrap CI) | Median | 10th percentile | 90th percentile |
|---|---:|---:|---:|---:|---:|
| constant | 100 | 0.2647 [[0.25505913725770074, 0.27569858163578437]] | 0.2578 | 0.2164 | 0.3215 |
| controller | 100 | 0.3418 [[0.3358330971317366, 0.34912549859393394]] | 0.3363 | 0.3131 | 0.3735 |
| high_predicted_return | 25 | 0.3322 [[0.3257601005275547, 0.33868585605159396]] | 0.3227 | 0.3144 | 0.3542 |
| onset_open_loop | 100 | 0.3256 [[0.3152279878208041, 0.3357377616653591]] | 0.3327 | 0.2515 | 0.3871 |
| ordinary | 70 | 0.2195 [[0.20378829582965532, 0.23549092574248623]] | 0.2083 | 0.1367 | 0.3086 |
| random | 100 | 0.3983 [[0.3708101933950744, 0.43050083585323756]] | 0.3394 | 0.2913 | 0.6125 |

Timestep-band disagreement summaries:

**constant**

| Timestep | Mean | Median | 10th percentile | 90th percentile | Steps |
|---|---:|---:|---:|---:|---:|
| 0-20 | 0.3252 | 0.3033 | 0.1902 | 0.4701 | 2000 |
| 20-40 | 0.2820 | 0.2634 | 0.1905 | 0.3955 | 2000 |
| 40-70 | 0.2479 | 0.2241 | 0.1571 | 0.3640 | 3000 |
| 70-100 | 0.2298 | 0.2092 | 0.1391 | 0.3522 | 3000 |

**controller**

| Timestep | Mean | Median | 10th percentile | 90th percentile | Steps |
|---|---:|---:|---:|---:|---:|
| 0-20 | 0.3789 | 0.3568 | 0.2483 | 0.5246 | 2000 |
| 20-40 | 0.3546 | 0.3464 | 0.2369 | 0.4734 | 2000 |
| 40-70 | 0.3243 | 0.3300 | 0.2319 | 0.4003 | 3000 |
| 70-100 | 0.3261 | 0.3380 | 0.2301 | 0.4020 | 3000 |

**high_predicted_return**

| Timestep | Mean | Median | 10th percentile | 90th percentile | Steps |
|---|---:|---:|---:|---:|---:|
| 0-20 | 0.3520 | 0.3522 | 0.2566 | 0.4434 | 500 |
| 20-40 | 0.3491 | 0.3463 | 0.2438 | 0.4595 | 500 |
| 40-70 | 0.3166 | 0.3308 | 0.2324 | 0.3889 | 750 |
| 70-100 | 0.3232 | 0.3414 | 0.2290 | 0.3989 | 750 |

**onset_open_loop**

| Timestep | Mean | Median | 10th percentile | 90th percentile | Steps |
|---|---:|---:|---:|---:|---:|
| 0-20 | 0.4104 | 0.3930 | 0.2641 | 0.5735 | 2000 |
| 20-40 | 0.3301 | 0.3206 | 0.2117 | 0.4553 | 2000 |
| 40-70 | 0.2985 | 0.2831 | 0.1669 | 0.4475 | 3000 |
| 70-100 | 0.2931 | 0.2726 | 0.1642 | 0.4475 | 3000 |

**ordinary**

| Timestep | Mean | Median | 10th percentile | 90th percentile | Steps |
|---|---:|---:|---:|---:|---:|
| 0-20 | 0.2382 | 0.2034 | 0.1100 | 0.4003 | 1400 |
| 20-40 | 0.2162 | 0.1957 | 0.1143 | 0.3417 | 1400 |
| 40-70 | 0.2147 | 0.1928 | 0.1106 | 0.3423 | 2100 |
| 70-100 | 0.2140 | 0.1999 | 0.1181 | 0.3239 | 2100 |

**random**

| Timestep | Mean | Median | 10th percentile | 90th percentile | Steps |
|---|---:|---:|---:|---:|---:|
| 0-20 | 0.4666 | 0.4355 | 0.2705 | 0.6888 | 2000 |
| 20-40 | 0.4249 | 0.3805 | 0.2394 | 0.6578 | 2000 |
| 40-70 | 0.3846 | 0.3269 | 0.2162 | 0.6201 | 3000 |
| 70-100 | 0.3488 | 0.2884 | 0.2009 | 0.5730 | 3000 |


## Matched model and simulator returns

- Matched controller sequences: 25
- Spearman disagreement vs overoptimism: -0.12461538461538461
- Bootstrap 95% interval for correlation: [-0.5713854828705245, 0.36085786182419055]
- Spearman disagreement vs eventual simulator failure: 0.11322770341445956
- Bootstrap 95% interval for failure correlation: [-0.02839113254129243, 0.32158878875928876]
- High predicted-return matched overoptimism: 116.58905105670314 (n=7)
- Mean disagreement, simulator failure / success: 0.34641447845225537 / 0.31889104142785074

Matched overoptimism by action policy:

| Policy | Matched n | Mean predicted − real return | 95% bootstrap CI |
|---|---:|---:|---:|
| controller | 25 | 102.05080983804066 | [87.79106297071941, 114.12441840957715] |
| constant | 25 | 53.5421964594737 | [33.670611502107725, 73.63565496980436] |
| random | 25 | -5.919442455466323 | [-9.377319130088091, -2.850513373024226] |

Paired model-overoptimism differences (positive means the controller exploit gap is larger):

- controller_minus_constant_overoptimism: 48.50861337856697 (95% bootstrap CI [29.66350819059837, 67.92088004418986]; n=25).
- controller_minus_random_overoptimism: 107.970252293507 (95% bootstrap CI [94.58254096031139, 119.547729829242]; n=25).

Paired change in controller disagreement over time:

- 20-40_minus_0-20: mean difference -0.024276732377707957; 95% bootstrap CI [-0.041555032608844335, -0.007751928831823174] (n=100).
- 40-70_minus_20-40: mean difference -0.030258140375216805; 95% bootstrap CI [-0.04427175660530725, -0.017089749651774767] (n=100).
- 70-100_minus_40-70: mean difference 0.0017311009715000774; 95% bootstrap CI [-0.006433559090768299, 0.00924769117174048] (n=100).

## Interpretation checks

- CMA controller − ordinary mean disagreement: 0.12230213028639556 (bootstrap 95% CI [0.10509687934570007, 0.13853996733537205]).
- CMA controller − constant-input mean disagreement: 0.07705097398124634 (bootstrap 95% CI [0.06700160810253585, 0.08620259566496126]).
- High-return vs remaining CMA runs: -0.012816863199075101 (bootstrap 95% CI [-0.024055998246123332, -0.0019224339334666706]).
- Matched simulator-failure vs success disagreement: 0.02752343702440463 (bootstrap 95% CI [0.014818106548550208, 0.04108219391154122]).

Interpret the comparisons only after reviewing sample sizes and the confidence intervals. The behavior policy follows the selected single model's deterministic mixture-mean rollout; all members are scored on the same latent/action inputs. This isolates disagreement conditional on one trajectory and does not claim that the ensemble mean is an optimized control policy.
