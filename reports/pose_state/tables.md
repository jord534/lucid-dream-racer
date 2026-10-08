
### Test-track windows from 30 steps before leaving the road (275): off-road steps recognised, steps 20-40

| Model | Fed | tau 1.15 | tau 0.05 |
|---|---|---|---|
| ctrl_s0 | own predictions | 0.34 [0.30, 0.37] | 0.33 [0.28, 0.37] |
| ctrl_s0 | real frames (and true pose): 1-step prediction | 0.86 [0.84, 0.89] | 0.90 [0.87, 0.93] |
| - | probe on the real frames (model-independent ceiling) | 0.91 [0.88, 0.93] | 0.91 [0.88, 0.93] |
| ctrl_s1 | own predictions | 0.38 [0.34, 0.41] | 0.39 [0.34, 0.44] |
| ctrl_s1 | real frames (and true pose): 1-step prediction | 0.86 [0.84, 0.89] | 0.90 [0.87, 0.93] |
| pose_s0 | own predictions | 0.32 [0.29, 0.36] | 0.35 [0.31, 0.40] |
| pose_s0 | oracle: own latent, true pose | 0.74 [0.71, 0.77] | 0.78 [0.73, 0.81] |
| pose_s0 | real frames, own pose | 0.78 [0.75, 0.81] | 0.82 [0.78, 0.85] |
| pose_s0 | real frames (and true pose): 1-step prediction | 0.88 [0.85, 0.90] | 0.92 [0.89, 0.95] |
| pose_s1 | own predictions | 0.28 [0.24, 0.32] | 0.24 [0.20, 0.29] |
| pose_s1 | oracle: own latent, true pose | 0.82 [0.79, 0.85] | 0.87 [0.83, 0.90] |
| pose_s1 | real frames, own pose | 0.83 [0.80, 0.85] | 0.86 [0.83, 0.89] |
| pose_s1 | real frames (and true pose): 1-step prediction | 0.90 [0.87, 0.93] | 0.93 [0.90, 0.95] |
| ref_headline | own predictions | 0.40 [0.36, 0.44] | 0.39 [0.34, 0.44] |
| ref_headline | real frames (and true pose): 1-step prediction | 0.88 [0.85, 0.91] | 0.91 [0.88, 0.94] |
| ref_loop_head_r02 | own predictions | (0.63 published, no interval) | 0.69 [0.64, 0.73] |
| ref_loop_head_r02 | real frames (and true pose): 1-step prediction | (0.63 published, no interval) | 0.92 [0.89, 0.95] |

### Validation windows from 30 steps before leaving the road (52): off-road steps recognised, steps 20-40

| Model | Fed | tau 1.15 | tau 0.05 |
|---|---|---|---|
| ctrl_s0 | own predictions | 0.75 [0.67, 0.82] | 0.82 [0.73, 0.90] |
| ctrl_s0 | real frames (and true pose): 1-step prediction | 0.86 [0.79, 0.92] | 0.88 [0.79, 0.95] |
| - | probe on the real frames (model-independent ceiling) | 0.87 [0.80, 0.94] | 0.87 [0.80, 0.94] |
| ctrl_s1 | own predictions | 0.64 [0.56, 0.72] | 0.73 [0.62, 0.82] |
| ctrl_s1 | real frames (and true pose): 1-step prediction | 0.85 [0.78, 0.91] | 0.88 [0.79, 0.95] |
| pose_s0 | own predictions | 0.71 [0.62, 0.79] | 0.74 [0.63, 0.85] |
| pose_s0 | oracle: own latent, true pose | 0.82 [0.75, 0.89] | 0.87 [0.78, 0.94] |
| pose_s0 | real frames, own pose | 0.84 [0.77, 0.91] | 0.85 [0.76, 0.93] |
| pose_s0 | real frames (and true pose): 1-step prediction | 0.87 [0.79, 0.93] | 0.88 [0.80, 0.95] |
| pose_s1 | own predictions | 0.67 [0.59, 0.74] | 0.77 [0.67, 0.86] |
| pose_s1 | oracle: own latent, true pose | 0.84 [0.76, 0.90] | 0.90 [0.81, 0.96] |
| pose_s1 | real frames, own pose | 0.86 [0.79, 0.92] | 0.89 [0.81, 0.95] |
| pose_s1 | real frames (and true pose): 1-step prediction | 0.87 [0.80, 0.93] | 0.89 [0.81, 0.96] |
| ref_headline | own predictions | 0.69 [0.62, 0.76] | 0.78 [0.69, 0.86] |
| ref_headline | real frames (and true pose): 1-step prediction | 0.86 [0.79, 0.92] | 0.88 [0.80, 0.95] |
| ref_loop_head_r02 | own predictions | (0.63 published, no interval) | 0.71 [0.61, 0.80] |
| ref_loop_head_r02 | real frames (and true pose): 1-step prediction | (0.63 published, no interval) | 0.88 [0.80, 0.95] |

### Test-track random starts (1,800): road-status accuracy, fed own predictions

| Model | 20-40, tau 1.15 | 70-100, tau 1.15 | 20-40, tau 0.05 | 70-100, tau 0.05 |
|---|---|---|---|---|
| pose_s0 | 0.67 [0.66, 0.68] | 0.35 [0.33, 0.36] | 0.72 [0.70, 0.73] | 0.29 [0.27, 0.31] |
| pose_s1 | 0.63 [0.62, 0.65] | 0.33 [0.31, 0.35] | 0.64 [0.63, 0.66] | 0.30 [0.28, 0.32] |
| ctrl_s0 | 0.81 [0.80, 0.82] | 0.46 [0.45, 0.48] | 0.82 [0.80, 0.83] | 0.52 [0.50, 0.54] |
| ctrl_s1 | 0.79 [0.78, 0.80] | 0.37 [0.35, 0.38] | 0.80 [0.78, 0.81] | 0.32 [0.31, 0.34] |
| ref_headline | 0.88 [0.87, 0.89] | 0.45 [0.44, 0.47] | 0.92 [0.91, 0.93] | 0.54 [0.53, 0.56] |
| ref_loop_head_r02 | - | - | 0.87 [0.86, 0.88] | 0.31 [0.30, 0.33] |

### Validation random starts (210): road-status accuracy, fed own predictions

| Model | 20-40, tau 1.15 | 70-100, tau 1.15 | 20-40, tau 0.05 | 70-100, tau 0.05 |
|---|---|---|---|---|
| pose_s0 | 0.94 [0.93, 0.96] | 0.82 [0.79, 0.85] | 0.97 [0.95, 0.98] | 0.87 [0.83, 0.90] |
| pose_s1 | 0.94 [0.92, 0.95] | 0.83 [0.80, 0.86] | 0.96 [0.94, 0.98] | 0.84 [0.80, 0.88] |
| ctrl_s0 | 0.94 [0.93, 0.96] | 0.84 [0.81, 0.87] | 0.97 [0.95, 0.98] | 0.87 [0.84, 0.90] |
| ctrl_s1 | 0.93 [0.90, 0.94] | 0.84 [0.81, 0.86] | 0.95 [0.93, 0.97] | 0.88 [0.84, 0.91] |
| ref_headline | 0.94 [0.93, 0.96] | 0.83 [0.81, 0.86] | 0.97 [0.95, 0.98] | 0.88 [0.85, 0.91] |
| ref_loop_head_r02 | - | - | 0.95 [0.93, 0.97] | 0.86 [0.83, 0.90] |

### Pose error, Test-track windows from 30 steps before leaving the road, tau 1.15 (lateral offset in half-widths / heading in degrees / off-road rate implied by the predicted offset vs the true off-road share)

| Model | Fed | 0-5 | 5-10 | 10-20 | 20-40 | 40-70 | 70-100 |
|---|---|---|---|---|---|---|---|
| pose_s0 | own predictions | 0.16 / 6 / 0.05 vs 0.06 | 0.28 / 14 / 0.10 vs 0.04 | 0.62 / 26 / 0.36 vs 0.01 | 1.62 / 37 / 0.27 vs 0.55 | 2.56 / 61 / 0.59 vs 0.89 | 2.71 / 88 / 0.69 vs 0.69 |
| pose_s0 | oracle: own latent, true pose | 0.09 / 3 / 0.05 vs 0.06 | 0.10 / 5 / 0.04 vs 0.04 | 0.12 / 7 / 0.02 vs 0.01 | 0.12 / 4 / 0.56 vs 0.55 | 0.17 / 5 / 0.83 vs 0.89 | 0.16 / 5 / 0.63 vs 0.69 |
| pose_s0 | real frames, own pose | 0.15 / 6 / 0.04 vs 0.06 | 0.28 / 11 / 0.09 vs 0.04 | 0.42 / 15 / 0.16 vs 0.01 | 0.88 / 24 / 0.28 vs 0.55 | 1.33 / 34 / 0.71 vs 0.89 | 1.71 / 61 / 0.63 vs 0.69 |
| pose_s0 | real frames (and true pose): 1-step prediction | 0.09 / 3 / 0.05 vs 0.06 | 0.10 / 4 / 0.03 vs 0.04 | 0.10 / 5 / 0.01 vs 0.01 | 0.08 / 2 / 0.58 vs 0.55 | 0.09 / 2 / 0.85 vs 0.89 | 0.09 / 2 / 0.66 vs 0.69 |
| pose_s1 | own predictions | 0.21 / 7 / 0.04 vs 0.06 | 0.37 / 14 / 0.05 vs 0.04 | 0.65 / 33 / 0.15 vs 0.01 | 1.71 / 40 / 0.24 vs 0.55 | 2.84 / 61 / 0.53 vs 0.89 | 3.03 / 86 / 0.71 vs 0.69 |
| pose_s1 | oracle: own latent, true pose | 0.11 / 4 / 0.04 vs 0.06 | 0.10 / 6 / 0.03 vs 0.04 | 0.08 / 8 / 0.02 vs 0.01 | 0.09 / 4 / 0.59 vs 0.55 | 0.13 / 5 / 0.89 vs 0.89 | 0.16 / 5 / 0.68 vs 0.69 |
| pose_s1 | real frames, own pose | 0.21 / 7 / 0.05 vs 0.06 | 0.33 / 13 / 0.07 vs 0.04 | 0.55 / 29 / 0.19 vs 0.01 | 1.14 / 34 / 0.34 vs 0.55 | 1.43 / 48 / 0.83 vs 0.89 | 1.68 / 52 / 0.64 vs 0.69 |
| pose_s1 | real frames (and true pose): 1-step prediction | 0.11 / 3 / 0.04 vs 0.06 | 0.11 / 5 / 0.04 vs 0.04 | 0.09 / 7 / 0.02 vs 0.01 | 0.07 / 3 / 0.62 vs 0.55 | 0.09 / 3 / 0.86 vs 0.89 | 0.09 / 3 / 0.67 vs 0.69 |

### Pose error, Test-track random starts, tau 1.15 (lateral offset in half-widths / heading in degrees / off-road rate implied by the predicted offset vs the true off-road share)

| Model | Fed | 0-5 | 5-10 | 10-20 | 20-40 | 40-70 | 70-100 |
|---|---|---|---|---|---|---|---|
| pose_s0 | own predictions | 0.14 / 4 / 0.12 vs 0.12 | 0.27 / 9 / 0.13 vs 0.12 | 0.49 / 16 / 0.19 vs 0.13 | 1.14 / 32 / 0.46 vs 0.13 | 2.34 / 51 / 0.70 vs 0.14 | 3.03 / 57 / 0.77 vs 0.15 |
| pose_s0 | oracle: own latent, true pose | 0.08 / 3 / 0.12 vs 0.12 | 0.08 / 3 / 0.12 vs 0.12 | 0.09 / 4 / 0.12 vs 0.13 | 0.10 / 5 / 0.13 vs 0.13 | 0.11 / 5 / 0.14 vs 0.14 | 0.11 / 4 / 0.14 vs 0.15 |
| pose_s0 | real frames, own pose | 0.14 / 4 / 0.12 vs 0.12 | 0.24 / 7 / 0.13 vs 0.12 | 0.32 / 10 / 0.14 vs 0.13 | 0.39 / 13 / 0.13 vs 0.13 | 0.46 / 16 / 0.14 vs 0.14 | 0.49 / 20 / 0.14 vs 0.15 |
| pose_s0 | real frames (and true pose): 1-step prediction | 0.08 / 3 / 0.12 vs 0.12 | 0.08 / 3 / 0.12 vs 0.12 | 0.08 / 3 / 0.12 vs 0.13 | 0.08 / 3 / 0.13 vs 0.13 | 0.08 / 3 / 0.14 vs 0.14 | 0.08 / 3 / 0.14 vs 0.15 |
| pose_s1 | own predictions | 0.14 / 5 / 0.12 vs 0.12 | 0.23 / 10 / 0.13 vs 0.12 | 0.43 / 19 / 0.15 vs 0.13 | 1.22 / 38 / 0.44 vs 0.13 | 2.76 / 62 / 0.76 vs 0.14 | 3.47 / 74 / 0.84 vs 0.15 |
| pose_s1 | oracle: own latent, true pose | 0.08 / 3 / 0.12 vs 0.12 | 0.08 / 3 / 0.12 vs 0.12 | 0.10 / 4 / 0.13 vs 0.13 | 0.12 / 4 / 0.14 vs 0.13 | 0.15 / 4 / 0.15 vs 0.14 | 0.13 / 4 / 0.15 vs 0.15 |
| pose_s1 | real frames, own pose | 0.14 / 5 / 0.12 vs 0.12 | 0.23 / 9 / 0.13 vs 0.12 | 0.31 / 12 / 0.14 vs 0.13 | 0.42 / 15 / 0.17 vs 0.13 | 0.53 / 18 / 0.19 vs 0.14 | 0.61 / 21 / 0.20 vs 0.15 |
| pose_s1 | real frames (and true pose): 1-step prediction | 0.08 / 3 / 0.12 vs 0.12 | 0.09 / 3 / 0.12 vs 0.12 | 0.09 / 3 / 0.13 vs 0.13 | 0.09 / 3 / 0.13 vs 0.13 | 0.09 / 3 / 0.14 vs 0.14 | 0.09 / 3 / 0.15 vs 0.15 |
