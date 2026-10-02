# Head accuracy against dream step

Model `runs/loop_head/r02/mdnrnn/best.pt`, real actions replayed, 16 dream samples per window, temperature 1.15. Made by `python diagnostics/head_horizon.py` (16.7 min). See the script's docstring for the question and how to read it.

### original data, held-out validation episodes | random start  (210 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.45 | 0.99 / 1.00 | 0.99 / 0.99 | 0.96 / 0.95 | 0.96 / 0.97 |
| 5-10 | 0.46 | 0.99 / 0.99 | 0.99 / 0.99 | 0.97 / 0.96 | 0.98 / 0.98 |
| 10-20 | 0.46 | 0.99 / 0.99 | 0.98 / 0.98 | 0.96 / 0.95 | 0.98 / 0.97 |
| 20-40 | 0.45 | 0.99 / 0.99 | 0.95 / 0.95 | 0.93 / 0.92 | 0.96 / 0.96 |
| 40-70 | 0.46 | 0.99 / 0.99 | 0.90 / 0.93 | 0.88 / 0.91 | 0.95 / 0.94 |
| 70-100 | 0.49 | 0.99 / 0.99 | 0.83 / 0.92 | 0.81 / 0.90 | 0.95 / 0.93 |

### original data, held-out validation episodes | 30 steps before leaving the road  (52 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.00 | 1.00 / - | 1.00 / - | 0.93 / - | 0.94 / - |
| 5-10 | 0.02 | 0.99 / 0.29 | 0.99 / 0.29 | 0.95 / 0.62 | 0.97 / 0.96 |
| 10-20 | 0.02 | 0.98 / 0.05 | 0.97 / 0.14 | 0.85 / 0.53 | 0.93 / 0.68 |
| 20-40 | 0.55 | 0.95 / 0.95 | 0.72 / 0.61 | 0.65 / 0.64 | 0.75 / 0.88 |
| 40-70 | 0.88 | 0.98 / 0.99 | 0.75 / 0.79 | 0.74 / 0.76 | 0.91 / 0.90 |
| 70-100 | 0.74 | 0.99 / 1.00 | 0.76 / 0.84 | 0.76 / 0.83 | 0.97 / 0.98 |

### test-track rollouts of dream-trained drivers | random start  (1500 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.12 | 0.99 / 0.95 | 0.99 / 0.94 | 0.96 / 0.86 | 0.97 / 0.89 |
| 5-10 | 0.12 | 0.99 / 0.95 | 0.99 / 0.92 | 0.96 / 0.86 | 0.97 / 0.90 |
| 10-20 | 0.13 | 0.99 / 0.95 | 0.97 / 0.88 | 0.95 / 0.85 | 0.97 / 0.89 |
| 20-40 | 0.14 | 0.99 / 0.94 | 0.87 / 0.83 | 0.83 / 0.80 | 0.96 / 0.88 |
| 40-70 | 0.15 | 0.99 / 0.94 | 0.50 / 0.80 | 0.49 / 0.77 | 0.96 / 0.88 |
| 70-100 | 0.16 | 0.99 / 0.94 | 0.29 / 0.80 | 0.31 / 0.77 | 0.96 / 0.87 |

### test-track rollouts of dream-trained drivers | 30 steps before leaving the road  (264 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.06 | 0.99 / 0.97 | 0.99 / 0.96 | 0.89 / 0.40 | 0.90 / 0.48 |
| 5-10 | 0.05 | 0.98 / 0.78 | 0.97 / 0.70 | 0.86 / 0.30 | 0.88 / 0.40 |
| 10-20 | 0.01 | 0.99 / 0.48 | 0.93 / 0.36 | 0.92 / 0.51 | 0.93 / 0.22 |
| 20-40 | 0.55 | 0.95 / 0.94 | 0.70 / 0.59 | 0.67 / 0.65 | 0.79 / 0.91 |
| 40-70 | 0.89 | 0.96 / 0.96 | 0.65 / 0.69 | 0.65 / 0.69 | 0.88 / 0.87 |
| 70-100 | 0.68 | 0.96 / 0.95 | 0.54 / 0.67 | 0.54 / 0.66 | 0.84 / 0.83 |
