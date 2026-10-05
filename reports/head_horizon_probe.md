# Head accuracy against dream step

Model `runs/ft_probe/last.pt`, real actions replayed, 16 dream samples per window, temperature 1.15. Made by `python diagnostics/head_horizon.py` (3.8 min). See the script's docstring for the question and how to read it.

### original data, held-out validation episodes | random start  (210 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.45 | 0.99 / 1.00 | 0.99 / 1.00 | 0.97 / 0.97 | 0.96 / 0.97 |
| 5-10 | 0.46 | 0.99 / 1.00 | 0.99 / 0.98 | 0.97 / 0.96 | 0.98 / 0.98 |
| 10-20 | 0.46 | 0.99 / 0.99 | 0.97 / 0.96 | 0.95 / 0.93 | 0.98 / 0.97 |
| 20-40 | 0.45 | 0.99 / 0.99 | 0.94 / 0.93 | 0.92 / 0.90 | 0.96 / 0.96 |
| 40-70 | 0.46 | 0.99 / 0.98 | 0.91 / 0.93 | 0.89 / 0.88 | 0.95 / 0.94 |
| 70-100 | 0.49 | 0.99 / 0.99 | 0.91 / 0.94 | 0.87 / 0.85 | 0.95 / 0.93 |

### original data, held-out validation episodes | 30 steps before leaving the road  (52 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.00 | 1.00 / - | 1.00 / - | 0.96 / - | 0.94 / - |
| 5-10 | 0.02 | 1.00 / 0.88 | 0.98 / 0.06 | 0.97 / 0.10 | 0.97 / 0.96 |
| 10-20 | 0.02 | 0.99 / 0.24 | 0.97 / 0.00 | 0.92 / 0.51 | 0.93 / 0.68 |
| 20-40 | 0.55 | 0.94 / 0.95 | 0.66 / 0.48 | 0.58 / 0.40 | 0.75 / 0.88 |
| 40-70 | 0.88 | 0.99 / 0.99 | 0.69 / 0.70 | 0.52 / 0.50 | 0.91 / 0.90 |
| 70-100 | 0.74 | 1.00 / 1.00 | 0.77 / 0.84 | 0.58 / 0.55 | 0.97 / 0.98 |

### test-track rollouts of dream-trained drivers | random start  (1800 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.12 | 0.99 / 0.94 | 0.99 / 0.94 | 0.98 / 0.89 | 0.97 / 0.90 |
| 5-10 | 0.12 | 0.99 / 0.93 | 0.99 / 0.90 | 0.98 / 0.86 | 0.97 / 0.90 |
| 10-20 | 0.13 | 0.99 / 0.94 | 0.97 / 0.84 | 0.96 / 0.81 | 0.97 / 0.90 |
| 20-40 | 0.13 | 0.99 / 0.93 | 0.95 / 0.74 | 0.92 / 0.73 | 0.96 / 0.89 |
| 40-70 | 0.14 | 0.99 / 0.93 | 0.70 / 0.70 | 0.77 / 0.61 | 0.96 / 0.90 |
| 70-100 | 0.15 | 0.99 / 0.93 | 0.28 / 0.73 | 0.63 / 0.53 | 0.96 / 0.89 |

### test-track rollouts of dream-trained drivers | 30 steps before leaving the road  (275 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.06 | 0.98 / 0.96 | 0.99 / 0.97 | 0.96 / 0.73 | 0.91 / 0.48 |
| 5-10 | 0.04 | 0.98 / 0.85 | 0.96 / 0.59 | 0.94 / 0.45 | 0.88 / 0.40 |
| 10-20 | 0.01 | 0.99 / 0.50 | 0.97 / 0.00 | 0.96 / 0.01 | 0.93 / 0.23 |
| 20-40 | 0.55 | 0.93 / 0.91 | 0.53 / 0.19 | 0.64 / 0.60 | 0.79 / 0.91 |
| 40-70 | 0.89 | 0.95 / 0.94 | 0.48 / 0.51 | 0.30 / 0.26 | 0.88 / 0.87 |
| 70-100 | 0.69 | 0.94 / 0.93 | 0.41 / 0.58 | 0.36 / 0.26 | 0.85 / 0.84 |
