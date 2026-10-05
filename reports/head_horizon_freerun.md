# Head accuracy against dream step

Model `runs/ft_free/last.pt`, real actions replayed, 16 dream samples per window, temperature 1.15. Made by `python diagnostics/head_horizon.py` (4.3 min). See the script's docstring for the question and how to read it.

### original data, held-out validation episodes | random start  (210 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.45 | 0.99 / 0.98 | 0.98 / 0.98 | 0.91 / 0.89 | 0.96 / 0.97 |
| 5-10 | 0.46 | 0.99 / 1.00 | 0.98 / 0.97 | 0.91 / 0.88 | 0.98 / 0.98 |
| 10-20 | 0.46 | 0.99 / 0.99 | 0.98 / 0.97 | 0.89 / 0.86 | 0.98 / 0.97 |
| 20-40 | 0.45 | 0.99 / 0.98 | 0.94 / 0.97 | 0.86 / 0.83 | 0.96 / 0.96 |
| 40-70 | 0.46 | 0.98 / 0.97 | 0.89 / 0.95 | 0.82 / 0.82 | 0.95 / 0.94 |
| 70-100 | 0.49 | 0.98 / 0.98 | 0.87 / 0.96 | 0.80 / 0.82 | 0.95 / 0.93 |

### original data, held-out validation episodes | 30 steps before leaving the road  (52 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.00 | 0.99 / - | 0.99 / - | 0.88 / - | 0.94 / - |
| 5-10 | 0.02 | 0.99 / 0.46 | 0.98 / 0.30 | 0.89 / 0.60 | 0.97 / 0.96 |
| 10-20 | 0.02 | 0.99 / 0.66 | 0.94 / 0.52 | 0.76 / 0.73 | 0.93 / 0.68 |
| 20-40 | 0.55 | 0.89 / 0.85 | 0.63 / 0.48 | 0.55 / 0.51 | 0.75 / 0.88 |
| 40-70 | 0.88 | 0.98 / 0.98 | 0.76 / 0.79 | 0.65 / 0.66 | 0.91 / 0.90 |
| 70-100 | 0.74 | 0.99 / 0.99 | 0.82 / 0.96 | 0.70 / 0.77 | 0.97 / 0.98 |

### test-track rollouts of dream-trained drivers | random start  (1800 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.12 | 0.99 / 0.91 | 0.98 / 0.89 | 0.93 / 0.74 | 0.97 / 0.90 |
| 5-10 | 0.12 | 0.99 / 0.91 | 0.98 / 0.85 | 0.92 / 0.70 | 0.97 / 0.90 |
| 10-20 | 0.13 | 0.99 / 0.92 | 0.96 / 0.78 | 0.89 / 0.67 | 0.97 / 0.90 |
| 20-40 | 0.13 | 0.99 / 0.90 | 0.93 / 0.78 | 0.81 / 0.66 | 0.96 / 0.89 |
| 40-70 | 0.14 | 0.99 / 0.92 | 0.76 / 0.84 | 0.63 / 0.67 | 0.96 / 0.90 |
| 70-100 | 0.15 | 0.99 / 0.91 | 0.43 / 0.88 | 0.47 / 0.69 | 0.96 / 0.89 |

### test-track rollouts of dream-trained drivers | 30 steps before leaving the road  (275 windows)

| Steps | Off-road share | teacher-forced head acc / off-road recall | free-running head acc / off-road recall | free-running latent probe acc / off-road recall | teacher-forced latent probe acc / off-road recall |
|---|---|---|---|---|---|
| 0-5 | 0.06 | 0.99 / 0.91 | 0.98 / 0.82 | 0.85 / 0.42 | 0.91 / 0.48 |
| 5-10 | 0.04 | 0.98 / 0.69 | 0.96 / 0.31 | 0.82 / 0.26 | 0.88 / 0.40 |
| 10-20 | 0.01 | 0.99 / 0.47 | 0.97 / 0.07 | 0.86 / 0.11 | 0.93 / 0.23 |
| 20-40 | 0.55 | 0.85 / 0.75 | 0.64 / 0.39 | 0.60 / 0.53 | 0.79 / 0.91 |
| 40-70 | 0.89 | 0.96 / 0.96 | 0.70 / 0.78 | 0.59 / 0.62 | 0.88 / 0.87 |
| 70-100 | 0.69 | 0.94 / 0.94 | 0.58 / 0.83 | 0.53 / 0.65 | 0.85 / 0.84 |
