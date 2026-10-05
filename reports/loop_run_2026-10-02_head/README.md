# On-road head, road penalty, control, and free-running training (2 to 5 October 2026)

All from the road-status run's round-2 model (`runs/loop_road/r02`), its three drivers reused as round 0.

## 1. Head run vs control (50 validation tracks, 3 drivers per round)

Head arm (`results_head.md`, `loop_head.log`): the fine-tune also trains an on-road output on exact labels
(weight 1; labels for the 1.36M original steps by replay, `ldr/label_road.py`, 45.1% off the road, every
replay matched the recorded rewards, and for the branches' own flags); drivers of rounds 1 and 2 are trained
with a penalty of 1 reward per dream step times the head's P(off road) (neither value tuned). Control arm
(`results_ctrl.md`, `loop_ctrl.log`): identical (same model, drivers, seeds, round-0 branches; road-ranked
selection, onset oversampling) with no head and no penalty. Round 0 is shared.

| Round | Arm | Real return | Reward gap | Real on-road share | Road gap | Dream says on road while real car is off |
|---|---|---|---|---|---|---|
| 0 | both | 688 | 48.2 | 0.56 | 0.19 | 0.71 |
| 1 | head | 231 (one driver -90) | 40.8 | 0.84 | 0.14 | 0.98 (175 of 450 branches have off-road steps) |
| 1 | control | 564 | 50.5 | 0.50 | 0.28 | 0.75 (399 branches) |
| 2 | head | 600 | 48.6 | 0.75 | 0.19 | 0.96 (251 branches) |
| 2 | control | 754 | 52.9 | 0.57 | 0.27 | 0.76 (402 branches) |

Head accuracy: 0.99 on real frames (off-road recall 0.97), but 0.83 (round 1) and 0.73 (round 2) on the
dream's own branches, and the head says "on road" 97% of the time while the real car is off the road.

What it means.
- The control did not keep improving: from round 0 to 2 its reward gap rose (48 to 53) and its road gap rose
  (0.19 to 0.27). More rounds of road-ranked data alone do not help beyond round 0 here.
- The head arm kept the average road gap lower (0.14 and 0.19 against 0.28 and 0.27) and its drivers kept the
  real car on the road far more (0.84 and 0.75 against 0.50 and 0.57), but at the cost of real return in round 1
  (231; one driver sat still, -90: with the penalty doing nothing is safe in the dream) and a lower return in
  round 2 (600 against 754, within the spread between drivers).
- The last column is not comparable between arms: it is measured only over branches where the real car goes
  off the road, and there are fewer (175 and 251 against about 400), the ones the dream gets most wrong.
- The head is not a reliable guard in the dream: it reads the dream's drifted state (section 3).

Independent checks on the head arm's round-2 drivers (`post_tests_head.log`, `post_tests_loop_head/`,
`../eval_loop_head_r02_p*.json`, `../dream_failure_analysis/loop_head_r02_p*/`): test return 518, 774, 475
(mean 589); tracks with an off-road window only 24, 19, 11 (the drivers stay on the road); closed-loop road
disagreement 0.60, 0.76, 0.53; dream actions in the simulator 0.34 (14 tracks), 0.60 (10), 0.32 (6). Not
better than the earlier runs and the medians rest on few tracks. The same checks have NOT been run on the
control's drivers (`runs/loop_ctrl/r02`, `run_post_tests.sh runs/loop_ctrl loop_ctrl`, about 45 minutes).

## 2. Where the dream loses the road: `../head_horizon.md` (and `../head_horizon.png`)

Real actions replayed in the head arm's round-2 model from real states, head and a latent probe read at
every step, teacher-forced against free-running (`diagnostics/head_horizon.py`, held-out original episodes
and test-track driver rollouts). The head is 0.99 accurate teacher-forced at every step. Free-running, it is
0.99 for the first 10 steps and then decays: on held-out original episodes 0.83 at steps 70 to 100; on driver
rollouts 0.29 (mostly false "off road"). Head and probe fall together: the dream's own state drifts, within
roughly 20 to 40 steps. At the moment the car leaves the road (steps 20 to 40 of the windows that start 30 steps
before it) the free-running head catches 59 to 61% of departures against 94 to 95% teacher-forced.

## 3. Training on the model's own output: `runs/ft_free`, `../head_horizon_freerun.md`

From that model: 4000 steps, windows of 120 steps with the first 20 recorded, then the model's own samples
(scheduled sampling ramped to 1.0 over the first half), on-road label at every step (`ft_free.log`, `ft_free_log.csv`). Command: `python -m ldr.train_mdnrnn --init-from runs/loop_head/r02/mdnrnn/best.pt
--extra runs/loop_head/extra_r01.npz --extra-frac 0.5 --extra-onset-frac 0.5 --on-weight 1.0 --seq-len 120
--free-warm 20 --ss-prob 1.0 --steps 4000 --batch 32 --lr 3e-4 --eval-every 400 --out runs/ft_free --seed 7`.
Checkpoint tested:
`last.pt` (the selected `best.pt` is chosen by teacher-forced loss and would be an early one).
Target set beforehand: free-running accuracy of at least 0.90 at steps 70 to 100 on driver rollouts.

| | Before | After |
|---|---|---|
| Driver rollouts, random start, steps 70 to 100: head accuracy | 0.29 | **0.43** (target 0.90: missed) |
| same, steps 40 to 70 | 0.50 | 0.76 |
| Held-out original episodes, steps 70 to 100 | 0.83 | 0.87 |
| Driver rollouts, 30 steps before leaving the road, steps 20 to 40: recall of off-road steps, free-running | 0.59 | **0.39** |
| same, teacher-forced | 0.94 | 0.75 |
| Latent probe on free-running latents, steps 0 to 10, held-out original | 0.96 | 0.91 |
| Teacher-forced validation NLL | about 0.98 | 1.07 |

What it means. Training on its own output reduced the long-horizon drift (fewer false "off road" calls late in
the window) but did not reach the target, and made the case that matters worse: at the moment the car leaves
the road the dream now catches fewer departures (59% to 39%), and even teacher-forced detection of departures
fell. The frames also got worse (validation NLL up, probe accuracy down). Scheduled sampling to 1.0 traded
sharpness and detection of departures for stability. This is one model, one training run, one setting.

## 4. A task loss through a frozen road probe: `runs/ft_probe`, `../head_horizon_probe.md` (added 5 October 2026)

Idea: the latent the dream samples must read, through a probe trained on real latents, as the true road
status. `ldr/road_probe.py` trains the probe (held-out AUC 0.999 on original episodes, 0.997 on branches;
`road_probe.json`; it is trained on episodes disjoint from those of the separate probe the test uses).
`ldr/train_mdnrnn.py --probe` adds, to the usual teacher-forced pass, the cross-entropy of the model's
predicted P(on road) (the probe averaged over the predicted mixture) against the exact flag, and, with
`--free-task-weight`, a second pass of 100 free-running steps (20 recorded, then own samples) scored only on
road status (head and probe) and reward, with no latent likelihood. Command: `python -m ldr.train_mdnrnn
--init-from runs/loop_head/r02/mdnrnn/best.pt --extra runs/loop_head/extra_r01.npz --extra-frac 0.5
--extra-onset-frac 0.5 --on-weight 1.0 --probe runs/road_probe/probe.pt --probe-weight 1.0 --free-task-weight 1.0
--free-batch 16 --free-len 120 --free-warm 20 --steps 4000 --lr 3e-4 --eval-every 400 --out runs/ft_probe
--seed 11` (`ft_probe.log`, `ft_probe_log.csv`). Teacher-forced validation loss 1.025 (about 0.98 before,
1.068 after the previous attempt). Checkpoint tested: `last.pt`.
Targets set beforehand: free-running accuracy of at least 0.90 at steps 70 to 100 on driver rollouts (random
starts), and at least 85% of departures caught at steps 20 to 40 of the windows that start 30 steps before the
car leaves the road. Both missed.

| Measure (free-running, replayed real actions) | Original model | After free-running training | After probe task loss |
|---|---|---|---|
| Driver rollouts, random start, steps 70 to 100: head accuracy | 0.29 | 0.43 | 0.28 |
| same, test probe (independent of the training probe) | 0.31 | 0.47 | 0.63 |
| Held-out original episodes, steps 70 to 100: head accuracy | 0.83 | 0.87 | 0.91 |
| Departures caught, steps 20 to 40, driver rollouts: head / test probe | 0.59 / 0.65 | 0.39 / 0.53 | 0.19 / 0.60 |
| Departures caught, steps 20 to 40, original episodes: head / test probe | 0.61 / 0.64 | 0.48 / 0.51 | 0.48 / 0.40 |

What it means. The long-horizon reading of the dream's latent improved in some regimes (test probe on driver
rollouts 0.31 to 0.63; held-out original episodes head 0.83 to 0.91), so the model did learn something that
carries to a probe it was not trained against. The case that matters is unchanged or worse: at the moment the
car leaves the road the dream still catches about 60% (test probe) and the head catches fewer (19 to 48%). Two
different ways of making the loss care about road status (free-running training; this one) have now failed to
fix this. The information needed (where the car is relative to the road edge, tens of steps from a real start)
does not appear to be tracked by the dream's state, and a loss on its outputs does not create it. One model,
one training run per attempt.

Note (5 October 2026): `../head_horizon.*` (the original model's curves) was re-run with the rollouts of all six
drivers (`loop_head` and `loop_road`, round 2), as the two later runs used; the first run had four. With six,
the test probe gives 0.35 (not 0.31) at steps 70 to 100 of random starts and 0.63 (not 0.65) of off-road steps
recognised at steps 20 to 40. The technical note uses the six-driver numbers throughout.
