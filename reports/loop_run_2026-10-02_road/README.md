# Loop run aimed at road status (2 October 2026)

Run from the round-2 world model of `reports/loop_run_2026-10-01/`, to try to shrink the disagreement
between dream and simulator about whether the car is on the road. It ran 17:40 to 19:40 (2 hours, one
11-minute stall while the laptop was busy or asleep), on the laptop, 6 workers, CPU.

Command: `python -m ldr.loop --rounds 2 --workers 6 --device cpu --price-eur-hour 0 --select-by road
--onset-frac 0.5 --start-from runs/loop/r02 --prior-branches runs/loop/r00/branches.pkl
runs/loop/r01/branches.pkl --round-offset 2 --out runs/loop_road`

## What was different from the first loop run
1. The critic ranked branches by the road-status gap (the dream's P(on road), read by a probe trained on
   real latents, minus the real on-road share over the window), not by the reward gap. 150 top plus
   30 random kept per round, as before.
2. Half of the branch share of every fine-tune batch was centred on the moment the real car leaves the road
   (only the new branches carry the on-road flags this needs; the 1,400 original episodes and the earlier
   two rounds' branches do not). Earlier branches were still included in every fine-tune.
3. Round 0 here is the round-2 model, with its three drivers reused. Round numbers below are this run's
   (they continue the first run's round 2). Seeds continue the first run's sequence.

## Result (50 validation tracks, 3 drivers per round)

| Round | Real return | Reward gap | On-road share (real) | Road gap | Dream says on road while real car is off |
|---|---|---|---|---|---|
| 0 (= first run's round 2) | 640 | 56.4 | 0.49 | 0.27 | 0.74 |
| 1 | 699 | 51.6 | 0.56 | 0.23 | 0.78 |
| 2 | 688 | 48.2 | 0.56 | 0.19 | 0.71 |

The road probe is reliable here: held-out AUC 0.99 on 110,000 to 140,000 real frames each round. The
road-ranked selection did pick out the worst places (kept branches had a mean road gap of 0.50 and 0.49, all
candidates 0.26 and 0.24).

## What it means
- The average road gap fell from 0.27 to 0.19 (about 30%) and the reward gap from 56 to 48. Real return
  went from 640 to 699 to 688, which is within the spread between drivers (about 640 to 757 in round 2).
- The number the targeting was meant to move did not move: when the real car is off the road, the dream
  still shows it on the road about 70 to 78% of the time in every round (0.74, 0.78, 0.71). The fall in
  the average road gap comes mostly from the real car being on the road more often (0.49 to 0.56 of the
  window), where there is less to disagree about.
- So, with this data-only approach and this amount of it, the dream's blind spot is not closing.
- Compared with the first run (reward gap 90 to 71 to 57 over its rounds), the reward gap now falls more
  slowly (56 to 52 to 48): diminishing returns, or an effect of the road targeting; this run cannot tell
  which, because the plain reward-ranked loop was not continued for the same rounds.

## Caveats
One run; three drivers per round, retrained from scratch each round; validation tracks only; the road
probe is retrained each round on that round's own real rollouts, so the road numbers are comparable
across rounds only roughly. The independent checks on this run's drivers are in the next section.

## Independent checks on this run's round-2 drivers (added 2 October 2026)

100 test tracks and the two dream-failure tests, per driver, each in the round-2 model it was trained in
(`run_post_tests.sh`, `post_tests.log`, `post_tests/`, `../eval_loop_road_r02_p*.json`,
`../dream_failure_analysis/loop_road_r02_p*/`). Same procedure and same table as in
`../loop_run_2026-10-01/README.md`: median over tracks of the dream's P(on road) minus the real car's
over the steps where the real car is off the road (0 would be agreement).

| Driver | Test return | Tracks with an off-road window | Closed-loop test | Dream actions in the simulator | Extra forgiveness when the controller steers in the dream |
|---|---|---|---|---|---|
| p0 | 653.5 | 53 | 0.38 | 0.37 (51 tracks) | 0.28 |
| p1 | 594.6 | 89 | 0.42 | 0.52 (89 tracks) | 0.14 |
| p2 | 754.9 | 79 | 0.52 | 0.68 (78 tracks) | 0.30 |
| **Mean of the 3** | **667.7** | | **0.44** | **0.52** | 0.24 |
| first run's round-2 drivers, for comparison | 630.7 | | 0.51 | 0.47 | 0.07 |

What it means. The test return of these drivers (668) is close to the first run's round-2 drivers (631):
no difference that can be told apart from the spread between drivers (595 to 755). The road-status
disagreement is the same story: the dream still shows the car on the road more than the simulator does,
for every driver (all medians clearly above 0), and the averages (0.44 and 0.52) are within the spread of the
first run's drivers (0.32 to 0.61). So the independent tests agree with the loop's own measure: ranking
branches by road status and oversampling the moment the car leaves the road did not close the gap.
The extra forgiveness specific to the controller's own steering is somewhat larger than in the first
run's drivers (0.24 against 0.07); with three drivers each I would not read much into that.

Caveats. Three drivers per run, different tracks produce off-road windows for different drivers (53 to 89
tracks), and the on-road probe is trained per driver on its own real rollouts.

## Files
`results.md`, `measure_round{0,1,2}.json`, `critic_round{0,1}.json`, `state.json`, `loop.log`. The models
and branches are in `runs/loop_road/` (not tracked by git). Reproduce: the command above, after the
first run's `runs/loop`.
