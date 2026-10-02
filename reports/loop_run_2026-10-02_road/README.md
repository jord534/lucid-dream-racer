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
across rounds only roughly. The independent failure tests and the 100-test-track evaluation have not
been run on this run's drivers (`runs/loop_road/r02/proposer{0,1,2}`, model `runs/loop_road/r02/mdnrnn`).

## Files
`results.md`, `measure_round{0,1,2}.json`, `critic_round{0,1}.json`, `state.json`, `loop.log`. The models
and branches are in `runs/loop_road/` (not tracked by git). Reproduce: the command above, after the
first run's `runs/loop`.
