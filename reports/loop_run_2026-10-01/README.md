# First run of the propose-critic loop (1 October 2026)

First full run of `ldr/loop.py`, on the laptop (Apple M1, 8 cores, 16 GB; `calibration.json`). It
started at 20:55 and finished at 23:05, about 2 h 10 min against a projection of 1.6 hours.

## What was run

`python -m ldr.loop --rounds 2 --workers 6 --device cpu --price-eur-hour 0 --out runs/loop`

Each round starts from a world model (round 0: the existing `runs/mdnrnn`). The loop
1. trains 3 dream controllers in it (100 generations each; they are the adversary, searching for
   whatever the dream overpays),
2. measures them on 50 fixed validation tracks (seeds 900,100 to 900,149; never the test tracks): their
   real return, and the exploit gap, which is the dream's predicted return minus the real return when the
   dream's own actions are run in the real simulator (3 branches per start state, up to 100 steps),
3. on 40 fresh training tracks per controller, finds where the dream is most wrong, and keeps the real
   outcome of the 150 highest-gap branches plus 30 random ones as new training episodes,
4. fine-tunes the world model (5000 steps, half of every batch from the branches collected so far).
The last round only does steps 1 and 2. Model, controllers and branches of each round are in
`runs/loop/rNN/` (not tracked by git).

## Result

| Round | World model | Real return of the 3 controllers (mean) | Per controller | Exploit gap (mean, median) |
|---|---|---|---|---|
| 0 | original | 270 | 251, 261, 299 | 90.4, 100.4 |
| 1 | fine-tuned once | 593 | 591, 493, 695 | 70.8, 73.2 |
| 2 | fine-tuned twice | 640 | 763, 600, 557 | 57.4, 59.1 |

Over the same 450 measured branches per round: the dream's predicted return fell 121.0 to 97.9 to 81.7;
the real return of those same actions barely moved (30.7, 27.1, 24.4); the share of the window the real
car stays on the road under the dream's actions rose 0.34 to 0.40 to 0.47. The branches kept for
training had a mean gap of 105.8 (round 0) and 107.7 (round 1), against 87.2 and 66.0 for all
candidates, so the selection did pick out the places where the dream is most wrong.

## What it means

- After one fine-tune on real branches, every one of the three dream-trained controllers beats every
  controller from the original model (lowest 493 against highest 299 on the same validation tracks).
  That is a clear improvement in real return, from about 270 to about 590. For reference, the
  simulator-trained controller scores about 900 on the test tracks and the earlier dream-trained one
  scored 252 there.
- The second fine-tune adds little to the real return that can be told apart from noise (640 against 593,
  with controllers ranging 493 to 763), but the exploit gap kept falling, from 90 to 71 to 57, a fall of
  about 37% in two rounds.
- The gap fell mainly because the dream became less generous (its predicted window return dropped about
  a third), not because the dream-chosen actions earn more in reality over the window. The windows start
  from each round's own controllers, so window numbers are not strictly comparable between rounds.
- The dream still overpays: a gap of 57 remains and the real car is on the road only 47% of the window.

## Caveats

One run, one seed per round. Three controllers per round, each retrained from scratch with a different
seed, so round-to-round differences include controller-seed variation (the spread above is the best
guide to it). Results are on validation tracks. The stronger tests used in the dream-failure analysis
(`diagnostics/dream_closed_loop.py`, `dream_actions_in_sim.py`, and the test-set evaluation of the final
controllers) have not been run on the new models. The projection of 1.6 hours was low by about 35%; the
measured stage times (`state.json`) are a better guide: about 10 min proposers, 13 min measure, 10 min
critic and 9 min fine-tune per round.

## Files

`results.md` (the loop's own table), `measure_round{0,1,2}.json` and `critic_round{0,1}.json` (details
per round), `state.json` (stage timings), `loop.log` (full log), `calibration.json` (the machine
measurements the run used). Reproduce: run the command above from the repository root, after
`python -m ldr.calibrate`.
