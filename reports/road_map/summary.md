# A road map around the car (technical note 5.1, candidate 2): result of stages 1-2

**Verdict: stopped at stage 2.** A map of the road the camera has shown, with the car moved across
it, does not hold the road for 20-40 steps for the drivers that matter. This holds even before any
world model is involved. Stages 3-5 (building the map into the world model, training, controllers)
were not run, as fixed beforehand in [preregistration.md](preregistration.md).

Two separate limits, each enough to fail. Both are measured on the 138 departure windows of the
development runs (the six dream-trained drivers on validation tracks), at steps 20-40:

1. **The map runs out.** These drivers are fast. Within 20-40 steps their wheels are off the
   ground the camera showed in the 40 steps of history on a third of steps (coverage 0.66). Where
   the map is blank it cannot tell that the car has left the road. With the car's **true** motion,
   the map recognises 0.48 [0.41, 0.55] of off-road steps. With the whole track instead, it
   recognises 1.00, so the readout itself is right.
2. **Guessed motion drifts.** A small network that guesses the car's motion from the actions,
   without the frames, puts the car 9.6 units off its true position by steps 20-40 (heading error
   32°), against a road half-width of 6.7. Even on a map of the whole track it recognises only 0.65
   [0.58, 0.71] of off-road steps.

Primary measure (guessed motion, map of what was seen): **0.27 [0.21, 0.33]**, threshold 0.80.
Accuracy on random starts: 0.91 [0.90, 0.93], threshold 0.90.

## Stage 1: exact geometry (passed)

`python -m ldr.label_geometry`. All 2,300 runs replayed exactly, and their recorded rewards and
on-road flags were reproduced:
- the 1,400 original episodes;
- the 600 test-track runs (labelled only, nothing evaluated on them);
- 300 new development runs: the six drivers on validation tracks 900,300-900,349.

The game reports road contact one step late: its flag after action t describes the wheels before
that action, which matches Box2D's contact timing. With that alignment, on the original data:

| Road status computed from | Agreement with the game | Threshold |
|---|---|---|
| The actual wheel shapes and the tiles | 0.9997 | 0.995 |
| The car's position and heading only | 0.9993 | 0.98 |

## Stage 2: does a map stay right? (failed)

`diagnostics/road_map_feasibility.py`. Windows as in `head_horizon.py`. The map holds the road the
camera showed (its exact visible rectangle) during the 40 real steps before the start; elsewhere it
is unknown, read as road. The car is moved by:
- true motion;
- held motion: the last real step repeated;
- guessed motion: a GRU trained on the original training episodes (6,000 steps) from the actions,
  its own previous motion and which wheels the map shows on the road.

Off-road steps recognised at steps 20-40 / coverage (share of steps whose wheels are on the seen part
of the map):

| Windows | True motion, seen map | True motion, whole track | Guessed, seen map | Guessed, whole track | Held, seen map |
|---|---|---|---|---|---|
| Development runs, departures (138) | 0.48 / 0.66 | 1.00 | **0.27** / 0.65 | 0.65 | 0.26 / 0.49 |
| Original data, departures (52) | 0.96 / 0.98 | 1.00 | 0.60 / 0.98 | 0.63 | 0.33 / 0.98 |

Accuracy at steps 20-40 on random starts, guessed motion and seen map: 0.91 on the development
runs (900 windows), 0.97 on the original data (210 windows).

Position error of guessed motion on the development departure windows: 3.4 units at steps 10-20 and
9.6 at steps 20-40. On the original data's departure windows it is 3.3 at steps 20-40.

The difference between the two sets is the driving. In the original data (collection policies) the
map is covered for 98% of steps and true motion keeps it right (0.96). The dream-trained drivers
outrun the camera's view, and their motion is harder to guess.

## What it implies

- A map built only from what the camera has shown cannot carry the road over the horizon where
  these drivers leave it. To hold it, the model would have to imagine new road beyond the view, the
  same thing the latent fails to do now. Alternatively it would need the track layout, which is
  privileged information.
- Dead reckoning from actions alone drifts by more than a road half-width within 20-40 steps. The
  world model's own motion head would also see the frames and might do better; that was not
  tested, because stage 2 is the cheaper test and failed on the map alone (true motion 0.48).
- Together with candidate 1 (reports/pose_state): on both counts, keeping track of where the car is,
  using only what the model can know, fails on the same horizon.

## What was prepared but not run

The code for stages 3-5 is in place behind flags that default off. It has been smoke-tested: a
200-step map model trains, loads, dreams on CPU and GPU, drives in the real game, and runs the
loop's dream branches and the stage-4 measurement. It was not run as an experiment:
- `ldr/road_map.py`;
- `--map-weight`/`--map-input` in `ldr/train_mdnrnn.py`;
- map support in `ldr/dream.py`, `ldr/agent.py` and `ldr/branch.py`;
- `ldr/label_patch.py`, `diagnostics/map_horizon.py` and `diagnostics/road_map_stage5.py`;
- `scripts/road_map_stages345.sh`.

The patch labels (`data/latents_patch.npy`, `data/latents_motion.npy`,
`reports/road_map/stage3_labels.json`; wheel agreement 0.9993) were written by that check, not
by a stage-3 run.

## Reproduction

```bash
nohup bash scripts/road_map_stages.sh 1 2 > runs/road_map/stages.log 2>&1 &   # ~33 min
python scripts/progress_watch.py runs/road_map/stages.log
```

Files: `stage1.json`, `stage2.json` (all readouts with 95% bootstrap intervals over windows),
`stage2_counts.npz` (per-window counts), `runs/road_map/motion_gru.pt`, `runs/road_map/stages.log`.
