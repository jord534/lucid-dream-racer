# Road map around the car (technical note 5.1, candidate 2): thresholds fixed before any result

Written 2026-10-08, before any code for this experiment produced a result. Nothing below is changed
after results are seen. Background: reports/pose_state/summary.md (candidate 1 failed its gate; the
model could use a true pose but could not keep its own pose right).

The idea: keep the road in a map that does not move, and move the car across it. Road status is
then read off the map at the car's wheels, so only the car's motion has to be predicted.

## Data used for development

A new development set that matches the test set's kind of driving: the same six dream-trained
drivers whose test-track runs define the target measure (loop_head r02 and loop_road r02,
proposers 0-2), each driven on 50 validation tracks, seeds 900,300-900,349 (not used anywhere
else). Also reported: the 70 held-out validation episodes of the original data. The test-track
runs are not used in stages 1-2 except to label them (a replay of logged actions; nothing is
evaluated on them).

## Stage 1: exact geometry labels

Every recorded episode is replayed from its track seed with its recorded actions. Per step: the
car's position and heading; per episode: the road tiles.
Pass if all of:
1. every replay reproduces the recorded rewards (and logged on-road flags where stored);
2. road status computed from the actual wheel shapes and the tiles agrees with the game's on-road
   flag on >= 99.5% of steps of the original data;
3. road status computed from the car's position and heading only (wheels placed at their fixed
   offsets, front-wheel steering ignored), which is what stage 2 reads, agrees on >= 98%.
The game may report contact with a one-step delay; both alignments (wheels after action t, wheels
before action t) are measured, and the better-agreeing one is used from then on.

## Stage 2: does a map stay right? (no world model involved)

Windows as in diagnostics/head_horizon.py: started 30 steps before the car first leaves the road
(>= 20 consecutive off-road steps), and 3 random starts per run; up to 100 steps; 40 steps of real
history before the start.

The map holds only the road the camera showed during those 40 steps of history (the exact visible
rectangle of each frame); elsewhere it is unknown, and unknown counts as road (so it can only miss
departures, never invent them). The car is moved across the map by:
- true motion (the map's own limit),
- held motion (the last real step repeated: a naive baseline),
- guessed motion: a small recurrent network trained on the training episodes of the original
  data, given the actions, its own previous motion and which wheels the map shows on the road.
  It is started with the 40 real steps of history, then runs on its own guesses.
Road status at each step: some wheel on a road cell of the map.

Primary measure: share of real off-road steps recognised at steps 20-40, on the development set's
departure windows, with guessed motion.
Pass if both:
1. that share >= 0.80;
2. accuracy at steps 20-40 of the development set's random starts >= 0.90.
Otherwise stop before building anything into the world model, and report.

Reported but not gating: true and held motion; a map of the whole track (upper limit of the
visible-only map); the held-out original-data episodes; position and heading error against horizon;
coverage (share of steps where the wheels are on a known part of the map). Intervals: 95%
bootstrap over windows.
Reading rule, fixed now: if true motion fails, the visible map is too small and the approach fails
regardless of motion; if true motion passes and guessed motion fails, the result is a motion-accuracy
requirement for stage 3, not a pass.

## Stages 3-5 (added 2026-10-08, while stage 1 was running and before any stage-1 or stage-2 result)

All stages are launched together and run unattended; each runs only if the one before passed.

### Design fixed now

- Road patch: road status at 42 fixed points around the car (car frame: 7 across at -12..12 units,
  6 along at -4..16 units, all inside the camera's view) plus the four wheels: 46 values in {0, 1}.
- World model: the headline recipe (`--shared-mixture --ss-prob 0.3`, 20,000 steps, seeds 0 and 1)
  plus `--map-weight 1 --map-input`: the patch at the current pose is an extra LSTM input, and a
  motion head predicts the car's motion over the step (du, dv, da, car frame), trained with the exact
  motion. In training the patch is always the true one (no scheduled sampling of the patch).
- In imagination the car's pose starts at the true pose and then moves by the model's own predicted
  motion; the patch is read from the map at that pose. Map for the stage-4 measure: the road the
  camera showed during the 40-step history, unknown read as road (as in stage 2). Also reported: the
  whole track.
- In the simulator (stage 5) the patch is read exactly from the current track at the car's true pose.
  All patch points lie inside the current camera view, so this stands in for perfect road perception
  from the current frame; it uses no information the frame does not show.
- Controllers (stage 5) train on 200-step dreams, longer than the visible map lasts, so their dreams
  use the whole track of the recorded episode they start from. Stated as a limitation.
- Controls: the candidate-1 controls (runs/pose_state/ctrl_s0, ctrl_s1: same recipe and seeds, no
  map); their test-window results already exist and are reused, not re-measured.

### Stage 3: labels and plumbing

Pass if all: the patch's wheel values agree with the game's flag (one-step delay respected) on
>= 98% of steps of the original data; the full test suite passes; a 200-step training run of the map
model completes and its checkpoint loads with the map enabled.

### Stage 4: train and test (gate)

Measured with the same windows, probe and readout as reports/pose_state (diagnostics/map_horizon.py).
Primary M: off-road steps recognised at imagined steps 20-40, 275 test-track departure windows,
latent probe, own predictions (own latents, own motion, map seen during the history), tau 1.15.
Pass only if, as in candidate 1: (1) M >= 0.80 for both map seeds; (2a) mean map M minus mean
control M >= 0.10; (2b) the lower map seed above the higher control seed; (2c) for each seed, the
95% bootstrap interval of map M minus control M (windows resampled jointly) is above zero.
Reported, not gating: tau 0.05; whole-track map; true pose (oracle); real frames; road status read
off the map at the wheels; motion and pose error; the development and validation windows.

### Stage 5: controllers (only if stage 4 passes)

For each map model and, for comparison, each control: ldr/loop.py round 0 (three controllers, 100
generations each, measured on validation tracks 900,100-900,149), then each controller on the 100
test tracks once. Success for the map models if all: mean exploit gap <= 60 (original model: 90);
mean dream P(on road) at the steps where the same actions leave the real car off the road <= 0.50;
pooled test IQM 95% interval lower bound above 209.6 (original dream-trained drivers). Compared
with the controls, 710.3 (two repair rounds) and 924.8 (simulator-trained).
