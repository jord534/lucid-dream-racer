# Pose-state experiment: thresholds fixed before any result

Written 2026-10-07, after the pose labels were checked (label_sanity.json) and before any pose
model was trained or evaluated. Nothing below was changed after results were seen.

## Primary measure (M)

Share of real off-road steps recognised at imagined steps 20-40, on windows started 30 steps
before the real car first leaves the road (>= 20 consecutive off-road steps), with the real
actions replayed and the model fed its own predictions (own latent samples and, for pose models,
its own predicted pose). Road status is read from each imagined latent by the independent road
probe of diagnostics/head_horizon.py (trained on one fifth of the training episodes, never used
in training). 16 samples per window, temperature 1.15. This is the quantity reported as 0.63 in
technical note 3.8 (Figure 3.5b), on the same 275 windows of the test-track runs of the six
dream-trained drivers (loop_head r02 and loop_road r02). Those logged runs are evaluated once per
final world-model checkpoint; development and debugging use only the 52 windows of the 70
held-out validation episodes of the original data.

Intervals: 95% percentile bootstrap over windows (2,000 resamples), each window keeping its 16
samples together.

## Step 3, oracle check (one pose model, seed 0)

The pose model is fed its own latent samples but the TRUE pose at every step.
Pass: oracle M >= 0.70 on the validation windows at temperature 1.15.
Fail: stop, report, do not run the gate or the controller.

## Step 5, gate

Pass only if all hold, at temperature 1.15, on the test-track windows:
1. M >= 0.80 for every pose seed;
2. clearly above the no-pose control trained with the same recipe and seeds:
   (a) mean M over pose seeds minus mean M over control seeds >= 0.10,
   (b) the lowest pose seed is above the highest control seed, and
   (c) for each seed, the bootstrap 95% interval of (pose M minus control M) excludes zero
       (windows resampled jointly for the two models).
Otherwise: stop and write up; no controller is trained.

Reported but not gating: the same at temperature 0.05; accuracy at steps 20-40 and 70-100 on
random starts; validation-episode versions; fed real frames (and true pose); fed real frames with
the model's own pose (what an agent in the simulator would have); pose error (lateral offset,
heading) against horizon; off-road rate and recall implied by the predicted lateral offset
(|offset| > 1.34 half-widths, the best-accuracy threshold on the training labels, 99.4%).

## Step 6, only if the gate passes

Controllers: as ldr/loop.py round 0, three per model, 100 generations each.
Validation (seeds 900,100-900,149): success if the mean exploit gap <= 60 (baseline 90) and the
median closed-loop road disagreement <= 0.50 (baseline 0.83). Test (once per final controller):
pooled IQM with 95% bootstrap CI over 100 test tracks; success if the CI lower bound is above
209.6 (original dream); compared with 710.3 (two repair rounds) and 924.8 (simulator).

## Recipe

Headline recipe: `python -m ldr.train_mdnrnn --steps 20000 --shared-mixture --ss-prob 0.3`,
same data (data/latents.npz), training seeds 0 and 1.
Pose models add `--pose-weight 1 --pose-input` (pose head, true pose fed back as input with
scheduled sampling at the same probability and ramp as --ss-prob, drawn independently of the
latent's coin). Controls: the same command and seeds without the pose flags.
