# What goes wrong when the dream-trained driver meets the real simulator?

Investigation log, written 1 October 2026. Everything here concerns one controller,
`runs/dream_v3/best.pt`, which was trained entirely inside the learned world model
(`runs/mdnrnn/best.pt`). On the 100 unseen test tracks (seeds 1,000,000 to 1,000,099) it scores
a mean of 251.7, against 908.4 for the controller trained in the simulator (`runs/controller_v2`).
The question was *where and how* it fails. All rollouts reproduce the stored evaluation
(`reports/eval_wm_dream_v3.json`) exactly, track by track (100 of 100 returns match).

## 1. Does the driver do worse on bendier tracks?  (`bendiness_corr.json`)

**Setup.** For each test track, three simple measures of bendiness (total heading change round
the lap, sharpest single turn, lap length in tiles) against that track's return, rank
correlation. Run `python diagnostics/bendiness_check.py` (about a minute).

**Result.** For the dream-trained driver there is no relationship: correlation -0.09 with total
turning (p = 0.36), -0.10 with sharpest turn (p = 0.32), -0.17 with length (p = 0.10). Its
returns are low everywhere: median 188, quarter-points 124 and 336, best 767. The first dream
model's driver (`wm_dream_v1`) and the simulator-trained driver both do show a negative link
(about -0.5 and -0.66 with total turning), but that is mostly because longer, windier tracks
cost everyone more time penalty.

**Meaning.** The dream-trained driver is not failing on the hard tracks and coping on the easy
ones. It is failing broadly. The bendiness measures are crude, so this does not rule out a local
effect at particular corners.

## 2. Where does it leave the road, and does it get back?  (`bend_check.json`, `bend_check.log`)

**Setup.** Drive the controller on all 100 test tracks (same code path as `ldr.evaluate`) and
log, per step, whether any wheel touches the road. An *excursion* is 20 or more consecutive steps
with no wheel on the road. *Recovered* means the car is later back on the road for 20 or more
steps and collects at least 5 more tiles. Bend sharpness is the track's heading change over the
next 10 tiles at the point where the excursion starts, expressed as a percentile of that same
track (0.5 = no sharper than a random point). The same was done for the simulator-trained driver
as a comparison. Run `python diagnostics/bend_check.py` (about 8 minutes, 6 workers).

**Result.**

| | Dream-trained | Simulator-trained |
|---|---|---|
| Tracks with an excursion | 100 of 100 | 11 of 100 |
| Median step of the first excursion | 194.5 | 394 |
| Excursions never recovered from | 41 | 2 |
| Median share of tiles collected | 29% | 100% |
| Mean bend percentile at the excursion | 0.43 | 0.60 |
| Excursions in the sharpest quarter of bends | 19 of 100 (25 expected, p = 0.20) | 2 of 11 |

**Meaning.** The dream-trained driver leaves the road early on every track, and about four times
in ten it does not come back. These departures do not start at unusually sharp bends. This
contradicts the idea that it only fails when the track gets bendy. It does not test whether the
car fails at the first corner of each track whatever its sharpness (absolute sharpness was not
saved).

## 3. A picture of one failure  (`dream_v3_side_filmstrip.png`)

Every 20th frame of `reports/dream_v3_side.gif` (frames are 2 environment steps apart), test
seed 1,000,000: dream on the left (temperature 1.15), real simulator on the right. Up to about
step 200 the two agree. Around step 240 the real car is off the road in a spin, its score goes
flat (about 230) and it stays there. In the dream the car stays on the road and the predicted
score keeps climbing. This is one track and one closed-loop run. It motivated section 4 but is
not evidence on its own.

## 4. Does the dream notice when the car leaves the road?  (`dream_fidelity.json`, logs)

**Setup.** Stage 1 (`python diagnostics/dream_fidelity.py --collect`, about 5 minutes): the same 100
rollouts, logging for every step the car's image encoding, action, reward, wheel-on-road flag,
and its sideways offset and heading error against the track centreline (kept in
`data/dream_failure/rollouts_dream_v3.npz`, 25 MB, not tracked by git). Stage 2
(`--analyse`, about 3 minutes): for each track, start the world model 30 steps before the first
excursion, give it 40 real steps of context, then feed it the *same real actions* for up to 100
steps and let it imagine the pictures. A small classifier reads "probability the car is on the
road" from an image encoding. It was trained on real frames labelled by wheel contact and is
scored on tracks it never saw (5-fold split by track); on real frames it is accurate (area under
the curve 0.996, 97% correct, catches 97.5% of off-road frames). The comparison was fixed before
looking at results: over the steps where the real car is off the road, the dream's probability
minus the probability from the real encoding, taken as a median over tracks. Positive would mean
the dream shows the car on the road more than it really is. Run at temperature 1.15 (what the
controller trained against, 32 imagined futures per track) and 0.05 (8 per track).

**Result (temperature 1.15; 0.05 is similar).**

- Primary: median difference +0.22 (95% interval 0.18 to 0.27); positive on 91 of 100 tracks;
  p < 0.001 (Wilcoxon). While the real car is off the road the dream gives a mean probability of
  0.32 that it is on the road, against 0.10 from the real images.
- But the dream is also less sure when the car is on the road: 0.69 against 0.89 from real images.
- Added after seeing the primary result (post hoc): the dream keeps about 42% of the real
  on-road versus off-road contrast (0.34 against 0.80), and separates the two states much worse
  (area under the curve 0.79 against 0.99).
- The dream shows the car leaving the road at some point in all 100 windows (98 at the low
  temperature). In this replay it pays slightly less reward than the simulator (median 31.1 against 40.0
  over the window), not more.

**Meaning.** When given the real actions, the dream does not hold the car on the road. It follows
the departure only partly: its signal about on-road versus off-road is blurred in *both* directions,
not tilted towards the road. "Forgiving" is therefore the wrong word; "loses track of the road" fits
better. The GIF's stable dream car is a closed-loop effect (the controller steers inside the
dream), which this replay does not test; sections 5 and 6 do.

**Caveats.** The classifier was trained on real image encodings; dream samples sit a little off
that distribution, so part of the blur may be the classifier transferring poorly rather than the
model not knowing. Telling these apart would need the imagined images decoded and inspected, which
has not been done. One controller, one world model, one set of 100 tracks.

## 5. When the controller steers inside the dream, does the dream still show the car on the road?  (`dream_closed_loop.json`, `closed_loop_frames.png`, `dream_closed_loop.log`)

**Setup.** Same 100 tracks and the same window as section 4, but the dream starts in *exactly* the
state the real controller was in (full real history, no noise; the controller's first action in the
dream matches the real recorded action to within 0.000001). It is started 30 steps before the real
car first leaves the road, while the car is still on it. From there, 16 imagined futures of up to
100 steps (temperature 1.15) in two versions: the dream is given the real recorded actions, or the
controller reads the dream's own images and memory and steers. The same on-road classifier reads
every imagined frame. The comparison was fixed before running: over the steps where the *real* car
is off the road, the dream's mean probability of "on road" minus the real frames' (median over
tracks). As a check on the classifier, every imagined latent was also passed through the VAE
(decode, then encode) before being read. Run `python diagnostics/dream_closed_loop.py` (about 13
minutes).

**Result.** Over the steps where the real car is off the road, the probability that the car is on
the road is 0.10 from the real frames, 0.32 from the dream given the real actions, and 0.87 from
the dream steered by the controller (0.90 after the VAE check). Controller-steered minus real:
median +0.83 (95% interval 0.78 to 0.85), positive on 100 of 100 tracks. The dream says "on road"
at 96% of those steps when the controller steers, 22% when it replays the real actions. The dream
pays the controller a median reward of 133 over the window, against 32 when replaying the real
actions and 40 in the real run. The controller drives very differently in the dream (average
throttle 0.64 against 0.22 in the real run over the same steps, and an average steering difference
of 0.62). In `closed_loop_frames.png` a straight stretch of road keeps reappearing right under the
imagined car, where the real frames show grass.

**Meaning.** The dream follows the real car off the road when given the real actions (blurred, but
mostly), and does not when the controller steers. That is the pattern in the GIF, now measured on
every track. It still left open whether the dream is wrong or the controller simply behaves
differently in the dream, which is what section 6 tests. The VAE check shows the result is not an
artefact of the classifier reading dream images badly.

## 6. Are the dream's own steering actions really safe?  (`dream_actions_in_sim.json`, `dream_actions_in_sim.log`)

**Setup.** Take the actions the controller chose inside the dream in section 5 (4 futures per
track, regenerated by the script) and run exactly those actions in the *real simulator* from the
same real state. That state is rebuilt by replaying the real action history from the track seed;
in all 400 replays the rebuilt rollout matched the logged one exactly (reward and wheel contact).
Real frames are encoded and read by the same classifier. The comparison was fixed before running:
over the steps where the real car is off the road under those actions, the dream's mean
probability of "on road" minus the real frames' (average over a track's futures, then median over
tracks). Run `python diagnostics/dream_actions_in_sim.py` (about 10 minutes, 4 workers).

**Result.** Under the dream's own actions the real car is on the road for only 35% of the window
and 11% of its last 20 steps. The dream says 0.89 and 0.90 for the same two spans. Dream minus
real, over the steps where the real car is off the road: median +0.82 (95% interval 0.80 to 0.83),
positive on 100 of 100 tracks (+0.84 after the VAE check).

**Meaning.** The actions the controller chose in the dream do not keep the car on the road in the
simulator, while the dream shows them working. So the dream is too forgiving about the controller's
own steering; the other explanation (the controller just chooses different actions that would have
worked in the simulator) is ruled out. Together with section 5, the controller has learned to drive
in a world that shows it staying on the road when the real one does not, which fits the transfer
failure. This is the model being exploited by the search, not just blurred.

**Caveats for sections 5 and 6.** One controller, one world model, one temperature, 100 tracks.
Section 6 applies the dream's actions open-loop in the simulator, which is not what the controller
would do on real frames (it would react), so it shows that the dream's prediction for those actions
is wrong, not what the controller would achieve. Section 7 tested the idea that the world model
rarely saw the car leave the road, and found it false. Starting the dream from a point already off
the road, to see whether the controller can recover there, was not done.

## 7. How much of the world model's training data shows the car off the road?  (`data_coverage.json`, `data_coverage.log`)

**Setup.** One candidate cause of sections 5 and 6 was that the training data rarely shows what
happens once the car leaves the road. This counts it. A random sample of 60 recorded training
episodes from each of the three recording policies (pursuit driver, random driver, on-policy
controller; 180 in all) was replayed in the real simulator. Replaying the recorded actions from the
track seed reproduced every episode exactly (180 of 180 recorded rewards matched), so the labels are
true. Counted: frames with no wheel on the road; frames where the car points more than 90 degrees
away from the track direction ("spun"); excursions (20 or more consecutive off-road steps); and
recoveries (an excursion followed by 20 or more consecutive on-road steps). Intervals are bootstrap
95% over episodes. The whole-dataset figures weight each policy by its number of frames. The same
counts were made for the dream-trained controller's own test rollouts for comparison. Run
`python diagnostics/data_coverage.py` (about 12 minutes, 6 workers).

**Result.**

| | Off the road | Spun | Episodes with an excursion | Recoveries per 1000 frames |
|---|---|---|---|---|
| Pursuit driver (500 episodes) | 52.5% (45 to 60) | 26.1% | 98% | 1.18 |
| Random driver (500 episodes) | 82.9% (77 to 88) | 43.5% | 98% | 0.75 |
| On-policy controller (400 episodes) | 1.6% (1.2 to 1.9) | 0.03% | 20% | 0.20 |
| **Whole dataset (estimate)** | **47.8%** | **24.3%** | | **0.75 (about 1,000 recoveries)** |
| Dream-trained controller, test rollouts | 61.6% (58 to 66) | 40.0% | 100% | 1.65 |

**Meaning.** The world model has seen a great deal of off-road driving: about half of all training
frames, a quarter of them spun, and about a thousand recoveries. The idea that the data is mostly
on-road, so the model never learned what leaving the road looks like, is false. The dream-trained
controller's own test rollouts (61.6% off the road) look like the pursuit and random driver data, not
unlike it. What is nearly absent is off-road driving *with controller-like actions*: the on-policy
controller data, the only recordings made by a controller, is 98.4% on the road. If the model has learned
that smooth, controller-style steering goes together with the road being under the car, that would
explain sections 5 and 6, and it matches the idea that the model has learned the action and the track
state as coupled. That is an inference; it was not tested.

**Caveats.** A sample of 60 episodes per policy, not the whole dataset. "Controller-like actions" were
not measured: it is assumed that the dream-trained controller steers in a style similar to the
real-trained controller whose data was recorded.

## 8. Do the actions alone reveal whether the car is on the road?  (`action_leak.json`, logs)

**Setup.** Tests the coupling idea from section 7: that a world model could read "road under the
car" off the actions instead of tracking the road itself. The same 180 training episodes as section
7 were replayed in the simulator (all 180 matched) and the true on-road label kept for every step.
For each step, a small network was given only the last 10 actions (steer, gas, brake, including the
action taken at that step) and asked whether the car was on the road. No images, no positions.
Scored on whole episodes it never saw (5-fold split by episode); chance is 0.5. Three versions: trained
and scored within each recording policy; trained on all three pooled; and a control trained on shuffled
labels. The pooled predictors were then applied to the dream-trained controller's actions on the
100 test tracks. Fixed before running: the within-policy scores, and the pooled predictors' reading on
the dream-trained controller's actions at steps where its car is really off the road. Run
`python diagnostics/action_leak_check.py --collect` (about 10 minutes) then `--analyse` (about 3).

**Result.**

| Predicting "on the road" from the last 10 actions alone | AUC (95% interval) |
|---|---|
| Pursuit driver, within policy | 0.93 (0.90 to 0.95) |
| On-policy controller, within policy | 0.92 (0.91 to 0.94) |
| Random driver, within policy | 0.73 (0.64 to 0.81) |
| All three pooled | 0.94 (0.92 to 0.95) |
| Control, shuffled labels | 0.49 |

On the dream-trained controller's own actions (its car is on the road 38% of the time), the pooled
predictors read the probability of "on road" as 0.66 at the steps where the car is really off the road
and 0.37 where it is on the road; they call 81% of the off-road steps "on road". Their AUC there is
0.27 (0.25 to 0.28), below chance: the reading is inverted.

**Meaning.** The actions alone say a lot about whether the car is on the road: 0.92 to 0.93 for drivers
that react to the track, and still 0.73 for the random driver. So the shortcut exists in the data, and
a world model could use it. On the dream-trained controller the shortcut misleads: when its car is off
the road it keeps applying the kind of driving actions the data associates with being on the road, so
actions-only reading says "road". That would produce the pattern in sections 5 and 6 (road drawn under
the car), but this test does not show the world model actually relies on it.

**Caveats.** The shuffled-label control rules out a pipeline fault. The random driver's 0.73 shows some
of the signal comes from how actions persist in time, not only from reacting to the road. Whether the
world model uses the shortcut would need a separate test, for example changing the actions it is given
and seeing whether the imagined road follows them.

## Files

| File | What it is |
|---|---|
| `bendiness_corr.json` | Section 1 numbers |
| `bend_check.json`, `bend_check.log` | Section 2 per-track results, summary and run log |
| `dream_v3_side_filmstrip.png` | Section 3 picture |
| `dream_fidelity.json`, `dream_fidelity_collect.log`, `dream_fidelity_analyse.log` | Section 4 summary, per-track rows, and logs |
| `dream_closed_loop.json`, `closed_loop_frames.png`, `dream_closed_loop.log` | Section 5 summary, per-track rows, picture, log |
| `dream_actions_in_sim.json`, `dream_actions_in_sim.log` | Section 6 summary, per-track rows, log |
| `data_coverage.json`, `data_coverage.log` | Section 7 summary and log |
| `action_leak.json`, `action_leak_collect.log`, `action_leak_analyse.log` | Section 8 summary and logs |
| `../../data/dream_failure/coverage_replays.npz` | True per-step on-road labels for the 180 replayed episodes (section 8) |
| `../../data/dream_failure/rollouts_dream_v3.npz` | Per-step real rollouts behind sections 4 to 6 |
| `../../diagnostics/bendiness_check.py`, `bend_check.py`, `dream_fidelity.py`, `dream_closed_loop.py`, `dream_actions_in_sim.py`, `data_coverage.py`, `action_leak_check.py` | The scripts |
