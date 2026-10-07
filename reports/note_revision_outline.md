# Outline for revising the technical note (LDR-TN-01)

Status: archived plan for issue 2, completed in `reports/technical_note.md` (tag `v1.1-tn01`). The
on-road-head run `runs/loop_head` and control `runs/loop_ctrl` are complete; the former PENDING row
was resolved in section 3.9. Issue 3 adds the ensemble pilot in section 3.11. Every number below
comes from a file named in the Evidence column; nothing is estimated.

## What the note can now claim, and on what evidence

| # | Claim | Evidence | How firm |
|---|---|---|---|
| 1 | The simulator-trained controller reproduces the paper: 908.4 (sd 68.0) vs 906 (sd 21), 100 test tracks | `reports/eval_wm_real_v2.json`; note section 3.1 | measured, one controller |
| 2 | The dream-trained controller scores 251.7 (sd 184.8) on the same tracks | `reports/eval_wm_dream_v3.json` | measured, one controller |
| 3 | For the original dream-trained controller the dream shows the car on the road (89% mean P) while the real car, run with the dream's own actions, is on the road 35% of the window | `reports/dream_failure_analysis/dream_actions_in_sim.json` | measured, 100 tracks, median gap 0.82 [0.80, 0.83] |
| 4 | The dream is accurate for ordinary actions and too forgiving of the controller's own steering | `dream_fidelity.json`, `dream_closed_loop.json` (closed minus open 0.56) | measured on that controller only |
| 5 | The training data is mostly off-road (45.1% of the 1.36M steps by exact replay; 48% in the earlier sample) against 1.6% in controller-style data | `data_coverage.json`, `data/label_road.log` (not tracked) | measured |
| 6 | Actions alone predict whether the car is on the road | `action_leak.json` | measured; **not shown that the model uses this** |
| 7 | A propose-critic loop (real replays of dream-steered actions, keep where the dream is most wrong, fine-tune) raised the real return of dream-trained drivers from about 270 to about 640 on validation tracks in two rounds, and from 252 to 631 (drivers 493 to 784) on the test tracks | `reports/loop_run_2026-10-01/` | one run, three drivers per round |
| 8 | Over four rounds the reward gap (dream's predicted minus real return for the dream's own actions) fell from 90 to 48; the fall comes from the dream paying less, while the real return of those actions stayed about flat | `loop_run_2026-10-01/`, `loop_run_2026-10-02_road/` | one run each |
| 9 | The road-status disagreement fell (0.83 to about 0.5 in both failure tests) but remains clearly above zero for every driver of both runs | `loop_run_2026-10-01/README.md`, `loop_run_2026-10-02_road/README.md` | 6 drivers, all medians > 0 |
| 10 | Ranking branches by road status and oversampling the moment the car leaves the road did not change this beyond the spread between drivers (test 668 vs 631; closed-loop 0.44 vs 0.51; in-simulator 0.52 vs 0.47; validation "dream says on road while real is off" 0.74, 0.78, 0.71) | `loop_run_2026-10-02_road/README.md`, `run_comparison.md` | one run vs one run; no control for extra rounds |
| 11 | PENDING: an on-road output trained on exact labels, plus a penalty in the dream, against a control without them | `runs/loop_head`, `runs/loop_ctrl` | to be decided by those results |

## What the note must not claim

- That the loop "fixes" or "closes" the gap: the reward gap is still 48 and the road disagreement is still
  0.3 to 0.7. The honest statement is a large improvement in real return and a partial reduction of the gap.
- That the dream-trained drivers approach the simulator-trained one (about 650 against 908).
- Why the dream is forgiving. Only that it is, where, and that the shortcut hypothesis (actions predict road
  status) is untested as a mechanism.
- That any difference between the first and the road-ranked run is real: one run each, 3 drivers each, the
  spread between drivers (about 300 return points) is larger than the differences.
- That road targeting helped, or hurt (claim 10), until the control in claim 11 exists.
- Statistical significance for the loop results: none was computed and the runs are not replicated.

## Sections that change

- **Title.** The current "Why ... fail: a diagnosis" no longer fits. Candidates (to decide once the head run
  is in): (a) "A CarRacing-v3 world model overpays the controller that steers inside it: diagnosis and a
  real-data repair loop"; (b) "Model exploitation in a CarRacing-v3 world model: diagnosis, and a loop that
  lifts dream-trained driving from 252 to about 630". (b) carries the value but must carry the final numbers.
- **Title block.** New date, new tag (a `v1.1-tn01` style tag after the final commit), figures source list
  adds the loop runs.
- **Summary.** Rewrite around: the failure (251.7), the mechanism found (dream too forgiving of its own
  controller's steering, data 45% off-road), the repair loop and its result (claim 7), and what remains
  (claim 9). It must stand alone (IET guide).
- **1.1 Scope.** Add the repair loop and the on-road experiments.
- **3.3 Diagnosis, 3.5 Residual limitation, Appendix A.3.** Section 3.5 and A.3 conclude that the model
  handles steering consequences; the failure tests contradict that for the controller's own steering. They must
  be rewritten, not appended to. Table 3.2 and the Summary conclusion depend on the same statement.
- **New 3.7 The repair loop** (design in one figure, the two runs, claims 7, 8) and **3.8 Road status**
  (claims 9, 10, 11).
- **4 Conclusions, 5 Recommendations.** Conclusions follow only the firm rows; recommendations: a
  structural change (the on-road output, or track memory decoupled from actions) if claim 11 is negative;
  replicate the loop over seeds either way.
- **Appendix A** gets the failure-test procedures (closed-loop, actions in the simulator, data coverage,
  action leak) and the loop; **B** the compute cost (first loop 2 h 10, road-status run 2 h, labelling 84 min,
  tests about 1.5 h per three drivers); **C** the commands.
- **Figures.** `reports/run_comparison.png` (made by `diagnostics/compare_runs.py`) is the candidate for the
  main results figure; `diagnostics/make_figures.py` needs the loop curves added.

## Before editing

1. Wait for the head run and its control, then fill row 11 and regenerate `run_comparison.md`/`.png`.
2. Decide whether one more repetition of the first loop with another seed is needed before the 631 is
   quoted as typical; today it is one run.
3. Decide the title, then the Summary, then the rest, then rebuild the PDF and tag.
