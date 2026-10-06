# Ensemble uncertainty experiment

This diagnostic tests whether independent MDN-RNNs disagree on trajectories that a controller exploits. It is separate from the existing single-model training and rollout path.

## Train

```bash
python -m ldr.train_ensemble --members 2 --parallel-members 2 --out runs/mdnrnn_ensemble_pilot
python -m ldr.train_ensemble --members 5 --parallel-members 2 --out runs/mdnrnn_ensemble
```

`--parallel-members` runs independent members at the same time; choose a value that fits the
available CPU or GPU resources. It defaults to one.

Training reuses `ldr.train_mdnrnn` for every member. The members use the same encoded dataset (`data/latents.npz`), architecture inferred from `runs/mdnrnn/best.pt` by default, sequence sampler, loss, Adam optimizer, learning-rate schedule, batch size and step count. The default scheduled-sampling probability is 0.3, matching the current base run. Seed `s+i` controls independent random initialization, sequence-window draws, and posterior-latent augmentation for member `i`. The trainer records all seeds, settings, dataset paths, and elapsed training time. There is no trajectory bootstrap: the established sampler selects overlapping windows directly, so adding episode-level resampling would need a separate, explicit training-data change. An optional common repair dataset can be supplied using `--extra`, `--extra-frac`, and `--extra-onset-frac`; every member receives the same source and mixing settings.

Each member remains a normal MDN-RNN checkpoint. The ensemble manifest and convenience checkpoints are additional files; existing `load_mdnrnn`, `DreamSim`, and the controller pipeline are unchanged.

## Run the diagnostic

First create the cached real test-track rollouts if needed:

```bash
python diagnostics/dream_fidelity.py --collect
```

Then collect ensemble predictions and figures:

```bash
python diagnostics/ensemble_uncertainty.py --ensemble runs/mdnrnn_ensemble
```

The default driver and cached rollout source are the original model-trained controller and its real test rollouts. Four conditions (logged-action open loop, CMA-ES controller, constant input, and random input) use the existing window that begins 30 steps before a 20-step off-road run. In addition, the ordinary baseline samples held-out validation episodes uniformly, without selecting windows by their road outcomes. The top `ceil(25%)` controller trajectories by predicted return form the high-return category. Up to 25 onset windows are selected randomly before looking at model outputs; controller, constant-input, and random action sequences from each selected window are replayed from the same simulator start state. These replays provide real returns and road status for both the optimized policy and action controls. Increase `--simulator-replays` for a larger matched sample.

For each transition, all models receive the same current latent and action but retain separate recurrent states. The original single model (`--behavior-model`) generates the behavior actions and deterministic state trajectory; the ensemble models are only measured on those inputs. The per-transition representative prediction is each member's mixture expectation, `sum_k pi_k mu_k`. Epistemic disagreement is maximum pairwise Euclidean distance across those deterministic representatives. This excludes stochastic sampling from the disagreement metric. Mixture predictive scale is saved separately as an aleatoric descriptor. Distances use all latent dimensions; the technical note's inactive-dimension caveat applies, so disagreement should be interpreted alongside action controls and transfer outcomes rather than as a standalone OOD score.

A simulator failure is an excursion of 20 or more consecutive off-road transitions, matching the repository's road-excursion definition. Brief single-frame wheel contacts do not count as failures.

## Outputs

The output folder contains `results.json` (summary and per-trajectory data) and `per_step.csv.gz` (per-member, per-step predictions and outcomes); `disagreement_by_timestep.png`; `disagreement_vs_overoptimism.png`; `predicted_vs_real_return.png`; and a generated `summary.md` with quantiles, timestep bands (0–20, 20–40, 40–70, 70–100), episode-bootstrap intervals, rank correlations, and inference timing. The plots show interquartile bands across trajectories. The matched correlation uses a bootstrap over trajectories. No scientific conclusion is encoded in advance; review the generated sample sizes and intervals before interpreting the hypothesis.

For a repaired-model comparison, train a second independent ensemble on the exact common repair mixture, using the matching `--extra` data and the same architecture and training settings. Then run this diagnostic with that ensemble and the repaired model, controller, and stored rollout file via `--ensemble`, `--behavior-model`, `--controller`, and `--store`. The repair loop itself is not changed.
