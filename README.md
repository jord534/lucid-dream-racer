# Lucid Dream Racer
World model (VAE + MDN-RNN) and linear controller for CarRacing-v3, with a playable dream.
[Live demo](https://huggingface.co/spaces/<user>/lucid-dream-racer) · GIF here

## Results (100 unseen tracks, IQM with 95% bootstrap CI)
| Method | Env steps | IQM | Mean |
| World model, controller trained on real track | ... | ... | ... |
| World model, controller trained in dream (tau=1.15) | ... | ... | ... |
| PPO (rl-zoo3 hyperparameters, 3 seeds) | 8M | ... | ... |

## What I found
3-5 sentences: transfer gap, temperature ablation, robustness degradation, one surprise.

## Reproduce
pip install -e ".[app,baseline,dev]" && bash scripts/run_all.sh

## How it works / Limitations / Referencess