"""Single source of truth for paths and default hyperparameters."""
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" # rollouts, packed frames, latents
RUNS = ROOT / "runs" # checkpoints, logs, figures
REPORTS = ROOT / "reports" # final plots and tables
# Disjoint seed ranges so no track is ever reused across stages.
SEED_DATA = 0 # data collection tracks: 0 .. n_episodes-1
SEED_TRAIN = 100_000 # controller training tracks
SEED_VAL = 900_000 # periodic validation tracks
SEED_TEST = 1_000_000 # final 100-track evaluation (touch once)
@dataclass(frozen=True)
class Cfg:
    env_id: str = "CarRacing-v3"
    img: int = 64 # VAE input resolution (Ha & Schmidhuber)
    max_steps: int = 1000 # CarRacing-v3 time limit
    z_dim: int = 32 # VAE latent size
    h_dim: int = 256 # LSTM hidden size
    a_dim: int = 3 # steer, gas, brake
    n_gauss: int = 5 # MDN mixture components per latent dim
    kl_free_nats: float = 0.5 # per latent dim (KL tolerance)
    seq_len: int = 64 # MDN-RNN training window
    tau: float = 1.15 # dream sampling temperature

C = Cfg()