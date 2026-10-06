"""Train an independent MDN-RNN ensemble with the existing training procedure.

The default is a two-member pilot. Expand to the requested five members with
``python -m ldr.train_ensemble --members 5``. Each member uses the same latents
file, architecture, objective, optimizer, and schedule, with a distinct seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .check_dream import load_mdnrnn
from .config import DATA, RUNS, C
from .ensemble import MDNRNNEnsemble


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_member(cmd, member_out):
    member_out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (member_out / "train_stdout.log").open("a") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    measured = time.perf_counter() - start
    log_path = member_out / "log.csv"
    if log_path.exists():
        with log_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[-1].get("wall_s"):
            measured = float(rows[-1]["wall_s"])
    return measured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--members", type=int, default=2, help="2 for pilot; 5 for the full ensemble")
    p.add_argument("--parallel-members", type=int, default=1,
                   help="number of independent member trainers to run concurrently")
    p.add_argument("--seed", type=int, default=72_000, help="base seed; member i uses seed+i")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=C.seq_len)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ss-prob", type=float, default=0.3,
                   help="match the current base model's scheduled-sampling setting")
    p.add_argument("--on-weight", type=float, default=0.0)
    p.add_argument("--extra", type=Path, default=None,
                   help="shared cached repair-loop branch dataset for a repaired-data ensemble")
    p.add_argument("--extra-frac", type=float, default=0.5)
    p.add_argument("--extra-onset-frac", type=float, default=0.0)
    arch = p.add_mutually_exclusive_group()
    arch.add_argument("--shared-mixture", dest="shared_mixture", action="store_const", const=True)
    arch.add_argument("--per-dimension-mixture", dest="shared_mixture", action="store_const", const=False)
    p.add_argument("--architecture-from", type=Path, default=RUNS / "mdnrnn" / "best.pt",
                   help="match the original model's mixture architecture (default: current base model)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=RUNS / "mdnrnn_ensemble")
    a = p.parse_args()
    if a.members < 2 or a.parallel_members < 1:
        p.error("--members must be at least two and --parallel-members at least one")
    if not (DATA / "latents.npz").exists():
        p.error(f"training data not found: {DATA / 'latents.npz'}")
    if a.extra and not a.extra.exists():
        p.error(f"repair dataset not found: {a.extra}")
    if a.shared_mixture is None and not a.architecture_from.exists():
        p.error(f"architecture checkpoint not found: {a.architecture_from}")
    if a.shared_mixture is None:
        a.shared_mixture = load_mdnrnn(a.architecture_from).shared

    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [a.seed + i for i in range(a.members)]
    cfg = {"members": a.members, "parallel_members": a.parallel_members,
           "seeds": seeds, "steps": a.steps, "batch": a.batch,
           "lr": a.lr, "seq_len": a.seq_len, "eval_every": a.eval_every,
           "ss_prob": a.ss_prob, "on_weight": a.on_weight,
           "shared_mixture": a.shared_mixture, "dataset": str(DATA / "latents.npz"),
           "dataset_sha256": _sha256(DATA / "latents.npz"),
           "extra_dataset": str(a.extra) if a.extra else None,
           "extra_dataset_sha256": _sha256(a.extra) if a.extra else None,
           "architecture_source": str(a.architecture_from), "loss": "ldr.train_mdnrnn.loss_fn",
           "extra_fraction": a.extra_frac if a.extra else 0.0,
           "extra_onset_fraction": a.extra_onset_frac if a.extra else 0.0,
           "diversity": ["independent random initialization", "independent RNG seeds",
                         "independent minibatch/window sampling and latent augmentation"],
           "trajectory_bootstrap": False}
    (a.out / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    tasks = []
    for i, seed in enumerate(seeds):
        member_out = a.out / f"member_{i:02d}"
        cmd = [sys.executable, "-m", "ldr.train_mdnrnn", "--steps", str(a.steps),
               "--batch", str(a.batch), "--lr", str(a.lr), "--seq-len", str(a.seq_len),
               "--eval-every", str(a.eval_every), "--seed", str(seed), "--out", str(member_out)]
        if a.ss_prob:
            cmd += ["--ss-prob", str(a.ss_prob)]
        if a.on_weight:
            cmd += ["--on-weight", str(a.on_weight)]
        if a.extra:
            cmd += ["--extra", str(a.extra), "--extra-frac", str(a.extra_frac),
                    "--extra-onset-frac", str(a.extra_onset_frac)]
        if a.shared_mixture:
            cmd.append("--shared-mixture")
        if a.device:
            cmd += ["--device", a.device]
        print(f"Queued ensemble member {i + 1}/{a.members}; seed {seed}", flush=True)
        tasks.append((cmd, member_out))

    run_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.parallel_members) as pool:
        elapsed = list(pool.map(lambda task: _train_member(*task), tasks))
    cfg["current_invocation_wall_seconds"] = time.perf_counter() - run_started

    cfg["member_training_seconds"] = elapsed
    cfg["total_training_seconds"] = sum(elapsed)
    (a.out / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    models = [load_mdnrnn(a.out / f"member_{i:02d}" / "best.pt") for i in range(a.members)]
    MDNRNNEnsemble(models, seeds).save(a.out, metadata=cfg)
    wall_minutes = cfg["current_invocation_wall_seconds"] / 60
    member_minutes = sum(elapsed) / 60
    print(f"Trained {a.members} members in {wall_minutes:.1f} minutes wall time "
          f"({member_minutes:.1f} cumulative member-minutes); checkpoints: {a.out}", flush=True)


if __name__ == "__main__":
    main()
