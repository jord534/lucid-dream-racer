"""Propose-critic-collect and retrain-measure loop, built to run unattended on a cloud machine.

Each round r starts from a world model M_r (round 0: runs/mdnrnn):
  proposers  train a few dream controllers in M_r (they are the adversary: they search for the
             actions the dream overpays)
  measure    on fixed validation tracks: real return of the proposers, and the exploit gap
             (the dream's predicted return minus the real return, for the dream's own actions)
  critic     on fresh training tracks: find where the dream is most wrong, keep the real outcome
             of the highest-gap branches plus some random ones as new episodes
  finetune   M_{r+1} = M_r fine-tuned on the old data plus every branch collected so far
After the last round it only measures. Success looks like: exploit gap falling and real return
rising from one round to the next. Every stage is resumable (marker files), the cost is capped,
and results are written to <out>/results.md in plain language after each measure.

    python -m ldr.loop --rounds 2 --price-eur-hour 1.2 --budget-eur 5 --out runs/loop
    python -m ldr.loop --tiny --price-eur-hour 0 --out runs/loop_tiny      # local smoke test
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from . import branch
from .check_dream import load_mdnrnn
from .config import REPORTS, ROOT, SEED_TRAIN, SEED_VAL, C

BASE = ROOT / "runs"


def say(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def last_line(path: Path) -> str:
    """The latest progress line of a log (tqdm rewrites its line with carriage returns)."""
    tail = path.read_bytes()[-2000:].decode(errors="ignore")
    parts = [x.strip() for x in tail.replace("\r", "\n").split("\n") if x.strip()]
    return parts[-1] if parts else "(no output yet)"


def run(cmd, log: Path, env=None, heartbeat=60):
    """Run a sub-process, echoing its latest progress line into the main log every minute."""
    with log.open("w") as f:
        proc = subprocess.Popen([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT)
        while True:
            try:
                proc.wait(timeout=heartbeat)
                break
            except subprocess.TimeoutExpired:
                say(f"    {last_line(log)}")
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


CALIBRATION = REPORTS / "calibration" / "latest.json"
EVAL_TRACKS, REPLAY_STEPS = (
    8,
    300,
)  # train_dream's real-track check size; assumed mean replay length


def estimate_minutes(a, r):
    """Minutes per stage from measured machine rates (see ldr/calibrate.py)."""
    rate = r["sim_steps_per_s"]
    evals = a.generations // min(25, a.generations)

    def sim(tracks):
        return a.proposers * tracks * (1000 + a.futures * REPLAY_STEPS) / rate / 60

    return {
        "proposers": a.proposers
        * (a.generations * r["dream_s_per_gen"] + evals * EVAL_TRACKS * 1000 / rate)
        / 60,
        "measure": sim(a.measure_tracks),
        "critic": sim(a.tracks),
        "finetune": a.finetune_steps / r["finetune_steps_per_s"] / 60,
    }


# Rough minutes per stage at the default settings on a laptop like the one used so far (an estimate
# from earlier timings, replaced by the measured times as soon as a stage has finished).
PRIOR_MIN = {"proposers": 15, "measure": 10, "critic": 9, "finetune": 17}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--proposers", type=int, default=3)
    p.add_argument(
        "--generations", type=int, default=100, help="dream CMA-ES generations per proposer"
    )
    p.add_argument(
        "--tracks", type=int, default=40, help="training tracks per proposer for the critic"
    )
    p.add_argument("--futures", type=int, default=3, help="dream-steered branches per start state")
    p.add_argument("--keep-top", type=int, default=150)
    p.add_argument("--keep-random", type=int, default=30)
    p.add_argument("--finetune-steps", type=int, default=5000)
    p.add_argument("--finetune-lr", type=float, default=3e-4)
    p.add_argument("--extra-frac", type=float, default=0.5)
    p.add_argument("--measure-tracks", type=int, default=50)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="default: the value measured by ldr.calibrate, else 6",
    )
    p.add_argument("--device", default=None, help="torch device for training: cpu, mps, cuda")
    p.add_argument("--budget-eur", type=float, default=5.0)
    p.add_argument(
        "--price-eur-hour", type=float, required=True, help="hourly price of this machine"
    )
    p.add_argument("--out", type=Path, default=BASE / "loop")
    p.add_argument(
        "--sync-cmd", default=None, help="shell command run after every stage (e.g. rclone sync)"
    )
    p.add_argument(
        "--finish-cmd", default=None, help="shell command run at the end (e.g. delete the machine)"
    )
    p.add_argument("--tiny", action="store_true", help="smallest settings, to test the plumbing")
    a = p.parse_args()
    if a.tiny:
        a.rounds, a.proposers, a.generations, a.tracks, a.futures = 1, 1, 4, 4, 2
        a.keep_top, a.keep_random, a.finetune_steps, a.measure_tracks = 4, 2, 40, 4
    calib = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else None
    a.workers = a.workers or (calib["recommended"]["workers"] if calib else 6)
    dev_args = ["--device", a.device] if a.device else []
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = (
        json.loads(state_path.read_text()) if state_path.exists() else {"hours": 0.0, "ledger": []}
    )
    py = sys.executable
    rng = np.random.default_rng(0)

    def finish(reason):
        write_results()
        say(
            f"finished: {reason}; about {state['hours'] * a.price_eur_hour:.2f} euros of "
            f"{a.budget_eur:.2f} used"
        )
        if a.sync_cmd:
            subprocess.run(a.sync_cmd, shell=True, check=False)
        if a.finish_cmd:
            subprocess.run(a.finish_cmd, shell=True, check=False)
        sys.exit(0)

    def write_results():
        rows = "\n".join(
            f"| {e['round']} | {e['real_return_mean']:.0f} | {e['gap_mean']:.1f} | "
            f"{e['dream_return_window_mean']:.1f} | {e['real_return_window_mean']:.1f} | "
            f"{e['real_on_road_share_mean']:.2f} | {e['candidates']} |"
            for e in state["ledger"]
        )
        (out / "results.md").write_text(f"""# Propose-critic loop results

Settings: rounds {a.rounds}, proposers {a.proposers}, dream generations {a.generations},
critic tracks {a.tracks} per proposer, {a.futures} branches per start, keep {a.keep_top} top-gap
plus {a.keep_random} random per round, fine-tune {a.finetune_steps} steps. Reproduce with
`python -m ldr.loop` and the same options. Cost so far about
{state["hours"] * a.price_eur_hour:.2f} euros ({state["hours"]:.2f} hours at {a.price_eur_hour}
per hour).

Round 0 is the world model we started from; each later round is that model fine-tuned on the real
branches found where the dream was most wrong. Everything is measured on the same fixed
validation tracks (seeds {SEED_VAL + 100} and up), never on the test tracks.

| Round | Real return | Exploit gap | Dream ret. | Real ret. | On-road share | Branches |
|---|---|---|---|---|---|---|
{rows}

**Columns.** Real return: mean real return of the dream-trained controllers on the validation
tracks. Exploit gap: the dream's predicted return minus the real return for the dream's own
actions, over a window of up to 100 steps (large and positive means the dream is being exploited).
Dream return and Real return (window): the two halves of that gap. Real on-road share: how much of
the window the real car stays on the road under those actions. Branches: how many dream-steered
branches were measured.

**How to read it.** If the loop works, the gap falls and the real return rises from round to round.
One run, one seed per round: treat small
changes as noise.
""")

    def typical(name):
        done = state.get("stage_min", {}).get(name)
        if done:
            return float(np.mean(done))
        if a.tiny:
            return 0.3
        return estimate_minutes(a, calib["rates"])[name] if calib else PRIOR_MIN[name]

    def remaining_min():
        total = 0.0
        for k in range(a.rounds + 1):
            names = ["proposers", "measure"] + (["critic", "finetune"] if k < a.rounds else [])
            total += sum(
                typical(n) for n in names if not (out / f"r{k:02d}" / f"{n}.done").exists()
            )
        return total

    def stage(name, rd, fn):
        marker = rd / f"{name}.done"
        if marker.exists():
            say(f"{rd.name}/{name}: already done, skipping")
            return
        if state["hours"] * a.price_eur_hour >= 0.9 * a.budget_eur:
            finish("budget cap reached")
        say(f"{rd.name}/{name}: start   (about {remaining_min():.0f} min left in total, estimate)")
        t0 = time.time()
        fn()
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        state["hours"] += (time.time() - t0) / 3600
        state.setdefault("stage_min", {}).setdefault(name, []).append((time.time() - t0) / 60)
        state_path.write_text(json.dumps(state, indent=1))
        say(
            f"{rd.name}/{name}: done in {(time.time() - t0) / 60:.1f} min "
            f"(about {state['hours'] * a.price_eur_hour:.2f} euros so far)"
        )
        if a.sync_cmd:
            subprocess.run(a.sync_cmd, shell=True, check=False)

    def round_dir(r):
        rd = out / f"r{r:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        if not (rd / "vae").exists():
            os.symlink(BASE / "vae", rd / "vae")
        if r == 0 and not (rd / "mdnrnn").exists():
            os.symlink(BASE / "mdnrnn", rd / "mdnrnn")
        return rd

    def thetas_of(rd):
        return [
            torch.load(rd / f"proposer{s}" / "best.pt", weights_only=False)["theta"]
            for s in range(a.proposers)
        ]

    def pool_for(rd):
        return Pool(
            a.workers,
            initializer=branch.init_worker,
            initargs=(rd / "vae" / "best.pt", rd / "mdnrnn" / "best.pt"),
        )

    def proposers(rd, r):
        env = dict(os.environ, LDR_RUNS=str(rd))
        for s in range(a.proposers):
            if (rd / f"proposer{s}" / "best.pt").exists():
                continue
            say(f"  training proposer {s}")
            run(
                [
                    py,
                    "-m",
                    "ldr.train_dream",
                    "--out",
                    rd / f"proposer{s}",
                    "--seed",
                    10 * r + s,
                    "--generations",
                    a.generations,
                    "--real-every",
                    min(25, a.generations),
                    "--workers",
                    a.workers,
                    *dev_args,
                ],
                rd / f"proposer{s}.log",
                env,
            )

    def measure(rd, r):
        rnn = load_mdnrnn(rd / "mdnrnn" / "best.pt")
        seeds = SEED_VAL + 100 + np.arange(a.measure_tracks)
        with pool_for(rd) as pool:
            logs, plan = branch.evaluate_candidates(
                rnn, thetas_of(rd), seeds, pool, a.futures, C.tau, rng
            )
        ret = np.array([float(l["rew"].sum()) for l in logs]).reshape(a.proposers, -1)
        g = lambda k: np.array([c[k] for c in plan])
        e = {
            "round": r,
            "model": str(rd / "mdnrnn" / "best.pt"),
            "real_return_mean": float(ret.mean()),
            "real_return_by_proposer": ret.mean(1).tolist(),
            "gap_mean": float(g("gap").mean()),
            "gap_median": float(np.median(g("gap"))),
            "dream_return_window_mean": float(g("dream_ret").mean()),
            "real_return_window_mean": float(g("real_ret").mean()),
            "real_on_road_share_mean": float(g("real_on_share").mean()),
            "candidates": len(plan),
        }
        state["ledger"] = [x for x in state["ledger"] if x["round"] != r] + [e]
        state["ledger"].sort(key=lambda x: x["round"])
        (rd / "measure.json").write_text(json.dumps(e, indent=1))
        say(
            f"  round {r}: real return {e['real_return_mean']:.1f}, "
            f"exploit gap {e['gap_mean']:.1f}, "
            f"real on-road share {e['real_on_road_share_mean']:.2f}"
        )
        write_results()

    def critic(rd, r):
        rnn = load_mdnrnn(rd / "mdnrnn" / "best.pt")
        seeds = SEED_TRAIN + 1000 * (r + 1) + np.arange(a.tracks)
        with pool_for(rd) as pool:
            logs, plan = branch.evaluate_candidates(
                rnn, thetas_of(rd), seeds, pool, a.futures, C.tau, rng
            )
        pick = branch.select(plan, a.keep_top, a.keep_random, rng)
        episodes = [branch.build_episode(plan[i], logs[plan[i]["log"]]) for i in pick]
        branch.save_episodes(episodes, rd / "branches.pkl")
        gaps = np.array([c["gap"] for c in plan])
        (rd / "critic.json").write_text(
            json.dumps(
                {
                    "candidates": len(plan),
                    "kept": len(episodes),
                    "gap_mean_all": float(gaps.mean()),
                    "gap_mean_kept": float(np.mean([e["gap"] for e in episodes]))
                    if episodes
                    else None,
                },
                indent=1,
            )
        )
        say(f"  kept {len(episodes)} of {len(plan)} candidates (mean gap {gaps.mean():.1f})")

    def finetune(rd, r):
        eps = [
            e
            for k in range(r + 1)
            for e in branch.load_episodes(out / f"r{k:02d}" / "branches.pkl")
        ]
        extra = out / f"extra_r{r:02d}.npz"
        branch.assemble(eps, extra)
        nxt = round_dir(r + 1)
        run(
            [
                py,
                "-m",
                "ldr.train_mdnrnn",
                "--init-from",
                rd / "mdnrnn" / "best.pt",
                "--extra",
                extra,
                "--extra-frac",
                a.extra_frac,
                "--steps",
                a.finetune_steps,
                "--lr",
                a.finetune_lr,
                "--ss-prob",
                0.3,
                "--eval-every",
                max(1, a.finetune_steps // 10),
                "--out",
                nxt / "mdnrnn",
                "--seed",
                r + 1,
                *dev_args,
            ],
            nxt / "finetune.log",
        )
        say(f"  fine-tuned on {len(eps)} branch episodes")

    for r in range(a.rounds + 1):
        rd = round_dir(r)
        stage("proposers", rd, functools.partial(proposers, rd, r))
        stage("measure", rd, functools.partial(measure, rd, r))
        if r == a.rounds:
            break
        stage("critic", rd, functools.partial(critic, rd, r))
        stage("finetune", rd, functools.partial(finetune, rd, r))
    finish("completed")


if __name__ == "__main__":
    main()
