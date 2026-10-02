"""Propose, critic, collect: where does the dream disagree with the simulator?

For a controller and a track: drive the real simulator and log it; start the dream in the exact
real state 30 steps before the car first leaves the road (or at a random step if it never does);
let the controller steer inside the dream; run the same actions in the real simulator. The
exploit gap is the dream's predicted return minus the real return over that window. The real
outcomes of the chosen candidates become new training episodes for the world model.
(Used by ldr/loop.py; the same measurements as diagnostics/dream_closed_loop.py and
dream_actions_in_sim.py.)"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .agent import WorldModelAgent
from .config import C
from .controller import act_batched
from .envs import make_env
from .mdnrnn import mdn_sample
from .utils import preprocess, to_tensor

RUN, LEAD, PREFIX, WINDOW, MIN_WINDOW = 20, 30, 60, 100, 20
_AG: WorldModelAgent | None = None


def init_worker(vae_path, rnn_path):
    global _AG
    _AG = WorldModelAgent(vae_path, rnn_path)


def _wheels_on(env) -> bool:
    return any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels)


def _first_run(flags, start=0):
    n = 0
    for i in range(start, len(flags)):
        n = n + 1 if flags[i] else 0
        if n == RUN:
            return i - RUN + 1
    return None


def _encode(obs):
    with torch.no_grad():
        m, lv = _AG.vae.encode(to_tensor(preprocess(obs, C.img)[None], _AG.dev))
    return m[0].numpy(), lv[0].numpy()


def rollout_log(job):
    """Drive the controller for up to 1000 steps (no early stop) and log everything."""
    theta, seed, *rest = job
    env = make_env()
    obs, _ = env.reset(seed=seed)
    _AG.reset()
    rng = np.random.default_rng(seed)
    mu, lv, act, rew, on = [], [], [], [], []
    for _ in range(rest[0] if rest else 1000):
        m, l = _encode(obs)
        mu.append(m), lv.append(l)
        a = _AG.act(theta, obs, rng)
        obs, r, term, trunc, _ = env.step(a)
        act.append(a), rew.append(r), on.append(_wheels_on(env))
        if term or trunc:
            break
    env.close()
    f32 = lambda x: np.asarray(x, np.float32)
    return {
        "seed": seed,
        "mu": f32(mu),
        "logvar": f32(lv),
        "act": f32(act),
        "rew": f32(rew),
        "on": np.array(on, bool),
    }


def real_branch(job):
    """Re-create the real state by replaying the real history, then run the proposed actions."""
    seed, prefix, acts = job
    env = make_env()
    env.reset(seed=seed)
    for a in prefix:
        env.step(a)
    mu, lv, rew, on, term_flag = [], [], [], [], False
    for a in acts:
        obs, r, term, trunc, _ = env.step(a)
        rew.append(r), on.append(_wheels_on(env))
        m, l = _encode(obs)
        mu.append(m), lv.append(l)
        if term or trunc:
            term_flag = bool(term)
            break
    env.close()
    f32 = lambda x: np.asarray(x, np.float32)
    return {
        "mu": f32(mu),
        "logvar": f32(lv),
        "rew": f32(rew),
        "on": np.array(on, bool),
        "terminated": term_flag,
    }


def start_state(log, rng):
    onset = _first_run(~log["on"])
    T = len(log["act"])
    if onset is not None:
        t0 = onset + 1 - LEAD
    elif T > PREFIX + MIN_WINDOW + 1:
        t0 = int(rng.integers(PREFIX, T - MIN_WINDOW))
    else:
        return None
    if t0 < 1:
        return None
    W = min(WINDOW, T - 1 - t0)
    return (t0, W) if W >= MIN_WINDOW else None


@torch.no_grad()
def dream_branch(rnn, theta, log, t0, W, B, tau):
    """The controller steers inside the dream, started in the exact real state at step t0."""
    mu, act = torch.from_numpy(log["mu"]), torch.from_numpy(log["act"])
    *_, state = rnn(mu[None, :t0], act[None, :t0])
    st = tuple(s.expand(1, B, -1).contiguous() for s in state)
    th = torch.as_tensor(np.asarray(theta), dtype=torch.float32)[None].expand(B, -1)
    z, acts, rs = mu[t0][None].expand(B, -1).clone(), [], []
    for _ in range(W):
        a = act_batched(th, z, st[0][0])
        logit, m, ls, r, _, st = rnn.step(z, a, st)
        z = mdn_sample(logit, m, ls, tau)
        acts.append(a), rs.append(r)
    return torch.stack(acts).numpy(), torch.stack(rs).numpy()  # (W,B,3), (W,B)


def _progress(it, n, label):
    out, t0 = [], time.time()
    for r in it:
        out.append(r)
        if len(out) % max(1, n // 10) == 0 or len(out) == n:
            el = time.time() - t0
            eta = el / len(out) * (n - len(out))
            print(
                f"  {label}: {len(out)}/{n}  ({el / 60:.1f} min elapsed, "
                f"about {eta / 60:.1f} min left)",
                flush=True,
            )
    return out


def evaluate_candidates(rnn, thetas, seeds, pool, futures, tau, rng):
    """Real rollouts of each controller on each track, then dream-steered branches replayed in
    the real simulator. Returns (logs, candidates); every candidate has its exploit gap."""
    jobs = [(th, int(s)) for th in thetas for s in seeds]
    logs = _progress(pool.imap(rollout_log, jobs, chunksize=1), len(jobs), "real rollouts")
    plan, replay = [], []
    for k, log in enumerate(logs):
        st = start_state(log, rng)
        if st is None:
            continue
        t0, W = st
        theta = thetas[k // len(seeds)]
        acts, rs = dream_branch(rnn, theta, log, t0, W, futures, tau)
        for b in range(futures):
            plan.append(
                {
                    "log": k,
                    "t0": t0,
                    "W": W,
                    "acts": acts[:, b].astype(np.float32),
                    "dream_r": rs[:, b],
                }
            )
            replay.append((log["seed"], log["act"][:t0], plan[-1]["acts"]))
    real = _progress(pool.imap(real_branch, replay, chunksize=1), len(replay), "real replays")
    for c, r in zip(plan, real):
        m = len(r["rew"])
        c["real"], c["m"] = r, m
        c["dream_ret"] = float(c["dream_r"][:m].sum())
        c["real_ret"] = float(np.clip(r["rew"], -1, 10).sum())
        c["gap"] = c["dream_ret"] - c["real_ret"]
        c["real_on_share"] = float(r["on"].mean())
    return logs, plan


def build_episode(c, log):
    """Prefix tail from the real log, then the real branch: one new training episode."""
    t0, m, r = c["t0"], c["m"], c["real"]
    P = min(PREFIX, t0)
    done = np.zeros(P + m, bool)
    done[-1] = r["terminated"]
    return {
        "mu": np.concatenate([log["mu"][t0 - P : t0 + 1], r["mu"]]),
        "logvar": np.concatenate([log["logvar"][t0 - P : t0 + 1], r["logvar"]]),
        "act": np.concatenate([log["act"][t0 - P : t0], c["acts"][:m]]),
        "rew": np.concatenate([log["rew"][t0 - P : t0], r["rew"]]),
        "done": done,
        "seed": log["seed"],
        "gap": c["gap"],
    }


def select(plan, keep_top, keep_random, rng):
    ok = [i for i, c in enumerate(plan) if min(PREFIX, c["t0"]) + c["m"] >= C.seq_len]
    ok.sort(key=lambda i: -plan[i]["gap"])
    top, rest = ok[:keep_top], ok[keep_top:]
    rand = list(rng.choice(rest, min(keep_random, len(rest)), replace=False)) if rest else []
    return top + [int(i) for i in rand]


def assemble(episodes, path: Path, val_frac=0.1, seed=0):
    """Write episodes in the latents.npz format that train_mdnrnn.SequenceSampler reads."""
    T = np.array([len(e["act"]) for e in episodes])
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(episodes), bool)
    is_val[rng.choice(len(episodes), max(1, round(len(episodes) * val_frac)), replace=False)] = True
    cat = lambda k: np.concatenate([e[k] for e in episodes])
    np.savez(
        path,
        mu=cat("mu"),
        logvar=cat("logvar"),
        ep_len=T,
        obs_start=np.r_[0, np.cumsum(T + 1)[:-1]],
        act_start=np.r_[0, np.cumsum(T)[:-1]],
        is_val=is_val,
        actions=cat("act"),
        rewards=cat("rew"),
        dones=cat("done"),
        curv=np.zeros(int((T + 1).sum()), np.float32),
    )


def save_episodes(episodes, path: Path):
    path.write_bytes(pickle.dumps(episodes))


def load_episodes(path: Path):
    return pickle.loads(path.read_bytes())
