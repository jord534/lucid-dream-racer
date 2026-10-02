"""Propose, critic, collect: where does the dream disagree with the simulator?

For a controller and a track: drive the real simulator and log it; start the dream in the exact
real state 30 steps before the car first leaves the road (or at a random step if it never does);
let the controller steer inside the dream; run the same actions in the real simulator. The
exploit gap is the dream's predicted return minus the real return over that window. The real
outcomes of the chosen candidates become new training episodes for the world model.
The road-status gap is the same comparison for "is the car on the road": a small probe reads
P(on road) from each dream latent, and the real on-road flags come from the simulator.
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
ONSET_RUN = 10  # off-road run that counts as "the car left the road" in a training episode
MIN_OFF_FRAMES = 200  # fewer real off-road frames than this: the road probe is not trusted
_AG: WorldModelAgent | None = None


def init_worker(vae_path, rnn_path):
    global _AG
    _AG = WorldModelAgent(vae_path, rnn_path)


def _wheels_on(env) -> bool:
    return any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels)


def _first_run(flags, start=0, run=RUN):
    n = 0
    for i in range(start, len(flags)):
        n = n + 1 if flags[i] else 0
        if n == run:
            return i - run + 1
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
    z, acts, rs, zs, ons = mu[t0][None].expand(B, -1).clone(), [], [], [], []
    head = getattr(rnn, "has_on", False)
    for _ in range(W):
        a = act_batched(th, z, st[0][0])
        logit, m, ls, r, _, st, *on = rnn.step(z, a, st, return_on=head)
        z = mdn_sample(logit, m, ls, tau)
        acts.append(a), rs.append(r), zs.append(z)
        if on:
            ons.append(torch.sigmoid(on[0]))
    # actions (W,B,3), rewards (W,B), latents after each step (W,B,z), on-road head P (W,B) or None
    return (
        torch.stack(acts).numpy(),
        torch.stack(rs).numpy(),
        torch.stack(zs),
        torch.stack(ons).numpy() if ons else None,
    )


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


def _auc(p, y):
    from scipy.stats import rankdata

    r = rankdata(p)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_probe(X, y):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(C.z_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(m.parameters(), 3e-3, weight_decay=1e-4)
    X, y = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    with torch.enable_grad():
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(m(X)[:, 0], y).backward()
            opt.step()
    return m.eval()


def road_probe(logs, rng, folds=5):
    """Probe latent -> P(car on road), trained on the real latents of these logs (sampled the way
    the world model sees them), with the real on-road flag as the label. Returns (probe, info);
    info has the held-out AUC (tracks held out in turn) and the number of real off-road frames."""
    X = [
        lg["mu"] + rng.standard_normal(lg["mu"].shape).astype(np.float32) * np.exp(0.5 * lg["logvar"])
        for lg in logs
    ]
    y = [np.r_[True, lg["on"][:-1]] for lg in logs]  # label of obs_t = state after step t-1
    off = int(sum((~v).sum() for v in y))
    info = {"off_road_frames": off, "frames": int(sum(len(v) for v in y)), "auc": float("nan")}
    info["ok"] = off >= MIN_OFF_FRAMES
    if not info["ok"]:
        return None, info
    cat = lambda xs, ix: np.concatenate([xs[i] for i in ix])
    folds = min(folds, len(logs))
    fold = np.arange(len(logs)) % folds
    P, Y = [], []
    for f in range(folds if folds > 1 else 0):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        pr = _fit_probe(cat(X, tr), cat(y, tr))
        with torch.no_grad():
            P.append(torch.sigmoid(pr(torch.from_numpy(cat(X, te)))[:, 0]).numpy())
        Y.append(cat(y, te))
    if P:
        info["auc"] = _auc(np.concatenate(P), np.concatenate(Y))
    return _fit_probe(cat(X, range(len(logs))), cat(y, range(len(logs)))), info


def evaluate_candidates(rnn, thetas, seeds, pool, futures, tau, rng):
    """Real rollouts of each controller on each track, then dream-steered branches replayed in
    the real simulator. Returns (logs, candidates, info); every candidate has its exploit gap
    (reward) and its road-status gap (dream's P(on road) minus the real on-road share, over the
    window; None when the probe could not be trusted, see info["probe"])."""
    jobs = [(th, int(s)) for th in thetas for s in seeds]
    logs = _progress(pool.imap(rollout_log, jobs, chunksize=1), len(jobs), "real rollouts")
    probe, pinfo = road_probe(logs, rng)
    if probe is None:
        print(
            f"  road probe not trusted: only {pinfo['off_road_frames']} real off-road frames",
            flush=True,
        )
    else:
        print(
            f"  road probe: held-out AUC {pinfo['auc']:.3f} on {pinfo['frames']} real frames "
            f"({pinfo['off_road_frames']} off-road)",
            flush=True,
        )
    plan, replay = [], []
    for k, log in enumerate(logs):
        st = start_state(log, rng)
        if st is None:
            continue
        t0, W = st
        theta = thetas[k // len(seeds)]
        acts, rs, zs, head_p = dream_branch(rnn, theta, log, t0, W, futures, tau)
        with torch.no_grad():
            p_on = (
                torch.sigmoid(probe(zs.reshape(-1, C.z_dim))[:, 0]).view(W, futures).numpy()
                if probe is not None
                else None
            )
        for b in range(futures):
            plan.append(
                {
                    "log": k,
                    "t0": t0,
                    "W": W,
                    "acts": acts[:, b].astype(np.float32),
                    "dream_r": rs[:, b],
                    "dream_p_on": p_on[:, b] if p_on is not None else None,
                    "head_p_on": head_p[:, b] if head_p is not None else None,
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
        if c["dream_p_on"] is not None and m:
            p, on = c["dream_p_on"][:m], r["on"]
            c["dream_p_on_mean"] = float(p.mean())
            c["road_gap"] = c["dream_p_on_mean"] - c["real_on_share"]
            c["p_on_when_real_off"] = float(p[~on].mean()) if (~on).any() else None
        else:
            c["dream_p_on_mean"] = c["road_gap"] = c["p_on_when_real_off"] = None
        if c["head_p_on"] is not None and m:
            h, on = c["head_p_on"][:m], r["on"]
            c["head_p_on_when_real_off"] = float(h[~on].mean()) if (~on).any() else None
            c["head_acc"] = float(((h > 0.5) == on).mean())
        else:
            c["head_p_on_when_real_off"] = c["head_acc"] = None
    return logs, plan, {"probe": pinfo}


def build_episode(c, log):
    """Prefix tail from the real log, then the real branch: one new training episode."""
    t0, m, r = c["t0"], c["m"], c["real"]
    P = min(PREFIX, t0)
    done = np.zeros(P + m, bool)
    done[-1] = r["terminated"]
    on = np.concatenate([log["on"][t0 - P : t0], r["on"]])  # flag after each action
    onset = _first_run(~on, run=ONSET_RUN)
    return {
        "on": on,
        "onset": -1 if onset is None else onset,
        "mu": np.concatenate([log["mu"][t0 - P : t0 + 1], r["mu"]]),
        "logvar": np.concatenate([log["logvar"][t0 - P : t0 + 1], r["logvar"]]),
        "act": np.concatenate([log["act"][t0 - P : t0], c["acts"][:m]]),
        "rew": np.concatenate([log["rew"][t0 - P : t0], r["rew"]]),
        "done": done,
        "seed": log["seed"],
        "gap": c["gap"],
        "road_gap": c["road_gap"],
    }


def select(plan, keep_top, keep_random, rng, key="gap"):
    """The keep_top candidates with the largest `key` ("gap": reward; "road_gap": road status),
    plus keep_random of the rest."""
    ok = [i for i, c in enumerate(plan) if min(PREFIX, c["t0"]) + c["m"] >= C.seq_len]
    ok.sort(key=lambda i: -plan[i][key])
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
        onset=np.array([e.get("onset", -1) for e in episodes], int),  # -1: unknown or none
        onroad=np.concatenate(  # flag after each action; -1 where the episode carries none
            [
                np.asarray(e["on"], np.int8) if "on" in e else np.full(len(e["act"]), -1, np.int8)
                for e in episodes
            ]
        ),
    )


def save_episodes(episodes, path: Path):
    path.write_bytes(pickle.dumps(episodes))


def load_episodes(path: Path):
    return pickle.loads(path.read_bytes())
