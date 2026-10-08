"""Does a pose state keep the model on the road? head_horizon.py's test, extended to pose models.

Same real runs, windows, independent road probe and readout as diagnostics/head_horizon.py
(technical note 3.8, A.8): the model is started after up to 40 real steps, then the REAL actions are
replayed for up to 100 steps, 16 samples per window, and P(on road) is read from each imagined latent
by a probe the model never trained against. The windows and the probe are drawn with head_horizon's
random sequence, so they are the same windows. What drives the model:
  real        every input latent is a real frame; for pose models the pose fed is the true one.
              Two readouts: the probe on the real next frame (head_horizon's "teacher-forced latent
              probe", the 0.91 of the note: a ceiling, independent of the model), and the probe on
              the model's own one-step prediction.
  own         own latent samples and, for pose models, its own predicted pose (as in the dream)
  oracle      own latent samples, but the TRUE pose at every step (pose-input models only)
  real+own q  real frames, own predicted pose (what an agent in the simulator would feed it)
For models with a pose head, also: lateral offset error (half-widths, labels clipped at +-5 like
the model's features), heading error (degrees), and road status implied by the predicted offset
(|offset| > 1.34 half-widths, the best-accuracy threshold on the training labels).

Per-window counts are saved so that intervals (bootstrap over windows) and paired differences
between models can be computed (diagnostics/pose_gate.py).

    python diagnostics/pose_horizon.py --model runs/pose_state/pose_s0/best.pt --name pose_s0 \
        --tau 1.15 --sets val          (development: held-out validation episodes only)
    ... --sets val test                (final checkpoints only: adds the test-track runs, once)
Writes reports/pose_state/horizon/<name>_tau<tau>.json and .npz."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from head_horizon import BINS, LEAD, WARM, B, H, base_episodes, first_off_run, rollout_episodes

from ldr.branch import _fit_probe
from ldr.check_dream import load_mdnrnn
from ldr.config import DATA, REPORTS, C
from ldr.label_pose import pose_by_obs
from ldr.mdnrnn import mdn_sample, pose_features, pose_lateral

torch.set_num_threads(2)
OUT = REPORTS / "pose_state" / "horizon"
OFF_LAT = 1.34          # |lateral| above this reads as off the road (reports/pose_state/label_sanity.json)
CHUNK = 128             # windows per batch (x16 samples)
ROLLOUT_TAGS = ("loop_head", "loop_road")   # head_horizon's six drivers, in its order
SETS = {"val": "original data, held-out validation episodes",
        "test": "test-track rollouts of dream-trained drivers"}
KINDS = ("random start", "30 steps before leaving the road")
# readouts: (name, latent fed, pose fed, which latent the probe reads)
READOUTS = [("real: probe on real frame", "real", "true", "real"),
            ("real: probe on 1-step prediction", "real", "true", "own"),
            ("own", "own", "own", "own"),
            ("oracle (own latent, true pose)", "own", "true", "own"),
            ("real + own pose", "real", "own", "own")]


def with_pose_base(eps, d, pose_obs):
    for ep, e in zip(eps, d["_ids"]):
        o = int(d["obs_start"][e])
        ep["q"] = pose_obs[o : o + len(ep["mu"])]
        yield ep


def with_pose_rollouts(path):
    pose, pose0 = (np.load(Path(path).with_name(Path(path).stem + s)) for s in ("_pose.npy", "_pose0.npy"))
    z = np.load(path)
    for ep, s, n, k in zip(rollout_episodes(path), z["start"], z["length"], range(len(pose0))):
        ep["q"] = np.concatenate([pose0[k : k + 1], pose[int(s) : int(s) + int(n)]])   # obs-aligned
        yield ep


def windows_like_head_horizon(sets_wanted):
    """Rebuild head_horizon's probe and windows with its exact random sequence."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    z = np.load(DATA / "latents.npz")
    d = {k: z[k] for k in ("mu", "logvar", "actions", "obs_start", "act_start", "ep_len", "is_val")}
    side = np.load(DATA / "latents_onroad.npy")
    pose_obs = pose_by_obs(d["obs_start"], d["act_start"], d["ep_len"], np.load(DATA / "latents_pose.npy"),
                           np.load(DATA / "latents_pose0.npy"))
    val = np.flatnonzero(d["is_val"])
    train = np.flatnonzero(~d["is_val"])[::5]
    X, y = [], []
    for ep in base_episodes(d, side, train):
        X.append(ep["mu"] + rng.standard_normal(ep["mu"].shape).astype(np.float32) * np.exp(0.5 * ep["lv"]))
        y.append(np.r_[True, ep["on"]])
    probe = _fit_probe(np.concatenate(X), np.concatenate(y))

    def random_starts(ep, T, rng, n=3):
        return [int(rng.integers(WARM, max(WARM + 1, T - H - 1))) for _ in range(n)] if T > WARM + H + 2 else []

    def onset_starts(ep, T, rng):
        o = first_off_run(ep["on"])
        return [] if o is None else [o + 1 - LEAD]

    d["_ids"] = val
    sets = {"val": list(with_pose_base(list(base_episodes(d, side, val)), d, pose_obs))}
    sets["test"] = [ep for tag in ROLLOUT_TAGS for i in range(3)
                    for ep in with_pose_rollouts(DATA / "dream_failure" / f"rollouts_{tag}_r02_p{i}.npz")]
    out = {}
    for key in ("val", "test"):   # head_horizon's order: the random draws of the later set depend on it
        for kind, fn in zip(KINDS, (random_starts, onset_starts)):
            wins = []
            for ep in sets[key]:
                T = len(ep["act"])
                for t0 in fn(ep, T, rng):
                    W = min(H, T - 1 - t0)
                    if t0 < 1 or W < 20:
                        continue
                    wins.append((ep, t0, W))
            if key in sets_wanted:
                out[f"{SETS[key]} | {kind}"] = wins
    return probe, out


@torch.no_grad()
def run_chunk(rnn, probe, wins, tau, gen, readouts):
    """wins: windows sharing the same warm-up length. Returns per-readout arrays (H, N, B):
    P(on road) and predicted pose features (or None), plus truth (H, N), true pose (H, N, 4), mask."""
    N, R, k = len(wins), len(wins) * B, min(WARM, wins[0][1])
    L = k + H + 1
    mu = np.zeros((N, L, C.z_dim), np.float32)
    lv = np.full((N, L, C.z_dim), -20, np.float32)
    act = np.zeros((N, L, 3), np.float32)
    q = np.zeros((N, L, 4), np.float32)
    q[..., 2] = 1
    truth = np.ones((N, H), bool)
    mask = np.zeros((N, H), bool)
    for n, (ep, t0, W) in enumerate(wins):
        s = t0 - k
        mu[n, : k + W + 1], lv[n, : k + W + 1] = ep["mu"][s : t0 + W + 1], ep["lv"][s : t0 + W + 1]
        act[n, : k + W] = ep["act"][s : t0 + W]
        q[n, : k + W + 1] = ep["q"][s : t0 + W + 1]
        truth[n, :W], mask[n, :W] = ep["on"][t0 : t0 + W], True
    rep = lambda x: torch.from_numpy(x).repeat_interleave(B, 0)
    mu_t, sd_t, act_t, qf = rep(mu), rep(np.exp(0.5 * lv)), rep(act), rep(pose_features(q))
    samp = lambda i: mu_t[:, i] + torch.randn(R, C.z_dim, generator=gen) * sd_t[:, i]
    pin, rp = rnn.pose_input, bool(rnn.pose_dim)
    st0 = rnn.initial_state(R, "cpu")
    for i in range(k):
        *_, st0 = rnn.step(samp(i), act_t[:, i], st0, q=qf[:, i] if pin else None)
    res = {}
    modes = {(zs, qs) for _, zs, qs, _ in readouts}
    for zs, qs in sorted(modes):
        st, z, qq = tuple(s.clone() for s in st0), samp(k), qf[:, k]
        p_real, p_own, poses = [], [], []
        for j in range(H):
            o = rnn.step(z, act_t[:, k + j], st, q=qq if pin else None, return_pose=rp)
            logit, m, ls, st = o[0], o[1], o[2], o[5]
            own = mdn_sample(logit, m, ls, tau)
            real = samp(k + j + 1)
            p_own.append(torch.sigmoid(probe(own)[:, 0]))
            p_real.append(torch.sigmoid(probe(real)[:, 0]))
            if rp:
                poses.append(o[-1])
            z = real if zs == "real" else own
            qq = qf[:, k + j + 1] if qs == "true" else (o[-1] if rp else None)
        sh = lambda v: torch.stack(v).view(H, N, B, *v[0].shape[1:]).numpy()
        for name, z_, q_, reads in readouts:
            if (z_, q_) == (zs, qs):
                res[name] = (sh(p_real if reads == "real" else p_own), sh(poses) if rp else None)
    true_q = q[:, k + 1 : k + H + 1].transpose(1, 0, 2)   # pose after action t0 + j, (H, N, 4)
    return res, truth.T, true_q, mask.T


def window_counts(p, pose, truth, true_q, mask):
    """Per window and bin: counts for accuracy/recall from probe P(on) (H,N,B), and pose errors."""
    out = {}
    pred_on = p > 0.5
    y = truth[..., None]
    m = np.broadcast_to(mask[..., None], p.shape)
    off = m & ~y
    for bi, (a, b) in enumerate(BINS):
        sl = slice(a, b)
        c = {"n": m[sl].sum((0, 2)), "correct": (m & (pred_on == y))[sl].sum((0, 2)),
             "n_off": off[sl].sum((0, 2)), "off_correct": (off & ~pred_on)[sl].sum((0, 2))}
        if pose is not None:
            lat_true = np.clip(true_q[..., 0], -5, 5)[..., None]
            lat_hat = pose_lateral(pose)                     # pose: (H,N,B,4) features
            hd_true = np.arctan2(true_q[..., 1], true_q[..., 2])[..., None]
            hd_hat = np.arctan2(pose[..., 1], pose[..., 2])
            dh = np.abs((hd_hat - hd_true + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
            imp_on = np.abs(lat_hat) <= OFF_LAT
            c.update(lat_abs_err=np.where(m, np.abs(lat_hat - lat_true), 0)[sl].sum((0, 2)),
                     hdg_abs_err_deg=np.where(m, dh, 0)[sl].sum((0, 2)),
                     implied_off=(m & ~imp_on)[sl].sum((0, 2)),
                     implied_correct=(m & (imp_on == y))[sl].sum((0, 2)),
                     implied_off_correct=(off & ~imp_on)[sl].sum((0, 2)))
        out[bi] = c
    return out


def evaluate(rnn, probe, wins, tau, seed):
    gen = torch.Generator().manual_seed(seed)
    # without pose input the pose fed does not matter: only the real and own readouts
    readouts = READOUTS if rnn.pose_input else READOUTS[:3]
    order = sorted(range(len(wins)), key=lambda i: min(WARM, wins[i][1]))
    counts = {r[0]: {bi: {} for bi in range(len(BINS))} for r in readouts}
    idx = []
    i = 0
    while i < len(order):
        k = min(WARM, wins[order[i]][1])
        grp = [j for j in order[i:] if min(WARM, wins[j][1]) == k][:CHUNK]
        res, truth, true_q, mask = run_chunk(rnn, probe, [wins[j] for j in grp], tau, gen, readouts)
        for name, (p, pose) in res.items():
            for bi, c in window_counts(p, pose, truth, true_q, mask).items():
                for key, v in c.items():
                    counts[name][bi].setdefault(key, []).append(v)
        idx += grp
        i += len(grp)
    inv = np.argsort(idx)   # back to the original window order, so models can be paired
    return {name: {bi: {key: np.concatenate(v)[inv] for key, v in c.items()} for bi, c in byb.items()}
            for name, byb in counts.items()}


def summarise_counts(c, reps=2000, seed=0):
    """Point values and 95% bootstrap intervals over windows."""
    rng = np.random.default_rng(seed)
    n_w = len(c["n"])
    boot = rng.integers(n_w, size=(reps, n_w))
    def ratio(num, den):
        if c[den].sum() == 0:
            return None
        s = c[num][boot].sum(1) / np.maximum(c[den][boot].sum(1), 1)
        return [float(c[num].sum() / c[den].sum()), *np.percentile(s, [2.5, 97.5]).tolist()]
    r = {"windows": int((c["n"] > 0).sum()), "steps": int(c["n"].sum() // B),
         "off_share": float(c["n_off"].sum() / max(c["n"].sum(), 1)),
         "acc": ratio("correct", "n"), "off_recall": ratio("off_correct", "n_off")}
    if "lat_abs_err" in c:
        r.update(lat_abs_err=ratio("lat_abs_err", "n"), hdg_abs_err_deg=ratio("hdg_abs_err_deg", "n"),
                 implied_off_rate=ratio("implied_off", "n"), implied_acc=ratio("implied_correct", "n"),
                 implied_off_recall=ratio("implied_off_correct", "n_off"))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path)
    p.add_argument("--name")
    p.add_argument("--tau", type=float, nargs="+", default=[C.tau])
    p.add_argument("--sets", nargs="+", default=["val"], choices=list(SETS))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rnn = load_mdnrnn(a.model)
    print(f"{a.model}: pose head {rnn.pose_dim}, pose input {rnn.pose_input}, trained pose {rnn.has_pose}", flush=True)
    probe, wins = windows_like_head_horizon(set(a.sets))
    for tau in a.tau:
        t0 = time.time()
        out = {"model": str(a.model), "name": a.name, "tau": tau, "sets": a.sets, "bins": [f"{x}-{y}" for x, y in BINS],
               "results": {}}
        arrays = {}
        for key, w in wins.items():
            cnt = evaluate(rnn, probe, w, tau, a.seed)
            out["results"][key] = {"windows": len(w), "readouts": {
                name: {out["bins"][bi]: summarise_counts(c) for bi, c in byb.items()} for name, byb in cnt.items()}}
            for name, byb in cnt.items():
                for bi, c in byb.items():
                    for k2, v in c.items():
                        arrays[f"{key}||{name}||{bi}||{k2}"] = v
            print(f"  tau {tau} | {key}: {len(w)} windows", flush=True)
        out["minutes"] = (time.time() - t0) / 60
        stem = OUT / f"{a.name}_tau{tau}{'_val' if a.sets == ['val'] else ''}"
        Path(f"{stem}.json").write_text(json.dumps(out, indent=1))   # not with_suffix: the tau has a dot
        np.savez_compressed(f"{stem}.npz", **arrays)
        for key, r in out["results"].items():
            print(f"\n{key} ({r['windows']} windows), tau {tau}")
            for name, byb in r["readouts"].items():
                cells = []
                for b, s in byb.items():
                    f = lambda v: "-" if v is None else f"{v[0]:.2f}"
                    cells.append(f"{b}: {f(s['acc'])}/{f(s['off_recall'])}"
                                 + (f" lat {s['lat_abs_err'][0]:.2f} hdg {s['hdg_abs_err_deg'][0]:.0f}"
                                    if "lat_abs_err" in s else ""))
                print(f"  {name:34s} " + "  ".join(cells))
        print(f"written {stem}.json ({out['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
