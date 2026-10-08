"""Does the dream keep the car on the road when the simulator does not?

Stage 1 (--collect): drive the dream-trained controller (runs/dream_v3) on the 100 test
tracks exactly as ldr.evaluate does and log, per step, the VAE posterior (mu, logvar), the
action, the reward, whether any wheel touches the road, and the car's signed lateral offset
and heading error against the track centreline. Written to data/dream_failure/.

Stage 2 (--analyse): for each track, start the MDN-RNN 30 steps before the car first leaves
the road (>= 20 consecutive off-road steps), teacher-force 40 real steps of context, then
feed it the SAME real actions open-loop for up to 100 steps (the replay used in section 3.3
of the note). A probe on the VAE latent, trained on real frames labelled by wheel contact
and cross-fitted by track (each track is scored by a probe that never saw it), reads
P(on road) from the real latent and from the dream's sampled latent.

Primary metric, fixed before looking at any result: per track, over the window steps where
the real car is off the road, mean P(on road | dream latent) minus mean P(on road | real
latent). Positive means the dream shows the car on the road more than the simulator does.
Reported as the median over tracks with a bootstrap 95% CI over tracks and a Wilcoxon
signed-rank p-value. Repeated at tau = 1.15 (the training temperature) and tau = 0.05.

    python diagnostics/dream_fidelity.py --collect
    python diagnostics/dream_fidelity.py --analyse
Writes reports/dream_failure_analysis/dream_fidelity.json."""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, wilcoxon

from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import C, DATA, REPORTS, RUNS, SEED_TEST
from ldr.envs import make_env
from ldr.mdnrnn import mdn_sample
from ldr.utils import preprocess, to_tensor

# Which controller and world model are diagnosed. The defaults are the original analysis
# (controller runs/dream_v3, world model runs/vae and runs/mdnrnn). To diagnose another, e.g. a
# controller of the propose-critic loop (all three scripts read these):
#   LDR_DIAG_TAG=loop_r02_p0 LDR_DIAG_CTRL=runs/loop/r02/proposer0/best.pt LDR_DIAG_MODEL=runs/loop/r02
# Its rollouts go to data/dream_failure/rollouts_<tag>.npz and its results to
# reports/dream_failure_analysis/<tag>/.
TAG = os.environ.get("LDR_DIAG_TAG", "")
CTRL = Path(os.environ.get("LDR_DIAG_CTRL", RUNS / "dream_v3" / "best.pt"))
_MODEL = Path(os.environ["LDR_DIAG_MODEL"]) if "LDR_DIAG_MODEL" in os.environ else RUNS
VAE_PATH, RNN_PATH = _MODEL / "vae" / "best.pt", _MODEL / "mdnrnn" / "best.pt"
OUT = REPORTS / "dream_failure_analysis" / TAG
STORE = DATA / "dream_failure" / f"rollouts_{TAG or 'dream_v3'}.npz"
TRACKS, WORKERS = 100, 6
RUN, LEAD, WARM, HORIZON, FOLDS = 20, 30, 40, 100, 5
TAUS = {1.15: 32, 0.05: 8}                  # temperature -> dream samples per track

_AGENT = None


def _init():
    global _AGENT
    _AGENT = WorldModelAgent(VAE_PATH, RNN_PATH)


def _first_run(flags, start=0):
    n = 0
    for i in range(start, len(flags)):
        n = n + 1 if flags[i] else 0
        if n == RUN:
            return i - RUN + 1
    return None


def _collect_one(job):
    theta, seed = job
    ag = _AGENT
    env = make_env()
    obs, _ = env.reset(seed=seed)
    ag.reset()
    rng = np.random.default_rng(seed)
    xy = np.asarray(env.unwrapped.track)[:, 2:4]
    mu, lv, act, rew, on, lat, hdg = [], [], [], [], [], [], []
    for _ in range(1000):
        with torch.no_grad():
            m, l = ag.vae.encode(to_tensor(preprocess(obs, C.img)[None], ag.dev))
        mu.append(m[0].cpu().numpy())
        lv.append(l[0].cpu().numpy())
        a = ag.act(theta, obs, rng, env=env)
        obs, r, term, trunc, _ = env.step(a)
        car = env.unwrapped.car
        pos = np.asarray(car.hull.position)
        i = int(np.argmin(((xy - pos) ** 2).sum(1)))
        d = xy[(i + 1) % len(xy)] - xy[i]
        d = d / (np.linalg.norm(d) + 1e-9)
        rel, f = pos - xy[i], np.array([-np.sin(car.hull.angle), np.cos(car.hull.angle)])
        lat.append(d[0] * rel[1] - d[1] * rel[0])
        hdg.append(np.arctan2(f[0] * d[1] - f[1] * d[0], f @ d))
        act.append(a)
        rew.append(r)
        on.append(any(len(w.tiles) > 0 for w in car.wheels))
        if term or trunc:
            break
    env.close()
    f32 = lambda x: np.asarray(x, np.float32)
    return dict(seed=seed, mu=f32(mu), logvar=f32(lv), act=f32(act), rew=f32(rew),
                onroad=np.asarray(on, bool), lat=f32(lat), hdg=f32(hdg))


def collect():
    theta = torch.load(CTRL, weights_only=False)["theta"]
    ref = json.loads((REPORTS / f"eval_{TAG or 'wm_dream_v3'}.json").read_text())["returns"]
    jobs = [(theta, SEED_TEST + i) for i in range(TRACKS)]
    res, t0 = [], time.time()
    print(f"collecting {len(jobs)} rollouts, {WORKERS} workers", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(_collect_one, jobs, chunksize=1):
            res.append(r)
            eta = (time.time() - t0) / len(res) * (len(jobs) - len(res))
            print(f"[{len(res):3d}/{len(jobs)}] seed {r['seed']}  steps {len(r['act']):4d}  "
                  f"return {r['rew'].sum():7.1f}  off-road {1 - r['onroad'].mean():4.0%}  "
                  f"eta {eta / 60:4.1f} min", flush=True)
    res.sort(key=lambda r: r["seed"])
    match = sum(abs(float(r["rew"].sum()) - x) < 1e-2 for r, x in zip(res, ref))
    print(f"returns matching the stored evaluation: {match}/{len(res)}", flush=True)
    length = np.array([len(r["act"]) for r in res])
    cat = lambda k: np.concatenate([r[k] for r in res])
    STORE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE, seed=np.array([r["seed"] for r in res]), length=length,
                        start=np.r_[0, np.cumsum(length)[:-1]], **{k: cat(k) for k in
                        ("mu", "logvar", "act", "rew", "onroad", "lat", "hdg")})
    print(f"wrote {STORE}", flush=True)


def _train_probe(X, y):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(C.z_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(m.parameters(), 3e-3, weight_decay=1e-4)
    X, y = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(m(X)[:, 0], y).backward()
        opt.step()
    return m.eval()


def _auc(p, y):
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ci(x, reps=2000):
    rng = np.random.default_rng(0)
    b = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(reps)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


@torch.no_grad()
def _replay(rnn, probe, tr, z_real, lab, p_real, onset_obs, tau, B):
    T, t0 = len(tr["act"]), onset_obs - LEAD
    W = min(HORIZON, T - 1 - t0)
    if t0 < 1 or W < 20:
        return None
    mu, lv, act = (torch.from_numpy(tr[k]) for k in ("mu", "logvar", "act"))
    sample = lambda k: mu[k] + torch.randn(B, C.z_dim) * (0.5 * lv[k]).exp()
    state = rnn.initial_state(B, "cpu")
    for k in range(t0 - min(WARM, t0), t0):
        *_, state = rnn.step(sample(k), act[k][None].expand(B, -1), state)
    z, p, r_hat = sample(t0), [], []
    for j in range(W):
        logit, m, ls, r, _, state = rnn.step(z, act[t0 + j][None].expand(B, -1), state)
        z = mdn_sample(logit, m, ls, tau)
        p.append(torch.sigmoid(probe(z)[:, 0]).mean().item())
        r_hat.append(r.mean().item())
    sl = slice(t0 + 1, t0 + 1 + W)
    return dict(t0=t0, W=W, p_dream=np.array(p), p_real=p_real[sl], real_on=lab[sl],
                r_dream=np.array(r_hat), r_real=np.clip(tr["rew"][t0:t0 + W], -1, 10))


def analyse():
    d = np.load(STORE)
    tracks = []
    for s, n, seed in zip(d["start"], d["length"], d["seed"]):
        sl = slice(int(s), int(s + n))
        tracks.append({k: d[k][sl] for k in ("mu", "logvar", "act", "rew", "onroad")})
        tracks[-1]["seed"] = int(seed)
    rng = np.random.default_rng(0)
    z_real = [t["mu"] + rng.standard_normal(t["mu"].shape).astype(np.float32)
              * np.exp(0.5 * t["logvar"]) for t in tracks]
    lab = [np.r_[True, t["onroad"][:-1]] for t in tracks]       # label of obs_t = state after step t-1
    fold = np.arange(len(tracks)) % FOLDS

    probes, p_real, held = {}, [None] * len(tracks), []
    for f in range(FOLDS):
        tr_ix = [i for i in range(len(tracks)) if fold[i] != f]
        probes[f] = _train_probe(np.concatenate([z_real[i] for i in tr_ix]),
                                 np.concatenate([lab[i] for i in tr_ix]))
        for i in np.flatnonzero(fold == f):
            with torch.no_grad():
                p_real[i] = torch.sigmoid(probes[f](torch.from_numpy(z_real[i]))[:, 0]).numpy()
            held.append(i)
    P, Y = np.concatenate([p_real[i] for i in held]), np.concatenate([lab[i] for i in held])
    probe_stats = dict(heldout_auc=_auc(P, Y), heldout_accuracy=float(((P > .5) == Y).mean()),
                       off_road_recall=float((P[~Y] < .5).mean()), on_road_recall=float((P[Y] > .5).mean()),
                       off_road_share=float((~Y).mean()))
    print("probe on real latents, held-out tracks:", json.dumps(
        {k: round(v, 3) for k, v in probe_stats.items()}), flush=True)

    rnn = load_mdnrnn(RNN_PATH, device="cpu")
    out = dict(probe=probe_stats, tau={})
    for tau, B in TAUS.items():
        torch.manual_seed(0)
        rows, pool = [], []
        for i, tr in enumerate(tracks):
            onset = _first_run([not o for o in tr["onroad"]])
            if onset is None:
                continue
            rp = _replay(rnn, probes[fold[i]], tr, z_real[i], lab[i], p_real[i], onset + 1, tau, B)
            if rp is None:
                continue
            pool.append(rp)
            off, on = ~rp["real_on"], rp["real_on"]
            row = dict(seed=tr["seed"], onset_step=onset + 1, window=rp["W"], off_steps=int(off.sum()),
                       reward_dream=float(rp["r_dream"].sum()), reward_real=float(rp["r_real"].sum()))
            if off.sum() >= 10:
                row.update(p_dream_off=float(rp["p_dream"][off].mean()),
                           p_real_off=float(rp["p_real"][off].mean()))
                row["diff"] = row["p_dream_off"] - row["p_real_off"]
                below = np.flatnonzero(rp["p_dream"] < .5)
                first_off = int(np.flatnonzero(off)[0])
                row["dream_notices_after"] = int(below[0]) - first_off if len(below) else None
            if on.sum() >= 10:
                row.update(p_dream_on=float(rp["p_dream"][on].mean()),
                           p_real_on=float(rp["p_real"][on].mean()))
            rows.append(row)
            print(f"tau {tau}: {len(rows):3d} tracks replayed", end="\r", flush=True)
        diff = np.array([r["diff"] for r in rows if "diff" in r])
        notice = [r["dream_notices_after"] for r in rows if "diff" in r]
        s = dict(tracks_replayed=len(rows), tracks_with_off_road_window=len(diff),
                 median_diff=float(np.median(diff)), median_diff_ci95=_ci(diff),
                 share_positive=float((diff > 0).mean()), wilcoxon_p=float(wilcoxon(diff).pvalue),
                 mean_p_dream_off=float(np.mean([r["p_dream_off"] for r in rows if "diff" in r])),
                 mean_p_real_off=float(np.mean([r["p_real_off"] for r in rows if "diff" in r])),
                 mean_p_dream_on=float(np.mean([r["p_dream_on"] for r in rows if "p_dream_on" in r])),
                 mean_p_real_on=float(np.mean([r["p_real_on"] for r in rows if "p_real_on" in r])),
                 dream_never_notices=int(sum(n is None for n in notice)),
                 median_notice_delay_steps=float(np.median([n for n in notice if n is not None]))
                 if any(n is not None for n in notice) else None,
                 median_reward_dream=float(np.median([r["reward_dream"] for r in rows])),
                 median_reward_real=float(np.median([r["reward_real"] for r in rows])))
        # Secondary, added after the primary result was seen (post hoc): pooled over all
        # window steps, how well each latent's P(on road) separates real on- from off-road.
        PD, PR = (np.concatenate([r[k] for r in pool]) for k in ("p_dream", "p_real"))
        ON = np.concatenate([r["real_on"] for r in pool])
        contrast = lambda p: float(p[ON].mean() - p[~ON].mean())
        s["secondary_post_hoc"] = dict(
            auc_real_latent=_auc(PR, ON), auc_dream_latent=_auc(PD, ON),
            on_minus_off_real_latent=contrast(PR), on_minus_off_dream_latent=contrast(PD),
            share_of_real_contrast_kept_by_dream=contrast(PD) / contrast(PR))
        out["tau"][str(tau)] = dict(summary=s, rows=rows)
        print(f"\ntau {tau}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                              for k, v in s.items()}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dream_fidelity.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'dream_fidelity.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--analyse", action="store_true")
    a = p.parse_args()
    if a.collect:
        collect()
    if a.analyse:
        analyse()
