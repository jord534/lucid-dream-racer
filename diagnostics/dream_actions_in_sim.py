"""When the controller steers inside the dream, are its actions really as safe as the dream
says? Run the dream's own steering in the real simulator and compare.

Follow-up to dream_closed_loop.py. That test found that when the controller steers inside the
dream, the dream shows the car on the road even where the real car is off it. Two explanations:
  (1) the dream is too forgiving: it shows the controller's recovery steering working when the
      real physics would not;
  (2) the controller sees a slightly different state in the dream and simply chooses
      different actions, which would also have worked in the simulator.
This test separates them. Per test track, as in dream_closed_loop.py, start the world model at
the step 30 before the real car first leaves the road, in exactly the real state, and let the
controller steer inside it (4 imagined futures per track, temperature 1.15, up to 100 steps).
Then take those SAME actions and run them in the real simulator from the same real state
(re-created by replaying the real action history from the track seed; checked against the logged
rollout). Real frames are encoded and read by the same cross-fitted on-road probe.

Primary comparison, fixed before any result was seen: over the window steps where the real car,
driven by the dream's actions, is off the road, per track (averaged over its imagined futures),
mean P(on road | dream latent) minus mean P(on road | real latent). Median over tracks,
bootstrap 95% CI, Wilcoxon p. Positive means that, for identical actions, the dream shows the
car on the road more than the simulator does (explanation 1); near zero means the dream
agrees with the simulator about those actions (explanation 2).
Also reported: the share of window steps the real car is on the road under the dream's actions,
against the dream's mean P(on road); and the same at the last 20 steps of the window.

    python diagnostics/dream_actions_in_sim.py        (about 10 minutes, 4 workers)
Needs data/dream_failure/rollouts_dream_v3.npz from dream_fidelity.py --collect.
Writes reports/dream_failure_analysis/dream_actions_in_sim.json."""
import json
import time
from multiprocessing import Pool

import numpy as np
import torch

from dream_closed_loop import _stats
from dream_fidelity import FOLDS, HORIZON, LEAD, OUT, STORE, _first_run, _train_probe
from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import C, RUNS
from ldr.controller import act_batched
from ldr.encode import load_vae
from ldr.envs import make_env
from ldr.mdnrnn import mdn_sample
from ldr.utils import preprocess, to_tensor

TAU, B, WORKERS = 1.15, 4, 4
_AG = None


def _init():
    global _AG
    _AG = WorldModelAgent()                              # used only for its VAE (CPU, 1 thread)


def _replay(job):
    k, b, seed, prefix, acts = job
    env = make_env()
    env.reset(seed=seed)
    wheels = lambda: any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels)
    pre_r, pre_on = 0.0, []
    for a in prefix:
        pre_r += env.step(a)[1]
        pre_on.append(wheels())
    on, mu, lv = [], [], []
    for a in acts:
        obs, _, term, trunc, _ = env.step(a)
        on.append(wheels())
        with torch.no_grad():
            m, l = _AG.vae.encode(to_tensor(preprocess(obs, C.img)[None], _AG.dev))
        mu.append(m[0].numpy()), lv.append(l[0].numpy())
        if term or trunc:
            break
    env.close()
    return dict(k=k, b=b, prefix_reward=pre_r, prefix_on=np.array(pre_on, bool),
                on=np.array(on, bool), mu=np.array(mu, np.float32), lv=np.array(lv, np.float32))


def main():
    torch.manual_seed(0)
    d = np.load(STORE)
    tracks = []
    for s, n, seed in zip(d["start"], d["length"], d["seed"]):
        sl = slice(int(s), int(s + n))
        tracks.append({k: d[k][sl] for k in ("mu", "logvar", "act", "rew", "onroad")})
        tracks[-1]["seed"] = int(seed)
    rng = np.random.default_rng(0)                       # same draws as the other two scripts
    z_real = [t["mu"] + rng.standard_normal(t["mu"].shape).astype(np.float32)
              * np.exp(0.5 * t["logvar"]) for t in tracks]
    lab = [np.r_[True, t["onroad"][:-1]] for t in tracks]
    fold = np.arange(len(tracks)) % FOLDS
    with torch.enable_grad():
        probes = {f: _train_probe(np.concatenate([z_real[i] for i in range(len(tracks)) if fold[i] != f]),
                                  np.concatenate([lab[i] for i in range(len(tracks)) if fold[i] != f]))
                  for f in range(FOLDS)}
    prob = lambda f, z: torch.sigmoid(probes[f](z)[:, 0])

    rnn, vae = load_mdnrnn(device="cpu"), load_vae(device="cpu")
    theta = torch.load(RUNS / "dream_v3" / "best.pt", weights_only=False)["theta"]
    theta = torch.as_tensor(np.asarray(theta), dtype=torch.float32)[None].expand(B, -1)

    plan, jobs, t_start = {}, [], time.time()
    with torch.no_grad():
        for i, tr in enumerate(tracks):                  # stage A: the dream, steered by the controller
            onset = _first_run([not o for o in tr["onroad"]])
            if onset is None:
                continue
            T, t0 = len(tr["act"]), onset + 1 - LEAD
            W = min(HORIZON, T - 1 - t0)
            if t0 < 1 or W < 20:
                continue
            mu, act = torch.from_numpy(tr["mu"]), torch.from_numpy(tr["act"])
            *_, state = rnn(mu[None, :t0], act[None, :t0])
            st = tuple(s.expand(1, B, -1).contiguous() for s in state)
            z, zs, acs = mu[t0][None].expand(B, -1).clone(), [], []
            for _ in range(W):
                a = act_batched(theta, z, st[0][0])
                logit, m, ls, _, _, st = rnn.step(z, a, st)
                z = mdn_sample(logit, m, ls, TAU)
                zs.append(z), acs.append(a)
            zs, acs = torch.stack(zs), torch.stack(acs)            # (W,B,32), (W,B,3)
            flat = zs.reshape(-1, C.z_dim)
            plan[i] = dict(t0=t0, W=W,
                           p=prob(fold[i], flat).view(W, B).numpy(),
                           pp=prob(fold[i], vae.encode(vae.decode(flat))[0]).view(W, B).numpy())
            for b in range(B):
                jobs.append((i, b, tr["seed"], tr["act"][:t0], acs[:, b].numpy().astype(np.float32)))
            print(f"stage A (dream): track {len(plan):3d}  ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    res, t_b = [], time.time()
    with Pool(WORKERS, initializer=_init) as pool:       # stage B: the same actions, in the real simulator
        for r in pool.imap_unordered(_replay, jobs, chunksize=1):
            res.append(r)
            eta = (time.time() - t_b) / len(res) * (len(jobs) - len(res))
            print(f"stage B (simulator): {len(res):3d}/{len(jobs)}   eta {eta / 60:4.1f} min", flush=True)

    rows = {}
    for r in res:
        i, b, pl = r["k"], r["b"], plan[r["k"]]
        tr = tracks[i]
        t0 = pl["t0"]
        ok = (abs(r["prefix_reward"] - float(tr["rew"][:t0].sum())) < 1e-2
              and np.array_equal(r["prefix_on"], tr["onroad"][:t0]))
        m = len(r["on"])
        zr = torch.from_numpy(r["mu"] + np.random.default_rng(i * 100 + b).standard_normal(r["mu"].shape)
                              .astype(np.float32) * np.exp(0.5 * r["lv"]))
        with torch.no_grad():
            p_real = prob(fold[i], zr).numpy()
        rows.setdefault(i, []).append(dict(ok=bool(ok), on=r["on"], p_real=p_real, p=pl["p"][:m, b],
                                           pp=pl["pp"][:m, b]))
    track_rows = []
    for i, rs in sorted(rows.items()):
        out = dict(seed=tracks[i]["seed"], futures=len(rs), replay_matches_log=all(x["ok"] for x in rs))
        diffs, pdiffs = [], []
        for x in rs:
            off = ~x["on"]
            if off.sum() >= 10:
                diffs.append(x["p"][off].mean() - x["p_real"][off].mean())
                pdiffs.append(x["pp"][off].mean() - x["p_real"][off].mean())
        if diffs:
            out.update(diff=float(np.mean(diffs)), diff_projected=float(np.mean(pdiffs)))
        out.update(real_on_share=float(np.mean([x["on"].mean() for x in rs])),
                   dream_p_on=float(np.mean([x["p"].mean() for x in rs])),
                   real_on_last20=float(np.mean([x["on"][-20:].mean() for x in rs])),
                   dream_p_on_last20=float(np.mean([x["p"][-20:].mean() for x in rs])))
        track_rows.append(out)
    g = lambda k: np.array([r[k] for r in track_rows if k in r])
    summary = dict(
        tracks=len(track_rows), tracks_in_primary=len(g("diff")),
        replays_not_matching_log=sum(not r["replay_matches_log"] for r in track_rows),
        primary_dream_minus_real_latent=_stats(g("diff")),
        projected_dream_minus_real_latent=_stats(g("diff_projected")),
        window_share_on_road=dict(real_under_dream_actions=float(g("real_on_share").mean()),
                                  dream_mean_p_on_road=float(g("dream_p_on").mean())),
        last20_share_on_road=dict(real_under_dream_actions=float(g("real_on_last20").mean()),
                                  dream_mean_p_on_road=float(g("dream_p_on_last20").mean())))
    print("\n=== summary ===\n" + json.dumps(summary, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dream_actions_in_sim.json").write_text(json.dumps(dict(summary=summary, rows=track_rows), indent=1))
    print(f"wrote {OUT / 'dream_actions_in_sim.json'}")


if __name__ == "__main__":
    main()
