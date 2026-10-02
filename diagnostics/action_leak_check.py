"""Can the recent actions alone tell whether the car is on the road?

Tests the coupling idea from the dream-failure analysis: that the world model may have learned
"controller-style steering goes with road under the car". If actions alone predict on-road
status, there is a shortcut a model can take from actions to track, instead of tracking the
road in its memory. data_coverage.py showed the off-road data is plentiful, but that the only
controller recordings are almost all on the road.

Stage 1 (--collect): replay the same 180 training episodes sampled by data_coverage.py (60 per
recording policy; the recorded actions from the track seed reproduce each episode exactly) and
save the true per-step on-road labels, to data/dream_failure/coverage_replays.npz.

Stage 2 (--analyse): from only the last 10 actions (steer, gas, brake; the action taken at the
step and the 9 before it), predict whether the car was on the road at the step (no images, no
other state). A small network, scored on whole episodes it never saw (5-fold split by episode):
  - trained and scored within each recording policy
  - trained on all three policies pooled, scored per policy and overall
  - a control trained on shuffled labels (should be at chance, 0.5)
Then the pooled predictors are applied to the actions of the dream-trained controller on the
100 test tracks (its states are on the road only 38% of the time).

Fixed before any result was seen. Primary: held-out area under the curve (AUC) of the
within-policy predictors, with bootstrap 95% intervals over episodes (0.5 = actions say nothing).
Key comparison: the pooled predictors' mean P(on road) on the dream-trained controller's actions,
at the steps where that car is actually off the road, against the steps where it is on the road.
A high value at off-road steps would mean the actions mislead about the track, as a shortcut would.
This shows the shortcut exists in the data; it does not show the world model uses it.

    python diagnostics/action_leak_check.py --collect        (about 10 minutes, 6 workers)
    python diagnostics/action_leak_check.py --analyse        (about 3 minutes)
Writes reports/dream_failure_analysis/action_leak.json."""
import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch

from data_coverage import PER_TYPE, ROLLOUTS, WORKERS, _replay
from dream_fidelity import OUT, STORE, _auc
from ldr.config import DATA

STORE2 = DATA / "dream_failure" / "coverage_replays.npz"
K, FOLDS, BOOT = 10, 5, 200


def collect():
    groups = defaultdict(list)
    for ep in range(1400):
        groups[str(np.load(ROLLOUTS / f"ep_{ep:05d}.npz")["policy"])].append(ep)
    rng = np.random.default_rng(0)                         # same sample as data_coverage.py
    jobs = [e for v in {p: sorted(rng.choice(v, PER_TYPE, replace=False).tolist())
                        for p, v in groups.items()}.values() for e in v]
    res, t0 = [], time.time()
    print(f"replaying {len(jobs)} episodes, {WORKERS} workers", flush=True)
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(_replay, jobs, chunksize=1):
            res.append(r)
            eta = (time.time() - t0) / len(res) * (len(jobs) - len(res))
            print(f"[{len(res):3d}/{len(jobs)}] episode {r['ep']:4d} ({r['policy']:10s}) "
                  f"eta {eta / 60:4.1f} min", flush=True)
    res.sort(key=lambda r: r["ep"])
    print(f"replays matching recorded rewards: {sum(r['reward_matches'] for r in res)}/{len(res)}", flush=True)
    STORE2.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STORE2, ep=np.array([r["ep"] for r in res]),
                        policy=np.array([r["policy"] for r in res]),
                        length=np.array([len(r["on"]) for r in res]),
                        on=np.concatenate([r["on"] for r in res]))
    print(f"wrote {STORE2}", flush=True)


def _windows(A, on):
    L = np.r_[True, on[:-1]]                               # on-road status at the step (before it)
    idx = np.arange(K - 1, len(A))
    X = np.stack([A[idx - j] for j in range(K)], 1).reshape(len(idx), -1)
    return X.astype(np.float32), L[idx]


def _train(X, y):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(K * 3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    X, y = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(net(X)[:, 0], y).backward()
        opt.step()
    return net.eval()


def _predict(net, X):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(X))[:, 0]).numpy()


def _crossfit(Xs, ys, shuffle=False):
    """Held-out predictions for every episode, from models that never saw that episode."""
    fold = np.arange(len(Xs)) % FOLDS
    preds, models = [None] * len(Xs), []
    for f in range(FOLDS):
        tr = [i for i in range(len(Xs)) if fold[i] != f]
        y = np.concatenate([ys[i] for i in tr])
        if shuffle:
            y = np.random.default_rng(f).permutation(y)
        net = _train(np.concatenate([Xs[i] for i in tr]), y)
        models.append(net)
        for i in np.flatnonzero(fold == f):
            preds[i] = _predict(net, Xs[i])
    return preds, models


def _auc_ci(ps, ys):
    value = _auc(np.concatenate(ps), np.concatenate(ys))
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(BOOT):
        ix = rng.integers(0, len(ps), len(ps))
        boot.append(_auc(np.concatenate([ps[i] for i in ix]), np.concatenate([ys[i] for i in ix])))
    return dict(auc=float(value), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])])


def analyse():
    d = np.load(STORE2)
    starts = np.r_[0, np.cumsum(d["length"])[:-1]]
    Xs, ys, pol = [], [], []
    for ep, p, s, n in zip(d["ep"], d["policy"], starts, d["length"]):
        A = np.load(ROLLOUTS / f"ep_{int(ep):05d}.npz")["actions"]
        X, y = _windows(A, d["on"][int(s):int(s + n)])
        Xs.append(X), ys.append(y), pol.append(str(p))
    pol = np.array(pol)
    out = dict(window_actions=K, episodes=len(Xs),
               share_on_road=dict({p: float(np.concatenate([ys[i] for i in np.flatnonzero(pol == p)]).mean())
                                   for p in sorted(set(pol))}))
    print("share of steps on the road:", {k: round(v, 3) for k, v in out["share_on_road"].items()}, flush=True)

    out["within_policy_auc"] = {}
    for p in sorted(set(pol)):
        ix = np.flatnonzero(pol == p)
        preds, _ = _crossfit([Xs[i] for i in ix], [ys[i] for i in ix])
        out["within_policy_auc"][p] = _auc_ci(preds, [ys[i] for i in ix])
        print(f"within {p:10s}: AUC {out['within_policy_auc'][p]['auc']:.3f} "
              f"{[round(x, 3) for x in out['within_policy_auc'][p]['ci95']]}", flush=True)

    preds, models = _crossfit(Xs, ys)
    out["pooled_auc_overall"] = _auc_ci(preds, ys)
    out["pooled_auc_by_policy"] = {p: _auc_ci([preds[i] for i in np.flatnonzero(pol == p)],
                                              [ys[i] for i in np.flatnonzero(pol == p)])
                                   for p in sorted(set(pol))}
    sp, _ = _crossfit(Xs, ys, shuffle=True)
    out["control_shuffled_labels_auc"] = float(_auc(np.concatenate(sp), np.concatenate(ys)))
    print("pooled overall AUC", out["pooled_auc_overall"], "| shuffled-label control",
          round(out["control_shuffled_labels_auc"], 3), flush=True)

    t = np.load(STORE)
    tp, ty = [], []
    for s, n in zip(t["start"], t["length"]):
        X, y = _windows(t["act"][int(s):int(s + n)], t["onroad"][int(s):int(s + n)])
        tp.append(np.mean([_predict(m, X) for m in models], 0))
        ty.append(y)
    P, Y = np.concatenate(tp), np.concatenate(ty)
    out["dream_trained_controller_actions"] = dict(
        share_on_road=float(Y.mean()), mean_p_on_road_at_off_road_steps=float(P[~Y].mean()),
        mean_p_on_road_at_on_road_steps=float(P[Y].mean()),
        share_of_off_road_steps_predicted_on_road=float((P[~Y] > .5).mean()),
        auc=_auc_ci(tp, ty))
    print("dream-trained controller's actions:", json.dumps(out["dream_trained_controller_actions"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "action_leak.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'action_leak.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--analyse", action="store_true")
    a = p.parse_args()
    if a.collect:
        collect()
    if a.analyse:
        analyse()
