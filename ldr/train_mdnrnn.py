"""Phase 3: train the MDN-RNN on encoded sequences.

    python -m ldr.train_mdnrnn --steps 20000
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .check_dream import load_mdnrnn
from .config import DATA, RUNS, C
from .mdnrnn import MDNRNN, POSE_DIM, mdn_nll, mdn_sample, pose_features
from .progress import progress
from .utils import CSVLogger, get_device, load_ckpt, save_ckpt, seed_everything

ONSET_AT = 16  # an onset sits at least this many steps into a training window and this far from its end


class SequenceSampler:
    def __init__(self, path: Path = DATA / "latents.npz", L: int = C.seq_len, pose: bool = False,
                 road_map: bool = False):
        """pose=True also loads the exact pose labels (ldr.label_pose sidecar files) for with_pose;
        road_map=True the road patch and motion labels (ldr.label_patch) for with_map."""
        d = np.load(path)
        self.mu, self.logvar = d["mu"], d["logvar"]
        self.act, self.rew, self.done = d["actions"], d["rewards"], d["dones"]
        # on-road flag after each action: 1 on, 0 off, -1 unknown. Branches carry it in the npz;
        # the original data in a sidecar file written by ldr.label_road.
        side = Path(path).with_name(Path(path).stem + "_onroad.npy")
        if "onroad" in d.files:
            self.on = d["onroad"].astype(np.int8)
        elif side.exists():
            self.on = np.load(side).astype(np.int8)
        else:
            self.on = np.full(len(self.act), -1, np.int8)
        self.pose = None
        if pose:
            from .label_pose import load_pose_obs
            raw = load_pose_obs(path, d)
            assert raw is not None, f"{path} has no pose labels: run python -m ldr.label_pose"
            self.pose = pose_features(raw).astype(np.float32)   # indexed like mu
        self.patch = self.motion = None
        if road_map:
            from .road_map import MOTION_SCALE
            side = lambda s: Path(path).with_name(Path(path).stem + s)
            assert side("_patch.npy").exists(), f"{path} has no road-patch labels: run python -m ldr.label_patch"
            self.patch = np.load(side("_patch.npy"))                        # indexed like mu
            self.motion = (np.load(side("_motion.npy")) / MOTION_SCALE).astype(np.float32)
        self.L = L
        onset = d["onset"] if "onset" in d.files else np.full(len(d["ep_len"]), -1)
        starts = {"train": [], "val": []}
        near = {"train": [], "val": []}  # windows that contain an off-road onset (see below)
        for e, T in enumerate(d["ep_len"]):
            if T < L:
                continue
            t = np.arange(T - L + 1)
            pair = np.stack([d["obs_start"][e] + t, d["act_start"][e] + t], 1)
            split = "val" if d["is_val"][e] else "train"
            starts[split].append(pair)
            if onset[e] >= 0:  # onset falls between steps ONSET_AT and L - ONSET_AT of the window
                ok = (t <= onset[e] - ONSET_AT) & (t >= onset[e] - (L - ONSET_AT))
                if ok.any():
                    near[split].append(pair[ok])
        empty = np.zeros((0, 2), int)
        cat = lambda v: np.concatenate(v) if v else empty
        self.starts = {k: cat(v) for k, v in starts.items()}
        self.near_onset = {k: cat(v) for k, v in near.items()}

    def sample(self, batch, split, rng, device, onset_frac=0.0, with_on=False, with_pose=False, with_map=False):
        """onset_frac: share of the batch drawn from windows around an off-road onset (only
        episodes that record one, i.e. branches collected with on-road flags).
        with_pose appends two tensors (B,L,4) of pose features: the pose at each input observation
        (q_t) and after each action (q_{t+1}, the pose head's target).
        with_map appends the road patch at each input observation (B,L,MAP_DIM) and the scaled motion
        over each step (B,L,3), the motion head's target."""
        pool = self.near_onset[split]
        n_on = round(batch * onset_frac) if len(pool) else 0
        s = self.starts[split][rng.integers(len(self.starts[split]), size=batch - n_on)]
        if n_on:
            s = np.concatenate([s, pool[rng.integers(len(pool), size=n_on)]])
        ar = np.arange(self.L)
        o = s[:, :1] + ar                       # obs indices t .. t+L-1
        a = s[:, 1:] + ar                       # action indices t .. t+L-1
        def z(ix):                              # sample z ~ N(mu, sigma): free augmentation
            mu, lv = torch.from_numpy(self.mu[ix]), torch.from_numpy(self.logvar[ix])
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        t = lambda x: torch.from_numpy(x).to(device)
        out = (z(o).to(device), t(self.act[a]), z(o + 1).to(device),
               t(np.clip(self.rew[a], -1, 10)), t(self.done[a].astype(np.float32)))
        if with_on:
            out = (*out, t(self.on[a].astype(np.float32)))
        if with_pose:
            assert self.pose is not None, "sampler built without pose=True"
            out = (*out, t(self.pose[o]), t(self.pose[o + 1]))
        if with_map:
            assert self.patch is not None, "sampler built without road_map=True"
            out = (*out, t(self.patch[o].astype(np.float32)), t(self.motion[o]))
        return out


def probe_prob(probe, logit, mu, logsig):
    """P(on road) under the model's predicted next-latent distribution, as judged by the frozen
    road probe: sum_k pi_k * sigmoid(probe(mu_k + sigma_k * eps)). Differentiable in every MDN
    output. Shared-mixture models only (one component for the whole frame)."""
    assert logit.dim() == mu.dim() - 1, "probe loss needs a shared-mixture world model"
    zk = (mu + logsig.exp() * torch.randn_like(mu)).transpose(-1, -2)   # (..., K, z)
    pk = torch.sigmoid(probe(zk)[..., 0])                               # (..., K)
    return (F.softmax(logit, -1) * pk).sum(-1)


def probe_bce(p, y, stats=None, key="probe"):
    """Cross-entropy of P(on road) against the exact flag y (1 on, 0 off, -1 unknown: ignored)."""
    known = y >= 0
    if not known.any():
        return p.sum() * 0
    p, y = p[known].clamp(1e-4, 1 - 1e-4), y[known]
    loss = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
    if stats is not None:
        off = y < 0.5
        stats.update({f"{key}_bce": loss.item(), f"{key}_acc": ((p > 0.5) == (y > 0.5)).float().mean().item(),
                      f"{key}_off_recall": (p[off] < 0.5).float().mean().item() if off.any() else float("nan")})
    return loss


def free_task_loss(model, batch, probe, warm, probe_weight=1.0, on_weight=1.0, r_weight=1.0,
                   stats: dict | None = None):
    """Free-running supervision of what matters for control, with no latent likelihood. The first
    `warm` inputs are recorded; after that the model is fed its own samples, as when a controller
    drives the dream. From then on it is scored only on quantities with exact labels: the on-road
    head, the road status read from its predicted latent by the frozen probe, and the reward.
    (The latent likelihood is left to the teacher-forced pass: scoring it against the real future
    from a drifted history only teaches smoothing, as the free-running run showed.)"""
    z, a, _zn, r, _d, on = batch
    B, L, _ = z.shape
    state, zin, outs = None, z[:, 0], []
    for t in range(L):
        step = model.step(zin, a[:, t], state, return_on=True)
        outs.append((step[0], step[1], step[2], step[3], step[6]))
        state = step[5]
        if t + 1 < L:
            zin = z[:, t + 1] if t + 1 < warm else mdn_sample(step[0], step[1], step[2], 1.0)
    logit, mu, logsig, r_hat, on_hat = [torch.stack(o, 1)[:, warm:] for o in zip(*outs)]
    r, on = r[:, warm:], on[:, warm:]
    known = on >= 0
    on_loss = (F.binary_cross_entropy_with_logits(on_hat[known], on[known]) if known.any()
               else on_hat.sum() * 0)
    pl = probe_bce(probe_prob(probe, logit, mu, logsig), on, stats, "free_probe")
    rl = F.mse_loss(r_hat, r)
    if stats is not None:
        stats.update(free_on_bce=on_loss.item(), free_r_mse=rl.item())
        if known.any():
            pred = on_hat[known] > 0
            stats["free_on_acc"] = (pred == (on[known] > 0.5)).float().mean().item()
    return probe_weight * pl + on_weight * on_loss + r_weight * rl


def loss_fn(model, batch, done_pos_weight=50.0, ss_prob: float = 0.0, on_weight: float = 0.0,
            stats: dict | None = None, free_warm: int = 0, probe=None, probe_weight: float = 0.0,
            pose_weight: float = 0.0, pose_ss_prob: float | None = None, map_weight: float = 0.0):
    """ss_prob > 0 is scheduled sampling: at each step, with that probability, the input
    latent is the model's own sample from the previous step instead of the recorded one.
    Teacher forcing alone never scores the model on its own outputs, which is exactly the
    regime the dream runs in. free_warm: the first free_warm inputs are always the recorded ones
    (the dream is started from a real warm-up); own samples are used only after that.
    Models with a pose head (model.pose_dim) need a batch sampled with_pose; pose_weight scales
    the head's squared error. With pose input, the pose fed in is the true one, or with
    probability pose_ss_prob (default: ss_prob; its own coin, independent of the latent's) the
    model's own prediction from the step before, detached, as in the dream.
    Road-map models (model.map_dim) need a batch sampled with_map: the true patch is fed at every
    step, and map_weight scales the motion head's squared error."""
    z, a, z_next, r, d, *rest = batch
    ron = on_weight > 0
    rp, pin, rmap = bool(model.pose_dim), model.pose_input, bool(model.map_dim)
    patch, m_next = rest[-2:] if rmap else (None, None)
    rest = rest[:-2] if rmap else rest
    q_in, q_next = rest[-2:] if rp else (None, None)
    pss = ss_prob if pose_ss_prob is None else pose_ss_prob
    if ss_prob <= 0 and (not pin or pss <= 0):
        logit, mu, logsig, r_hat, d_hat, _, *extra = model(z, a, return_on=ron, q=q_in if pin else None,
                                                           return_pose=rp, patch=patch, return_motion=rmap)
    else:
        B, L, _ = z.shape
        state, zin, qin, outs = None, z[:, 0], q_in[:, 0] if pin else None, []
        for t in range(L):
            step = model.step(zin, a[:, t], state, return_on=ron, q=qin, return_pose=rp,
                              patch=patch[:, t] if rmap else None, return_motion=rmap)
            outs.append(step[:5] + step[6:])
            state = step[5]
            if t + 1 < L:
                own = mdn_sample(step[0], step[1], step[2], 1.0)
                use = (torch.rand(B, 1, device=z.device) < ss_prob).float()
                use = use * float(t + 1 >= free_warm)
                zin = use * own + (1 - use) * z[:, t + 1]
                if pin:
                    useq = (torch.rand(B, 1, device=z.device) < pss).float() * float(t + 1 >= free_warm)
                    qin = useq * step[6 + ron].detach() + (1 - useq) * q_in[:, t + 1]
        logit, mu, logsig, r_hat, d_hat, *extra = [torch.stack(o, 1) for o in zip(*outs)]
    on_hat = extra[:1] if ron else []
    pose_hat = extra[int(ron)] if rp else None
    motion_hat = extra[-1] if rmap else None
    nll = mdn_nll(logit, mu, logsig, z_next)
    r_loss = F.mse_loss(r_hat, r)
    d_loss = F.binary_cross_entropy_with_logits(
        d_hat, d, pos_weight=torch.tensor(done_pos_weight, device=d.device))
    total = nll + r_loss + d_loss
    if rp:
        pose_loss = F.mse_loss(pose_hat, q_next)
        total = total + pose_weight * pose_loss
        if stats is not None:
            stats.update(pose_mse=pose_loss.item(),
                         pose_lat_mae=(pose_hat[..., 0] - q_next[..., 0]).abs().mean().item() * 2)
    if rmap:
        motion_loss = F.mse_loss(motion_hat, m_next)
        total = total + map_weight * motion_loss
        if stats is not None:
            stats.update(motion_mse=motion_loss.item())
    if probe is not None and rest:
        pl = probe_bce(probe_prob(probe, logit, mu, logsig), rest[0], stats)
        total = total + probe_weight * pl
    if ron:
        on = rest[0]
        known = on >= 0
        if known.any():
            lg, y = on_hat[0][known], on[known]
            on_loss = F.binary_cross_entropy_with_logits(lg, y)
            total = total + on_weight * on_loss
            if stats is not None:
                pred, off = lg > 0, y < 0.5
                stats.update(on_bce=on_loss.item(), on_acc=(pred == (y > 0.5)).float().mean().item(),
                             off_recall=(~pred[off]).float().mean().item() if off.any() else float("nan"))
    return total, nll, r_loss, d_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ss-prob", type=float, default=0.0,
                   help="final scheduled-sampling probability, ramped in over the run")
    p.add_argument("--shared-mixture", action="store_true",
                   help="draw one mixture component per frame instead of per dimension")
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm start from this checkpoint (shared/per-dimension is read from it)")
    p.add_argument("--extra", type=Path, default=None,
                   help="npz of extra episodes in the latents.npz format (e.g. real branches)")
    p.add_argument("--extra-frac", type=float, default=0.5,
                   help="share of every training batch drawn from --extra")
    p.add_argument("--seq-len", type=int, default=C.seq_len,
                   help="training window length (long windows let own samples compound)")
    p.add_argument("--free-warm", type=int, default=0,
                   help="steps at the start of every window that always use the recorded inputs")
    p.add_argument("--probe", type=Path, default=None,
                   help="frozen road probe (ldr.road_probe): task loss on the predicted latent")
    p.add_argument("--probe-weight", type=float, default=1.0,
                   help="weight of the probe task loss in the teacher-forced pass")
    p.add_argument("--free-task-weight", type=float, default=0.0,
                   help="weight of the extra free-running pass scored on road status and reward only")
    p.add_argument("--free-batch", type=int, default=16)
    p.add_argument("--free-len", type=int, default=120)
    p.add_argument("--on-weight", type=float, default=0.0,
                   help="weight of the on-road head loss (0: head not trained)")
    p.add_argument("--extra-onset-frac", type=float, default=0.0,
                   help="share of the --extra part of every batch centred on an off-road onset")
    p.add_argument("--pose-weight", type=float, default=0.0,
                   help="weight of a pose head (ldr.label_pose: lateral offset, heading, speed after "
                        "the action); 0: no pose head")
    p.add_argument("--pose-input", action="store_true",
                   help="also feed the pose to the LSTM: the true one in training, the model's own "
                        "prediction in the dream (needs --pose-weight)")
    p.add_argument("--pose-ss-prob", type=float, default=None,
                   help="final probability of feeding the model's own predicted pose instead of the "
                        "true one, ramped like --ss-prob (default: --ss-prob)")
    p.add_argument("--map-weight", type=float, default=0.0,
                   help="road map (ldr.road_map): weight of the motion head; 0: no road map")
    p.add_argument("--map-input", action="store_true",
                   help="feed the road patch read off the map to the LSTM (needs --map-weight)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=RUNS / "mdnrnn")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    dev = get_device(args.device)
    assert args.pose_weight > 0 or not args.pose_input, "--pose-input needs --pose-weight"
    wp = args.pose_weight > 0
    assert (args.map_weight > 0) == args.map_input, "--map-weight and --map-input go together"
    wm = args.map_input
    data = SequenceSampler(L=args.seq_len, pose=wp, road_map=wm)
    extra = SequenceSampler(args.extra, L=args.seq_len, pose=wp, road_map=wm) if args.extra else None
    if wm:
        from .road_map import MAP_DIM
    model = (load_mdnrnn(args.init_from, dev) if args.init_from
             else MDNRNN(shared=args.shared_mixture, pose_dim=POSE_DIM if wp else 0,
                         pose_input=args.pose_input, map_dim=MAP_DIM if wm else 0).to(dev)).train()
    assert bool(model.pose_dim) == wp and model.pose_input == args.pose_input, \
        "pose flags do not match the --init-from checkpoint"
    assert bool(model.map_dim) == wm, "road-map flags do not match the --init-from checkpoint"

    probe = None
    if args.probe:
        from .road_probe import load_probe
        probe = load_probe(args.probe, dev)
    won = args.on_weight > 0 or probe is not None   # the probe loss needs the on-road labels
    if args.free_task_weight > 0:
        assert not wp and not wm, "the free-running pass does not carry the pose or the road map"
        assert probe is not None and args.on_weight > 0, "free-running pass needs --probe and --on-weight"
        data_f = SequenceSampler(L=args.free_len)
        extra_f = SequenceSampler(args.extra, L=args.free_len) if args.extra else None
    trained_on = args.on_weight > 0 or (args.init_from is not None
                                        and bool(load_ckpt(args.init_from, "cpu").get("on_trained", False)))

    def draw(split, n, d=None, x=None):
        d, x = (data, extra) if d is None else (d, x)
        if x is None or not len(x.starts[split]):
            return d.sample(n, split, rng, dev, with_on=won, with_pose=wp, with_map=wm)
        k = round(n * args.extra_frac)
        on = args.extra_onset_frac if split == "train" else 0.0
        a = d.sample(n - k, split, rng, dev, with_on=won, with_pose=wp, with_map=wm)
        b = x.sample(k, split, rng, dev, on, with_on=won, with_pose=wp, with_map=wm)
        return tuple(torch.cat([u, v]) for u, v in zip(a, b))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    start, ck = 0, args.out / "last.pt"
    if ck.exists():
        s = load_ckpt(ck, dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); start = s["step"]
    log, rng, best = CSVLogger(args.out / "log.csv"), np.random.default_rng(args.seed + start), 1e9

    t_start = time.time()
    ptask = os.environ.get("LDR_PROGRESS_TASK")   # progress lines for scripts/progress_watch.py
    if ptask:
        progress(ptask, start, args.steps)
    for step in range(start, args.steps):
        ramp = min(1.0, 2 * step / args.steps)   # ramp in over the first half
        ss = args.ss_prob * ramp
        pss = None if args.pose_ss_prob is None else args.pose_ss_prob * ramp
        tstats = {}
        loss, nll, rl, dl = loss_fn(model, draw("train", args.batch), ss_prob=ss,
                                    on_weight=args.on_weight, stats=tstats,
                                    free_warm=args.free_warm, probe=probe,
                                    probe_weight=args.probe_weight,
                                    pose_weight=args.pose_weight, pose_ss_prob=pss,
                                    map_weight=args.map_weight)
        if args.free_task_weight > 0:
            loss = loss + args.free_task_weight * free_task_loss(
                model, draw("train", args.free_batch, data_f, extra_f), probe, args.free_warm,
                stats=tstats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # LSTMs spike
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vstats = {}
                vl, vn, vr, vd = loss_fn(model, data.sample(256, "val", rng, dev, with_on=won,
                                                         with_pose=wp, with_map=wm),
                                         on_weight=args.on_weight, stats=vstats, probe=probe,
                                         probe_weight=args.probe_weight, pose_weight=args.pose_weight,
                                         map_weight=args.map_weight)
                xstats = {}
                if won and extra is not None and len(extra.starts["val"]):
                    loss_fn(model, extra.sample(256, "val", rng, dev, with_on=True, with_pose=wp, with_map=wm),
                            on_weight=args.on_weight, stats=xstats)
                # validation always teacher-forced, so the number stays comparable across runs
                ve = (loss_fn(model, extra.sample(256, "val", rng, dev, with_pose=wp, with_map=wm))[1].item()
                      if extra is not None and len(extra.starts["val"]) else float("nan"))
            model.train()
            pose_cols = (dict(pose_mse=tstats.get("pose_mse", float("nan")),
                              val_pose_mse=vstats.get("pose_mse", float("nan")),
                              val_pose_lat_mae=vstats.get("pose_lat_mae", float("nan"))) if wp else {})
            if wm:
                pose_cols.update(motion_mse=tstats.get("motion_mse", float("nan")),
                                 val_motion_mse=vstats.get("motion_mse", float("nan")))
            log.log(step=step + 1, nll=nll.item(), r_mse=rl.item(), d_bce=dl.item(),
                    val_nll=vn.item(), val_r_mse=vr.item(), val_d_bce=vd.item(), val_nll_extra=ve,
                    on_bce=tstats.get("on_bce", float("nan")),
                    val_on_acc=vstats.get("on_acc", float("nan")),
                    val_on_off_recall=vstats.get("off_recall", float("nan")),
                    val_on_acc_extra=xstats.get("on_acc", float("nan")),
                    val_on_off_recall_extra=xstats.get("off_recall", float("nan")),
                    probe_bce=tstats.get("probe_bce", float("nan")),
                    val_probe_acc=vstats.get("probe_acc", float("nan")),
                    val_probe_off_recall=vstats.get("probe_off_recall", float("nan")),
                    free_on_acc=tstats.get("free_on_acc", float("nan")),
                    free_probe_acc=tstats.get("free_probe_acc", float("nan")),
                    free_probe_off_recall=tstats.get("free_probe_off_recall", float("nan")),
                    **pose_cols)
            eta = (time.time() - t_start) / (step + 1 - start) * (args.steps - step - 1)
            print(f"step {step + 1}/{args.steps}  nll {nll.item():.3f}  "
                  f"val nll {vn.item():.3f}  val r_mse {vr.item():.3f}  "
                  + (f"probe acc {vstats.get('probe_acc', float('nan')):.3f} (off-road recall "
                     f"{vstats.get('probe_off_recall', float('nan')):.2f}; free-running pass "
                     f"{tstats.get('free_probe_acc', float('nan')):.2f}/"
                     f"{tstats.get('free_probe_off_recall', float('nan')):.2f})  " if probe is not None else "")
                  + (f"on-road acc {vstats.get('on_acc', float('nan')):.3f} (branches "
                     f"{xstats.get('on_acc', float('nan')):.3f}, off-road recall "
                     f"{xstats.get('off_recall', float('nan')):.2f})  " if won else "")
                  + (f"pose mse {vstats.get('pose_mse', float('nan')):.4f} (lateral error "
                     f"{vstats.get('pose_lat_mae', float('nan')):.3f} half-widths)  " if wp else "")
                  + (f"motion mse {vstats.get('motion_mse', float('nan')):.4f} (scaled)  " if wm else "")
                  + f"about {eta / 60:.1f} min left", flush=True)
            payload = dict(model=model.state_dict(), opt=opt.state_dict(),
                           sched=sched.state_dict(), step=step + 1, on_trained=trained_on)
            if wp:
                payload["pose_trained"] = True
            if wm:
                payload.update(map_trained=True, map_dim=model.map_dim)
            save_ckpt(ck, **payload)
            sel = vl.item() if np.isnan(ve) else 0.5 * (vn.item() + ve)   # with extra, judge both
            if ptask:
                progress(ptask, step + 1, args.steps)
            if sel < best:
                best = sel
                save_ckpt(args.out / "best.pt", **payload)


if __name__ == "__main__":
    main()
