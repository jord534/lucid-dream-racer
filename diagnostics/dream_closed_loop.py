"""Does the dream keep the car on the road when the controller steers inside it?

Follow-up to dream_fidelity.py, which fed the dream the real actions. Here the dream-trained
controller (runs/dream_v3) chooses the actions itself, as it did during training and in the
dream GIF.

Setup, per test track: take the real rollout logged by `dream_fidelity.py --collect`, and
start the world model at the step 30 before the real car first leaves the road. It is given
the full real history up to that step with exact (noise-free) inputs, so the dream starts in
exactly the state the real controller was in; the controller's action there is checked
against the real recorded action. From that state, two imagined futures of up to 100 steps
(16 samples each, temperature 1.15, the training temperature):
  open   - the real recorded actions (as in dream_fidelity.py)
  closed - the controller reads the dream's own latent and memory and steers.
The same cross-fitted probe as before reads P(on road) from each latent.

Primary comparison, fixed before any result was seen: over the steps where the real car is
off the road, per track, mean P(on road | closed-loop dream) minus mean P(on road | real
latent); median over tracks, bootstrap 95% CI over tracks, Wilcoxon p. Positive means the
dream shows the car on the road more than reality does when the controller drives it.
Secondary (also fixed beforehand): the same for closed minus open; the same three
comparisons after "projecting" each imagined latent through the VAE (decode, then encode),
which checks whether the probe reads dream latents badly only because they sit off the
real-latent distribution; predicted reward against real reward.

    python diagnostics/dream_closed_loop.py        (about 5 minutes)
Writes reports/dream_failure_analysis/dream_closed_loop.json and closed_loop_frames.png."""
import json
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

from dream_fidelity import FOLDS, HORIZON, LEAD, OUT, STORE, _ci, _first_run, _train_probe
from ldr.check_dream import load_mdnrnn
from ldr.config import C, RUNS
from ldr.controller import act_batched
from ldr.encode import load_vae
from ldr.mdnrnn import mdn_sample

TAU, B = 1.15, 16
SHEET_TRACKS, SHEET_STEPS = [0, 25, 60], [0, 20, 40, 60, 80]


def _stats(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), ci95=_ci(x), share_positive=float((x > 0).mean()),
                wilcoxon_p=float(wilcoxon(x).pvalue))


@torch.no_grad()
def main():
    torch.manual_seed(0)
    d = np.load(STORE)
    tracks = []
    for s, n, seed in zip(d["start"], d["length"], d["seed"]):
        sl = slice(int(s), int(s + n))
        tracks.append({k: d[k][sl] for k in ("mu", "logvar", "act", "rew", "onroad")})
        tracks[-1]["seed"] = int(seed)
    rng = np.random.default_rng(0)                       # same draws as dream_fidelity.py
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
    project = lambda z: vae.encode(vae.decode(z))[0]
    theta = torch.load(RUNS / "dream_v3" / "best.pt", weights_only=False)["theta"]
    theta = torch.as_tensor(np.asarray(theta), dtype=torch.float32)[None].expand(B, -1)

    rows, sheet, t_start = [], {}, time.time()
    for i, tr in enumerate(tracks):
        onset = _first_run([not o for o in tr["onroad"]])
        if onset is None:
            continue
        T, t0 = len(tr["act"]), onset + 1 - LEAD
        W = min(HORIZON, T - 1 - t0)
        if t0 < 1 or W < 20:
            continue
        mu, act = torch.from_numpy(tr["mu"]), torch.from_numpy(tr["act"])
        *_, state = rnn(mu[None, :t0], act[None, :t0])           # exact real history
        state0 = tuple(s.expand(1, B, -1).contiguous() for s in state)
        z0 = mu[t0][None].expand(B, -1)
        check = float((act_batched(theta, z0, state0[0][0]) - act[t0]).abs().max())

        def run(closed):
            st, z, zs, rs, acs = tuple(s.clone() for s in state0), z0.clone(), [], [], []
            for j in range(W):
                a = act_batched(theta, z, st[0][0]) if closed else act[t0 + j][None].expand(B, -1)
                logit, m, ls, r, _, st = rnn.step(z, a, st)
                z = mdn_sample(logit, m, ls, TAU)
                zs.append(z), rs.append(r), acs.append(a)
            return torch.stack(zs), torch.stack(rs), torch.stack(acs)

        sl = slice(t0 + 1, t0 + 1 + W)
        real_on = lab[i][sl]
        off = ~real_on
        if off.sum() < 10:
            continue
        with torch.no_grad():
            p_real = prob(fold[i], torch.from_numpy(z_real[i]))[sl].numpy()
        out = dict(seed=tr["seed"], t0=t0, window=W, off_steps=int(off.sum()), action_check=check,
                   p_real_off=float(p_real[off].mean()), p_real_on=float(p_real[real_on].mean())
                   if real_on.any() else None,
                   reward_real=float(np.clip(tr["rew"][t0:t0 + W], -1, 10).sum()))
        for name, closed in (("open", False), ("closed", True)):
            zs, rs, acs = run(closed)
            flat = zs.reshape(-1, C.z_dim)
            p = prob(fold[i], flat).view(W, B).mean(1).numpy()
            pp = prob(fold[i], project(flat)).view(W, B).mean(1).numpy()
            out.update({f"p_{name}_off": float(p[off].mean()), f"pp_{name}_off": float(pp[off].mean()),
                        f"false_on_{name}": float((p[off] > .5).mean()),
                        f"reward_{name}": float(rs.mean(1).sum()),
                        f"gas_{name}": float(acs[..., 1].mean()),
                        f"steer_abs_diff_{name}": float((acs[..., 0].mean(1) - act[t0:t0 + W, 0]).abs().mean())})
            if closed:
                out["gas_real"] = float(act[t0:t0 + W, 1].mean())
            if i in SHEET_TRACKS:
                sheet.setdefault(i, dict(t0=t0, W=W, real_on=real_on, p_real=p_real))[name] = (zs[:, 0], p)
        rows.append(out)
        print(f"[{len(rows):3d}] seed {tr['seed']}  real off-road steps {out['off_steps']:3d}  "
              f"P(on road) there: real {out['p_real_off']:.2f}  dream replaying real actions "
              f"{out['p_open_off']:.2f}  dream steered by controller {out['p_closed_off']:.2f}  "
              f"({(time.time() - t_start) / 60:.1f} min)", flush=True)

    g = lambda k: np.array([r[k] for r in rows])
    summary = dict(
        tracks=len(rows), max_action_check=float(g("action_check").max()),
        primary_closed_minus_real=_stats(g("p_closed_off") - g("p_real_off")),
        closed_minus_open=_stats(g("p_closed_off") - g("p_open_off")),
        open_minus_real=_stats(g("p_open_off") - g("p_real_off")),
        projected_closed_minus_real=_stats(g("pp_closed_off") - g("p_real_off")),
        projected_open_minus_real=_stats(g("pp_open_off") - g("p_real_off")),
        mean_p_off=dict(real=float(g("p_real_off").mean()), open=float(g("p_open_off").mean()),
                        closed=float(g("p_closed_off").mean()),
                        open_projected=float(g("pp_open_off").mean()),
                        closed_projected=float(g("pp_closed_off").mean())),
        share_of_off_steps_dream_says_on_road=dict(open=float(g("false_on_open").mean()),
                                                   closed=float(g("false_on_closed").mean())),
        median_reward=dict(real=float(np.median(g("reward_real"))), open=float(np.median(g("reward_open"))),
                           closed=float(np.median(g("reward_closed")))),
        mean_gas=dict(real=float(g("gas_real").mean()), closed=float(g("gas_closed").mean())),
        mean_abs_steer_difference_controller_in_dream_vs_real=float(g("steer_abs_diff_closed").mean()))
    print("\n=== summary ===\n" + json.dumps(summary, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dream_closed_loop.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))

    cell, lw, rows_lab = 3 * 64, 150, ["real frame", "dream, real actions", "dream, controller steers"]
    keys = sorted(sheet)
    img = Image.new("RGB", (lw + cell * len(SHEET_STEPS), (cell * 3 + 18) * len(keys)), "white")
    dr = ImageDraw.Draw(img)
    for gi, i in enumerate(keys):
        s, y0 = sheet[i], gi * (cell * 3 + 18)
        dr.text((4, y0 + 3), f"seed {tracks[i]['seed']}", fill="black")
        for c, j in enumerate(SHEET_STEPS):
            dr.text((lw + c * cell + 4, y0 + 3), f"step +{j + 1}: real car "
                    f"{'on' if s['real_on'][j] else 'OFF'} road", fill="red" if not s["real_on"][j] else "black")
        ok = [c for c, j in enumerate(SHEET_STEPS) if j < s["W"]]
        steps = [SHEET_STEPS[c] for c in ok]
        real = vae.decode(torch.from_numpy(tracks[i]["mu"][[s["t0"] + 1 + j for j in steps]]))
        dream = {n: vae.decode(s[n][0][steps]) for n in ("open", "closed")}
        for r, (lab_, ims, ps) in enumerate([(rows_lab[0], real, s["p_real"]),
                                              (rows_lab[1], dream["open"], s["open"][1]),
                                              (rows_lab[2], dream["closed"], s["closed"][1])]):
            dr.text((4, y0 + 18 + r * cell + cell // 2), lab_, fill="black")
            for k, c in enumerate(ok):
                j = SHEET_STEPS[c]
                im = Image.fromarray((ims[k].permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy())
                img.paste(im.resize((cell, cell), Image.NEAREST), (lw + c * cell, y0 + 18 + r * cell))
                dr.rectangle([lw + c * cell, y0 + 18 + r * cell, lw + c * cell + 78, y0 + 30 + r * cell], fill="black")
                dr.text((lw + c * cell + 3, y0 + 18 + r * cell + 1), f"P(on) {ps[j]:.2f}", fill="yellow")
    img.save(OUT / "closed_loop_frames.png")
    print(f"wrote {OUT / 'dream_closed_loop.json'} and {OUT / 'closed_loop_frames.png'}")


if __name__ == "__main__":
    main()
