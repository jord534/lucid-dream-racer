"""Measure this machine and recommend settings for the propose-critic loop (ldr/loop.py).

    python -m ldr.calibrate        (about 3 to 4 minutes)

Run it on the machine that will run the loop, plugged in, with nothing heavy running beside it.
It records the hardware (CPU, cores, memory, GPU, power state) and times the three real workloads
the loop is made of:
  - real-simulator rollouts (the same code the loop uses) with 1 to all cores, to find how many
    workers are worth using;
  - dream CMA-ES training, one generation, on each available torch device;
  - MDN-RNN fine-tuning steps (with scheduled sampling, as the loop does), on each device.
From those it recommends a worker count and training device, and projects the time of every loop
stage at the default settings. Writes reports/calibration/latest.json (read by ldr.loop for its
default worker count and its time estimates) and a dated plain-language report beside it.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import torch

from . import branch
from .check_dream import load_mdnrnn
from .config import RUNS
from .controller import N_PARAMS
from .dream import DreamSim
from .loop import CALIBRATION, estimate_minutes
from .train_dream import dream_fitness
from .train_mdnrnn import SequenceSampler, loss_fn

# Must match the defaults of ldr.loop.
DEFAULT = SimpleNamespace(
    rounds=2,
    proposers=3,
    generations=100,
    tracks=40,
    futures=3,
    measure_tracks=50,
    finetune_steps=5000,
)
SIM_STEPS, FT_STEPS = 250, 20


def sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gb(rss):  # ru_maxrss is bytes on macOS and kilobytes on Linux
    return rss / (2**30 if sys.platform == "darwin" else 2**20)


def hardware():
    h = {
        "system": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }
    if sys.platform == "darwin":
        h["cpu"] = sh("sysctl -n machdep.cpu.brand_string")
        h["physical_cores"] = int(sh("sysctl -n hw.physicalcpu") or 0)
        h["performance_cores"] = int(sh("sysctl -n hw.perflevel0.physicalcpu") or 0)
        h["efficiency_cores"] = int(sh("sysctl -n hw.perflevel1.physicalcpu") or 0)
        h["ram_gb"] = round(int(sh("sysctl -n hw.memsize") or 0) / 2**30, 1)
        h["power"] = sh("pmset -g batt | head -1")
        gpu = sh(
            "system_profiler SPDisplaysDataType | grep -E 'Chipset Model|Total Number of Cores'"
        )
        h["gpu"] = " | ".join(x.strip() for x in gpu.splitlines())
    else:
        h["cpu"] = sh("lscpu | grep 'Model name' | sed 's/.*: *//'")
        h["ram_gb"] = round(int(sh("awk '/MemTotal/ {print $2}' /proc/meminfo") or 0) / 2**20, 1)
        h["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    h["torch_cuda"] = torch.cuda.is_available()
    h["torch_mps"] = torch.backends.mps.is_available()
    return h


def _init():
    branch.init_worker(RUNS / "vae" / "best.pt", RUNS / "mdnrnn" / "best.pt")


def _work(job):
    n = len(branch.rollout_log(job)["act"])
    return n, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def sim_sweep(counts):
    theta = np.zeros(N_PARAMS, np.float32)
    rate, rss = {}, 0
    for w in counts:
        with Pool(w, initializer=_init) as pool:
            pool.map(_work, [(theta, 10_000 + i, 20) for i in range(w)], chunksize=1)  # warm up
            t0 = time.time()
            res = pool.map(
                _work, [(theta, 20_000 + i, SIM_STEPS) for i in range(2 * w)], chunksize=1
            )
            dt = time.time() - t0
        rate[w] = sum(n for n, _ in res) / dt
        rss = max(rss, max(r for _, r in res))
        print(f"  {w:2d} workers: {rate[w]:7.0f} simulator steps per second", flush=True)
    return rate, gb(rss)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench_dream(dev):
    """Seconds per CMA-ES generation (population 64, 16 runs each, 200 steps), as in train_dream."""
    sim = DreamSim(load_mdnrnn(device=dev), dev)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, N_PARAMS)).astype(np.float32) * 0.1
    dream_fitness(sim, X, 16, 200, rng, dev, 40)  # warm up
    _sync(dev)
    t0 = time.time()
    for _ in range(2):
        dream_fitness(sim, X, 16, 200, rng, dev, 40)
    _sync(dev)
    return (time.time() - t0) / 2


def bench_finetune(dev, data):
    """Training steps per second (batch 64, sequence 64, scheduled sampling 0.3)."""
    model = load_mdnrnn(device=dev).train()
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    rng = np.random.default_rng(0)

    def step():
        loss = loss_fn(model, data.sample(64, "train", rng, dev), ss_prob=0.3)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    _sync(dev)
    t0 = time.time()
    for _ in range(FT_STEPS):
        step()
    _sync(dev)
    return FT_STEPS / (time.time() - t0)


def main():
    t_start = time.time()
    hw = hardware()
    print("hardware:", json.dumps(hw, indent=1), flush=True)
    cores = hw["logical_cores"]
    counts = sorted(
        {c for c in (1, 2, 4, 6, 8, 12, 14, 16, 24, 32, 48, 64) if c <= cores} | {cores}
    )
    print("\nsimulator rollouts:", flush=True)
    rate, worker_rss = sim_sweep(counts)
    allowed = {w: r for w, r in rate.items() if w <= max(1, cores - 2)}  # keep two cores free
    best = max(allowed.values())
    workers = min(w for w, r in allowed.items() if r >= 0.95 * best)

    devs = [torch.device("cpu")]
    devs += [torch.device("cuda")] if torch.cuda.is_available() else []
    devs += [torch.device("mps")] if torch.backends.mps.is_available() else []
    data = SequenceSampler()
    dream, ft = {}, {}
    print("\ntraining workloads:", flush=True)
    for dev in devs:
        dream[dev.type] = bench_dream(dev)
        ft[dev.type] = bench_finetune(dev, data)
        print(
            f"  {dev.type:5s}: dream generation {dream[dev.type]:5.2f} s, "
            f"fine-tune {ft[dev.type]:5.1f} steps/s",
            flush=True,
        )
    d = DEFAULT
    cost = {k: d.proposers * d.generations * dream[k] + d.finetune_steps / ft[k] for k in dream}
    device = min(cost, key=cost.get)

    rates = {
        "sim_steps_per_s": rate[workers],
        "dream_s_per_gen": dream[device],
        "finetune_steps_per_s": ft[device],
    }
    est = estimate_minutes(d, rates)
    total = (d.rounds + 1) * (est["proposers"] + est["measure"]) + d.rounds * (
        est["critic"] + est["finetune"]
    )
    notes = []
    if "Battery" in hw.get("power", ""):
        notes.append(
            "The machine was on battery power: plug it in, speed is usually lower on battery."
        )
    if workers * worker_rss + 4 > hw["ram_gb"]:
        notes.append("Workers may not fit in memory: lower --workers.")
    result = {
        "hardware": hw,
        "sim_steps_per_s_by_workers": rate,
        "worker_peak_rss_gb": round(worker_rss, 2),
        "dream_s_per_generation": dream,
        "finetune_steps_per_s": ft,
        "rates": rates,
        "recommended": {"workers": workers, "device": device},
        "projected_minutes_default_settings": est,
        "projected_total_hours_default": total / 60,
        "notes": notes,
        "calibration_seconds": time.time() - t_start,
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(result, indent=1))
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M")
    (CALIBRATION.parent / f"{stamp}.json").write_text(json.dumps(result, indent=1))
    report = report_text(result, d, stamp)
    (CALIBRATION.parent / f"{stamp}.md").write_text(report)
    print("\n" + report)


def report_text(r, d, stamp):
    hw, est = r["hardware"], r["projected_minutes_default_settings"]
    sweep = "\n".join(f"| {w} | {v:.0f} |" for w, v in r["sim_steps_per_s_by_workers"].items())
    train = "\n".join(
        f"| {k} | {r['dream_s_per_generation'][k]:.2f} | {r['finetune_steps_per_s'][k]:.1f} |"
        for k in r["dream_s_per_generation"]
    )
    cores = ", ".join(
        f"{k.replace('_', ' ')} {hw[k]}"
        for k in ("physical_cores", "performance_cores", "efficiency_cores")
        if k in hw
    )
    return f"""# Machine calibration, {stamp}

Measured by `python -m ldr.calibrate` on this machine. It times the real workloads of the
propose-critic loop and recommends settings; `python -m ldr.loop` reads `latest.json` for its
default worker count and its time estimates. Re-run it whenever the machine or its load changes.

**Machine.** {hw.get("cpu", "?")}; {hw["logical_cores"]} logical cores ({cores}); {hw["ram_gb"]} GB
RAM; GPU: {hw.get("gpu") or "none found"}; power: {hw.get("power", "unknown")}; torch {hw["torch"]}
(cuda {hw["torch_cuda"]}, mps {hw["torch_mps"]}).

**Real-simulator rollouts** (where the critic, measure and real-track checks spend their time):

| Workers | Simulator steps per second |
|---|---|
{sweep}

Peak memory of one worker: {r["worker_peak_rss_gb"]} GB.

**Training workloads** (what the proposers and the fine-tune spend their time on):

| Device | Seconds per dream CMA-ES generation | Fine-tune steps per second |
|---|---|---|
{train}

**Recommended:** `--workers {r["recommended"]["workers"]} --device {r["recommended"]["device"]}`
(fewest workers within 5% of the best rollout speed, using at most all cores but two so the
machine stays usable; the training device that gives the shortest projected training time).

**Projected time at the default settings** ({d.proposers} proposers x {d.generations} generations,
{d.tracks} critic tracks and {d.measure_tracks} measure tracks per proposer, {d.futures}
branches per start, {d.finetune_steps} fine-tune steps): proposers {est["proposers"]:.0f} min,
measure
{est["measure"]:.0f} min, critic {est["critic"]:.0f} min, fine-tune {est["finetune"]:.0f} min per
round; **about {r["projected_total_hours_default"]:.1f} hours for {d.rounds} rounds.**

**Caveats.** Short benchmarks (simulator {SIM_STEPS} steps per rollout, {FT_STEPS} training
steps), so the projection is an estimate: the loop replaces it with measured stage times as it
runs. It assumes an
average replay of 300 steps. Thermal throttling during a long run is not captured.
{chr(10).join("**Note:** " + n for n in r["notes"])}

Reproduce: `python -m ldr.calibrate`.
"""


if __name__ == "__main__":
    main()
