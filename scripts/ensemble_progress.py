"""Show member progress and a live ETA for ``ldr.train_ensemble``."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from itertools import pairwise
from pathlib import Path


def read_log(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{"wall_s": float(r["wall_s"]), "step": int(r["step"])}
                for r in csv.DictReader(f) if r.get("step") and r.get("wall_s")]


def recent_rate(rows):
    if len(rows) < 2:
        return None
    recent = rows[-5:]
    rates = [(b["wall_s"] - a["wall_s"]) / (b["step"] - a["step"])
             for a, b in pairwise(recent) if b["step"] > a["step"]]
    return statistics.median(rates) if rates else None


def render(directory: Path):
    cfg_path = directory / "training_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    n, target = int(cfg.get("members", 2)), int(cfg.get("steps", 20_000))
    logs = [read_log(directory / f"member_{i:02d}" / "log.csv") for i in range(n)]
    rates = [recent_rate(rows) for rows in logs]
    fallbacks = [r for r in rates if r]
    if not fallbacks:
        prior = read_log(Path("runs/mdnrnn/log.csv"))
        if prior:
            r = recent_rate(prior)
            if r:
                fallbacks.append(r)
    fallback = statistics.median(fallbacks) if fallbacks else 3600 / 20_000
    rates = [r or fallback for r in rates]
    member_remaining = []
    lines = []
    for i, (rows, rate) in enumerate(zip(logs, rates)):
        step = rows[-1]["step"] if rows else 0
        elapsed = rows[-1]["wall_s"] if rows else 0
        left = max(0, target - step) * rate
        member_remaining.append(left)
        log_path = directory / f"member_{i:02d}" / "log.csv"
        if step >= target:
            status = "complete"
        elif not rows:
            status = "queued"
        elif time.time() - log_path.stat().st_mtime < 150:
            status = "training"
        else:
            status = "no recent log update"
        lines.append(f"member {i + 1}/{n}: {step:>6,}/{target:,} steps  "
                     f"elapsed {elapsed / 60:5.1f} min  ETA {left / 60:5.1f} min  [{status}]")
    print("\n".join(lines))
    parallel = max(1, int(cfg.get("parallel_members", 1)))
    remaining = sum(max(member_remaining[i:i + parallel])
                    for i in range(0, len(member_remaining), parallel))
    print(f"estimated time remaining for ensemble: {remaining / 60:.1f} min "
          f"({remaining / 3600:.2f} h)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", default=Path("runs/mdnrnn_ensemble"))
    p.add_argument("--watch", action="store_true", help="refresh continuously until Ctrl-C")
    p.add_argument("--interval", type=int, default=30, help="refresh interval in seconds")
    a = p.parse_args()
    try:
        while True:
            render(a.directory)
            if not a.watch:
                break
            time.sleep(max(1, a.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
