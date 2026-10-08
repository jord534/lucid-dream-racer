"""Follow a long run's log: one line per task with its progress and ETA, a total ETA, and the results.

    python scripts/progress_watch.py runs/road_map/stages.log [--every 5] [--once]

Reads the PLAN / PROGRESS / RESULT lines written through ldr/progress.py. Tasks not started yet are
estimated from the plan, corrected by how the finished tasks compared with their own estimates.
Stops by itself when the run has finished or failed."""

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path


def parse(text):
    plan, prog, results, events = {}, {}, [], {}
    for line in text.splitlines():
        if line.startswith("PLAN "):
            for item in line[5:].split():
                k, v = item.rsplit("=", 1)
                plan[k] = float(v)                      # a later plan refines an earlier one
        elif line.startswith("PROGRESS "):
            _, ts, done, total, task = line.split(maxsplit=4)
            p = prog.setdefault(task, {"t0": float(ts), "done": 0, "total": int(total)})
            p.update(t=float(ts), done=int(done), total=int(total))
            plan.setdefault(task, 0.0)
        elif line.startswith("RESULT "):
            results.append(line[7:])
        elif line.startswith(("START ", "EXIT ")):
            parts = line.split()
            events[parts[0]] = (float(parts[1]), int(parts[2]) if len(parts) > 2 and parts[0] == "EXIT" else 0)
    return plan, prog, results, events


def fmt(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def render(path):
    text = Path(path).read_text(errors="ignore")
    plan, prog, results, ev = parse(text)
    now = time.time()
    finished = "EXIT" in ev and ev["EXIT"][1] == 0
    failed = "EXIT" in ev and ev["EXIT"][1] != 0
    # how the finished tasks compare with their estimates (median ratio), to correct the rest
    ratios = sorted((p["t"] - p["t0"]) / plan[k] for k, p in prog.items()
                    if p["done"] >= p["total"] and plan.get(k, 0) > 5 and p["t"] - p["t0"] > 1)
    corr = ratios[len(ratios) // 2] if ratios else 1.0
    rows, remaining = [], 0.0
    for k, est in plan.items():
        p = prog.get(k)
        if p and p["done"] >= p["total"]:
            rows.append(f"  ✓ {k:42s} {p['total']:>5}/{p['total']:<5}  took {fmt(p['t'] - p['t0'])}")
        elif p and p["done"] > 0:
            el = now - p["t0"]
            left = el / p["done"] * (p["total"] - p["done"])
            remaining += left
            rows.append(f"  ▶ {k:42s} {p['done']:>5}/{p['total']:<5}  {100 * p['done'] / p['total']:3.0f}%  ETA {fmt(left)}")
        elif p:
            left = est * corr
            remaining += max(0, left - (now - p["t0"]))
            rows.append(f"  ▶ {k:42s} {'starting':>11}         ETA ~{fmt(left)}")
        else:
            remaining += est * corr
            rows.append(f"  · {k:42s} {'waiting':>11}         est ~{fmt(est * corr)}")
    start = ev["START"][0] if "START" in ev else min((p["t0"] for p in prog.values()), default=os.path.getmtime(path))
    out = [f"{Path(path).name}   started {datetime.fromtimestamp(start):%H:%M}   elapsed {fmt(now - start)}"]
    if finished:
        out.append(f"FINISHED in {fmt(ev['EXIT'][0] - start)}")
    elif failed:
        out.append("STOPPED AT A GATE (a result, see below)" if ev["EXIT"][1] == 3
                   else f"STOPPED WITH AN ERROR (exit code {ev['EXIT'][1]}); last lines of the log below")
    else:
        out.append(f"ETA {fmt(remaining)}  (about {datetime.fromtimestamp(now + remaining):%H:%M})"
                   + (f"   estimates scaled x{corr:.2f} from finished tasks" if ratios else ""))
    out += [""] + rows
    if results:
        out += ["", "Results:"] + [f"  {r}" for r in results]
    if failed and ev["EXIT"][1] != 3:
        tail = [ln for ln in text.splitlines() if not ln.startswith(("PROGRESS", "PLAN"))][-12:]
        out += [""] + [f"  {ln}" for ln in tail]
    return "\n".join(out), finished, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="+", help="one or more logs, shown one after the other")
    p.add_argument("--every", type=float, default=5)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    while True:
        views, states = [], []
        for log in a.log:
            if not Path(log).exists():
                views.append(f"{Path(log).name}: not started yet")
                states.append((False, False))
                continue
            view, fin, fail = render(log)
            views.append(view)
            states.append((fin, fail))
        if not a.once:
            print("\033[2J\033[H", end="")
        print(("\n\n" + "-" * 78 + "\n\n").join(views), flush=True)
        if a.once or all(f for f, _ in states) or any(x for _, x in states):
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
