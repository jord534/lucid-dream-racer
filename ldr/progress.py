"""Machine-readable progress lines for long runs, read by scripts/progress_watch.py.

    PLAN <task>=<estimated seconds> ...       (once, at the top of a run's log)
    PROGRESS <unix time> <done> <total> <task>
    RESULT <text>                              (shown at the end of the watch view)
"""

import time

_last: dict = {}


def plan(tasks: dict) -> None:
    print("PLAN " + " ".join(f"{k.replace(' ', '_')}={int(v)}" for k, v in tasks.items()), flush=True)


def progress(task: str, done: int, total: int, every: float = 2.0) -> None:
    """Print a progress line, at most every `every` seconds per task (always the first and the last)."""
    now = time.time()
    if done in (0, total) or now - _last.get(task, 0) >= every:
        _last[task] = now
        print(f"PROGRESS {now:.1f} {done} {total} {task.replace(' ', '_')}", flush=True)


def result(text: str) -> None:
    print(f"RESULT {text}", flush=True)
