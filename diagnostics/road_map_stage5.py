"""Stage 5 of the road-map experiment: controllers trained inside each model (reports/road_map/preregistration.md).

    python diagnostics/road_map_stage5.py setup NAME MODEL_DIR     (round-0 folder for ldr.loop --start-from)
    python diagnostics/road_map_stage5.py summarise                (after the loop runs and the test evaluations)

For each model: ldr.loop round 0 (three controllers, 100 generations, measured on validation tracks
900,100-900,149) in runs/road_map/stage5/NAME_loop, and each controller on the 100 test tracks
(reports/eval_roadmap_NAME_p{0,1,2}.json). The summary gives, per model, the validation exploit gap,
the dream's P(on road) at the steps where the same actions leave the real car off the road, real
return, and the pooled test IQM with its 95% bootstrap interval; and the preregistered criteria for
the road-map models. Writes reports/road_map/stage5.json."""

import json
import os
import sys
from pathlib import Path

import numpy as np

from ldr.analysis import bootstrap_ci, iqm
from ldr.config import REPORTS, ROOT, RUNS
from ldr.progress import result

S5 = RUNS / "road_map" / "stage5"
MAPS, CONTROLS = ["map_s0", "map_s1"], ["ctrl_s0", "ctrl_s1"]


def setup(name, model_dir):
    d = S5 / name
    d.mkdir(parents=True, exist_ok=True)
    for link, target in (("vae", RUNS / "vae"), ("mdnrnn", Path(model_dir).resolve())):
        if not (d / link).exists():
            os.symlink(target, d / link)
    print(f"{d}: vae -> {RUNS / 'vae'}, mdnrnn -> {Path(model_dir).resolve()}")


def summarise():
    rows = {}
    for name in MAPS + CONTROLS:
        mfile = S5 / f"{name}_loop" / "r00" / "measure.json"
        evals = [REPORTS / f"eval_roadmap_{name}_p{s}.json" for s in range(3)]
        if not mfile.exists() or not all(e.exists() for e in evals):
            continue
        m = json.loads(mfile.read_text())
        rets = [np.array(json.loads(e.read_text())["returns"]) for e in evals]
        pooled = np.concatenate(rets)
        rows[name] = {
            "validation_real_return": m["real_return_mean"], "validation_real_return_by_driver": m["real_return_by_proposer"],
            "exploit_gap": m["gap_mean"], "dream_p_on_when_real_off": m.get("p_on_when_real_off_mean"),
            "road_gap": m.get("road_gap_mean"), "real_on_road_share": m["real_on_road_share_mean"],
            "test_iqm_pooled": [iqm(pooled), *bootstrap_ci(pooled, iqm)],
            "test_iqm_by_driver": [[iqm(r), *bootstrap_ci(r, iqm)] for r in rets],
            "test_mean_pooled": float(pooled.mean())}
    crit = {}
    for name in MAPS:
        r = rows.get(name)
        if r is None:
            continue
        p_off = r["dream_p_on_when_real_off"]
        crit[name] = {"exploit gap <= 60": r["exploit_gap"] <= 60,
                      "dream P(on road) where the real car is off <= 0.50": p_off is not None and p_off <= 0.50,
                      "test IQM interval above 209.6": r["test_iqm_pooled"][1] > 209.6}
    passed = bool(crit) and len(crit) == len(MAPS) and all(all(c.values()) for c in crit.values())
    out = {"models": rows, "criteria": crit, "passed": passed,
           "baselines": {"original dream-trained drivers, test IQM": 209.6, "after two repair rounds": 710.3,
                         "simulator-trained": 924.8, "original model, exploit gap": 90}}
    (REPORTS / "road_map" / "stage5.json").write_text(json.dumps(out, indent=1))
    for name, r in rows.items():
        t = r["test_iqm_pooled"]
        p = r["dream_p_on_when_real_off"]
        result(f"{name}: exploit gap {r['exploit_gap']:.1f}, dream P(on road) where the real car is off "
               f"{'-' if p is None else f'{p:.2f}'}, validation return {r['validation_real_return']:.0f}, "
               f"test IQM {t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]")
    for name, c in crit.items():
        for k, v in c.items():
            result(f"{'PASS' if v else 'FAIL'}  {name}: {k}")
    result(f"STAGE 5 {'PASSED' if passed else 'FAILED'}  (reports/road_map/stage5.json)")


if __name__ == "__main__":
    os.chdir(ROOT)
    {"setup": lambda: setup(*sys.argv[2:4]), "summarise": summarise}[sys.argv[1]]()
