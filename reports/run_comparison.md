# Comparison across runs

Made by `python diagnostics/compare_runs.py` from the files in `reports/`. Test return: the three
last-round drivers of each run on the 100 test tracks (the original controller is a single one).
Closed-loop and in-simulator columns: median over tracks of the dream's P(on road) minus the real car's,
over the steps where the real car is off the road (0 would be agreement; one value per driver, then the
mean). One run per group and three drivers each: differences smaller than the spread between drivers are
not evidence of anything.

| Group | Test return per driver | Mean | Closed-loop road disagreement per driver | Mean | Dream actions in the simulator, per driver | Mean |
|---|---|---|---|---|---|---|
| original dream controller | 252 | 252 | 0.83 | 0.83 | 0.82 | 0.82 |
| first loop | 784, 616, 493 | 631 | 0.61, 0.61, 0.32 | 0.51 | 0.40, 0.64, 0.36 | 0.47 |
| road-ranked loop | 653, 595, 755 | 668 | 0.38, 0.42, 0.52 | 0.44 | 0.37, 0.52, 0.68 | 0.52 |

Validation measurements per round (50 validation tracks, 3 drivers; rounds of consecutive runs are
placed end to end; the road columns exist from the road-ranked run on; head accuracy only for runs with an
on-road head):

| Run | Cumulative round | Real return | Reward gap | Road gap | Dream says on road while real car is off | Head accuracy on branches |
|---|---|---|---|---|---|---|
| first loop | 0 | 270 | 90.4 | - | - | - |
| first loop | 1 | 593 | 70.8 | - | - | - |
| first loop | 2 | 640 | 57.4 | - | - | - |
| road-ranked loop | 2 | 640 | 56.4 | 0.27 | 0.74 | - |
| road-ranked loop | 3 | 699 | 51.6 | 0.23 | 0.78 | - |
| road-ranked loop | 4 | 688 | 48.2 | 0.19 | 0.71 | - |
