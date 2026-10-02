# Propose-critic loop results

Settings: rounds 2, proposers 3, dream generations 100,
critic tracks 40 per proposer, 3 branches per start, keep 150 top-gap
plus 30 random per round, fine-tune 5000 steps. Reproduce with
`python -m ldr.loop` and the same options. Cost so far about
0.00 euros (2.15 hours at 0.0
per hour).

Round 0 is the world model we started from; each later round is that model fine-tuned on the real
branches found where the dream was most wrong. Everything is measured on the same fixed
validation tracks (seeds 900100 and up), never on the test tracks.

| Round | Real return | Exploit gap | Dream ret. | Real ret. | On-road share | Branches |
|---|---|---|---|---|---|---|
| 0 | 270 | 90.4 | 121.0 | 30.7 | 0.34 | 450 |
| 1 | 593 | 70.8 | 97.9 | 27.1 | 0.40 | 450 |
| 2 | 640 | 57.4 | 81.7 | 24.4 | 0.47 | 450 |

**Columns.** Real return: mean real return of the dream-trained controllers on the validation
tracks. Exploit gap: the dream's predicted return minus the real return for the dream's own
actions, over a window of up to 100 steps (large and positive means the dream is being exploited).
Dream return and Real return (window): the two halves of that gap. Real on-road share: how much of
the window the real car stays on the road under those actions. Branches: how many dream-steered
branches were measured.

**How to read it.** If the loop works, the gap falls and the real return rises from round to round.
One run, one seed per round: treat small
changes as noise.
