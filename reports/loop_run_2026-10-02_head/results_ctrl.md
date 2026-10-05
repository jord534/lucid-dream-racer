# Propose-critic loop results

Settings: rounds 2, proposers 3, dream generations 100,
critic tracks 40 per proposer, 3 branches per start, keep 150 top-gap
plus 30 random per round, fine-tune 5000 steps. Reproduce with
`python -m ldr.loop` and the same options. Cost so far about
0.00 euros (1.79 hours at 0.0
per hour).

Round 0 is the world model we started from; each later round is that model fine-tuned on the real
branches found where the dream was most wrong. Everything is measured on the same fixed
validation tracks (seeds 900100 and up), never on the test tracks.

| Round | Real return | Exploit gap | Dream ret. | Real ret. | On-road share | Road gap | Dream on-road when real off | Branches |
|---|---|---|---|---|---|---|---|---|
| 0 | 688 | 48.2 | 78.8 | 30.6 | 0.56 | 0.19 | 0.71 | 450 |
| 1 | 564 | 50.5 | 81.0 | 30.5 | 0.50 | 0.28 | 0.75 | 450 |
| 2 | 754 | 52.9 | 83.3 | 30.5 | 0.57 | 0.27 | 0.76 | 450 |

**Columns.** Real return: mean real return of the dream-trained controllers on the validation
tracks. Exploit gap: the dream's predicted return minus the real return for the dream's own
actions, over a window of up to 100 steps (large and positive means the dream is being exploited).
Dream return and Real return (window): the two halves of that gap. Real on-road share: how much of
the window the real car stays on the road under those actions. Road gap: the dream's P(on road)
minus the real on-road share over the window (positive: the dream shows the car on the road more
than the simulator does; "-" where the road probe could not be trusted or the round predates it).
Dream on-road when real off: the dream's mean P(on road) over the window steps where the real car
is off the road, among branches that have such steps (the closer to 0 the better; the failure tests
measure the same thing). Branches: how many dream-steered branches were measured. Critic settings:
rank by road gap, off-road-onset share of branch batches 0.5.

**How to read it.** If the loop works, the gap falls and the real return rises from round to round.
One run, one seed per round: treat small
changes as noise.
