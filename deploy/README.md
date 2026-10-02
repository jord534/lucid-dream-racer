# Running the propose-critic loop on a cloud machine (Scaleway)

What the loop does is described at the top of `ldr/loop.py`. In short: each round it trains a few
dream controllers (the adversary), measures how badly the dream overpays them compared with the real
simulator (the exploit gap), keeps the real outcomes where the dream was most wrong, and fine-tunes the
world model on them. Success is the exploit gap falling and the real return rising round by round.

## Status

- Tested on a laptop only. A tiny run (1 round, 1 controller, 4 tracks) completes end to end in about
  1.5 minutes; re-running a finished directory skips every stage; the budget cap stops a run and
  writes results. The numbers from the tiny run are noise and prove only that the plumbing works.
- Not yet run on a cloud machine. `bootstrap.sh` has never run on a real VM.
- How long a full-size round takes, and how long MDN-RNN fine-tuning takes on a CPU-only machine,
  are not measured. Use the tiny run on the VM first, then one full round, and look at the stage times
  in the log before trusting any estimate.

## Steps

1. **Account (you).** Create a Scaleway account (card and ID verification), set a billing alert, add your
   SSH public key (`~/.ssh/id_ed25519.pub`), and create an Object Storage bucket plus an API key for it.
   Keep the key out of the repo and out of chat.
2. **Machine (you).** Create one Ubuntu instance in Compute Optimized, zone PAR-1: **COMPUTE3-X16C-32G**
   (16 vCPUs, 32 GB RAM, 0.4682 euros per hour), with at least 30 GB of disk (the PyTorch environment
   takes several GB). At that price a 5-euro cap is about 10.7 hours of runtime. Why not bigger: the
   cost per vCPU-hour is nearly the same at every size, rollouts parallelise across cores, but the
   model training probably does not use more than about 16 threads well, and while we are still
   debugging a larger machine only burns money faster (COMPUTE3-X32C-64G is 0.9363 per hour, about
   5.3 hours of cap). The POP2-HC family is about 9% cheaper per vCPU, but I cannot tell how it
   compares in speed. Use `--workers 14` and `--price-eur-hour 0.4682`.
3. **Set up the machine.**
   ```
   ssh root@<ip>      # or the user Scaleway gives you
   git clone https://github.com/jord534/lucid-dream-racer.git
   bash lucid-dream-racer/deploy/bootstrap.sh
   ```
4. **Copy the inputs from your laptop (about 0.6 GB).** The loop does not need the 16 GB of packed frames.
   ```
   rsync -av data/latents.npz root@<ip>:lucid-dream-racer/data/
   rsync -av runs/vae runs/mdnrnn root@<ip>:lucid-dream-racer/runs/
   ```
5. **Smoke test on the machine (about 2 minutes).**
   ```
   cd lucid-dream-racer && source .venv/bin/activate && export SDL_VIDEODRIVER=dummy
   python -m ldr.loop --tiny --price-eur-hour 0 --out runs/loop_tiny
   ```
6. **Real run, inside `tmux` so it survives you disconnecting.**
   ```
   tmux new -s loop
   python -m ldr.loop --rounds 2 --workers <vCPUs minus 2> --price-eur-hour <price> \
       --budget-eur 5 --out runs/loop \
       --sync-cmd "rclone sync runs/loop scw:<bucket>/loop" \
       --finish-cmd "<command that deletes this machine>"
   ```
   Detach with Ctrl-b then d. Reconnect later and run `tmux attach -t loop`, or just read
   `runs/loop/results.md` and the bucket. If the machine or the run dies, run the same command again: it
   resumes at the first unfinished stage.

## Things to watch

- **A forgotten machine costs more than the whole budget.** At 0.4682 euros per hour, one day is about
  11 euros. Make sure the machine is deleted when the run finishes.
- **Billing after the run.** On many clouds, powering a machine off from inside does not stop the charge.
  Check how Scaleway bills a stopped instance. Use `--finish-cmd` to delete it with the Scaleway CLI
  (`scw`, needs its own credentials on the machine), or delete it yourself from the console when
  `results.md` shows the run has finished.
- **The budget cap is checked between stages.** It stops at 90% of the budget at the start of a stage,
  but one long stage (for example fine-tuning on a slow CPU) can run past it. It also does not count
  time spent in a stage that was interrupted. Keep the provider's billing alert on as the real backstop.
- **Test seeds.** The loop measures on validation tracks (seeds 900100 and up) and collects on fresh
  training tracks. It never touches the test tracks.
- **Results go to** `runs/loop/results.md` (plain-language table of every round) and `runs/loop/state.json`;
  per-round files are in `runs/loop/rNN/`. Copy them into `reports/` when you want them in the repo.
