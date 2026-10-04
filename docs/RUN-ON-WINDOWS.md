# Generating the eddy-current response table, on two Windows machines

Everything needed is in this zip. Nothing downloads at run time.

Below, **A** is the first machine and **B** the second. They do disjoint halves
and you merge at the end.

---

## 1. Environment (once per machine, Miniforge prompt)

```bat
conda create -n ecsim python=3.12 numpy scipy h5py -y
conda activate ecsim
```

**If the machine has an NVIDIA GPU**, add CuPy as well — the solver has a GPU
backend and the run uses it *alongside* the CPU cores, not instead of them:

```bat
conda install -c conda-forge cupy -y
python scripts\gpu_check.py
```

`gpu_check.py` verifies the GPU backend works and times it against the CPU. If
it fails or CuPy is absent, nothing breaks — just leave `--gpu-workers` off.

### Read the GPU check before you trust the GPU

**Run it in both precisions and compare.** On an RTX 2070 SUPER, double
precision made the GPU **2.7x slower than the CPU** for identical work — 5.75 s
against 2.15 s on 2000 cells, agreeing to 4.9e-08. The cause is not the card
being a consumer part: the solver carries `complex128`, and that silicon runs
FP64 at **1/32** of its FP32 rate.

Two caveats on that measurement, both since addressed in the tool, so your own
numbers should be cleaner than the ones above. The iteration counts it reported
(126 against 2) were not a convergence difference — the two paths were counting
different things, and the check now counts operator applications on both. And
the GPU timing included first-call setup; the check now warms the GPU up first.
Re-measure rather than reading the figures above as the verdict.

**This is why `--precision single` matters, and why "workstation GPU" does not
by itself fix it.** The RTX A-series workstation cards are the same silicon as
the consumer ones — an A6000 is GA102 at **1/64** FP64, worse in ratio than the
2070 SUPER. Only data-centre parts (V100, A100, H100) run FP64 at 1/2 rate. So
the question is never consumer-vs-workstation, it is **what this card's FP64
throughput actually is** — and on anything RTX-branded, double precision will
lose.

Single precision sidesteps it entirely: 32-64x more peak rate on that silicon,
half the memory, and the FFTs stay single (the solver uses scipy's FFT rather
than numpy's, which would quietly promote back to double).

**Measured on the 2070 at full size, single and double both took about 360 s
for a 101x101 field and gave the same peak |dZ| to five figures.** Two things
follow. The answer is the same, so single is safe to use. And it bought no time
at that size, because the GMRES solve is no longer the bottleneck there -- the
per-position field evaluation is, and that does not care about precision. So
**choose single for the MEMORY, which is what limits how many workers fit, not
for speed.**

**And it costs nothing we keep.** The table is *stored* as `complex64` whatever
you choose, so double precision through the solve is more precision than the
deliverable retains. The 4.9e-08 CPU-vs-GPU agreement says the headroom is
there.

What to do:

1. Run `gpu_check.py`. Note the per-solve time and the agreement.
2. If single precision agrees with double to ~1e-6 or better on your flaws and
   is faster, use `--precision single` for the whole run.
3. If the GPU is still slower, leave `--gpu-workers` off and run CPU-only. That
   is a perfectly good outcome, not a failure.

A slow GPU lane is not free: it holds one pose at a time and can become the
straggler at the very end of a long run.

Then, from the folder this file is in:

```bat
pip install -e .
```

`-e` so `import ecsim` works from any directory. Nothing else is needed:
`numba` is listed as an optional extra but the package never imports it.

Check it took:

```bat
python -c "import ecsim, numpy, scipy, h5py; print('ok')"
```

---

## 2. Run it

**Always pass `--yes` for an unattended run.** Without it the parent stops at
`continue? [y/N]` after its first timing estimate, while every worker carries
on burning CPU on the task it already holds. The prompt is easy to miss because
it only appears after one full wave of poses has completed.

**Machine A:**
```bat
python tools\generate_ec_table.py --out-dir out --of 2 --shard 0 --workers 16 ^
    --precision single --gpu-workers 1 --yes
```

**Machine B:**
```bat
python tools\generate_ec_table.py --out-dir out --of 2 --shard 1 --workers 16 ^
    --precision single --gpu-workers 1
```

Drop `--precision single` if `gpu_check.py` says the accuracy does not hold on
your flaws; `double` is the default. Whichever you pick is recorded in the
table's provenance, so a file always says how it was computed.

Drop `--gpu-workers` on a machine with no NVIDIA GPU. Use `--gpu-workers 2` if
there are two GPUs; more than one worker per GPU usually just contends for its
memory.

### How the GPU and CPU share the work

They do not split it up front. **Both lanes pull from one queue and are topped
up as they finish**, so whichever is faster simply takes more, with no ratio for
you to guess and no lane left idle at the end. The run reports the split and
the per-pose time for each lane when it finishes, e.g.

```
   lane throughput:
      gpu      310 pose(s)     41 s each
      cpu      150 pose(s)    290 s each
```

which also tells you whether the GPU was worth using at all.

`--workers` defaults to HALF your cores if you leave it off. **On a machine
with 1.5 TB that default is far too timid — use most of the cores.**

What ends a run is memory, not CPU. Each worker is a separate process holding
its own solver state: ecsim measures about **2.6 GB for one 101x101
double-precision solve**, and a run on a 2070-class box with modest RAM failed
exactly that way, on allocation, after a long time. The script now prints its
estimate per worker at startup — multiply it by `--workers` and compare with
what the machine has. At 1.5 TB you will not get close.

Watch the first few minutes in Task Manager anyway and adjust; the run is
resumable, so changing your mind costs only the pose in flight.

### Windows processor groups — this one will catch you

Windows puts processes in a **processor group of at most 64 logical
processors**, and a process does not span groups without asking. On a dual
socket machine with more than 64 threads, launching 61 workers lands them all
in one group: you will see about half the machine busy and wonder why.

The `--of` / `--shard` split is the right tool — run one instance per group
rather than one giant instance. Two machines x two groups is simply `--of 4`
with `--shard 0..3`, and the merge step takes all four. Setting affinity per
instance also works if you prefer.

### The field is now a flag, and worth thinking about

```bat
--field-step-mm 0.1      --field-halfwidth-mm 6
```

Default is 0.25 mm over +/-6 mm (49x49), which is still 4x oversampled against
our 1 mm trace pitch.

**The field grid barely affects run time at all.** Per-pose cost is two coil
incident-field maps plus roughly 150 basis solves, and neither depends on how
many points you sample the field at. An earlier note here claimed 0.25 mm saved
4.3x over 0.1 mm; that was wrong. What the step actually changes is memory per
worker.

So with 1.5 TB, 0.1 mm (121x121) is close to free and buys the ability to study
trace pitches finer than 1 mm in simulation later. Your call — the table is
usable either way, and the choice is recorded in the file.

It prints a measured seconds-per-pose and an ETA after the first few poses and
**asks before committing to the rest**. That estimate is the real one for your
machine; treat any number I have quoted as indicative only.

**If it is interrupted** — power cut, closed window, Ctrl-C — just run the same
command again. It reads back what is already written and does only the rest.

---

## 3. Merge

Copy B's `out\al6061_2mhz_response.shard1of2.h5` to A, then on A:

```bat
python tools\generate_ec_table.py --out-dir final --merge ^
    out\al6061_2mhz_response.shard0of2.h5 ^
    out\al6061_2mhz_response.shard1of2.h5
```

That writes `final\al6061_2mhz_response.h5`, says how many poses are present,
**warns if any are missing**, and validates the file against the schema.

Also copy back `out\al6061_2mhz_baseline.npz` from either machine — it is the
no-flaw lift-off/tilt trajectory, it is quick to compute, and both machines
produce it identically.

---

## 4. What to send back

```
final\al6061_2mhz_response.h5      the table
out\al6061_2mhz_baseline.npz       the no-flaw trajectory
```

---

## Notes worth having

**Why two files.** The schema's own `/baseline` dataset is only valid in the
factorised lift-off/tilt model, and ecsim measured that lift-off and tilt do
*not* factor out of the flaw response (15-60% residual). So this table is dense
in both, which leaves nowhere in the file for the flaw-free trajectory. We need
that trajectory: with no flaw under the probe it *is* the whole signal.

**Both lanes write through the parent.** HDF5 is single-writer here: workers
compute and hand the field back, the parent writes it. So you can stop the run
at any time without risking the file.

**What drives the run time.** Measured here: one pose took 635 s on a 101x101
field and still over 200 s on a 9x9 one. So the cost is dominated by per-pose
setup, not by the number of field points — which means the spatial grid is a
weak lever and the levers that matter are the number of poses and the number of
workers. Do not expect a finer or coarser field to change the total much.

**The probe geometry is a placeholder.** Only the shaft diameter of our probe
is known (0.187 in). The coil dimensions in `tools\generate_ec_table.py` are
plausible-but-invented and are recorded in the file's provenance. The table is
for exercising the scanning pipeline, not for predicting absolute amplitudes.
Once there is an unclipped tilt-ladder measurement with the probe's own
rotation recorded, `scripts\fit_d20.py` shows how to fit them, and the table can
be regenerated.
