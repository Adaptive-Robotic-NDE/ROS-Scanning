# Cell visit, next session

Written 2026-10-22 from `results26`. **Supersedes the 2026-10-17 version.**

That one was about speed and the disk. Speed is now understood and the limit
has moved; the stall is now diagnosed rather than instrumented; and the two
things to carry to the cell are a digitizer that can go faster than 10 blocks
per second and a scan that holds its tool axis still.

Undated. Nothing here goes stale with time, only with the next run.

---

## 0. What results26 settled

| | |
|---|---|
| **The NVMe is not the problem** | and neither is the antivirus. The 10-19 stall capture has **no threads in D state**, PSI io all zeros, and `delayacct blkio ticks = 0`. The 10-14 diagnosis was wrong and this replaces it. |
| **The stall is a deadlock in spatialnde2** | not a slowdown, not a livelock, not contention. `snde2 recdb gmn` burned **549 ticks at 17:08 and 549 ticks at 17:13** — zero CPU in five minutes — and that thread is the only thing that can release a blocked mutable computation. §1. |
| **The card was the speed limit, by coincidence** | 1 kHz / 100 samples = 10 blocks/s, and `max_trace_rate_hz["live"]` is 10.0 Hz. Two independent ceilings at the same number, so neither showed up as the binding one. §2. |
| **-200279 immediately on a rate change** | 2657 of them. Expected: `SampleRate` and `SampsPerChan` could only be set one at a time, so 250 kHz with the block still 100 samples asked for 2500 blocks/s. Fixed in the module. §2. |
| **bench_ec's measurements were good and its verdict was wrong** | the numbers below stand; the conclusion drawn from them did not. §2.3. |
| **The capture script did not capture** | gdb never attached (no sudo), and when it did it timed out after 4 of 314 threads. Both fixed; §1.4 is how to run it. |

---

## 1. The stall is a deadlock, and here is the chain

### 1.1 What the capture says

`py-spy` puts the stalled thread here:

```
stage_ec_reading   (ec_acquisition.dpi:946)
  project          (surface_projection.dpi:1072, wait=True)
    globalrev_wait                                  k = 1967
```

So the thing not completing is the **projection's** globalrev, not the trace's.

`threads.txt`: 0 threads in D state, 237 in `futex_wait`. CPU deltas over the
five minutes between the two captures:

| thread | 17:08 | 17:13 | delta |
|---|---|---|---|
| `ros_spinner` | | | +67 s |
| `dgpy_initializa` | | | +64 s |
| Qt main | | | +8 s |
| **`snde2 recdb gmn`** | **549** | **549** | **0** |
| `snde2 acrd 00/01/02` | 1316 / 43179 / 9018 | same | 0 |
| `snde2 ttm tbec` x8 | 547 | 547 | 0 |

The process is busy. The math pipeline is not merely slow — it did not execute
a single instruction for five minutes.

### 1.2 Why that is a deadlock and not a slow queue

`recmath_compute_resource.cpp:586` is the gate:

```cpp
if (!computation->function_to_execute->is_mutable || !computation_rss_is_globalrev
    || !prior_globalrev
    || (prior_globalrev->ready && (!prior_globalrev_globalrev
        || !prior_globalrev_globalrev->mutable_recordings_still_needed))) {
  _queue_computation_into_database_acrdb_locked(...)
} else {
  blocked_list.emplace(...)           // <-- the projector lands here
}
```

A **mutable** computation in globalrev N is refused the queue unless N-1 is
ready *and* nobody still needs its mutable recordings. The surface projector is
`mutable=True` and documented as requiring it (`surface_projection.dpi:827`):
it accumulates on top of its own previous revision.

`blocked_list` has exactly **one** drain: `recdatabase::globalrev_mutablenotneeded_code()`,
which runs on the `snde2 recdb gmn` thread and is woken only by
`~globalrev_mutable_lock()`. Zero ticks on that thread over five minutes means
no `globalrev_mutable_lock` destructor ever ran, which means a holder is pinned,
which means every later mutable computation stays blocked forever. The
projection globalrev can then never complete, and `project(wait=True)` never
returns. That is a cycle with no timer in it anywhere — a deadlock.

### 1.3 Who is holding it

`osg_compositor::perform_ondemand_calcs`, `openscenegraph_compositor.cpp:495-556`:

```cpp
if (mutable_channels.size() > 0) {
  { std::lock_guard<std::mutex> recdb_admin(recdb_strong->admin);
    globalrev = recdb_strong->latest_defined_globalrev();
    std::lock_guard<std::mutex> globalrev_admin(globalrev->admin);
    incomplete_mutable_recording_locker = globalrev->mutable_recordings_need_holder;
    ... }
  globalrev->wait_complete();   // still holding the holder
}
```

It takes a **copy of the holder** for the *latest defined* globalrev, then waits
for that globalrev to complete while holding it. A globalrev whose own mutable
math is on `blocked_list` cannot complete. The compositor waits for the
globalrev; the globalrev waits for the holder; the holder is in the
compositor's own stack frame.

The two release paths cannot help: `notify.cpp:757-760` nulls a completed
globalrev's holder only if it is **not** the latest defined, and
`recstore.cpp:4048-4054` nulls the *previous* latest's holder when a new
globalrev is defined. The compositor's copy keeps the lock object alive past
both.

**Not yet proven:** which globalrev's holder is pinned, and whether the
compositor is the only holder. The compositor's native stack is among the 310
threads gdb missed on 10-19. §1.4 is the capture that settles it.

### 1.4 What to capture next time, and how

```bash
sudo ~/Desktop/tyler_test9_ws/ROS-Scanning/tools/capture_stall.sh $(pgrep -f ECAutomatedScan)
```

**`sudo` is required.** `kernel.yama.ptrace_scope` is 1 on this machine, so gdb
cannot attach to a process it did not start; without it the script now refuses
in one line and writes the exact command to run into `native.txt` instead of
pretending. With it: 41 of 41 threads in about two seconds, where before it was
4 of 314 in 180 s (python-gdb was pretty-printing every frame argument).

In the output, `native.txt` is the file that matters. Look for:

* a thread in `osg_compositor::perform_ondemand_calcs` inside `wait_complete` —
  that confirms §1.3;
* any other frame holding a `globalrev_mutable_lock`;
* `recmath_compute_resource` frames, and whether `blocked_list` is non-empty.

### 1.5 What to do about it

Two real fixes, both in spatialnde2, and neither is "do not plot it":

1. **Drop the holder once the arrays have been read.** The compositor needs the
   mutable data to be stable while it copies it into the render pass, not for
   the whole ondemand → render → composite pass and certainly not across a
   `wait_complete()`. Releasing it before the wait breaks the cycle outright.
2. **Make the inhibit per-channel instead of per-globalrev.** An off-screen
   mutable channel is blocked today by a holder taken for a globalrev. This is
   also the fix for the GUI stall in §1.6.

The workaround that does not defeat the purpose, if a scan has to run before
either lands: make the projector non-mutable, the way `compute_cscan` in
`virtual_scanner.dpi` already is. It costs one 512x512 copy per trace and keeps
the live image. `EC_PROJECTION_MODE = "deferred"` also avoids it, at the price of
no live image, which is the thing worth having.

### 1.6 The related, separate thing: mutable math on screen costs 0.7-2 s per globalrev

Measured 2026-08-09 on `virtual_scanner.dgp`: instantiating a math function with
`mutable=True` on a channel the live viewer displays costs 0.7-2 s of Qt
GUI-thread stall per new globalrev (blocked time far exceeds total process CPU,
so the thread is waiting, not working). Same mechanism as §1.3 — the compositor
holds `mutable_recordings_need_holder` for the whole pass, so compositor and
math pipeline alternate instead of overlapping — but the outcome there is slow,
not stuck. `compute_cscan` was left non-mutable for exactly this reason; the
incremental copy-forward path already gives ~100x and only the nx*ny copy is
given up.

Do not conflate either of these with the **globalrev completion wedge**, which
was a different defect (a producing mutable channel permanently stopping
globalrev completion, so the database retained every recording and the process
died of "Too many open files"). That is **fixed**, spatialnde2 `976aa6b`. With
it fixed, `mutable=True` is fine on channels nothing displays live: a full 60x60
scan with the mutable surface projector ran 658.6 s with flat fds and
`latest_defined == latest_ready` throughout.

---

## 2. The digitizer: 10 blocks per second was the card, not the pipeline

### 2.1 The arithmetic

The correlator takes **one block per pitch crossing**, so the block rate is the
hard ceiling on trace rate:

```
block rate = SampleRate / SampsPerChan = 1000 / 100 = 10 blocks/s
```

and `max_trace_rate_hz["live"] = 10.0`. Those are two independent ceilings that
happen to be the same number, which is why raising one alone would have changed
nothing and why the scan speed looked like a software clamp.

At `SCAN_STEP_SIZE = 0.00025` that is 10 x 0.25 mm = **2.5 mm/s**, which is
exactly what was commanded.

### 2.2 Smear does not improve with sample rate alone

Each trace is the mean of one block, so

```
smear / pitch = block_duration / block_period = 1
```

at any sample rate, as long as the blocks are contiguous. Running the card at
250 kHz with `SampsPerChan` scaled to keep 10 blocks/s gives a 100 ms block —
2500 samples averaged instead of 100 — and the **same 0.25 mm of smear**. More
samples in the average, not finer resolution.

**Shorter blocks are what buys resolution.** 250 kHz with `SampsPerChan = 2500`
is a 10 ms block at 100 blocks/s; the correlator still takes one per crossing at
10 crossings/s, and each trace is now a 10 ms average — 0.025 mm of smear, a
tenth of the pitch. The blocks between crossings are discarded, which is the
right thing to discard.

### 2.3 What bench_ec measured, and what it got wrong

The measurements stand:

| | |
|---|---|
| block rate | 10.21 Hz, jitter 15.37 ms |
| output filter corner | 15.6 Hz horizontal, 24.4 Hz vertical |
| averaging efficiency | 13% horizontal, 5% vertical |
| vertical noise when averaged | **0.5x** — it got *worse* |

The last line is the finding: averaging made the vertical channel noisier, so
the dominant noise is **below** the block rate — drift, not white noise — and no
amount of averaging can help it. That is a reason to go faster, not slower.

Two things it said that were wrong, and are fixed in `ec_bench.dpi`'s verdict:
it compared the achieved speed against `SCAN_VELOCITY` (the *requested* 10 mm/s)
instead of the 2.5 mm/s commanded and 1.1 mm/s achieved; and it called the
instrument's front-panel filter setting an instrument limit. The filter is
adjustable, and at 15.6 Hz it is aggressive for a 10 Hz block rate — worth
checking against the gantry instrument's setting, which is the comparison that
matters.

### 2.4 At the cell: set the card, then measure it

`dgpython-nidaqmx` now sets rate and block length together, sizes the input
buffer for a second of data, and reads into a preallocated array. **Nothing
moves for any of this.** Put the probe on the part and leave the arm still.

```
daq.AcquisitionStatus()
```

Expect `{'alive': True, 'blocks': <rising>, 'read_errors': 0,
'sample_rate_hz': 1000.0, 'block_samples': 100, 'block_rate_hz': 10.0,
'input_buf_size': 1000, ...}`. `read_errors` is the -200279 count; it should be
0 before you change anything.

```
daq.Configure(SampleRate=250*ur.kHz, SampsPerChan=2500)
```

Returns every parameter. One reconfiguration, so there is no instant at which
the card is asked for 2500 blocks/s. Then:

```
daq.AcquisitionStatus()
```

Expect `block_rate_hz: 100.0`, `input_buf_size: 250000`, `read_errors` still 0
and `blocks` climbing by about 100/s. **If `read_errors` climbs, that is the
pipeline not keeping up with 100 blocks/s, not the card** — `CalcSync` makes
every block wait for its math, so 100 blocks/s is 100 blocking globalrevs/s.
Step down (`SampsPerChan=12500` for 20 blocks/s, `25000` for 10) until it is
clean, and report which number held. That measurement is the input to the next
change, §2.6.

```
bench_ec(seconds=20)
```

Expect the block rate to read back what you set, the jitter to fall with it, and
the filter corners to be unchanged (they are the instrument's, not the card's).

### 2.5 Then the scan, at the resolution the card can now afford

Nothing to type: the step size, the line spacing, the preload and the
conditioning rotation are config defaults since 2026-10-22 (§3.4). Just run it.

That is the results26 scan with a tenth of the smear and nothing else changed —
the comparison that isolates the block length. If `read_errors` stayed 0 at 100
blocks/s, this is the run to do first.

### 2.6 What is still owed here

* **Option 5, the reader thread.** One thread reading the card into a bounded
  queue, another publishing. Not needed if §2.4 holds at 100 blocks/s; needed if
  it does not.
* **A replacement for CalcSync.** `max_pending_globalrevs=` is in the module
  now and is **off**: it waits only once the number of defined-but-not-ready
  globalrevs exceeds the bound, so the card runs free until the pipeline falls
  behind, and then the card slows instead of memory growing. It does not give up
  — an advisory bound is not a bound. It is not enabled because the right number
  for it is a measurement, not a guess, and §2.4 is that measurement. Note
  `globalrev_backlog_warning_threshold` (default 100) is a *warning* and not a
  throttle; there is no native backpressure.

---

## 3. The axis-locked scan, start to finish

**What it is.** One tool orientation for every waypoint of the eddy-current
path, instead of the commanded axis following the mesh's normals. On a plate
flat to about a degree, normal-following costs 61% of the commanded speed and
makes the arm rock along a pass; the axis turn rate is what TOTG prices the
trajectory on. Locking it should cost nothing on a plate, and this is the run
that says whether it does.

The locked orientation is the probe's own vertical reference with
`EC_PROBE_ROTATION_DEG` in it — chosen, not taken from wherever the arm happened
to be standing — so a locked plan is the same plan every time.

### 3.1 Deploy

From the workspace root on the cell PC:

```bash
./tari.bash --layer site
```

That is the whole deployment. Do not install a package by hand -- the layer
chain fetches the manifests, builds and verifies, and it is idempotent, so
running it again is the way to fix a partial state.

### 3.2 Bring the cell up

```bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hw
```

(From the cell PC's own desktop session, so `DISPLAY` is already set. The
explicit `DISPLAY=:10` is a container thing and does not belong here.)

Wait for it to settle, then check that the pieces are there rather than assuming:

```bash
ros2 run tari_bringup tari_cell_info --cell ex-ur10e-1 --mode hw
```

### 3.3 Start the scan config

```bash
cd ~/Desktop/tyler_test9_ws/ROS-Scanning/dgpfiles
dataguzzler-python ECAutomatedScan.dgp
```

It leaves you at a `dgpy>` prompt. Everything below is typed there.

### 3.4 Confirm what the run will use, before it moves

**The four settings that used to be typed before every scan are defaults now**
(2026-10-22). Nothing to set — check that they are what you expect:

| | default | was |
|---|---|---|
| `SCAN_STEP_SIZE` | 0.25 mm | 1 mm |
| `SCAN_LINE_SPACING` | 0.25 mm | 0.3 mm |
| `EC_PRELOAD_M` | 2.00 mm | 0 |
| `EC_ROTATION_DEG` | 86.5 deg | 0 |

```
describe_calibration()
scan_plan()
ec_preload()
ec_rotation()
probe_rotation()
daq.AcquisitionStatus()
```

`describe_calibration()` must show the 0.822 mm Azure solve and a **non-zero
Keyence lateral correction** — deploying a calibration has silently zeroed that
one before.

`scan_plan()` prints the step size, the velocity it will actually command, the
trace rate, and the time and memory estimate. **Read the commanded velocity, not
the requested one**: at a 0.25 mm step the 10 mm/s request is clamped to
2.5 mm/s by the live trace-rate ceiling, and the plan says so. That clamp is
deliberate — see the note on `SCAN_VELOCITY` in the config for why the velocity
was not simply lowered to match.

`ec_preload()` is 2.00 mm, which is exactly `EC_PRELOAD_CEILING_M`. If the
tooling has changed and the ceiling has dropped, `plan_ec_path()` refuses this
default by name rather than clipping it.

`ec_rotation()` is the rotation the scan STARTS with, and it is still a property
of this probe and this instrument today: `EC_AUTO_ROTATION` re-measures it from
the scan's own traces at the end and re-paints, and `set_rotation()` overrides
it at the prompt. If the measured value comes back far from 86.5, that is the
number to bring home.

`probe_rotation()` is 0: the PHYSICAL rotation of the probe about its own axis,
which is the one the locked path in §3.6 holds. Not the same quantity as
`ec_rotation()`, which is a rotation of the impedance plane in software.

### 3.5 The reference run: surface-following

```
run_automated_scan()
```

Five stages, each printed: detect, Azure sweep, ROI, Keyence raster, mesh
filter, toolpath. It then scans. At the end it prints the outcome check — passes
completed, traces staged, `Fault`, `Acquisition Errors`, `Staging Timeouts`,
worst correlation wait, observed probe speed. Note the elapsed time and the
observed speed; they are the baseline.

### 3.6 The locked run: one command, one argument

```
run_automated_scan(orient="fixed")
```

Everything before the toolpath is identical, so it measures the same part. At
the planning stage it will print the probe rotation and the reachability report
as usual; the difference to look for is that the 45-degree tilt crop drops
**nothing** (`0 waypoints cropped`), because a vertical axis tilts zero degrees
everywhere.

Compare: elapsed time, observed probe speed, and the notch's peak and SNR from
the NORMS figures. If the locked run is faster at the same quality on this
plate, locking is the default for flat work.

### 3.7 If you want to see the locked path before flying it

```
run_automated_scan(scan=False, orient="fixed")
dry_run_ec_scan(passes=[0, 1, 2], sample=5)
```

`scan=False` stops after the plan and leaves it in `ec_scan_path`;
`dry_run_ec_scan` flies the named passes with no instrument and no contact, at
one waypoint in `sample`. Use it the first time; the full run does not need it.

### 3.8 Safety, unchanged

* The contact probe never goes beyond the taught points. Shrink
  `EC_SCAN_REGION` for margin; never grow the motion.
* Absolute wrist-3 angle governs the cable envelope. A refusal is never resolved
  by turning the wrist a whole revolution to the same probe orientation.
* The Azure's power cable is a collision hazard and the planner has no model of
  it. It caused the 2026-10-06 e-stop. Watch it.
* Stopping External Control on the pendant is a **pause** inside
  `run_automated_scan` (it runs supervised). Resume it, or the run ends with
  `ScanAborted` rather than a mesh with holes in it.

---

## 4. Checklist

1. `sudo lspci -s 89:00.0 -vv | grep -iE 'LnkSta|LnkCap'` — the NVMe link is
   still erroring and still not the cause of anything. One line, for the record.
2. §2.4 — the block rate the pipeline sustains with `CalcSync` on. **This is the
   number the next change depends on.**
3. §2.5 — the results26 scan at a tenth of the smear.
4. §3.5 and §3.6 — the surface-following and axis-locked pair.
5. If anything stalls: `sudo .../capture_stall.sh <pid>`, and keep `native.txt`.
