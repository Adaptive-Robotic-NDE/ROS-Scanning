# Site visit procedure

What to set, what to run, and what to check, for the next eddy-current scan at
cell ex-ur10e-1. Undated: nothing here goes stale with time, only with the next
run.

Three things this procedure exists to hold in place:

- The Azure reads every surface about 12.45 mm too far (8 measurements over 4
  days, sd 0.68 mm), and a bring-up adapter subtracts it.
- `table_top.stl` was 14 mm below the touched-off surface and is raised.
- `run_automated_scan()` refuses if the reconstructed mesh disagrees with the
  Keyence by more than 2 mm.

---

## 1. Deploy

From the workspace root on the controlling computer:

```bash
./tari.bash --layer site
```

That is the whole deployment. Do not install a package by hand -- the layer
chain fetches the manifests, builds and verifies, and it is idempotent, so
running it again is the way to fix a partial state.

## 2. Bring the work cell up

```bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hw
```

From the controlling computer's own desktop session, so `DISPLAY` is already set. Then check
that the pieces are there rather than assuming:

```bash
ros2 run tari_bringup tari_cell_info --cell ex-ur10e-1 --mode hw
```

## 3. Start the scan configuration

```bash
cd ~/Desktop/tyler_test9_ws/ROS-Scanning/dgpfiles
dataguzzler-python ECAutomatedScan.dgp
```

It leaves a `dgpy>` prompt. Everything below is typed there.

## 4. Confirm what the run will use, before anything moves

The four scan settings are configuration defaults now. Nothing to set -- check
that they are what you expect:

| | default |
|---|---|
| `SCAN_STEP_SIZE` | 0.25 mm |
| `SCAN_LINE_SPACING` | 0.25 mm |
| `EC_PRELOAD_M` | 2.00 mm |
| `EC_ROTATION_DEG` | 86.5 deg |

```
describe_calibration()
scan_plan()
ec_preload()
ec_rotation()
probe_rotation()
daq.AcquisitionStatus()
```

`describe_calibration()` must show the 0.822 mm Azure solve and a **non-zero
Keyence lateral correction**. Deploying a calibration has silently zeroed that
one before.

`scan_plan()` prints the step size, the velocity it will actually command, the
trace rate, and the time and memory estimate. **Read the commanded velocity,
not the requested one**: at a 0.25 mm step the 10 mm/s request is clamped to
2.5 mm/s by the live trace-rate ceiling, and the plan says so. That clamp is
deliberate -- see the note on `SCAN_VELOCITY` in the configuration for why the
velocity is not simply lowered to match.

`ec_preload()` is 2.00 mm, which is exactly `EC_PRELOAD_CEILING_M`. If the
tooling has changed and the ceiling has dropped, `plan_ec_path()` refuses this
default by name rather than clipping it.

`ec_rotation()` is the rotation the scan starts with. It remains a property of
this probe and this instrument on the day: `EC_AUTO_ROTATION` re-measures it
from the scan's own traces at the end and re-paints, and `set_rotation()`
overrides it at the prompt. If the measured value comes back far from 86.5,
that is the number to bring home.

`probe_rotation()` is 0: the PHYSICAL rotation of the probe about its own axis,
which is what the axis-locked path in section 7 holds. Not the same quantity as
`ec_rotation()`, which is a rotation of the impedance plane in software.

## 5. Set the digitizer, then measure it

The card, not the software, used to cap the trace rate: at 1 kHz with 100-sample
blocks it delivers 10 blocks/s, and the correlator takes one block per pitch
crossing, so 10 blocks/s was 10 traces/s. That is numerically the same as the
live projection ceiling, which is why neither showed up as the binding one.

Raising the sample rate alone buys nothing. Each trace is the mean of one block,
so with contiguous blocks the smear equals the pitch at any sample rate.
**Shorter blocks** are what buys resolution: 250 kHz in 2500-sample blocks is a
10 ms block at 100 blocks/s, the correlator still takes one per crossing, and
each trace becomes a 10 ms average -- a tenth of the pitch.

Nothing moves for any of this. Put the probe on the part and leave the arm
still.

```
daq.AcquisitionStatus()
```

Expect `read_errors` 0 before changing anything.

```
daq.Configure(SampleRate=250*ur.kHz, SampsPerChan=2500)
daq.AcquisitionStatus()
```

Expect `block_rate_hz: 100.0`, `input_buf_size: 250000`, `read_errors` still 0,
and `blocks` climbing by about 100/s.

**If `read_errors` climbs, that is the pipeline not keeping up, not the card.**
`CalcSync` makes every block wait for its math, so 100 blocks/s is 100 blocking
globalrevs/s. Step down -- `SampsPerChan=12500` for 20 blocks/s, `25000` for 10
-- until it is clean, and record which number held. That measurement decides
whether the acquisition path needs a reader thread and a bounded backlog.

```
bench_ec(seconds=20)
```

Expect the block rate to read back what was set, the jitter to fall with it, and
the output-filter corners to be unchanged -- they are the instrument's, not the
card's. At 15.6 Hz horizontal and 24.4 Hz vertical the instrument's filter is
aggressive for a 10 Hz block rate; worth comparing against the gantry
instrument's setting, because that comparison is otherwise partly a filter
comparison.

One measurement from `bench_ec` worth keeping in mind: averaging made the
vertical channel noisier, not quieter. The dominant noise is below the block
rate -- drift, not white noise -- so more averaging cannot help it.

## 6. The reference run: surface-following

```
run_automated_scan()
```

Five printed stages -- detect, Azure sweep, region of interest, Keyence raster,
mesh filter, toolpath -- and then the scan. At the end it prints the outcome
check: passes completed, traces staged, `Fault`, `Acquisition Errors`,
`Staging Timeouts`, worst correlation wait, observed probe speed. Note the
elapsed time and the observed speed; they are the baseline for section 7.

## 7. The axis-locked run

```
run_automated_scan(orient="fixed")
```

One tool orientation for every waypoint instead of the commanded axis following
the mesh's normals. On a coarse reconstruction of a nearly flat plate those
normals wander: the arm rocks along a pass, the axis turn rate is what the time
parameterisation prices the trajectory on, and surface-normal tracking costs 61%
of the commanded speed on a part flat to one degree. Locking it should cost
nothing on a plate, and this run is what says whether it does.

The locked orientation is the probe's own vertical reference with
`EC_PROBE_ROTATION_DEG` in it -- chosen, not taken from wherever the arm
happened to be standing -- so a locked plan is the same plan every time.

Everything before the toolpath is identical to section 6, so both runs measure
the same part. The tell that the lock took: the 45-degree tilt crop drops
**nothing**, because a vertical axis tilts zero degrees everywhere.

Compare elapsed time, observed probe speed, and the notch's peak and SNR from
the generated figures. If the locked run is faster at the same quality on this
plate, locking becomes the default for flat work.

To see the locked path before flying it:

```
run_automated_scan(scan=False, orient="fixed")
dry_run_ec_scan(passes=[0, 1, 2], sample=5)
```

`scan=False` stops after the plan and leaves it in `ec_scan_path`;
`dry_run_ec_scan` flies the named passes with no instrument and no contact, at
one waypoint in `sample`. Worth doing the first time; the full run does not need
it.

## 8. Safety

- The contact probe never goes beyond the taught points. Shrink
  `EC_SCAN_REGION` for margin; never grow the motion.
- Absolute wrist-3 angle governs the cable envelope. A refusal is never resolved
  by turning the wrist a whole revolution to the same probe orientation.
- The Azure's power cable is a collision hazard and the planner has no model of
  it. It caused the 2026-10-06 emergency stop. Watch it.
- Stopping External Control on the pendant is a PAUSE inside
  `run_automated_scan`, which runs supervised. Resume it, or the run ends with
  `ScanAborted` rather than a mesh with holes in it.

## 9. If the scan stalls

A stall presents as the scan stopping with no error: no new traces, no progress
output, and Ctrl-C does nothing, because the console thread cannot interrupt a
blocked call inside the recording database.

Capture it before killing the process:

```bash
sudo ~/Desktop/tyler_test9_ws/ROS-Scanning/tools/capture_stall.sh $(pgrep -f ECAutomatedScan)
```

**`sudo` is required.** `kernel.yama.ptrace_scope` is 1 on this machine, so gdb
cannot attach to a process it did not start; without it the script refuses in
one line and writes the command to run into `native.txt`. With it, all threads
are dumped in a couple of seconds.

Keep the whole output directory. `native.txt` is the file that matters, and the
question it answers is which thread holds a `globalrev_mutable_lock` while
waiting for a globalrevision to complete. A mutable math function cannot be
queued until the previous globalrevision has released that lock, and the only
thing that releases it is the lock's destructor -- so one globalrevision that
never completes stops every later mutable computation, permanently and with no
diagnostic.

Storage is ruled out as a cause: `/proc/pressure/io` reads zero throughout, no
thread has a storage wait channel, and the process's `delayacct blkio ticks` is
0.

Workarounds that do not defeat the purpose, if a scan has to run before the
library is fixed: make the projector non-mutable, which costs one image copy per
trace and keeps the live image; or set `EC_PROJECTION_MODE = "deferred"`, which
builds the image at the end instead.

## 10. Checklist

1. `sudo lspci -s 89:00.0 -vv | grep -iE 'LnkSta|LnkCap'` -- the NVMe link is
   still throwing corrected PCIe receiver errors and is still not the cause of
   anything. One line, for the record.
2. Section 5 -- the block rate the pipeline sustains with `CalcSync` on. This is
   the number the next change depends on.
3. Section 5 -- the scan at a tenth of the smear.
4. Sections 6 and 7 -- the surface-following and axis-locked pair.
5. If anything stalls: section 9, and keep `native.txt`.
