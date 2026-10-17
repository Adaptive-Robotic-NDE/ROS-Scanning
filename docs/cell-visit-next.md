# Cell visit, next session

Written 2026-10-17 from `results25`. **Supersedes the 2026-10-15 version.**
That one was built around the stall and the Azure calibration; both of those
are now either fixed or instrumented, and what is left is a scan that works
and takes 43 minutes to cover 18 x 29 mm.

Undated. Nothing here goes stale with time, only with the next run.

---

## 0. What 10-16 settled

| | |
|---|---|
| **No stall** | 72 of 72 passes, 8469 of 8496 traces, `Fault none`, `Acquisition Errors 0`, `Staging Timeouts 0`, worst correlation wait **71 ms**. The scan ran to the end with the antivirus running and the disk still erroring. |
| **The clearance gate reports** | `gate: 2171/2171 poses, 0:00:01 elapsed`. It also took the SAMPLED path this time (2171 poses, over the 2000 threshold) and measured only 144 of them, in one second. §4 is the case that does not. |
| **The NORMS plug-in works at the cell** | `JSON Valid`, 41 properties, both figures hashed and backed up, `Horizontal Channel` / `Vertical Channel`. |
| **Scan quality** | the notch is a clean dipole at (5.5, 7.75) mm, peak 0.944 V, SNR 18.4. |
| **The disk is worse** | `RxErr` has reached **16 630** and was climbing by 7 between two consecutive commands. §1. |
| **Speed** | measured, explained, and NOT what the end-of-scan message said. §2 is the whole of it. |

---

## 1. The NVMe, again

`tmp.log` from 10-16:

```
== /sys/bus/pci/devices/0000:89:00.0
RxErr 16623   ...   TOTAL_ERR_COR 16623
(seconds later)
RxErr 16630
```

Seven corrected receiver errors between two consecutive shell commands. It has
not caused a failure yet and it is not why the scan is slow — but it is a
hardware fault that has been running for at least three sessions and it is the
thing that will eventually corrupt a run. `lspci -s 89:00.0 -vv` printed
nothing in that log, which means it needs `sudo`:

```bash
sudo lspci -s 89:00.0 -vv | grep -iE 'LnkSta|LnkCap'
```

If `LnkSta` is below `LnkCap` in width or speed, the link has already
degraded. Reseat it, try another socket, or move `/data/scandata` to a
different disk — it is one path and nothing depends on where it points.

---

## 2. SPEED: what is actually limiting the scan

The 43.6-minute scan breaks down exactly:

```
72 passes x 27.0 s of sweep   = 32.4 min     TOTG's trajectory
72 passes x  8.5 s of overhead = 10.2 min    lift, transit, descend, plan
                                 -------
                                  42.6 min   (43.6 observed)
```

### 2.1 The arm is doing exactly what it was told

The end-of-scan report said:

> observed probe speed 1.1 mm/s over 72 passes (asked for 2.5)
> that is 0.43x what was asked for: ROBOT_VELOCITY_CALIBRATION is wrong here.

**It is not.** mtc prints a trajectory check for every pass, and over all 72:

| | |
|---|---|
| peak tool speed | **2.80 mm/s** median (asked for 2.5) |
| mean tool speed | **1.09 mm/s** median |
| mean / peak | **0.39** |
| worst joint at peak | 91% of its scaled limit |
| duration | 26.95 s median, 24.96 min, 37.32 max |

The calibration maps a scaling to a **peak** and the peak is right. The arm
then executes that trajectory faithfully — pass duration tracks the planned
duration to within the 8.5 s of per-pass overhead, including the one 45.8 s
outlier, which was a planned 37.3 s trajectory and not a stall. That message
has been rewritten to name both causes and say which line of mtc's output
tells them apart.

### 2.2 What costs the other 61%

Compare the same arm on different paths in the same session:

| path | waypoints | tool axis | mean / peak |
|---|---|---|---|
| transit | 2 | fixed | **0.94** |
| Azure sweep row | 5 | locked to world z | 0.98 |
| Keyence raster row | 21 | locked to world z | 0.86 |
| **EC scan pass** | **30** | **follows the surface normal** | **0.39** |

The EC pass is the only one that tracks the surface normal, and it is the only
one that loses most of its speed. This is the mechanism from the 10-09 stall
([[ec-stall-is-totg-pricing-turn-rate]]) still alive at a much smaller
magnitude: TOTG decelerates through orientation changes.

**And on this part the orientation change buys nothing.** The whole toolpath
tilts **up to 1.0 degree from vertical, mean 0.4** (18 x 29 mm, 10.7 mm of
relief). The smoother is already at its cap and says so:

> tool axis smoothed (linear, 20.0 mm -- widened from 6.0): turn rate
> 0.763 -> 0.818 deg/mm worst, 0.540 -> 0.182 at p99
> ** still over the 0.40 deg/mm bound at the 20.0 mm window cap, and widening
> was still helping -- the mesh is noisier than this can fix.

So the axis is being made to chase mesh noise across a surface that is flat to
one degree, and paying 2.5x the scan time for it.

### 2.3 The experiment, and it is cheap

**Fly one pass with the tool axis locked to vertical, and compare.** Nothing
else changes — same mesh, same region, same pitch, same instrument.

```
dry_run_ec_scan()          # note the per-pass trajectory duration mtc prints
```

Then lock the axis and do it again. If mean/peak goes from 0.39 to ~0.86, the
orientation tracking is proven to be the cost and the remedy is a rule — lock
the axis where the surface is flatter than some threshold, follow it where it
is not. If it does not move, the cost is the waypoint spacing instead and the
next knob is `EC_WAYPOINT_SPACING_M` (1.0 mm now, 30 waypoints per pass).

Bring back mtc's `Trajectory check:` line for both, verbatim. The numbers to
compare are `N points, T s` and `at up to X mm/s`.

### 2.4 The ceiling above it

Even at 0.39, the commanded 2.5 mm/s is not the robot's choice. It comes from

```
velocity 10.0 mm/s REQUESTED -> 2.5 mm/s USED
         scaled down to hold 10.0 traces/s; the ceiling is 10.0 traces/s
```

`MAX_TRACE_RATE_HZ["live"] = 10.0`, and that number was measured on an
**ultrasound** scan whose trace is a 16000-point waveform. An eddy-current
trace is one complex number. It has never been measured for this instrument,
and the evidence says it is far too low: the staging queue **peaked at 2** all
run, and the hard floor from the transaction window is 25 Hz, not 10.

Raising it to 25 would command 6.25 mm/s, which is also back inside the range
the arm was actually calibrated over (5.1–32.7 mm/s) instead of extrapolated
below it. **Two problems, one change.**

Measure it rather than assume it — §3 and §5.2.

---

## 3. FIRST, AND IT NEEDS NOTHING TO MOVE: `bench_ec()`

Put the probe on the part, leave the arm still, and run:

```
bench_ec()
```

Twenty seconds. It answers the question nobody has ever asked of this cell:
**what does the instrument itself allow?** The US-454A's frequency, drive,
gain, filter and phase are all set on its front panel and this config reads
none of them back, so its output bandwidth — which decides how fast the probe
may cross a flaw before the flaw is smeared — has never been measured.

It reports four things and the last is the point:

| | |
|---|---|
| block rate and gaps | is the card delivering what it was asked for |
| averaging efficiency | is the 100-sample block average buying its sqrt(100), or is the instrument's filter slower than the block and 98 of those samples redundant |
| mains fraction | is the noise floor hum (fixable) or broadband (not) |
| **v_max** | **the scan speed the instrument's own bandwidth allows** |

Read `v_max` against the 2.5 mm/s the scan commands:

* **near or below it** — the scan is already at the instrument's limit, §2 is
  not worth doing, and the 43 minutes is the right answer. (A 10 Hz output
  filter gives 2.0 mm/s against a 1 mm flaw, which is almost exactly what the
  scan runs at. This outcome is entirely possible and it would settle the
  question.)
* **far above it** — the instrument is not what the scan is waiting for, and
  §2.3 and §2.4 are where the 43 minutes actually goes.

It prints a verdict line saying which. Bring the whole report back.

The estimators are checked against filters of known corner in
`ec_bench_test.py` — within about 15% from 25 Hz to 400 Hz, biased high at
5–10 Hz. Good enough for a decision with a factor-of-1.5 threshold on it, not
good enough to quote as an instrument specification.

---

## 4. The clearance gate, if the region is smaller

10-16 planned 2171 poses, which is over `TOOLING_GATE_SAMPLE_ABOVE` (2000), so
the gate sampled and took **1 second**. 10-15 planned 1749, which is under it,
so the gate measured every pose and took **231 seconds in silence**.

Both are fixed as far as reporting goes — the gate now prints progress at any
length over 120 poses — but the 80x cost difference across that threshold is
real and still there. A smaller region can plan for far longer than a big one.
If the plan says

> planning time 0:04:49 (the clearance gate, every pose measured -- under the
> 2000-pose sampling threshold, so it costs ~132 ms each; it reports its own
> progress while it runs)

that is now expected, announced, and not a hang.

---

## 5. The run

### 5.1 Settings

Unchanged. Rotation 86.5 again — but see 5.3.

```
set_probe_rotation(0)
set_ec_preload(0.002)
set_scan_step_size(0.00025)
set_rotation(86.5)
```

```
ec_serpentine()   probe_rotation()   ec_preload()
scan_velocity()   scan_step_size()   projection_mode()   ec_rotation()
```

### 5.2 One short scan with projection off

Before the real one, and only if §3 says the instrument has headroom:

```
set_projection_mode("deferred")
```

`MAX_TRACE_RATE_HZ["deferred"]` is 22 Hz, so the same scan commands 5.5 mm/s
instead of 2.5. Run **four passes** of it and read three things:

* the staging queue depth in the end-of-scan report (anything under 16 is fine)
* `Max Correlation Wait` in the pass metadata (71 ms on 10-16; it is the margin)
* mtc's per-pass peak and duration

If that holds, the live ceiling is the thing to raise and the image can be
rebuilt afterwards with `replay_ec_scan()` — deferred records the same traces
and projects them at the end for an identical picture.

### 5.3 Watch for the lift-off angle being refused

10-16 ended with:

> could not measure the lift-off rotation (anisotropy 2.86, under 3.00, so
> there is no lift-off direction to find and any angle would be arbitrary.
> That is what a scan over a part with no lift-off variation looks like —
> and also what a dead instrument looks like.)

The scan was fine, so it is the first of those two. But the ratio has been
falling — 6.45 on 10-15, 4.14 on 10-09, **2.30** on 10-16 — while the flaw
channel noise has been rising, 0.032 → 0.051 V. Both point the same way: the
signal is getting smaller relative to the noise floor. **`bench_ec()` is the
measurement that says whether that is the instrument's gain.**

The NORMS plug-in now applies the same threshold, so the record says
`Lift-off Angle Determinate: no` and withholds the angle rather than writing
down a number the pipeline calls arbitrary.

### 5.4 Lines to read in the plan

| line | expect |
|---|---|
| `planning time` | honest now; minutes for a small region, seconds for a big one (§4) |
| `Sample Rate` / `Block Span` in the record | **NEW and the point of §3.** The card's rate is recorded now, so `Block Smear` says how far the probe moved during one trace's average. If the smear approaches the 0.25 mm pitch, the resolution is set by the block and not by the step size |
| `tool axis smoothed ... still over the ... bound` | expected on this mesh; §2.2 |
| `region_coverage` | 77% on 10-16 — the ROI was drawn wider than the fine sweep measured. Draw inside the swept area |
| `IMPOSSIBLE PENETRATION` | absent |

---

## 6. Still open, unchanged

**The acquisition lag.** 0.140 s at 3.93 mm/s, one scan, one speed. Still the
oldest open question and still worth 25 minutes once §2 and §3 are settled —
and §5.2's faster scan would give the second speed for free.

**The per-pass overhead.** 8.5 s x 72 passes = 10.2 min of the 43.6. Two
2-pose moves at 2.54 mm/s plus a planning round trip. Not yet looked at.

**The rare spatialnde2 background-thread crash.** Two sightings, never
reproduced. Did not happen on 10-16.

---

## 7. What to bring back

1. **The whole `bench_ec()` report.** It is the one measurement that has never
   been made and it decides whether anything else in §2 is worth doing.
2. mtc's `Trajectory check:` line for a normal pass, and for a pass flown with
   the tool axis locked to vertical (§2.3).
3. The end-of-scan report, verbatim — it now names both causes of a slow scan
   instead of asserting one.
4. The `ec_scan_*` directory, `dataguzzler.log`, `ros.log`, `dmesg.log`.
5. `sudo lspci -s 89:00.0 -vv | grep -iE 'LnkSta|LnkCap'` and the AER counters.
6. Whether the lift-off ratio is still falling (§5.3).
