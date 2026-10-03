# Cell visit, Monday 2026-10-05

Written 2026-10-03, after the 2026-10-02 run (`results18`). Everything below
either **verifies a fix that has not yet run on hardware**, or **answers a
question that is blocking work**. Each item says what to run, what to write
down, and what the answer looks like, so that running it produces a number
rather than an impression.

The tests are independent. Run any subset in any order. **Section 1 costs about
twenty minutes and unblocks the most, so do it first even if nothing else gets
done.**

Nothing here needs a code change. Everything new defaults to the behaviour the
cell already flew, so a run that touches none of these knobs is the run you did
last time.

---

## 0. Before anything

```bash
cd ~/tari && ./tari.bash --layer site --resync
```

The workspace script was substantially repaired after the last run, where its
build phase finished **zero** packages. Expect it to behave differently:

- It now refuses to start if `ros_ws/build` or `ros_ws/install` holds another
  account's unwritable files, and names the file and both recoveries. That is
  the failure that produced "0 packages finished" last time.
- Dirt that is only line endings is now reported as line endings and discarded
  freely; **real** content changes need `--yes` as well. Last time 224 CRLF `.py`
  files were reported as work in progress and discarded on your behalf.
- It warns if `setuptools` in your own `~/.local` shadows the system one, which
  the previous warning claimed to have fixed and could not.

Machine-side items from that run, still outstanding — see the commit message on
`tari_workspace_staging@iron` for the full list:

```bash
# the global safe.directory the first run needed is no longer required, and it
# trusts every repository on the box, as root
sudo git config --global --unset-all safe.directory
     git config --global --unset-all safe.directory

# setuptools 84 in your own home; sudo cannot reach it. Every account that builds.
python3 -m pip uninstall -y setuptools
python3 -c 'import setuptools; print(setuptools.__version__, setuptools.__file__)'
```

Then bring the cell up as usual and **read three new lines before moving the
arm**:

| line | means |
|---|---|
| `Mount check: world -> base_link agrees with TF to … mm and … deg` | the planner and the rest of the cell agree about where the arm is bolted |
| `projection surface UV mirrored back…` or nothing | the parameterization is usable |
| `Raster time parameterization: TOTG (retime_backend 'totg')` | today's timing, unchanged |

If instead you see **`MOUNT MISMATCH`**, stop and send me the line. It means the
planner is interpreting every Cartesian waypoint in the wrong place, and
nothing else will look wrong because joint trajectories still execute.

---

## 1. The instrument — do this first, no robot motion needed

This is the highest-value twenty minutes of the visit. It unblocks auto-ranging,
the gain solve, and the re-fly of the tilt sweep.

**Setup:** instrument powered, probe hanging free in air, nothing touching it.
Config started as usual.

### 1a. The link, and what it is

```python
ec_instrument.link_check()
ec_instrument.identify()
```

`identify()` replaces the old `*IDN?` probe, which could never have worked —
`*IDN?` is SCPI and the protocol reserves `*` for the first character of a
*response*. Expect the US-454A banner: P/N 99195, S/N 335, app 1.8.

`link_check()` compares the configured baud rate against what the instrument
reports. **Note the module's own default is 115200 and this instrument runs at
9600**; the scan config passes 9600 explicitly, which is the only reason it has
ever worked.

**Record:** both outputs verbatim.

### 1b. WHICH CONTROL MOVES THE ±5 V MAPPING — the blocking question

You have said all four matter and `GAIN` dominates. This measures it, which is
what an auto-ranging routine needs.

With the probe **in free air and nulled**, step `GAIN` and watch the DAQ:

```python
ec_null()                      # or ec_instrument.null()
for g in (0, 6, 12, 18, 24):
    ec_instrument.set_gain(g)
    print(g, recdb.latest.ref['/ec/traces'].data[-1])
```

Then repeat for `set_sensitivity`, `set_spread`, `set_probe_amplitude`, one at a
time, returning `GAIN` to its starting value between each.

**Record:** a table of control setting against the signal value. **What you are
after is volts of DAQ swing per unit of each control.** With that, auto-ranging
is a solve rather than a search.

**Why it matters:** on 2026-10-02 the `+6°` tilt pass had **64 of 64 samples
pinned at exactly −5.0000** on the imaginary channel, and `+3°` had 4 of 64. Six
degrees of tilt consumed the entire input range. Until this table exists, any
gain change is guesswork.

### 1c. Re-range for the tilt sweep

Using 1b, set the gain so that **the whole ±6° tilt sweep stays inside ±5 V**,
with headroom. Target roughly 60–70% of full scale at the largest excursion.

**Record:** the final `GAIN`, `SENSITIVITY`, `SPREAD`, `PROBEAMP`, and the
frequency. `ec_instrument.setdict` dumps everything if that is easier.

---

## 2. Re-fly the tilt sweep (test 12), now that it is readable

**Specimen:** the same taught line as 2026-10-02 — `contact_a` to `contact_b`,
63 mm, on the flat. **Do not touch the instrument during the run.**

```python
capture_session(specimen="99073 flat", notes="re-fly at reduced gain")
capture_contact()
```

Defaults are tilts 0, ±3, ±6 about the travel axis, 20 mm approach, 10 mm/s,
`force_limit_n` 25.

Then, offline, on any machine:

```bash
python3 ROS-Scanning/dgpfiles/modules/ec_liftoff.py <take_dir>/ec_*.npz
```

**What good looks like:** no pass reported as `SATURATED`, and the axis
`linearity` above about 0.9. Last time two passes were railed and excluding them
moved linearity from 0.649 to 0.904 and the ruler residual from 0.96 to 0.27 —
the clipping was what made the whole take look untrustworthy.

### 2a. Fly each pass in BOTH directions

If there is time, repeat the sweep travelling `contact_b → contact_a`.

**Why:** the within-pass trend is currently ambiguous. A registration error is a
function of *position* and reverses sign with travel direction; drift is a
function of *time* and does not. All five passes last time went one way, so the
two cannot be separated — and I gave you the wrong answer on it once already,
calling the trend "registration" when the measured drift of **0.016 signal
units/s**, consistent across all five passes and nearly independent of tilt,
accounts for a quarter to a half of it.

---

## 3. The notches (Uniwest 99073 aluminium radius standard)

**Specimen:** the cal plate itself, flat side up.

### 3a. Photograph the markings

A phone photo, close, square on, of the markings at the **left third** of the
plate. They exist — I found them in `scene_0200.jpg` from 2026-09-20 — but at
roughly 1.3–2.5 mm per pixel a 4 mm character is 2–3 pixels, so no processing
recovers them. Thirty seconds with a phone settles it.

The published spec gives **lengths only**: 0.100″, 0.080″, 0.060″, 0.040″
(2.540, 2.032, 1.524, 1.016 mm), on the flat, on the corners, and along each of
the four radii (0.500″, 0.375″, 0.250″, 0.125″). **Depth is not published.** At
2 MHz the skin depth in aluminium is 0.071 mm, so any realistic EDM notch is
2–10 skin depths — electrically deep, response saturating with depth — which is
probably why length is what Uniwest quotes.

### 3b. Keyence raster of the plate

Measure the notches rather than reading the label: the label gives nominal
sizes, the Keyence gives **positions in part coordinates**, which is what a
calibration routine needs and the label does not carry.

**Caveat to record with the data:** EDM notches are typically 0.1–0.25 mm wide,
and laser triangulation cannot see the bottom of a narrow slot. **Treat Keyence
depth as a lower bound**; trust it for position.

**Record:** the reconstruction, plus which notch is which.

### 3c. Note for later — the tilt-over-notch run

Once 1, 2 and 3 are done, the calibration run becomes possible: the same tilt
ladder, flown so the probe crosses notches of known size. That gives, for the
first time, two *independent calibrated* axes — tilt in degrees along lift-off,
notch depth in mm along defect — and from them the conditioning rotation as a
solve rather than by eye, absolute sensitivity, and the **(tilt × notch) matrix**
that says how much detectability each degree of lift-off costs. That last number
is what would justify a spring change, and nothing we have measured can produce
it yet.

It needs 1b (gain control), 3a and 3b (sizes and positions), and a taught line
that crosses the notches. **Notches exist in two orientations**, so expect to fly
it twice with the tooling rotated 90° — and check wrist-3 reachability first,
because the measured no-tilt cable band is +16.1°..+301.1° and a 90° rotation
must land inside it rather than be resolved by ±360.

---

## 4. Verify the three fixes that have never run on hardware

All three are defaults-on and should need nothing but a normal scan.

### 4a. The projection actually lands

Run any scan. **Watch for the `NOTHING IS REACHING THE PROJECTED IMAGE` line
after pass 1.** It should not appear.

Two of the three runs on 2026-10-02 projected 1876 and 1673 traces into a
completely empty image with no error raised anywhere, because the UV
parameterization came out **mirrored** — a coin flip in the unwrap, which made
spatialnde2 take the square root of a negative determinant and zero every
weight. 18 of 34 stored surfaces across four result sets were affected. The
unwrap is now pinned and an existing bad surface is mirrored back before the arm
moves.

### 4b. The traces reach the disk

Mid-scan:

```python
ec_save_status()
```

Afterwards:

```python
list_ec_runs()
replay_ec_scan()                      # newest run, as conditioned
replay_ec_scan(rotation_deg=40.0)     # same traces, different rotation
```

An eddy-current scan previously wrote **nothing** to disk at any point — which
is why all 59 completed passes of the 2026-10-02 run were lost when the process
was killed. One `.npz` per completed pass now lands beside the meshes, carrying
the raw complex impedance as well as the conditioned value, so the conditioning
rotation can be chosen *after* the fact.

**Record:** the run directory, and whether `replay_ec_scan()` reproduces the
live image.

### 4c. If it freezes again

The 2026-10-02 freeze produced no output at all for 27 minutes. If staging
stalls now you should get, once:

```
** STAGING STALLED: still waiting after 120 s -- N trace(s) queued, …
   recdb: <latest ready>/<latest defined>
```

**Send me that line.** `ready` lagging `defined` means a globalrev that cannot
complete; equal means a stage that is merely slow. It distinguishes the
candidates in one line.

---

## 5. The arm tilting without advancing

**This is real, not an optical illusion, and it is measurable.**

Measured on the mesh the 2026-10-02 scan actually flew
(`…_ecplan_roi.ply`, 8073 edges), the angle between normals at the ends of each
mesh edge:

| | |
|---|---|
| p50 | 0.27° |
| p95 | 0.67° |
| p99 | 0.87° |
| p99.9 | 1.88° |
| **max** | **8.73°** |
| edges turning > 5° | **8 (0.10%)** |

So the surface is smooth almost everywhere and then has **eight discrete spots**
where the commanded normal jumps 5–8.7°. The pose is specified at `probe_end`,
so a tilt rotates about the tip: the probe tip barely moves while the wrist
swings through `0.192 m × θ`.

| commanded tilt | wrist travel | per 1 mm of advance |
|---|---|---|
| 0.27° (median) | 0.9 mm | 1× |
| 1.88° (p99.9) | 6.3 mm | 6× |
| **8.73° (max)** | **29.3 mm** | **29×** |

And because TOTG bounds *joint* speed rather than tool speed, that segment also
takes roughly 30× longer than a median one. **The arm really does stop advancing
and swing 29 mm of wrist to move the probe 1 mm, and it dwells there while doing
it.** Eight times per ROI.

### What to do about it, in increasing order of change

The machinery already exists. `mesh_reference_normals()` computes a smooth
normal field from **positions alone** — a plane from the local covariance, then
a quadric, so it is correct on a curved part rather than merely blurred. It is
currently used only as a *repair*, and only where the mesh disagrees with it by
more than **15°**. Every one of these eight outliers is under 15°, so the repair
leaves all of them alone. The 2026-10-02 log says exactly that: `0 of 2765
(0.00%) … nothing needed repairing; planning on the mesh as it is`.

**Step 1 — measure it on the day, before changing anything.** Per-trace poses
are now saved (4b), so after a scan the achieved orientation along a pass can be
plotted against arc length and compared with the commanded path. That confirms
whether the arm is tracking the command or overshooting it.

**Step 2 — tighten the repair tolerance.** One parameter. Setting it to about
**3°** would replace those eight outliers with the smooth field and touch
nothing else, since p99.9 is 1.88°. This is the smallest change, uses code that
is already tested, and the existing log line reports exactly what it did.

**Step 3 — if that is not enough, take the toolpath orientation from the
reference field directly**, keeping mesh *positions*. Position and orientation
have different noise requirements and there is no reason to take both from the
same noisy source: the probe must be *where* the surface is, but it only needs
to be *roughly* normal to it, because it is spring-loaded and has a finite
contact footprint.

**The risk, stated honestly:** over-smoothing makes the probe less normal to the
surface on genuinely curved regions, which is lift-off. The reason to prefer
step 2 is that it is bounded — it only touches points that disagree with their
own neighbourhood by more than 3°, which on a real surface is a reconstruction
artefact rather than geometry. The radius standard's 0.125″ (3.2 mm) radius is
the case where this would bite, and that is a good reason to measure on the flat
first.

**One more thing worth knowing:** the arc-length retimer added this week bounds
`ds/dt` in **translation only** — it explicitly refuses a pure reorientation
rather than bounding it. So if we move to that backend, an orientation-rate
bound has to be added alongside, or these eight spots get *worse*, not better.

---

## 6. Optional, if everything else is done

### 6a. One row through the arc-length retimer

Demonstrated in simulation on 2026-10-03: it holds **exactly** the commanded
10.0 mm/s where TOTG managed 9.2 mm/s peak and took 9× as long over the same
59 mm row. It has **never run on the arm.**

```python
# one row, nothing else, and watch it
# retime_backend is a runtime parameter on /mtc_hub_node
```
```bash
ros2 param set /mtc_hub_node retime_backend toppra
```

**Two things to set deliberately first, and a reason to be cautious:** with
`retime_joint_scale` at its default of 1.0 it used **95% of a joint limit** on
that row, against TOTG's 91% of a limit scaled to 0.47% — roughly 200× more
joint margin consumed. And it emitted 125 trajectory points where TOTG emitted
1129. Neither is a defect, both are parameters, but **do not fly a whole scan on
it**; fly one row, with a hand near the stop, and compare the
`Trajectory check:` lines.

### 6b. Confirm the stiffness measurement

`k_series = 10.0 ± 3.6 N/mm`, measured from the 2026-10-02 bag during the 20 mm
plunge. That lands exactly on the threshold that says *a softer spring is the
entire fix* rather than force control. It includes a dynamic component
(overshoot 1.2–1.8×), so the quasi-static value is probably somewhat lower.

A step-and-dwell ladder — command penetration in 0.5 mm steps, pause 2 s at
each, log the wrench — would settle it. It will not change the conclusion, but
it is the difference between a measurement and an estimate, and the spring
specification depends on it.

---

## Recommended order if the day is short

1. **§1** instrument — 20 min, no motion, unblocks the most
2. **§3a** photograph the markings — 30 seconds
3. **§1c + §2** re-range and re-fly the tilt sweep
4. **§4** verify the three fixes during whatever scan you run anyway
5. **§3b** Keyence raster of the plate
6. **§5 step 1** measure the orientation wobble from the saved poses

§5 step 2, §6a and §6b are all worth doing and none of them blocks anything
else.
