# Diagnostic test plan — the open questions, and exactly how to settle each

Every item here answers a question that is currently **open**, and each is
written so that running it produces a *number* rather than an impression. They
are independent: run any subset in any order, and skip the rest without
breaking anything.

Nothing in this document requires a code change. The code is on
`ROS-Scanning_staging@main`, `tari_ros_staging@iron` and
`dgpython-ros2motion_staging@iron`. Everything new defaults to the behaviour the
cell already flew, so **a run that touches none of these knobs is the run you
did last time.**

Budget: tests 1–3 cost 15 minutes total and should be done every visit. Tests
4–9 are the motion work and need a specimen. Tests 10–12 are the
force/lift-off question, and 12 is the one that can cancel a whole project.

---

## 0. Before anything moves

### 0.1 Pull and build

```bash
cd ~/dataguzzler-work-dir && ./tari.bash --resync
```

```bash
cd ~/ros_ws && colcon build --symlink-install --packages-select tari_cpp_mtc tari_ur_description tari_toppra
```

`tari_ur_description` matters even though only YAML changed — `gd_max_iters` has
to reach both nodes and ROS reads it at launch. `tari_toppra` is new and loaded
by nothing; building it only makes it importable.

### 0.2 Bring up the cell

```bash
source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
export TARI_CELL_PATH=tari_cell_configs
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```

Load and **play** External Control on the pendant.

### 0.3 Start the scan config

```bash
cd ~/dataguzzler-work-dir/ROS-Scanning && dataguzzler-python --ipython dgpfiles/ECAutomatedScan.dgp
```

Everything below that reads `dgpy>` is typed at that prompt. Everything in a
plain shell block is a second terminal with ROS sourced.

### 0.4 Specimen requirements, once, covering every test below

| test | specimen needed |
|---|---|
| 1, 2, 3 | none — bring-up only |
| 4–9 | **a part with a known ROI**, centred near x 0.85, y 0.00 |
| 10 | **none** — free space, nothing within a metre of the tool |
| 11 | a **flat scrap of the specimen material**, or the specimen itself |
| 12 | the specimen, with `contact_a` / `contact_b` taught on it |

For tests 4–9 any part that has produced rows before is fine; **reuse the same
one you benchmarked rows on**, because test 4 has to be compared against that
benchmark and not against a fresh part. Keep it clear of x ≥ 1.06 — the arm
cannot reach the Azure standoff beyond there.

---

## 1. The IK plugin actually loaded — 1 minute, every visit

**What is being tested.** A missing `pick_ik` makes **every scan pass move 0 mm**
while the logs say "planning failed", so it reads as a toolpath problem. It is
an `apt` package and **every container recreate loses it.** `mtc_node` now
asserts it at startup.

**Run.** Read the `mtc_hub_node` startup output.

**Pass.**

```
IK solver for group 'ur_manipulator' on this node: pick_ik/PickIkPlugin, loaded.
```

**Fail.** Either `No IK solver is configured...` or `...DID NOT LOAD`. Then:

```bash
ls /opt/ros/iron/lib/libpick_ik_plugin.so || sudo apt-get install -y --no-install-recommends ros-iron-pick-ik
```

**Do not check `/move_group` instead.** It loads the same `kinematics.yaml` with
only `mode` overridden, so it reports `pick_ik` too — which looks like a
confirmation and tells you nothing about `mtc_hub_node`.

## 2. The guard values and the tool-speed mode are what you think — 1 minute

**What is being tested.** Two runs with different speed semantics produce
identical logs everywhere else, so the mode has to be read at the top.

**Run.** Read two banner lines.

From `mtc_hub_node`:

```
Raster execution guards: jump ... speed excess 100% secant / 10% analytic,
tool ceiling 0.050 m/s, tool-speed scaling ceiling factor 1.0x, ...
```

From the `dgpy>` startup banner:

```
  tool speed: CONVERTED THROUGH THE CURVE (ROBOT_VELOCITY_CALIBRATION) ...
```

**Record** the ceiling factor and the tool-speed line. If the second says
`COMMANDED IN M/S`, a previous session left the solver on and tests 4–9 are not
comparable to earlier runs.

## 3. Where the 1327 mm lead-in comes from — free, on the first Cartesian move

**What is being tested.** In simulation **every** Cartesian move reported about
1327 mm of lead-in, even with waypoint 0 at the FK pose of the tool. That is
larger than the arm's reach from the tool, so it is a frame or origin problem,
not a motion — and it currently makes the simulated cell useless for motion
testing. MoveIt is ruled out as the source. Two candidates remain: a pose array
in a frame that was being silently discarded (**now fixed**, so this may already
be gone), or a toolpath planned in a **mesh's own coordinates** rather than
world, which puts waypoint 0 at the world origin.

**Run.** Nothing. Any Cartesian move prints it.

**Read.**

```
Lead-in geometry: tool 'probe_end' is at (x, y, z) |NNN mm from the model
origin|; waypoint 0 is at (x, y, z) |NNN mm from the model origin|;
lead-in NNN mm.
```

**Interpret.**

| what you see | conclusion |
|---|---|
| lead-in under ~10 mm | no problem; the frame fix resolved it |
| waypoint 0 within a few mm of the origin | **the toolpath is in mesh coordinates**, not world |
| lead-in ≈ the tool's own distance from the origin | same conclusion |
| a `Transforming N waypoints from '<frame>'` line appears | the frame was being discarded and is now corrected; note the offset |

**Record the three numbers verbatim.** This is the one test whose result unblocks
using simulation for motion work at all.

---

## 4. Does raising `cost_threshold` stop IK refusing large moves?

**The highest-value motion experiment here.** Allow 30 minutes.

**What is being tested.** `cost_threshold` and `minimal_displacement_weight`
have been treated as one knob and they are two. From pick_ik's source,
`cost_threshold` gates **only** the auxiliary goals — never position or
orientation, which have their own thresholds. So with minimal displacement as
the only auxiliary goal it is purely a **maximum displacement**, refusing once

```
|| (q - seed) . f ||  >=  cost_threshold / minimal_displacement_weight
```

with `f = [3/14, 3/14, 1/7, 1/7, 1/7, 1/7]` from this arm's velocity limits. At
today's `0.001 / 0.01 = 0.1` a single **wrist** joint is refused beyond
**0.700 rad = 40.1°** and an **inner** joint beyond **0.467 rad = 26.7°**.

A 90° wrist turn is 1.571 rad — which is exactly the refusal
`motion_profiles.yaml` already records as measured. Raising `cost_threshold` to
0.01 lifts those radii to **401°/267°**, past anything the arm can do, while
leaving the descent *steering* untouched — and steering is the half that
actually keeps a pass in one configuration.

**Mechanism confirmed from source; consequence NOT measured.** That is what this
test is for.

### Procedure

1. Run your **existing row benchmark first, unchanged**, and write down rows
   entered and rows completed. Without a before, there is no after.

2. Raise the threshold on the node that plans rows:

```bash
ros2 param set /mtc_hub_node robot_description_kinematics.ur_manipulator.cost_threshold 0.01
```

3. Re-run **the same rows on the same part in the same session.** Do not change
   the ROI, the spacing or the speed between the two halves.

4. Revert:

```bash
ros2 param set /mtc_hub_node robot_description_kinematics.ur_manipulator.cost_threshold 0.001
```

### What to record

| | before | after |
|---|---|---|
| rows entered / attempted | | |
| rows completed | | |
| median completion % of a short row | | |
| any pose refused that previously solved | | |

**Success** is more rows entered and completed with **no** new refusals and no
row changing configuration mid-pass. **Failure** is a row that completes but
flips branch partway — that would mean the steering mattered more than the
mechanism predicts, and the per-node weight split should stay.

Three currently-open symptoms are the right shape for this: the 156° home wrist,
the 180° cylinder-flank swing, and scan-profile IK failing on approaches of
about 0.25 m. If any of those clears, that is the result.

## 5. Is the analytic speed reading trustworthy enough to tighten the guard?

**What is being tested.** The guards differentiated *positions* across TOTG's
0.1 s grid, and a secant across an acceleration change reads fast — the worst
*honest* reading on a healthy run was **136%** of a scaled joint limit on a row
whose tool was doing 10.7 mm/s against a 10 mm/s command. That is why
`max_speed_excess` had to be opened to **1.00**, of which roughly half is
headroom over legitimate readings rather than over faults, so nothing inside it
can be told from nothing. The trajectory now also carries `q̇`, and tool speed is
`|J(q)·q̇|` at the waypoint.

**Run.** Nothing. Any raster row prints both.

**Read, once per pass:**

```
Trajectory check, derivative comparison: ANALYTIC governs. joint NN% of its
scaled limit (secant said NN%); tool N.N mm/s (secant said N.N mm/s).
```

**Record** both pairs over 20–30 rows — a quick `grep` of the log is enough.

**Interpret.** If the analytic joint figure sits near 50% where the secant
reported 100–136%, the secant was the problem and `max_speed_excess` can come
down from 1.00 toward 0.25 on evidence. If the two agree closely, the allowance
is real and must stay. **Do not tighten anything during this visit** — this test
only collects the pair.

A line saying `SECANT governs` means the trajectory carried no velocities, which
is worth knowing about on its own.

## 6. Is the ramp the reason short passes never reach speed?

**What is being tested.** Uniform time scaling divides velocity by *k* and
acceleration by *k²*. With `acc == vel` the ramp **time** is `v/a` whatever *k*
is, so the ramp never shrinks as the move slows while the row it has to fit
inside does. A pass shorter than two ramps never reaches its commanded speed and
peaks at `sqrt(a·L)`.

**Run.** Nothing. Every Cartesian move prints the ratio and the path length.

**Read.**

```
Scalings: velocity G, acceleration G (acc/vel N.NN) over NNN.N mm path. ...
```

and on short passes, a WARN when `acc/vel < 1.0`.

**Record**, for the shortest 10 passes: the path length, the `acc/vel` ratio, and
the **achieved peak** from the trajectory-check line above it.

**Interpret.** A peak well under the command on a *short* path with `acc/vel < 1`
is the ramp not fitting. That makes the case for deriving `acc` from the ramp
distance instead of pinning it to `vel` — a change deliberately **not** made
yet, because it would alter the speed of every pass that currently flies.

## 7. Where does the planning CPU actually go?

**What is being tested.** FCL caches collision geometry in a
`static thread_local` cache, and `mtc_node` hands **every action goal a freshly
detached thread** — so the cache starts empty on every goal and the specimen's
geometry, up to **60,000 triangles**, is rebuilt on the first check of every
move. A scan issues hundreds of moves. Upstream reports the same pattern
(moveit2 #3691; moveit #3515, where building the BVH for a large mesh took over
99% of planning time).

This is a **hypothesis**, and the point of the test is that it can be retired on
one line of evidence.

**Setup.** The specimen **must be in the planning scene** or there is nothing
large to rebuild:

```
dgpy> specimen_to_planning_scene()
```

**Run.** Any raster rows.

**Read.**

```
Collision checks: NNNN, N refused. First check NN.N ms; remaining NNNN averaged
N.NNN ms (NN.N ms total). First/rest ratio NNx -- ...
```

**Interpret.**

| ratio | conclusion |
|---|---|
| **> ~50x** | confirmed. The fix is to stop making a thread per goal — a persistent worker per handler keeps the cache warm. Worth real effort. |
| ~1–5x | retired. The CPU goes elsewhere; do not spend more time on it. |
| in between | record the absolute first-check milliseconds and the number of moves per scan, and multiply. That product is the whole prize. |

**Also record** the triangle count `specimen_to_planning_scene()` reports. If the
ratio is high, halving `SPECIMEN_SCENE_MAX_TRIANGLES` is a second, cheaper lever.

## 8. Optional: the absolute jump threshold

MoveIt's **relative** jump factor cannot fail on a path with fewer steps than the
factor — arithmetic, since each step is compared to the *mean*. The absolute
test is the right shape and **truncates** at the jump, reporting a fraction,
where our own check **refuses** the whole trajectory. For a surface-following
toolpath with `min_fraction < 1`, a truncation is strictly better: the pass
covers what it can instead of nothing.

```bash
ros2 param set /mtc_hub_node raster_jump_revolute 0.1
```

0.1 rad matches `raster_max_joint_jump`, so the two agree. **Record** whether
rows that previously came back refused now come back partial. Revert with `0.0`.

## 9. Does commanding m/s beat the calibration curve?

**What is being tested.** `vel` is MoveIt's `max_velocity_scaling_factor` — a
fraction of every joint's limit — and this pipeline converts a wanted m/s into
one of those fractions through a **scalar curve fitted by driving one straight
line at five speeds.** The curve does not transfer: re-measuring moved the
eddy-current figure from 0.0064 to 0.0047 for the same commanded 10 mm/s.
`mtc_hub_node` can instead solve for the scaling against the real kinematics of
the actual path — parameterise, measure peak tool speed by FK, correct, repeat.

**What it buys, honestly.** The caller's scaling is the solver's **ceiling** as
well as its guess, so a solved move can only come out *slower*. Every shortfall
measured last run went the other way:

| scaling | commanded | measured peak | ratio |
|---|---|---|---|
| 0.00254 | 6.0 mm/s | 5.8 | 97% |
| 0.00473 | 10.0 | 8.5 | 85% |
| 0.00630 | 13.0 | 8.0 | 62% |
| 0.01649 | 32.7 | 20.4 | 62% |

Reaching the commanded speed in those cases needs a scaling **above** the
curve's, which the ceiling forbids. So switching it on alone buys a *guarantee*
— the tool never overruns what it was commanded at, measured rather than
predicted — and **not** throughput. The throughput half needs the ceiling
raised.

### Procedure

1. Run a few passes with the curve, as normal, and note the
   `Trajectory check` peak for each.

2. Turn the solver on:

```
dgpy> set_tool_speed_solver(True)
```

3. Re-run the same passes. Read, per pass:

```
Tool speed solved: NN.N mm/s commanded, NN.N mm/s measured at scaling G.
The caller's scaling G was the guess; the ceiling was G (factor 1.0x). ...
```

**If that line ends with `THE CEILING BOUND IT`**, the commanded speed was not
reached *because the solver was not allowed a higher scaling* — which is exactly
the case the table above predicts. Only then:

```bash
ros2 param set /mtc_hub_node raster_tool_speed_ceiling_factor 3.0
```

4. Re-run again. **Record** commanded vs measured peak and the per-pass cycle
time for all three conditions.

**Revert** with `set_tool_speed_solver(False)` and ceiling factor `1.0`.

**Interpret.** Raising the ceiling is bounded by two things that still measure
the outcome before anything executes: the commanded m/s itself, and
`raster_max_tip_speed` (0.050 m/s), which refuses the whole trajectory. So the
risk is a refused pass, not an overspeed one. **If a pass is refused on tip
speed after raising the factor, lower it** — do not raise `raster_max_tip_speed`
to match.

---

## 10. Is the "very noisy" wrist force actually noise? — 10 minutes, free space, zero risk

**This is the test that answers the force-control question, and it never touches
the specimen.**

**What is being tested.** UR's own UR10e factsheet gives the tool-flange force
sensor as range ±100 N, precision ±5.0 N, **accuracy ±5.5 N**. On a 7.5 N
setpoint that is ±73%. But most of a *published* accuracy figure is bias, and
bias is removable by taring — so the live reading may be dominated by
**uncompensated tool gravity as the wrist reorients**, which is bookkeeping, not
sensor noise. This test separates the two.

**Specimen: none.** Park the tool with nothing within about a metre.

### Procedure

1. **Let the arm sit powered for 15 minutes before taking numbers** (thermal
   bias). Note the start time.

2. Confirm the configured payload — mass and CoG for the full EC tooling — and
   **write down what it is set to.** The URDF models ~0.39 kg with the probe end
   ~192 mm out; check the installation against that.

3. Park at the orientation a scan starts in. **Tare** the sensor. Hold still for
   60 s and log:

```bash
ros2 topic echo /force_torque_sensor_broadcaster/wrench --once
```

```bash
timeout 60 ros2 topic echo /force_torque_sensor_broadcaster/wrench > ~/ft_still.txt
```

→ **σ of Fz over that window is the real short-term noise floor.** This is the
number that replaces "very noisy". Expect it far below ±5 N.

4. **Without re-taring**, jog the wrist slowly through the full range of
   orientations the toolpath uses — at minimum ±30° about two axes — logging
   throughout.

```bash
timeout 120 ros2 topic echo /force_torque_sensor_broadcaster/wrench > ~/ft_reorient.txt
```

→ **This is the decisive measurement.** The driver publishes payload-compensated
tool-frame force, so with the payload right, Fz should stay flat to a few tenths
of a newton. **If it swings by order m·g — 3–4 N for this tooling — the
configured payload is wrong**, and most of what you watched is removable.

5. Repeat step 4 with a deliberately wrong payload (0 kg) and compare. The
difference *is* "how much is removable by calibration", measured.

6. Two-minute frame check: push the probe gently by hand along its own axis with
the tool at a non-axis-aligned orientation. The response must appear in **Fz**,
not spread across Fx/Fy.

### Deliverable — four numbers

| | N |
|---|---|
| σ Fz, stationary, 60 s | |
| Fz peak-to-peak through reorientation, payload configured | |
| …the same, payload 0 kg | |
| σ Fz after a 20 Hz low-pass | |

Those four settle the skepticism quantitatively and cost nothing.

## 11. What is `k_series`? — 20 minutes, light contact

**The highest-value single number in the whole force question**, and it needs no
force control.

**What is being tested.** The EC probe is **already spring-loaded** — the URDF
models the plunger (`outer_shell` → `bearing_tube` → `probe_holder`) as a chain
of *fixed* joints, i.e. the compliance exists in hardware and is modelled as
rigid. Today's contact force is therefore

```
F = k_series x (commanded preload + mesh error)
```

and `k_series` — the series combination of the spring, the 192 mm tooling tube,
the arm and the specimen — **has never been measured.** Everything about whether
force control is needed turns on it.

**Specimen.** A flat scrap of the specimen material, or the specimen itself.

### Procedure

1. **Lower the force limit for this run.** 60 N is 6–12× the intended setpoint;
   it protects against gross error but will happily let the probe be pressed to
   50 N.

```
dgpy> globals()['CAPTURE_CONTACT_FORCE_LIMIT'] = 25.0
```

**`globals()[...]`, not a bare assignment**, and this is not a style choice.
These settings are installed into the config's globals by a loop in
`cell_capture_node.dpi`, whose own comment says so. A bare
`CAPTURE_CONTACT_FORCE_LIMIT = 25.0` at the prompt lands somewhere that is not
the globals the capture code reads, so it appears to work, prints no error, and
**the limit stays at 60 N.** Confirm it took:

```
dgpy> CAPTURE_CONTACT_FORCE_LIMIT
```

2. Bring the probe to light touch.

3. Step the commanded position **into** the surface along the tool axis in known
   increments — 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 mm — pausing ~3 s at each.
   Log the wrench and the commanded and actual TCP pose at every step.

4. **Reverse the sequence** to get the unloading curve.

### What to record and what it means

Plot measured normal force against commanded penetration. **The slope is
`k_series` in N/mm**; the intercept is where contact actually begins relative to
the mesh; the **width of the hysteresis loop is the plunger's friction**, which
is a hard floor on any force regulation through that mechanism and is worth
knowing before anyone buys a sensor.

| slope | conclusion |
|---|---|
| **≤ 2 N/mm** | the mechanics already meet ±1 N at ±0.5 mm of mesh error. The real defect is preload depth, mesh registration or normal error — **not compliance**, and not force control. |
| **≥ 10 N/mm** | a softer spring with more stroke is the entire fix, and no software will beat it. |
| 2–10 N/mm | record it; it sets how much a spring change would buy. |

## 12. Does the EC measurement actually care? — one command, and it can cancel the project

**What is being tested.** An eddy-current reading cares about **lift-off**;
force is only the means by which a spring holds lift-off steady. This cell
**already** rotates lift-off onto its own axis — `/ec/cond/horizontal`, and
setting that rotation *is* the EC setup procedure. So the controlled variable is
already measured. What has never been measured is **how much lift-off actually
varies along a pass.** If it is small, a force loop has nothing to fix. If it is
large, the same number says how much softer the spring needs to be.

**Specimen.** `contact_a` and `contact_b` taught **on the part**, with the probe
**compressed as it would be when scanning**, both at the same orientation (the
probe's roll is locked during an EC scan, so the pass keeps the orientation
taught at the first point).

### Procedure

1. Jog to the start of a contact line, probe compressed, and teach:

```
dgpy> teach_point("contact_a")
```

2. Jog to the far end, same orientation, probe compressed:

```
dgpy> teach_point("contact_b")
```

3. Run the scenario — it already exists and needs no modification:

```
dgpy> capture_contact(points=("contact_a","contact_b"), tilts_deg=(0,3,-3,6,-6), tilt_axis="travel", ec=True, force_limit=25.0)
```

It flies the taught line at five tilts **about the probe tip**, so the tip stays
on the line, and writes one `ec_tilt<±deg>.npz` per pass containing the **raw
complex impedance**, `s_m`, positions and the trigger report.

4. Analyse, on any machine, with no ROS and no cell:

```bash
python3 ~/dataguzzler-work-dir/ROS-Scanning/dgpfiles/modules/ec_liftoff.py <take_dir>
```

### What it tells you, and why it is not just a plot of the channel

- It **derives the lift-off axis from the data**, from how tilt moves the
  per-pass means — because the rotation dialled in at setup was set by eye
  against a display. It prints the dialled-in angle beside it, and a large
  disagreement is itself a finding: the scan display has been separating
  lift-off from the part along the wrong axis.
- It uses the **commanded tilt steps as a ruler**, so spreads come out in
  *tilt-degrees* rather than uninterpretable volts. No probe calibration needed.
- It **separates trend from residual**, because they choose different work.

| what dominates | what to do |
|---|---|
| **trend** (> 65%) | registration — the mesh sits tilted or offset relative to the part. Fix the calibration; compliance would only hide it. |
| **residual** (< 35%) | mesh roughness. This is what compliance is for — and what an active loop **cannot reach**: at 10 mm/s a 1–2 Hz loop corrects wavelengths above ~5–10 mm, a spring reaches below 1 mm. |
| comparable | both, roughly half each. Do not read a few percent either side as a verdict. |

**Also record** the `lift-off / defect axis` ratio. Above 1.0 means lift-off
varies more than the part does — the image is largely a picture of mesh error.

---

## 13. Not at the cell: the arc-length timing backend

`tari_toppra` is built and loaded by **nothing**. It times a Cartesian path by
TCP arc length so that commanding m/s becomes a constant bound. Measured
offline, 1040 waypoints with a 24× spread in Cartesian pitch at 10.000 mm/s
commanded: an index-parameterized Jacobian bound gave **+17.5%** at 50
gridpoints degrading to **+182.5%** at 2000 — worse as the grid refines — while
arc length gave **+0.023%**.

It needs no cell time. Run its suite anywhere:

```bash
cd ~/ros_ws/src/tari_ros/tari_toppra && python3 -m pytest test -q
```

Wiring it in is a later step and wants the row data from tests 5 and 6 first.

---

## What to bring back

For each test run: the numbers in its "record" table, and the scan log. Tests 3,
7 and 12 each close out a question that has been open for several sessions, so
their numbers are worth more than a completed scan.

**If time is short, in this order:** 1, 2, 3 (15 min, free), then **12** (it can
retire a whole workstream), then **4** (the largest motion win), then 10.
