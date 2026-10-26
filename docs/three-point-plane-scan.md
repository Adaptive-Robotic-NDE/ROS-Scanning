# The three-point plane scan

Teach a scan area by driving the probe to three points and capturing the pose at
each. No mesh, no reconstruction, no registration — which is the whole reason
this path exists and why it is the one to reach for on a new part or a new cell.

**Part 1 is a runbook**: copy, paste, scan. Part 2 explains what the three
points mean and the traps behind them; read it before teaching a part you care
about.

The runbook below has been verified end to end in simulation: 183 of 183 traces
staged, every pass `move_ok` at `fraction 1.0`, 0 stager errors, 0 late
transactions.

---

# Part 1 — Runbook

## 0. What you need running

Two terminals. The first runs the robot cell (ROS 2), the second runs
dataguzzler-python. In simulation both are inside the ROS 2 simulation
container; on hardware they are wherever the robot's ROS installation lives.

**In simulation**, get a shell in the ROS 2 container first — from the workspace root
of the machine that runs it:

```bash
./ros2_docker.sh shell
```

Every command below assumes ROS is sourced in whichever shell you use:

```bash
source /opt/ros/iron/setup.bash
source ~/ros_ws/install/setup.bash
```

## 1. Terminal 1 — start the work cell

One command brings up the robot, MoveIt, RViz, both motion action servers and
the shared services. In simulation it also starts Gazebo:

```bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=sim
```

On hardware, the same command with the mode changed:

```bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```

Wait for the summary line `N started, 0 skipped`. Anything unavailable is
skipped **by name with a reason** rather than taking the bring-up down with it,
so read that list — a skipped motion server means nothing will move later.

To see what a cell would start without starting it:

```bash
ros2 run tari_bringup tari_cell_info --cell ex-ur10e-1 --mode sim
```

The plane scan needs only the robot and the motion servers, so if the machine is
busy you can leave the service layer out:

```bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=sim services:=false
```

In simulation, check the arm is actually there before going further:

```bash
ros2 topic echo /tf --once
```

## 2. Terminal 2 — start dataguzzler-python

```bash
cd ~/software_ws/ROS-Scanning/dgpfiles      # inside the ROS 2 container
dataguzzler-python UTPlaneScan.dgp
```

That is the whole command — the configuration takes no arguments. It opens the
SpatialNDE2 viewer, prints a summary of the current settings, and leaves you at
a `dgpy>` prompt. Everything from here on is typed at that prompt.

If you want the scan to talk to the **real digitizer** instead of the built-in
simulator, edit one line near the top of `UTPlaneScan.dgp` before starting it:

```python
ASCAN_SOURCE = "hardware"        # "simulated" is the default
```

`ASCAN_CHANNEL_PATH` follows automatically (`/CS_CH1` for the GaGe CompuScope,
`/ut_ascan` for the simulator) and nothing downstream is aware of which one is
loaded. See "Real digitizer" below for the one thing that does differ.

Expect the banner to end with:

```
  A-scan source: simulated
  step size 5.00 mm, line spacing 10.00 mm, velocity 50.0 mm/s, projection live
```

## 3. Set the scan up (optional, but do it once)

Open the technician panel and see what the current settings would cost, without
moving anything:

```python
show_scan_panel()
preview_scan_plan()
```

Step size, line spacing and velocity are the controls; everything else — trace
rate, scan time, memory, projection resolution — is derived from them and the
plan will clamp or refuse rather than let you exceed the measured limits.

## 4. Teach the three corners

Drive the probe to each corner however you normally do — pendant, servoing, or
`ros.MoveJoints(...)` from this prompt — and capture at each one:

```python
p1 = capture_point("origin")     # corner 1: origin, AND the scan orientation
p2 = capture_point("u axis")     # corner 2: far end of the first scan axis
p3 = capture_point("v extent")   # corner 3: the other axis
```

Each call prints where it captured, and refuses if this process has never heard
from TF rather than quietly handing you a corner at the origin:

```
  captured origin: x 0.7496  y -0.3502  z 1.0991  (world -> probe_end)
```

**Square the probe up before capturing `p1`.** Its orientation is the one held
for the entire scan; the other two contribute position only.

## 5. Choose the grid and scan

```python
m, n = plane_grid_size(0.30)                    # square area, side in metres
res = Scan2DPlaneTriggered(p1, p2, p3, m, n)
```

`m` is the number of points along a row, `n` the number of rows, both derived
from the step size and line spacing you set in step 3. For a **rectangle**,
derive them from the two taught lengths instead:

```python
import numpy as np
def taught_length(a, b):
    pa = np.array([a.transform.translation.x, a.transform.translation.y, a.transform.translation.z])
    pb = np.array([b.transform.translation.x, b.transform.translation.y, b.transform.translation.z])
    return float(np.linalg.norm(pb - pa))

m = int(round(taught_length(p1, p2) / SCAN_STEP_SIZE)) + 1
n = int(round(taught_length(p1, p3) / SCAN_LINE_SPACING)) + 1
res = Scan2DPlaneTriggered(p1, p2, p3, m, n)
```

**Try a couple of rows first** on anything new — same call, same geometry, stops
early:

```python
res = Scan2DPlaneTriggered(p1, p2, p3, m, n, max_rows=3)
```

## 6. Check what you got

```python
res['staged'], res['expected_traces']      # traces in the database vs planned
res['stager_errors'], res['late_transactions']    # both should be 0
[s['fraction'] for s in res['per_segment']]       # 1.0 on every pass
res['skipped']                                    # rows that did not run
```

A healthy short run:

```
segments 3, expected_traces 183, staged 183, stager_errors 0,
late_transactions 0, elapsed 31.9 s
```

The data is in the recording database: `/ut_scan_volume` is `(m, n, nsamples)`,
`/ut_cscan` the gated image, `/ut/traces` the flat trace list with a pose per
trace. Retune the gate live from the panel — it reads `/cscan_start` and
`/cscan_end` as channels, so the c-scan recomputes without rescanning:

```python
project_traces()        # rebuild the projected image from the trace list
```

## 7. The whole thing, condensed

```bash
# terminal 1
source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=sim
```

```bash
# terminal 2
source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
cd ~/software_ws/ROS-Scanning/dgpfiles && dataguzzler-python UTPlaneScan.dgp
```

```python
# at the dgpy> prompt, driving the probe to each corner in between
preview_scan_plan()
p1 = capture_point("origin")
p2 = capture_point("u axis")
p3 = capture_point("v extent")
m, n = plane_grid_size(0.30)
res = Scan2DPlaneTriggered(p1, p2, p3, m, n, max_rows=3)   # drop max_rows for the full scan
res['staged'], res['expected_traces'], res['stager_errors']
```

## Real digitizer

`ASCAN_SOURCE = "hardware"` loads the GaGe CompuScope. Record length, sample
rate and sample type are **not** set in this config — they are whatever the card
is configured for, they can change between runs, and everything that needs them
asks the data.

`ASCAN_TRIGGER_MODE` is the one setting that means something different on
hardware:

| mode | simulated | hardware |
|---|---|---|
| `"timer"` | free-runs at `ASCAN_RATE_HZ`, the scan samples it | the card free-runs, the scan samples it — `run_default_scan()` / `run_raster_scan()` |
| `"position"` | synthesises a trace at each crossing, modelling a trigger box | **refuses**: on hardware the trigger box fires the card by wire and this config cannot arrange it |
| `"timestamped"` | free-runs and correlates by measurement timestamp | identical, and this is how a real digitizer is actually driven |

So develop a triggered scan against `"timestamped"`: it is the mode that means
the same thing on both sides. It requires `RECDB_TRANSACTION_MANAGER = "timed"`,
which is the default here.

## When it does not work

| symptom | cause |
|---|---|
| `capture_point` raises "nothing has been heard from TF" | the work cell is not running, or this process cannot see it. Check terminal 1 and `ros2 topic echo /tf`. In simulation after the work cell has been up a long time, DDS discovery can go stale for *new* processes — restart the container |
| `no A-scan on /ut_ascan after 30 s` | with `ASCAN_TRIGGER_MODE="position"` the simulator only emits when the probe MOVES, so the first scan of a session needs the arm to have moved at least one trigger pitch. Jog it, then scan |
| a pass reports `fraction` well below 1.0 | the arm entered the row in a configuration that cannot follow it. The scan refuses to fill in the rest — see the trap below |
| everything refuses to plan | the arm may be parked in an invalid state. `ros2 service call /check_state_validity moveit_msgs/srv/GetStateValidity "{group_name: ur_manipulator}"` |
| nothing moves and nothing is logged | in simulation, restart the container first (`./ros2_docker.sh restart`); a Gazebo that has been up for hours takes the rest of the stack down with it in ways that read as unrelated failures |

---

# Part 2 — What the three points mean

| point | what it fixes |
|---|---|
| **t1** | the scan ORIGIN, and **the probe orientation used for the entire scan** |
| **t2** | the u axis: its DIRECTION and its LENGTH (`|t2 − t1|`) |
| **t3** | the v extent: only the part of `t3 − t1` PERPENDICULAR to u is used |

Read `_plane_grid()` in `modules/plane_scan.dpi` for the arithmetic. Three
consequences worth knowing before you teach a part:

- **t3 does not have to be square to t1→t2.** The v axis is re-derived as
  `w × u` where `w = u × (t3 − t1)`, so v is perpendicular to u by construction
  and t3 only contributes its perpendicular distance. Teaching a rough third
  corner gives a clean rectangle, not a parallelogram.
- **The scan plane is the plane through all three points.** If the three are not
  coplanar with the surface you meant, the scan is flat and the surface is not.
- **Orientation comes from t1 alone** and is held for every point.

## Which call to use

All three take the same `(t1, t2, t3, m, n)` and build identical geometry.

| call | motion | acquisition | use it when |
|---|---|---|---|
| `Scan2DPlane` | stops at every point | one A-scan while stationary | the geometry is in doubt, or the probe needs to settle. Costs a MoveIt plan-and-execute cycle per point — 0.3–0.5 s regardless of step size |
| `Scan2DPlaneRaster` | one continuous trajectory per row | free-running digitizer, samples binned to the nearest grid column | a quick continuous scan with no position trigger |
| `Scan2DPlaneTriggered` | one continuous trajectory per row | **one trace per position trigger** | the normal choice. Trigger k IS column k — nothing is binned, and the pose on a trace is the interpolated trigger crossing rather than whatever TF last published |

`Scan2DPlaneTriggered` requires `ASCAN_TRIGGER_MODE = "position"` or
`"timestamped"` and refuses otherwise. The trigger pitch is set from the grid at
the start of the scan — trigger k must be column k, so it is not yours to choose
independently of `m`.

**The volume is built from the trace poses, not from arrival order.** At the end
of a triggered scan `regrid_volume()` rewrites `/ut_scan_volume` by putting each
trace in the work cell its recorded pose falls in. That matters because a trace can go
missing -- a transaction the recording database refused because its timestamp
could no longer be ordered, a crossing the trigger never published, a pass that
came up short -- and filling the volume in arrival order turns one missing trace
into a shear across everything after it, with nothing in the volume to show for
it. Placed by pose, a missing trace leaves its own cell empty (NaN) and its
neighbours do not move. The trace list is untouched either way; this is about the
quick look telling the truth. You can re-run `regrid_volume()` from the prompt at
any time, and `res['regrid']` reports what it did.

**The triggered scan holds its last column half a pitch inside the far edge.**
The trigger for column m−1 fires when the probe has progressed (m−1) pitches,
and a pass that stopped exactly there would leave that crossing to a coin flip
against noise — on the UR7e cell it came out as rows of 49, 50 or 51 traces and
a sheared C-scan. Every physical scanner solves this with overtravel past the
last sample. Here the overtravel is taken *inside* the taught rectangle: the
columns are placed at `row/(m−1+½)` instead of `row/(m−1)`, and the row is still
commanded to t2's edge, so the probe goes exactly where it was taught and never
beyond it (it may be in contact with the part). The cost is a scan area shorter
by half a pitch on the t2 side — 0.6 mm at a 1.2 mm step — and a pitch about
1% finer than `Scan2DPlane` would use for the same `m`; the volume metadata
records the actual pitch. `TRIGGER_END_MARGIN` in `modules/plane_scan.dpi` is
the knob, and `_check_overtravel_within_taught` refuses a path whose rows end
anywhere but on the taught edge.

Both continuous forms take `max_rows=` and `force=`; the point-to-point form
takes `max_points=`. The volume is allocated at full size either way, so a short
run is a dry run of the real geometry rather than a different scan.

## Reading the result

| field | meaning |
|---|---|
| `staged` vs `expected_traces` | traces that reached the database against traces planned |
| `per_segment[k].move_ok` / `.fraction` | did the row execute, and how much of it the planner covered |
| `skipped` | rows that did not run at all |
| `short_passes` | rows whose trigger count disagreed with the grid |
| `regrid` | how the volume was rebuilt from the trace poses: cells `filled`, cells left `empty` because no trace reached them, traces that landed `outside_grid`, `collisions` where two traces claimed one cell, and `worst_cell_residual` -- how far the worst trace sat from its cell centre, in cells. Over about 0.5 means the grid does not describe where the probe actually went |
| `over_budget`, `per_segment[k].over_budget` | crossings past the row's m−1 that were counted and not staged. A few per row is normal (the probe overtravels to the taught edge); every row would mean the pitch is wrong |
| `per_segment[k].observed_m` vs `.raw_travel_m` | progress the trigger measured against the plain sum of every TF step; the excess in the second is wobble and noise, which used to fire the surplus trigger |
| `stager_errors`, `stager_last_error` | staging failures — should be 0 |
| `late_transactions`, `max_lateness_secs` | the timestamped-acquisition window — should be 0 |
| `stager_max_queue_depth` | how close the staging pipeline came to falling behind |

## Traps

**`ros.t` is a snapshot, but an empty one until TF arrives.** The TF thread
rebinds it on every poll rather than mutating the object you hold, so three
captures stay distinct while the arm keeps moving — verified: `p1` still read
x = 0.7496 after the arm had moved to x = 1.0499 and `p2` had been taken, no
`deepcopy` needed. What it is NOT is None before TF exists: the motion module
initialises it to a blank `TransformStamped`, so a bare `p1 = ros.t` on a
process that cannot see the work cell yields a corner at the origin, silently. That
is why `capture_point()` exists and why it checks.

**A pass that runs short is a fabricated C-scan, not a gap.** The volume is
indexed by grid position, so a row that stops three quarters of the way along
does not leave a hole — the acquisition fills the rest from a neighbour and the
image looks complete. That is why `RASTER_MIN_FRACTION` is 1.0 and why the
Cartesian stage refuses a short path outright. Do not lower it for a grid scan;
`PATH_MIN_FRACTION` exists for the surface-following scans, where a short pass
costs coverage rather than correctness.

**Pin the origin for anything you intend to compare.** `scan_corners()` with
`origin=None` starts from wherever the probe happens to be, which drifts between
sessions. Capturing t1 by hand is the pinned version of that.

**When the plane is a lie.** A flat scan plane over a curved part changes the
standoff across the scan, so the gate walks off the feature and the C-scan is
wrong even though every trace is real. For a curved part either follow the
surface (`run_mesh_scan()`, which needs a mesh) or use a tracking gate —
`cscan_gate` takes a reference recording precisely so a second gate can sit a
fixed distance behind a front-wall crossing.

**Velocity is derived, and the derivation can be wrong.** The continuous forms
clamp velocity against the measured trace-rate ceiling. If the arm's observed
speed disagrees with the commanded one the scan says so and names the
correction, e.g. *"observed probe speed 8.4 mm/s over 30 passes (asked for
50.0) … ROBOT_VELOCITY_CALIBRATION is wrong here. Replace or add the entry
(0.0424, 0.0084)"*. Believe it: the calibration is a measured curve, not a
constant, and it is per cell.

## Saving a taught area

The three transforms are ordinary messages. To keep an area across sessions,
store the numbers and rebuild them — do not pickle the message:

```python
def area_to_dict(t):
    tr, r = t.transform.translation, t.transform.rotation
    return {"xyz": [tr.x, tr.y, tr.z], "quat": [r.x, r.y, r.z, r.w]}
```

Rebuild with a `TransformStamped` whose `header.frame_id` is `world` and whose
`child_frame_id` is `probe_end`; only `transform` is read by `_plane_grid()`,
plus `t1.transform.rotation` for the scan orientation.
