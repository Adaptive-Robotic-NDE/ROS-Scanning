# Capture run sheet — ex-ur10e-1

> **Superseded.** This is the run sheet for one specific visit, kept for what
> that visit measured and for the pipeline measurements in sections 4a and 4c.
> The operating document is `cell-test-plan.md`.

What to run, in order, and what to bring back. The detail behind each scenario is in
`cell-data-capture.md`; the bring-up detail is in `tari_bringup/CELL_CHECKLIST.md`. This file is
the order of operations and the checks that go with it.

The goal of the visit, in priority order:

1. Confirm the known hardware failures are fixed (section 2). Each is a one-minute check.
2. Collect capture scenarios 1-6 (section 3). This is the data the pipeline work needs.
3. Contact and full-pipeline data (section 4), only if 1 and 2 leave time.

**Read section 4a before you go.** It is the measured state of the pipeline in simulation —
scenarios 5 (raster and cylinder), 7 and 8 flown end to end — and lists the three things that
still bite.

## 0. Before you leave

- Replace `ros_ws/src` and `ROS-Scanning` as whole trees; do not merge. Copying over an older
  tree leaves deleted files behind, and the work cell then runs an `ECAutomatedScan.dgp` that
  differs from the workspace copy — the acceptance suite's `installed but not in src:
  dgpfiles` warning is that condition.
- Delete `build/` and `install/` for anything that moved, then build. `azure_kinect_ros_driver`
  must be built WITHOUT `--symlink-install` or its `node` lands in `bin/`, where launch cannot
  find it.
- `ros2 run tari_bringup acceptance --cell ex-ur10e-1 --level 0` should pass before you drive
  there; it checks the builds, the Python dependencies and the descriptions. It also refuses a
  package configured against another ROS distro, which is what a node exiting 127 usually is:
  on a PC carrying both distros, `keyence_experimental` had `/opt/ros/humble` hard-coded and
  built against Humble's rclcpp.

## 1. Bring-up

```bash
source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
export TARI_CELL_PATH=tari_cell_configs
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```

The manifest carries the real robot controller's address and the measured work volume
(x 0.560..1.145, y -0.509..0.454, z 0.925..1.245). Load and **play** the External Control
program on the pendant.

## 2. Confirm the known hardware failures are fixed

| symptom seen on hardware | What to check now |
|---|---|
| `keyence_driver_node` exited 127 | it stays up, and `/profiles` runs at ~217 Hz with 800 points |
| `/yolo/enable` did not exist | `detect_part()` returns a region without you typing one |
| services reported missing | they were live: the check sampled the graph once, under load. It now waits, and also calls each service |
| the Keyence driver also segfaulted 6 s after connecting | it bounds-checks the profile now and logs a retry instead of dying |
| Keyence read 5.3 mm low | `capture_keyence_ladder()` on a known flat: the surface lands within about 0.3 mm |
| Azure depth 44% valid, 10 mm low | the sweep prints its camera-to-nearest-surface RANGE, refuses under 500 mm, and holds that range by standing off to the side rather than by flying to a height the arm cannot reach |
| the motion preview was invisible | RViz shows "Capture preview" and "Capture preview poses" |
| the Azure flew 696 mm off | pass a deliberately wrong region: it refuses before the arm moves |

## 3. Capture scenarios 1-6

Run them in this order. Scenario 1 first (everything needs the work volume), then 6 before
anything turns the wrist.

1. `capture_touch_off(...)` — re-teach the corners; the manifest's numbers are from an
   earlier visit.
2. `capture_cable_envelope(...)` — the wrist limit, with you stopping it when the cables tighten.
3. `capture_keyence_ladder(...)` — also the Keyence height check above.
4. `capture_keyence_timing(...)`.
5. `capture_azure_sweeps(...)` — normal lighting and one bad-lighting take, per specimen.
6. `capture_keyence_sweeps(raster=False, cylinder=(...))` — the cylinder passes. **Conditional.**

   **Why it was blocked, and what now holds it.** A raster over a cylinder places the Keyence
   correctly and drives the eddy-current probe 5-6 mm INTO the part: the probe sits 75 mm away
   on the same plate, and on a cylinder flank the sensor's own standoff puts the probe on the
   surface. The stage had no collision checking at all. Now `tari_cpp_mtc` collision-checks the
   raster, `obtain_specimen_mesh()` puts the specimen in the planning scene, and **this
   scenario's own clearance check measures the whole end-of-arm assembly** through
   `tooling_clearance.dpi` instead of the probe tip (`cell-data-capture.md`, "Known
   limitations"). Measured against the live `robot_description` and the taught fiducial
   cylinder, every pass from 0 to ±60° clears by **+32.0 mm at `ec_probe`** where the tip-only
   number is +34.3 mm; nothing is refused, and the seven checks cost 3.2 s. The pipeline's own
   raster over the same cylinder plans 582 poses in 6 passes with a worst clearance of
   **+20.8 mm** and drops nothing.

   The motion fault that blocked it is fixed. `checkPathFidelity` scored the lead-in that
   `computeCartesianPath` always prepends — the travel from wherever the arm is to waypoint 0 —
   against the requested polyline, so **every** Cartesian move that did not already start on
   its first waypoint was refused ("the tool leaves the requested path by 109.0 mm at point 0
   of 1008, limit 5.0 mm" for a 173 mm descent). It now splits the trajectory at the point
   where it first reaches waypoint 0: the lead-in is measured against the straight line to
   waypoint 0, the rest against the polyline as before. **Flown in simulation: the cylinder
   scenario completes all 7 passes, 30 Cartesian moves, none refused, none over the speed
   limit, nothing unreachable**, `/profiles` at 42.3 Hz with 800 points and none empty on the
   crest, and a mesh out the far end. Every one of those 30 moves is a single-waypoint
   Cartesian move — the case that was refused 100% of the time.

   **A flat plate is unaffected by the clearance question** — the same geometry leaves 39 mm
   of clearance over a flat surface, which is why a plate scans fine on hardware.

The exact argument lines are in `cell-data-capture.md`.

## 4. Only if there is time

- `capture_contact(...)` — spring-probe passes between taught points. Every one of its
  approach, touch-down, pass and lift moves is a single-pose Cartesian move, which is exactly
  what the trajectory guard used to refuse. Flown in simulation with the scripted stand-in for
  the pendant: **all 5 tilts (0, ±3, ±6°), 41 Cartesian moves,
  none refused, none stopped**, peaks 6.5-8.0 mm/s against the 10 mm/s near-the-part limit and
  19-24 mm/s against the 30 mm/s transit limit; one pass read 10.8 mm/s, 8% over, which is
  inside the monitor's tolerance and is reported rather than stopped. 10.8 MB of bag, 69 122
  messages over 439.9 s, 26 `/joint_states` lost. There is no wrench in simulation
  (`force_torque_sensor_broadcaster` is absent), so **the force limit is the one part of this
  scenario that only the work cell can exercise** — run the 0° pass first and watch the wrench.
- `capture_full_pipeline(...)` — the whole pipeline, recorded. Flown in simulation: the Azure
  dome's 5 rows all flew 100%, the Keyence raster planned 3041 Noether poses, cropped to 1364
  inside the region and the work volume, gave up 19.6% of the path to the tooling-clearance
  gate (worst −32.0 mm at `ec_probe`) and flew the 18 passes that were left, with every pass
  approach going through the transit. **Every motion stage completed and not one trajectory
  was refused.** The failure mode to expect here is memory, not motion: mesh filtering and UV
  parameterization took dataguzzler-python to **8.4 GB RSS and the kernel OOM-killed it** on a
  15 GB machine that was also running Gazebo, YOLO and the reconstruction nodes. **The memory
  and the gate are dealt with — see section 4c** — so it no longer needs `line_spacing=0.010`
  to finish and it releases YOLO itself. Two things still to know before you run it at the
  cell:
  * **Know how much RAM the controlling computer has, and print the cost first.** `preview_ec_scan()` with
    nothing planned now prices the area from the three technician controls alone and says what
    the plan will cost in memory and in time. `plan_ec_path()` refuses a plan that will not
    fit instead of discovering it by being OOM-killed.
  * **Inset the region so the dome's TOOL path stays in the work volume** (section 4a).
- One pre-existing failure to expect on either: at very small velocity scalings MoveIt's time
  parameterization can give up outright — "Time parameterization failed for 71 waypoints at
  velocity scaling 0.00220448" costs one raster row. The row is lost, nothing moves, and the
  next row proceeds; it is not a safety event.

## 4a. What the pipeline does, measured in simulation

The whole capture pipeline flown in the simulated cell. What it does, and what it still gets
wrong, so nobody rediscovers it on the production system.

### The guard, and what it now measures

`tari_cpp_mtc`'s Cartesian stage checks five things about a trajectory before it is sent:
where it starts (joint space), per-joint steps, per-joint speed against the scaling it was
given, the tool's speed in m/s, and how faithfully the tool follows the requested path. The
last one was scoring the lead-in `computeCartesianPath` always prepends — the straight travel
from wherever the arm is to waypoint 0 — as if it were off-path, which refused **every**
Cartesian move that did not already start on its first waypoint: a one-waypoint 173 mm
descent at 109.0 and 167.8 mm, a two-waypoint row entered 50 mm above its start at 45.0 mm,
and 4 of 5 Azure dome rows. The callers fell back to joint moves, which are scaled in joint
space, and the tool reached 50.6 mm/s against a commanded 4 mm/s.

It now splits the trajectory at the point where it first reaches waypoint 0 (found on
trajectory *segments*, because the parameterizer resamples on a 0.1 s grid and no sample need
land on the waypoint). Before that point it is measured against the straight line from the
tool's starting position to waypoint 0 — so an approach that bows or detours still fails, on
the same 5 mm — and its length is capped at 0.80 m, just over this cell's whole work-volume
diagonal. After that point, against the requested polyline, exactly as before. A new coverage
test walks the requested path the other way and refuses a trajectory that leaves more than
20 mm of it untravelled, which is the case distance-to-the-nearest-segment cannot see.

**Measured over 149 trajectories of a whole pipeline run** (dome, raster, cylinder passes,
contact passes): off-path **0.35 mm worst**, against the 5 mm limit; **0.0 mm untravelled**
on every single one; lead-ins of 0-338 mm, straight to **0.10 mm worst**; trajectories ending
**0.10 mm** from their last waypoint; start offsets 0.0000 rad; peak tool speed **24.5 mm/s**.
So the limits are not being spent: the off-path limit is 14x the worst honest reading and the
tool ceiling (50 mm/s, the loosest speed the approved motion rules allow anywhere) is 2x.

One trajectory in 149 was refused, and not by the path check: an 83-pose row read 136% of the
joint speed its scaling allowed, where the tolerance was 25%. Every row that flies sits at
80%, the runaway this tolerance exists to catch reads 11300%, and that row's **tool** was
doing 10.7 mm/s against a 10 mm/s command — so the tolerance is 1.0 (200%), still two orders
of magnitude from a runaway, with `max_tip_speed` doing the work of bounding metres per
second.

### The three things that still bite

- **A pass approach can be a whole-arm reconfiguration, and the joint-space speed model
  cannot bound it.** Coming off the Azure dome to the first raster row is 61.4° of arm and a
  515 mm tip path; it was classified "near the part" at 10 mm/s, measured 16.5 mm/s, peaked at
  **390 mm/s** and the monitor stopped the arm — correctly, and the run ended there. A pass
  approach over `CAPTURE_APPROACH_MAX_DEG` (20°, measured: honest row approaches are
  0.0-12.2°) now goes through `capture_transit` instead — up, across, down, each leg a
  Cartesian move whose speed *is* bounded correctly — and then re-solves the row-spanning
  configuration. **Watch this one on the production system**: the joint-space model is still what bounds
  every other joint move, and it is the model that was wrong, not the arm.
- **The Azure sweep is a PAN now, not a dome, and the standoff question is settled.** The
  depth mode's minimum is a distance from the camera to the surface, not a height, and the
  height a straight-down sweep needs -- camera 1.455, probe_end 1.380 -- is not reachable
  across this bench: solved against the work cell's own kinematics (analytic UR IK on the rendered
  `ur_cell.urdf`, 8 x 13 aim points over the measured work volume), it has **no solution at
  all** over x >= 1.145 and reaches 84.6% of the volume. Standing off to the side and aiming
  back buys the same distance at a reachable height, and every rung from 0.25 to 0.50 m above
  the part reaches 100% of the volume from at least three of eight directions. Most of what
  that buys is not the lower height -- 0.50 m is barely lower than 0.53 -- it is the freedom
  to choose **which side to stand on**: over the far edge of the bench the camera stands off
  towards the base instead of beyond it.

  The wide depth mode is ruled out, computed rather than recalled: the eddy-current probe and
  its holder sit 65-110 mm from the camera, 45.5 degrees off the optical axis, so they clear
  an NFOV frame's bottom edge by 11.8 degrees and sit 15.7 degrees **inside** a WFOV one,
  occupying 2.9% of every frame at a distance no depth mode can measure. The tooling is rigid
  to the camera, so no sweep can be planned around it.

  The old dome's own trap is gone with it -- it was planned from the region and not cropped to
  the work volume, so over the detected cal-plate region the tool path reached x 0.656 and the
  capture tool refused every row. The pan sweep's poses are checked against `/compute_ik` with
  collision checking before anything moves, and a direction that does not solve is dropped by
  name.

  Two things to watch on the production system. That check means **planning takes 2-5 minutes before
  anything moves** -- that is the search, not a hang. And the aim plane defaults to the work
  volume's floor, so pass a tall part's top as `surface_z=` or the range is measured to the
  bench while the part is that much closer.

  **Flown in simulation on both specimens, every row 100%:**

  | | fiducial cylinder | cal plate |
  |---|---|---|
  | region | 145 x 85 mm | 304 x 153 mm |
  | camera above the part | 500 mm | 500 mm |
  | clear of the region | 176 mm | 176 mm |
  | from vertical | 26-27 deg | 33-35 deg |
  | directions kept / dropped | 3 / 5 | 3 / 5 |
  | camera-to-nearest-surface | 530-547 mm | 530-572 mm |
  | lowest tooling point above the part | 465-467 mm | 477-480 mm |
  | rows flown | 6 of 6, 100% each | 6 of 6, 100% each |
  | crest error | -0.5 mm | -1.1 mm |
  | radial RMS on the cylinder | 1.55 mm | 1.65 mm |
  | plate top | -0.98 mm | -1.27 mm |
  | vertices | 9 547 | 9 712 |

  A top-down sweep, flown at a height this work cell cannot reach, gives crest
  +0.4 mm and plate -0.07 mm -- so the pan sweep costs roughly a millimetre of bias, low, in
  exchange for being flyable. On a 6 mm-voxel planning surface that is inside the voxel; do not
  spend it twice by also trusting the rough mesh where the Keyence should measure.

  **And the straight-down sweep refuses over that same region.** Asked to
  fly `style="dome"` at its derived height (probe_end 1.442) over the cylinder region -- which
  sits in the middle of the work volume, not at its edge -- it could not enter any of its first
  four rows from where the pan sweep left the arm, went home, and still could not:

  ```
    row 1 of the sweep will not be entered from here      (and rows 2, 3, 4)
    no approach from here; going home and trying again
  RuntimeError: the arm would not go to the first pose of the sweep ([ 0.83 -0.216  1.442])
  ```

  It has been flown before, so it is marginal rather than impossible over the middle of the
  bench -- which is the same thing the reach map says from the other direction. Over the far
  x column it is not marginal, it has no solution at all.
- **An eddy-current plan at the configured 0.30 mm line spacing is ~180 000 poses**, and the
  clearance gate walks every one at about 0.11 s each — roughly six hours before anything
  moves. Set `EC_SCAN_REGION`, or pass `line_spacing`, to the patch you actually want.

### What the scenarios produced

| scenario | result | bag | messages | peak tool |
|---|---|---|---|---|
| 4, Azure dome (region inset) | 5 of 5 rows, 0 refused | 16.4 MB | 18 365 in 111 s, 3 `/joint_states` lost | 24.4 mm/s of 30 |
| 5, Keyence raster (cal plate) | 19 rows, 1 refused (the 136% above) | 167.0 MB | 278 682 in 1407 s, 101 lost | 24.3 mm/s |
| 5, cylinder passes | 7 of 7, nothing unreachable | 71.5 MB | 131 724 in 662 s, 39 lost | 20.5 mm/s |
| 7, contact | 5 of 5 tilts, 41 Cartesian moves | 10.8 MB | 69 122 in 440 s, 26 lost | 24.5 mm/s |
| 8, whole pipeline | dome 5 of 5, raster 18 passes, 1 row lost to time parameterization; OOM in post-processing | 400 MB | ~0.05% `/joint_states` lost | 25.5 mm/s |

**The offline replay of a take works.** `ros2 run tari_bringup record replay TAKE
--start-nodes` must subscribe to `/clock` with a BEST_EFFORT profile, because that is what
`ros2 bag play --clock` publishes with; a RELIABLE subscription leaves the replay clock at 0,
both service calls fire at bag time 0 and `stop_reconstruction` answers "The TSDF is empty".
An Azure take recorded in simulation replays into `industrial_reconstruction` with the
cell and the simulator down and gives back **the same mesh**: 5 596 vertices against 5 573,
the same bounding box and centroid to 0.1 mm, nearest-vertex agreement 0.060 mm mean and
0.16 mm at p95, 99.5% of vertices within 1 mm. The Keyence path rebuilds the same surface
(88% of replayed vertices within 0.5 mm) but not the same point set, because on a 1 400 s bag
the recorded stop call lands tens of seconds of bag time late for a reason not yet identified.
The full numbers, and what that means for quoting them, are in `cell-data-capture.md`.

**So a bag you bring back is enough to re-run the reconstruction here.** That is what makes
scenario 8 worth recording even if its post-processing falls over on the production system.

`/profiles` delivered 800 points with none empty throughout (42-99 Hz depending on the pass).
`/joint_states` loss is 0.05-0.07% and is the recorder's own accounting, not a gap in the
trajectory. Four topics are **absent in simulation** and will only appear on the production system:
`io_states`, `robot_program_running`, `speed_scaling`, and the force/torque wrench — so the
contact scenario's force limit, and the speed-scaling slider, are untested until you are
there.

## 4c. The eddy-current stage

Three things that made the eddy-current stage of scenario 8 unusable at the 0.30 mm line
spacing this cell scans at.

**The clearance gate ran on the wrong path and measured every pose of it.** It sat inside
`plan_mesh_path()`, so it checked the RAW Noether toolpath — 911 511 poses on the
fiducial-cylinder specimen — before the tilt cap, the work-volume crop, `EC_SCAN_REGION` and
the resample to 1 mm that leave 180 539. It now runs once, at the end of `plan_ec_path()`, on
exactly the poses `run_ec_scan()` will command, and it **samples**: measured poses bound the
ones between them through a 1-Lipschitz certificate, and a window is left unmeasured only
when that bound clears the limit for every pose in it. Measured on the cal plate's own
Keyence reconstruction with a 933-pose path at the real spacings: **534.2 s → 17.9 s**
(0.573 → 0.019 s per pose), **4 625 MB → 402 MB**, worst clearance agreeing to 0.06 mm, 10.6%
of poses measured. The whole plate at those spacings — 165 141 poses, the shape of the real
thing — takes **30 minutes and 477 MB**, against **26 hours** at the old rate and more memory
than the machine has. `ec-automated-scan-first-run.md` has the detail and what the
certificate guarantees.

**Nothing knew what a plan would cost until it had been built.** `preview_ec_scan()` with
nothing planned now prices the area from the three technician controls alone — passes,
travel, waypoints, the planner's reply size, the gate's working set, planning time, then the
scan — and `plan_ec_path()` refuses a plan over `EC_PLAN_LIMITS` before it calls Noether.
Print it before you commit the work cell to anything at 0.30 mm.

**YOLO was never released.** `run_automated_scan(region=...)` skips detection, so it also
skipped the one place that turned the detector off, and `yolo_node` held 400% of a core and
its loaded model through the reconstruction and planning stages — the ones that were
OOM-killed. It is now released straight after stage 1 in both branches, by deactivating the
lifecycle node, which drops the model and the image subscription; a later `detect_part()`
brings it back.

## 5. What to bring back

- The whole `~/scandata/capture/<session>/` tree (bags, `scenario.json`, `session.log`).
- `~/.ros/log` for the bring-up, and the colcon build logs if anything was rebuilt.
- Every `tari_acceptance/*.json`.
- The terminal captures, which are the most useful single artifact of a visit.
- Any `~/scandata/<timestamp>/` reconstruction output.

## 6. Standing notes

- **Stop the External Control program before exiting dataguzzler-python**, or this UR firmware
  e-stops the robot.
- YOLO used 646% of a core with the detector enabled. Watch the effect on the camera's frame
  rate; the acceptance CPU check reports it.
- The Azure colour cloud costs 352 MB/s while anything subscribes. Untick RViz's Azure display
  before sweeps.
