# Cell data capture — ex-ur10e-1

The automated eddy-current pipeline (YOLO → Azure TSDF → Keyence raster → Noether →
contact scan) was built in Gazebo. This procedure records what the **real** sensors and the
real arm do, so every stage can be refined offline and the simulator made to behave like the
hardware. There are eight scenarios. Each take is its own ROS 2 bag with sidecars, driven from
the `ECAutomatedScan.dgp` prompt.

- Scenario functions: `ROS-Scanning/dgpfiles/modules/cell_capture*.dpi`, included at the end
  of `ECAutomatedScan.dgp`.
- Recorder: `ros2 run tari_bringup record` (`ros_ws/src/tari_ros/tari_bringup/tari_bringup/record.py`).
  The capture tool starts and stops it; it also summarises, expands and replays takes offline.

## Before leaving for the work cell

1. **Deploy by replacing, not merging**: copy `ros_ws/src` and `ROS-Scanning` over as whole
   trees, then do the clean build in `ec-automated-scan-first-run.md` section 0. Then check
   the recorder:

   ```bash
   source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
   ros2 run tari_bringup record --help | head -3
   python3 -c "import rosbag2_py, cv2, numpy, tf2_ros; print(rosbag2_py.get_registered_writers())"
   ```

   The last line must list `mcap`.
2. **Disk.** `df -h ~/scandata` — plan on **60 GB free** for a full visit (table under
   *Disk space*). Takes go to `~/scandata/capture/<session>/`. Every take is estimated before
   it starts and refused on hardware if it will not fit above `CAPTURE_RESERVE_GB` (5 GB);
   a running take closes its bag and stops itself if free space falls below the reserve.

## On the production system: the command sequence

Three terminals.

**T1 — bring-up** (leave it running all day):

```bash
source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
export TARI_CELL_PATH=tari_cell_configs
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```

Wait for `N started, 0 skipped` and the `sensor topic audit` line (no ERROR for
`/profiles`, the Azure topics or the wrench). On the pendant, load and **play** the External
Control program. RViz's config now ships both preview displays, **Capture preview** (MarkerArray on
`/capture/preview_markers`) and **Capture preview poses** (PoseArray on `/capture/preview`);
confirm both are ticked in the Displays panel — a config that does not subscribe to them shows
no path at all, however correctly it was published. Every planned motion is drawn there before
you confirm it (the path line green away from the part, orange near it; waypoints as dots; a label
naming the move), and the **MotionPlanning** display already in the config animates the arm
through it.

**T2 — the capture prompt** (every motion is confirmed here):

```bash
source /opt/ros/iron/setup.bash && source ~/ros_ws/install/setup.bash
export TARI_CELL_PATH=tari_cell_configs
cd <where ROS-Scanning was copied>/dgpfiles
dataguzzler-python ECAutomatedScan.dgp
```

Its banner ends with `cell capture loaded`, the speed limits and the work volume, then
`ECAutomatedScan ready.`. Before scenario 1 the hardware work volume reads "none declared"
and every motion scenario refuses. At `dgpy>`:

```python
capture_session(specimen="cal_plate", lighting="normal", notes="first visit")
capture_status()     # joint_states live? wrench live? robot program running? light/laser outputs?
```

**T3 — spare**: `ros2 topic hz /profiles`, `ros2 run tari_bringup record summary <take>`,
`df -h ~/scandata`.

Then run the scenarios **in this order** — each line is exactly what to type:

```python
# 1. work volume (everything else needs it): jog to 4 corners, Enter at each
capture_touch_off(corners=("corner_1", "corner_2", "corner_3", "corner_4"), height=0.32, floor_offset=0.0015)

# 6. cable envelope, BEFORE anything turns the wrist: answer t when the cables are tight
# full_circle=True sweeps the whole turn instead of +/-max_deg from the start angle, so the
# band is known around zero too -- which is where a scan goes and where no sweep reaches
capture_cable_envelope(step_deg=15.0, full_circle=True, wrist_deg_s=5.0)

# 2. Keyence ladder: probe tip on a flat, hole-free spot of the cal plate top
teach_point("flat")
capture_keyence_ladder(point="flat", line_deg=90.0, standoffs_mm=(-20, -10, 0, 10, 20), tilts_deg=(5, 10, 20), tilt_standoffs_mm=(-10, 0, 10), hold_s=2.0)

# 3. Keyence timing: probe tip on the raised top AT its long edge, ~100 mm apart
teach_point("edge_a"); teach_point("edge_b")
capture_keyence_timing(edge=("edge_a", "edge_b"), speeds_mm_s=(5.0, 10.0), round_trips=3, half_span_mm=30.0)

# 4. Azure sweeps, per specimen (region from the detector, or type it)
region = detect_part(index=0)
capture_azure_sweeps(region=region, specimen="cal_plate", lighting="normal", variants=("pipeline", "tilted", "high"), voxel_length=0.006, sdf_trunc=0.024)
#    ...change the lights (one bad-lighting take)...
capture_azure_sweeps(region=region, specimen="cal_plate", lighting="dim", variants=("pipeline",), voxel_length=0.006, sdf_trunc=0.024)

# 5. Keyence sweeps: the pipeline raster over the "pipeline" take's live Azure mesh,
#    then passes along the cylinder (probe tip on its crest ~10 mm in from each end)
#    NOT YET, SKIP THIS LINE: in simulation the pipeline raster makes a sudden ~3 deg wrist
#    jump starting a pass (tooling 159 mm/s near the part, limit 10) and the speed monitor
#    stops it. Run only the cylinder passes below until that is fixed.
# capture_keyence_sweeps(rough_mesh="<that take>/live_azure.ply", region=region, raster=True, cylinder=None, line_spacing=0.02)
teach_point("cyl_a"); teach_point("cyl_b")
capture_keyence_sweeps(raster=False, cylinder=("cyl_a", "cyl_b"), angles_deg=(0, 20, -20, 40, -40), margin_mm=10.0)

# 7. contact: probe ON the cylinder, compressed as when scanning, same orientation at both
teach_point("contact_a"); teach_point("contact_b")
capture_contact(points=("contact_a", "contact_b"), tilts_deg=(0, 3, -3, 6, -6), tilt_axis="travel", approach_mm=20.0, ec=False)

# 8. the whole pipeline, recorded
capture_full_pipeline(index=0, region=None, line_spacing=None, cutoff=None, park=True)
```

Repeat 4 and 5 for each specimen, with `specimen=` naming it. `list_taught()` shows the
taught points; they carry over to the next session on this cell (re-teach anything that
moved).

## Two things that bite

- **Stop the External Control program on the pendant BEFORE you exit dataguzzler-python.** On this
  UR firmware, exiting while it runs adds enough latency to e-stop the robot. That is the robot's
  firmware, not this stack.
- **A region is `(xmin, ymin, xmax, ymax)`.** Typed as `(xmin, xmax, ymin, ymax)` it can fly the
  Azure sweep hundreds of millimetres away from the part. `check_region()` refuses a region with
  no extent, one given in millimetres, or one outside the work volume, and prints the region you
  probably meant -- but read the order before you type.

## Answering the prompt, and stopping

Every motion prints what it will do — straight-line or joint move, length, "near" or "away"
the part, the speed — while RViz shows it.

| key | effect |
|---|---|
| **Enter** | go |
| **s** | skip this move (the scenario decides whether it can continue) |
| **q** | abort the scenario: nothing more moves, the laser and light go off, the take is closed and **kept** |
| **t** | cable test only: "the cables are tight here" |

**To stop the arm while it moves, use the pendant** (pause, or the e-stop). The move then
fails, the scenario ends, and the take is closed and kept. **Ctrl-C at the dgpy prompt does
not stop a running scenario** — the prompt runs in a thread that never receives it. After a
pause, press play on the pendant before the next motion; every capture motion refuses while
the robot program is not running.

`CAPTURE_AUTO_CONFIRM` exists for simulation and is refused in hardware mode.

## The scenarios

### 1 — Work-volume touch-off

For each corner of the usable table region the prompt says "jog the probe tip onto the table
at corner k". Jog with the pendant, stop, press Enter. `teach_point()` averages
world → `probe_end` for 1 s and refuses if the arm moved more than 0.3 mm meanwhile. The
recorder runs throughout (core topics), so the bag shows TF, joint states and the wrench while
jogging.

The box: x/y the extent of the corners; floor the highest corner + 1.5 mm; top floor + 0.32 m.
Motion is then limited to the box's x/y, never below its floor, and at most
`CAPTURE_ABOVE_VOLUME` (0.25 m) above its top, where the Azure sweep flies. Touch-off warns if
`SPECIMEN_SWEEP_HEIGHT` is outside that.

**Two different ceilings, and they are often confused.** The box's own top is where a part may
be placed and where the TSDF stops integrating -- which is why the Azure's tooling, 192 mm tall
above `probe_end` and flying well above it, never lands in a reconstruction. What bounds the
TOOL is that top **plus** `CAPTURE_ABOVE_VOLUME`. The pan sweep derives its height against the
motion ceiling; testing the whole tooling against the crop top instead collapsed every plan to
250 mm above the part and 68 degrees from vertical.

Files:
- `<session>/work_volume_hardware.yaml`: the block to paste into
  `tari_cell_configs/config/cells/ex-ur10e-1.yaml` in the site package;
- `~/scandata/capture/cell_override/ex-ur10e-1.work_volume.json`: the measured box. Every
  dgpy session that loads the capture tool uses it at once: it regenerates
  `cell_override/ex-ur10e-1.yaml` from the real manifest with only the box replaced, and puts
  it first on `TARI_CELL_PATH`. When the block is in the real manifest the override switches
  itself off; `clear_work_volume_override()` retires the files.

**Pendant jogging and External Control.** Jogging from the pendant needs the External
Control program stopped or paused. From the installed ur_robot_driver 2.3.12:
- `controller_stopper` keeps `joint_state_broadcaster`, `io_and_status_controller`,
  `force_torque_sensor_broadcaster` and `speed_scaling_state_broadcaster` running (its
  `consistent_controllers`), so **`/joint_states`, TF and the wrench keep publishing while you
  jog**, which is what teaching relies on.
- It stops the trajectory controller while the program is not running and restarts it when
  `robot_program_running` goes true.
- **Nothing on the ROS side needs restarting.** Press play on the pendant (or
  `ros2 service call /dashboard_client/play std_srvs/srv/Trigger`) before the next automated
  motion.

Scenario 7 teaches the same way. *Reasoned from the driver's source; not yet seen on the robot.*

### 6 — Cable envelope (second, before anything turns the wrist)

Over the middle of the work volume at transit height, at no tilt, ±40° about x and ±20° about
y (the extremes the Keyence and Azure views use), wrist 3 steps 15° at a time at 5°/s, each
step previewed and confirmed, one way and then the other. **Answer `t` when the cables are
tight**: the last step before it is recorded and the wrist returns.

**One band per tilt, and each is enforced only at its own tilt.** They are not close to one
another — 270° of travel untilted against 75° at −40° about x on this cell — so intersecting
them into one window and enforcing that everywhere refuses runs at angles that are well inside
the band for the tilt actually commanded. A tilt between the measured ones is interpolated
along each axis; a tilt outside them is refused rather than extrapolated. The model is
`modules/cable_envelope.dpi`.

The result is saved to `~/scandata/capture/cable_limits_<cell>_<mode>.json`, and the test prints
a `cable_envelope:` block to paste into the work-cell manifest — **do that**, because the JSON is
only on the machine that ran the test while the manifest travels with the work cell and is what a
fresh dgpy session reads.

From then on **every motion that would turn wrist 3 outside the band for its tilt is refused**
before it moves — not only capture motions. `install_cable_guard()` wraps `ros` for the whole
ECAutomatedScan.dgp session, so `run_automated_scan()` is covered too. Without that coverage a
scan can command wrist 3 far outside the band and only the emergency stop catches it.
`cable_guard_off()` lifts the guard for a block, and this scenario is the one caller that needs
it: it cannot measure the band without leaving it.

The band is in **absolute** wrist_3 joint angle and is known only where it was swept. `max_deg`
is measured from the start angle, so the default 180 covers about 16–327° from a typical
working angle and says **nothing about zero** — which is where a sweep can ask the arm to go.
`capture_cable_envelope(full_circle=True)` sweeps the whole turn instead; it is four times the
steps and is what any new measurement should use.

Each tilt is reached from the untilted pose; a tilt the arm cannot take there is recorded as
`unreachable` and the test goes on (in simulation −40° about x was: the only IK solution was
another arm configuration, and the joint move to it was refused because it would have swung the
tip out of the work volume). Where a scenario has a choice — a laser line has no sign — it
picks the orientation that turns wrist 3 least and stays inside the band for that orientation.

### 2 — Keyence static ladder

Standoffs −20…+20 mm about the reference distance; tilts ±5/10/20° about the laser line
("along") and across it, at −10/0/+10 mm; each held 2 s with the laser on. Tilts turn about
the point where the beam meets the flat, so the measured spot stays put. `line_deg=90` puts
the laser line along world y, which is the tool's own orientation (no wrist turn). Poses that
would bring the probe tip within `CAPTURE_CLEARANCE` (10 mm) of the flat are dropped and
listed. Every hold is an event in the bag with its pose and time window.

### 3 — Keyence timing

The laser line lies along the edge; the sensor crosses it ±30 mm, three round trips per speed,
its reference distance 8 mm below the top so both levels are in range. Each pass is an event
with speed and direction. Speeds above the approved 10 mm/s near the part are clamped. The
edge's apparent shift between directions, at two speeds, separates latency from everything
else — and `tari_pc_reconstruction` takes the *latest* TF for each profile, which is exactly
what this measures.

### 4 — Azure sweeps

"pipeline" is `reconstruct_specimen("azure")` exactly as `reconstruct_rough()` flies it
(`SPECIMEN_SWEEP_STYLE`, which is `"pan"` now, 6 mm voxels); "tilted" adds the old dome's
tilt and "high" flies it 0.10 m higher, both of which only mean anything for
`SPECIMEN_SWEEP_STYLE = "dome"` -- with the pan sweep the camera is already aimed and already
sees the part from several directions, and its height is derived rather than set. The live TSDF runs in every take; its mesh is copied
in as `live_azure.ply`, and every reconstruction service call is logged with all its fields and
ROS time, which is what `record replay` repeats.

### 5 — Keyence sweeps

The raster is `reconstruct_specimen("keyence")` with `orient="fixed"`, `roll="path"`,
`snake=False`, 20 mm lines, 10 mm/s. The cylinder passes run along the axis between the taught
crest points (10 mm in from each end), at 0, ±20 and ±40° round it with the sensor facing the
surface, into one live `tari_pc_reconstruction`. The axis comes from the crest points and the
known radius (31.69 mm), not from the Azure mesh, which is what is under test.

### 7 — Contact

Each pass: over `contact_a`; down onto it **along the tool axis**; straight to `contact_b`;
up **along the tool axis**. Never past the taught points. Tilts turn the probe about its tip,
about the travel direction (`tilt_axis="travel"`) or across it (`"across"`); the probe's roll
stays as taught. 10 mm/s. A wrench change over `CAPTURE_CONTACT_FORCE_LIMIT` (60 N) stops the
arm. `ec=True` also arms `ec_source` and writes `ec_<tilt>.npz` per pass — untested, there is
no DAQ off the work cell.

### 8 — Full pipeline

`run_automated_scan(scan=False)`, recorded, every motion it makes confirmed at the prompt:
the capture gate stands in for `ros` for the length of the call.

## When a scenario fails

The take is **always** closed and kept, with `scenario.json` saying what happened.

| message | what happened | what to do |
|---|---|---|
| `... Nothing moved.` (work volume, cable envelope, clearance, "cannot follow up-across-down") | refused before moving | read the reason; re-teach the point, or jog the arm nearer with the pendant, then run the scenario again |
| `... did not complete (planner reached N%)` | MoveIt could not plan or finish the move | the arm stopped where it was; run the scenario again (a new take); if it repeats, jog nearer or re-teach |
| `STOPPED (overspeed ...)` / `force change ...` | the monitor stopped the arm; on hardware the UR program is **paused** | check the work cell, press play on the pendant, run the scenario again. Later moves of that kind are already slowed by the measured ratio |
| `... The cables allow A..B deg at tilt (...) ... Nothing moved.` in scenario 4, 5 (raster) or 8 | the pipeline's own first move goes to its home configuration, `home_joints` in `mesh_reconstruction.dpi`, with **wrist 3 at 0°** — about 166° from where the arm works, and below the 31.3° the cable test confirmed as slack | nothing moved; the take is kept. Do not widen the band to get past it, and do not reach for the branch 360° away — the cable wraps, so +200° and −160° are the same tool orientation and opposite cable states. The pipeline has to use a home with wrist 3 near the working angle (reported as a pipeline fix) |
| `the robot program is not running` | External Control is paused/stopped (after jogging) | press play on the pendant |
| `not enough disk for ...` | the estimate does not fit above the reserve | free space, or run fewer variants |
| `the recorder did not start` | `recorder.log` in the take has the reason | `ros2 run tari_bringup record probe --cell ex-ur10e-1 --mode hardware --sets core,keyence` in T3 |
| `LOST MESSAGES on critical topics` after a take | DDS dropped TF/joint states/profiles/wrench | note it and repeat the take; `record summary <take>` shows how many |
| dgpy died | the recorder notices and closes the bag within seconds (`--stop-with-pid`) | switch the warning light off on the pendant's I/O tab; restart T2; the session's taught points and work volume come back |

Partial takes are data too — keep them; `scenario.json` marks them `aborted` or with `errors`.

## Disk space

What each take cost in simulation, and what to expect on hardware. The simulated Azure is
640×480 at ~2 Hz and the simulated Keyence 56 points at ~2 Hz, so on hardware the Azure and
Keyence rows are **estimates** — the recorder's probe prints the real rate and size before
every take (`~N MB for T s`), and refuses on hardware if it will not fit.

| take | simulation (measured) | hardware (expected) |
|---|---|---|
| core only (touch-off, cable test, contact) | ~4 MB/min — 100 Hz joint states | ~15–25 MB/min — joint states, TF and wrench at 500 Hz |
| + Keyence (ladder, timing, sweeps) | ~4.5–5 MB/min — 48 points × ~2–20 Hz | + the Keyence's own rate × ~13 KB per 800-point profile: **up to ~0.8 GB/min** at 1 kHz; the probe prints it |
| + Azure (sweeps, full pipeline) | see scenario 4 below | ~1 GB/min at 5 frames/s (JPEG colour + PNG depth, 2048×1536) |

A whole visit — every scenario once, two specimens, two lighting takes — is on the order
of 20–40 GB, dominated by the Azure takes and by the Keyence rate. Plan on 60 GB free.

## What a take contains

`~/scandata/capture/<session>/NN_<scenario>_<time>/`:

| file | contents |
|---|---|
| `bag/` | MCAP bag (zstd) |
| `capture.json` | recorder sidecar: git state (or content hashes) of every tree, the manifest as loaded (plus `manifest.yaml`), robot description (`robot_description.urdf`/`.srdf`), static TF, camera_info, the Keyence sampling rate from the driver's log plus the measured rate, and per topic: count, **DDS-level losses** (publisher sequence numbers), header-stamp gaps, bytes |
| `scenario.json` | taught points used, parameters, every move (class, limit, commanded, **measured peak** speed, scaling and its bound), every confirmation, every reconstruction service call with all its fields and ROS time, results, and a `speed` summary |
| `recorder.log` | the recorder's output |
| `live_*.ply` | the live meshes |

**Topics** come from the manifest for the mode. `core`: `/tf`, `/tf_static`, `/joint_states`,
`/robot_description(_semantic)`, the wrench (`/force_torque_sensor_broadcaster/wrench`), UR
IO, `robot_program_running`, speed scaling, `/rosout`, `/capture/events`, `/capture/preview`.
`keyence` adds `/profiles` (raw, every message). `azure` adds colour, depth and camera_info —
hardware `/rgb8/rgb/image_raw`, `/azure/depth_to_rgb/image_raw`, `/azure/rgb/camera_info`;
sim `/azure/image_raw`, `/azure/depth/image_raw_16uc1`, `/azure/depth/camera_info`. Colour is
written as `<topic>/compressed` (JPEG q90), depth **losslessly** as `<topic>/compressedDepth`
(PNG), both thinned together by header stamp to `CAPTURE_IMAGE_RATE` (5/s) so the kept frames
still pair. Nothing else is thinned.

```bash
ros2 run tari_bringup record summary <take>                  # counts, losses, sizes
ros2 run tari_bringup record expand  <take>                  # -> <take>/replay_bag, raw images
ros2 run tari_bringup record replay  <take> --start-nodes    # re-run the live reconstruction
```

`replay` plays the expanded bag with `--clock` and makes the recorded reconstruction calls when
the bag clock reaches their recorded times, writing outputs to `<take>/replay/`. With
`--start-nodes` it starts `industrial_reconstruction` / `tari_pc_reconstruction` itself with
`use_sim_time`. **Nothing else may serve those services**: run it with the work cell (and the
simulator) stopped.

### What replay has been proven to reproduce

The claim the rest of this document rests on is that a bag brought back from the work cell is
enough to re-run the reconstruction elsewhere. It has been tested by replaying a take
recorded in simulation with the work cell and the simulator down and comparing the replayed mesh
against the one the live run produced.

**The Azure dome into `industrial_reconstruction`** (89 s of sweep, 15 MB bag, 455 MB
expanded):

| | live | replayed |
|---|---|---|
| vertices / faces | 5 573 / 10 674 | 5 596 / 10 721 (+0.41%) |
| bounding box | x 0.6603–1.0497, y −0.3570–−0.0150, z 0.9167–0.9967 | identical to 0.1 mm |
| centroid | 0.8540, −0.1859, 0.9386 | 0.8541, −0.1859, 0.9386 |

Nearest-vertex distance replay→live: mean **0.060 mm**, rms 0.18, p95 0.16, max 4.72. The
other way: mean 0.054 mm, rms 0.13, p95 0.15, max 2.99. 92% of vertices within 0.1 mm of the
other mesh, 99.5% within 1 mm. The two service calls fired at bag clock 594.945 and 684.445
against recorded times of 594.943 and 684.230.

**The Keyence raster into `tari_pc_reconstruction`** (1 406 s, 159 MB bag, 57 950 profiles
recorded with none lost): the same surface comes back, but not the same point set. Run twice,
once on a loaded machine and once on an idle one:

| | loaded | idle |
|---|---|---|
| `pc_start` | at bag clock 1769.080 vs 1769.072 recorded | the same, 8 ms |
| `pc_stop` | 2599.260 vs **2578.254** — 21 s late | 2626.160 — **48 s late** |
| Poisson vertices | 78 043 (72.7% of live) | 78 922 (73.5%) |
| replay→live nearest vertex | mean 0.50 mm, p95 1.87 | mean 0.43 mm, p95 1.21 |
| within 0.5 mm | 85.5% | 88.0% |
| bounding box, centroid | within 2 mm of live | within 2 mm of live |

So the replay rebuilds the same surface to well under a millimetre, and it is **not** a
repeat of the live run: the stop call lands tens of seconds of bag time late on a 1 400 s
bag, the reconstruction therefore integrates data taken after the raster had finished, and
the merge / SOR / crop chain downstream of that produces a different point set. Load is not
the cause — the idle run was worse — and the mechanism is not identified: the replay loop
spins at over 20 Hz, so the clock must be arriving in gaps. `replay` records the largest gap
between two `/clock` samples (`clock_jump_max_s` in `replay.json`) and warns on any call more
than a second late, which is what the next long replay should be read for. The 89 s Azure
take's stop landed 0.215 s late, which is why it is the clean proof and the raster is not.

**The QoS trap behind "The TSDF is empty".** `ros2 bag play --clock` publishes with
`rclcpp::ClockQoS`, which is BEST_EFFORT, and a RELIABLE reader matches no best-effort
writer. A `/clock` subscription left on the default (reliable) profile therefore never
advances: the replay clock stays at 0 for the whole bag, every recorded service call fires at
once at bag time 0 with `after_bag_end: true`, and `stop_reconstruction` answers *"The TSDF is
empty"* — a failure that reads as a reconstruction problem and is a QoS problem. `record.py`
has one `clock_qos()` for this, BEST_EFFORT and depth 1, which is what `rclpy`'s own
`TimeSource` uses and what reads Gazebo's reliable `/clock` as well.

The recorder also runs on its own, e.g. while jogging:

```bash
ros2 run tari_bringup record --cell ex-ur10e-1 --mode hardware --sets core,keyence --out ~/scandata/capture/jog
```

## How the approved rules are enforced

| rule | enforcement |
|---|---|
| preview and confirm every motion | every motion — the tool's own and every one the pipeline makes in scenarios 4, 5 and 8 — is drawn in RViz, the arm animated through it, and confirmed at the prompt before it is sent |
| ≤ 50 mm/s away, ≤ 10 mm/s near | see below |
| contact only between taught points | contact passes run exactly taught → taught, approach and retract along the tool axis at those points; margins come out of the sampled length, never beyond |
| nothing outside the work volume | straight legs are checked every 5 mm, joint moves every 2° through FK, before anything moves; one sample outside refuses the move; no motion at all without a measured volume |
| warning light on | the scenario holds `warning_light()` (output 0) for its whole length and refuses to move on hardware if it cannot switch it on; the laser (output 1) is on only in the Keyence scenarios |
| wrist rotation only within the cable envelope | any move whose IK (Cartesian) or FK (joint) puts wrist 3 outside the band **for the tilt it commands** is refused. Per tilt, in absolute joint angle, interpolated between the measured tilts and refused outside them. In force for the whole session, not only inside a scenario — `install_cable_guard()`, lifted only by `cable_guard_off()`. An unmeasured cell says so loudly once and does not refuse, because the cable test needs a work volume and the work volume needs motion |

**Speed, three ways.** "Near" is within 0.10 m of the known part, taught points or table, or
anywhere below the work-volume top when a scenario knows no geometry.
1. **Before the move — a bound.** MoveIt's scaling is a fraction of the joint velocity limits,
   not a speed. The capture tool reads the limits MoveIt applies (from `mtc_hub_node`), solves
   IK along the Cartesian path (or FK along a joint move) and takes the largest scaling at
   which no checked point of the tooling (the probe tip and the Keyence head,
   `lj_v7080_frame`) can exceed
   0.8× (Cartesian) or 0.85× (joint) the class limit. The work cell's velocity calibration can only
   make it slower.
2. **During — a stop.** The probe tip's speed is measured from TF, timed by TF's own stamps,
   over ≥ 0.1 s. Above 1.25× the limit (`CAPTURE_OVERSPEED_FACTOR`) the arm is stopped: on
   hardware the UR program is paused through the dashboard (a controlled stop on the robot);
   if that fails, and always in simulation, the trajectory controller is given a hold. The
   scenario ends.
3. **After — reported, and corrected.** A measured peak above the limit (+5%,
   `CAPTURE_SPEED_TOLERANCE`) is reported as `OVER THE LIMIT`, never `ok`, counted in
   `scenario.json`, and every later move of that kind in the session is slowed by the measured
   ratio with 10% to spare.

## Known limitations

- Not run on hardware. Reasoned from source, not observed: pendant/External Control
  behaviour; the dashboard pause; the real Keyence rate from the driver log; image sizes at
  2048×1536.
- The **work-volume** envelope check covers the probe tip (and the Keyence optical frame for
  speed): the taught box is a box the tip must stay in, and that is what it tests.
- **Clearance to the specimen is measured against the whole end-of-arm assembly.** The
  cylinder passes of scenario 5 build the taught cylinder into a surface and run every pass
  through `tooling_clearance.dpi`, which samples the collision geometry of every link rigidly
  attached to `tool0` — the eddy-current probe and its holder, the Keyence, the Azure and its
  bracket — out of the live `robot_description`, and refuses the pass on the worst signed
  distance any of them reaches, naming the link. It reports the number, not a verdict, so the
  take records the margin each pass actually had. This replaced a check on the probe TIP
  alone: on the fixture the tip reads +34.3 mm at every angle while the tooling reads
  +32.3 mm, and a pose 45 mm into the part reads −10.6 mm at the tip and −22.6 mm at the
  Keyence body. The pipeline's own stages are checked the same way, and the specimen now
  reaches MoveIt's planning scene from `obtain_specimen_mesh()`/`plan_mesh_path()`, so the
  planner refuses it too. Where the description cannot be read the check says so loudly and
  falls back to the tip; a take that got the fallback records `clearance_check: "tip only"`.
- Drop counts are exact on single-publisher topics; on `/tf` (several publishers) they are an
  estimate, and marked as one.
- EC capture through `ec_source` is untested.
- **Replay is proven for the Azure/TSDF path and only indicative for the Keyence one** — see
  "What replay has been proven to reproduce" above. On a long bag the recorded service calls
  land tens of seconds of bag time late for a reason not identified, which changes what the
  reconstruction integrates. Quote the surface agreement, not the point counts.
- Replay repeats the reconstruction, not the motion. It says that a bag carries enough to
  rebuild the mesh; it says nothing about whether the arm would move the same way again.

## Simulation evidence

Every scenario run in Gazebo against `ex-ur10e-1` in `sim` mode, driven by a stand-in operator
(`scandata/capture_sim/sim_driver.py`, outside the repositories: scripted moves in place of
pendant teaching, `CAPTURE_AUTO_CONFIRM`, "cables tight" at ±150°). The cal plate and the
fiducial cylinder spawned as Gazebo models. **Speed peaks are the bag's ground truth**: the
probe tip's world position looked up from the recorded `/tf` at every stamp, chord speed over
≥ 0.1 s of stamp time, per executed move; the live monitor agreed with it to within 0.33 mm/s
on every move. "lost" is counted from publisher sequence numbers (exact on these topics).

| # | take | size | ROS time | messages | lost | moves | peak / limit, mm/s |
|---|---|---|---|---|---|---|---|
| 1 | touch-off (4 corners, stand-in teaching inside the take) | 16.3 MB | 237 s | 37 444 | `/joint_states` 10 | 19 | away 41.1 / 50, near 8.3 / 10 |
| 6 | cable envelope (4 of 5 tilts; −40° about x unreachable, refused before moving) | 17.2 MB | 261 s | — | `/joint_states` 6 | 48 | 23.7 / 30 |
| 2 | Keyence ladder, 40 holds | 74.2 MB | 937 s | 163 621 | `/joint_states` 26; `/profiles` 0 of 14 942 | 46 | away 39.8 / 50, near 8.3 / 10 |
| 3 | Keyence timing, 12 passes | 17.6 MB | 249 s | — | `/joint_states` 8; `/profiles` 0 of 2 154 | 18 | away 24.3 / 30, near 8.3 / 10 |
| 4 | Azure dome ("pipeline"), live TSDF, cal plate | 23.2 MB | 103 s | — | `/joint_states` 4; depth/colour 329 each (0 lost, 2 thinned) | 5 | away 23.7 / 30 |
| 5 | Keyence passes along the cylinder, 0/±20/±40°, live `tari_pc_reconstruction` | 33.9 MB | 507 s | — | `/joint_states` 9; `/profiles` 0 of 2 299 | 31 | away 26.2 / 30, near 8.8 / 10 |
<!-- EVIDENCE-MORE -->

**The clearance routing, measured on scenario 5's cylinder passes** against the live
`robot_description` and the taught fiducial cylinder (no motion needed — the check runs at
plan time). The tooling envelope is 12 links, 119 955 sample points at 4 mm, reaching 178 mm
from `probe_end`; the taught cylinder becomes a 3 977-vertex patch. Every pass from 0 to ±60°
about the axis: tip-only **+34.31 mm**, whole tooling **+32.03 mm**, closest link `ec_probe`,
all kept. Seven passes checked in 3.2 s.

**A single-waypoint Cartesian move can be refused by `tari_cpp_mtc`'s trajectory guard, and
the reason is not in this tool.** `checkPathFidelity` scores every executed point against the
requested polyline, and a single point is not a polyline, so the whole approach counts as
off-path ("the tool leaves the requested path by 109.0 mm at point 0 of 1008, limit 5.0 mm"
for a 173 mm descent). `capture_move()` then falls back to a joint move, which is scaled in
joint space: the tool reaches tens of mm/s against a commanded 4 mm/s and the monitor stops
the arm — the enforcement working exactly as designed, on a move that should never have been
a joint move. Scenarios 5, 7 and 8 are the ones that hit this.

Across every take: **no move over its limit; the worst peak was 0.83× the limit**. (The away
limit is `ECAutomatedScan.dgp`'s `TRANSIT_SPEED`, and the capture tool takes its limits from
there; the earliest takes in the table were run at 50 mm/s and the rest at 30.)

**The stop works.** With the trip lowered to 0.3× (15 mm/s) for one 100 mm descent commanded at
50 mm/s, the monitor measured 16.2 mm/s, replaced the trajectory with a hold, and the arm
stopped after 3.7 mm; the move raised `CaptureRefused` naming the overspeed.

**Losses.** `/joint_states` lost 6–26 messages per take (about 0.03%), counted exactly and
confirmed by the header stamps (one 10 ms period missing at a time, never more than one in a
row). The publisher is reliable, so these are DDS-level losses while the 8-core container ran
Gazebo, MoveIt, YOLO and dgpy at full load; `/tf`, `/tf_static` and `/profiles` lost nothing.
The recorder reports this after every take as `LOST MESSAGES on critical topics`.

Design decisions the simulation forced, each of which is a trap if it is ever undone:

- **The speed monitor is TF-stamped, not poll-timed.** Timing `live_pose`'s polls makes a
  50 mm/s move read 82.8 mm/s when TF arrives in bursts. The measurement is now compared
  against the bag, and an over-limit peak is never reported as "ok".
- **The transit checks all three legs before moving**, by IK every 20 mm in the arm's current
  configuration. Asking only "can the target be reached at some height" fails 90% of the way
  across.
- **After touch-off the tip sits on the table, below the new floor**, so an upward first move
  out of the volume is allowed.
- **The Keyence ladder's speed bound watches the sensor head, not the optical point** 80 mm
  out along the beam. Watching the optical point makes the ladder crawl at 1.3 mm/s at the tip.
- **The timing passes pick between the two equivalent sensor orientations by which keeps the
  tip inside the touch-off box** — the sensor is 76 mm beside the tip, so one of them does not.
- **Flying the pipeline through the capture gate needs three things.** An IK chain built 20 mm
  at a time, because in the scan motion profile a single IK call from a seed 0.3 m away does
  not converge and the pipeline falls back to its home configuration, which is outside the
  touch-off box. A refused row returned to the pipeline as "did not fly" rather than raised, so
  the sweep carries on and its mesh is saved — the dome's last row lies outside the touch-off
  box, because the box is where parts are and the dome is planned from the region. And a
  `home_joints` whose wrist 3 is near the working angle: at 0° it is ~156° away and the cable
  envelope refuses the pipeline's very first move.
