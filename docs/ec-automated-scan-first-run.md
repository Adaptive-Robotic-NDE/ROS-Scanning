# The automated eddy-current scan: first run on hardware

`dgpfiles/ECAutomatedScan.dgp` runs the eddy-current cell `ex-ur10e-1` without a
person answering prompts. It finds the part (YOLO), measures it roughly (Azure
dome sweep into a TSDF), then finely (a Keyence raster over the rough mesh). It
plans a contact toolpath over the fine mesh (Noether), then scans it in
continuous motion, correlating eddy-current readings with position by timestamp.
It was written from `ClientScannerTest.dgp`, which ran on this cell with an
operator at every step, and **it has never run on hardware.**

This page says what to run on the first day, in what order, what each stage
prints when it is healthy, how it fails and what to do. It also lists every
setting that differs from what ClientScannerTest used on the real cell, and why.

**What has been verified, and where.** Every stage up to and including a
motion-only dry run of the eddy-current toolpath has run end to end in Gazebo
(`CELL_MODE="sim"`, no eddy-current instrument), on the calibration plate and on
the fiducial cylinder; section 4 has the numbers.
`run_ec_scan()` itself has not run anywhere, because simulation has no
eddy-current instrument. The simulation has no spring and no contact, so
everything this page says about contact is geometry, not force. See section 5.

---

## 0. Before the first day

**Deploy by replacing, not merging.** Copy `ros_ws/src` and `ROS-Scanning` over
as whole trees (delete the old ones first), then do a clean
`colcon build --symlink-install`. Copying over an old tree leaves deleted files
behind. Everything this scan depends on is inside those two trees, including
three local patches that are not upstream:

| package | patch | without it |
|---|---|---|
| `industrial_reconstruction` | `tf_cache_time` (60 s TF cache) | every frame fails "extrapolation into the past" on a slow camera; "The TSDF is empty" |
| `tari_pc_reconstruction` | clustering opt-in, crop before cluster | `/stop_and_process` fails "DBSCAN did not find any clusters" |
| `tari_mesh_filter` | colourless-mesh fix | the filter node dies on the Keyence mesh; reads as a timeout |

`spatialnde2` is compiled and is not in those trees. The controlling computer needs it rebuilt
with the recstatus placement-race fix before any timestamped eddy-current
scan. Without it, the DAQ's acquisition thread can die mid-scan with "becomes
fully ready with recording not found in instantiated or mdonly".

**Budget time for two measurements the scan needs and does not have.** The
contact preload is unknown (`EC_PRELOAD_M = 0`, section 3), and the work cell's speed
curve was measured over a 300 mm line in free space while an eddy-current pass is
a few tens of millimetres — so the speeds that matter for a contact sweep have not
been measured. Both are cell work with the arm, not settings to pick: steps 7 and
12 below.

**The settings to look at before starting** (top of `ECAutomatedScan.dgp`):

| setting | value | why it matters on day one |
|---|---|---|
| `CELL_MODE` | `"hardware"` | the only switch between the rooms |
| `EC_DAQ_DEVICE`, `EC_CHANNELS` | `"PXI1Slot2"`, `("ai0","ai1")` | the Uniwest X/Y outputs on the NI card: check the wiring |
| `WARNING_LIGHT_PIN` | `0` | UR digital output 0 is the work cell's warning light |
| `NEAR_PART_SPEED`, `TRANSIT_SPEED` | 0.010, 0.030 m/s | 10 is the approved near-part limit. Transit is 30, not the 50 limit, because other directions run faster than the calibration: a vertical rise commanded at 50 peaked at 87.6 mm/s in simulation |
| `SCAN_VELOCITY` | 0.010 m/s | in contact for the whole sweep, so it must not exceed `NEAR_PART_SPEED` |
| `SCAN_LINE_SPACING` | 0.0003 m | ClientScannerTest's value; makes a whole part hours long -- see `EC_SCAN_REGION` |
| `EC_SCAN_REGION` | `None` | set it to the area to scan |
| `EC_PRELOAD_M` | `0.0` | the contact preload, in metres. **Not known**: measure the fixture. `set_ec_preload()` is how it is changed; ceiling `EC_PRELOAD_CEILING_M` = 2 mm (section 3) |
| `EC_INSTRUMENT` | `"uniwest"` on hardware | the instrument the scan nulls on the part, at port `EC_INSTRUMENT_PORT` (33). `EC_NULL_ON_CONTACT`, `EC_NULL_REQUIRED` and `set_ec_nulling()` decide what a failure means (section 3) |
| `NOETHER_ROLL_REFERENCE` | `(1, 0, 0)` | the locked probe roll; set it from the cable-envelope test |
| `NOETHER_ROTATION_OFFSET`, `EC_TRAVEL_DIRECTION` | pi/2, `(-1, 0, 0)` | passes along the part's longest side, travelling -x, as ClientScannerTest ran them |
| `EC_MAX_TILT_DEG` | 45 | **a placeholder**: the steepest pose the probe is planned onto; the fixture's real limit is unknown |
| `EC_AXIS_SMOOTHING_M` | 0.006 m | the probe axis is averaged along each pass over the probe face's width (5.8 mm); positions are not touched |
| `JOINT_MOVE_SCALING` | 0.015 | MoveIt scaling for every automated joint move; at MoveIt's 0.1 the tip reached 245 mm/s, at 0.02 still 55 (section 4) |
| `EC_AXIS_SPEED` | 0.006 m/s | the lift off and descent onto the part; 0.008 peaked at 11.9 mm/s |
| `EC_WAYPOINT_SPACING_M` | 0.001 m | every pass resampled to 1 mm between waypoints, ends kept, roll still locked |
| `EC_CLEAN_PLANNING_MESH` | `True` | drop non-manifold faces and crumbs under 1% before Noether plans; without it Noether refused the cylinder |
| `SPECIMEN_SWEEP_STYLE` | `"pan"` | hold the depth mode's minimum RANGE by standing off to the side and aiming back, instead of by flying high (step 5) |
| `SPECIMEN_SWEEP_SLANT` | `None` | the range to hold; `None` = the depth mode's minimum + 30 mm, so 0.53 m for NFOV_UNBINNED |
| `SPECIMEN_SWEEP_CAMERA_HEIGHT` | `None` | camera height above the part; `None` walks `AZURE_PAN_HEIGHTS` from 0.50 m down and takes the highest rung the arm reaches |
| `SPECIMEN_SWEEP_DIRECTIONS` | 4 | how many of the 8 candidate stand-off directions to fly; fewer than 2 reachable is a refusal |
| `SPECIMEN_SWEEP_HEIGHT`, `SPECIMEN_SWEEP_STOPS` | `None`, 5x5 | `"dome"` only: the old straight-down raster |
| `DETECT_POSE`, `PARK_POSE` | `"over_table"` | the named pose ClientScannerTest started and ended at |
| `ROBOT_VELOCITY_CALIBRATION` | the work-cell manifest's curve, measured over a 300 mm line | every m/s goes through it, and a pass is a tenth of that line — **re-measure it at pass length before the first scan**, step 7 |

---

## 1. The first day, in order

Each step assumes the previous one worked. The approved motion rules for this
cell are: at most 50 mm/s away from the part and 10 mm/s near it; contact only
on the planned surface; nothing outside the work volume; the warning light on
whenever an automated stage moves the arm. Preview every motion in RViz before
confirming it.

1. **Bring up the work cell.**
   `ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware`.
   Wait for `N started, 0 skipped` and the `sensor topic audit` line. A skipped
   motion server means nothing will move later.
2. **Start the config**, in the directory above `dgpfiles` or in `dgpfiles`:
   `TARI_CELL_PATH=tari_cell_configs dataguzzler-python ECAutomatedScan.dgp`
   (or source `tari_setup.bash` first). The banner should end with the
   stage list. On hardware it will say
   `cell ex-ur10e-1 mode hardware declares no work volume` -- expected, see
   step 4.
3. **Check the warning light.** At the prompt:
   `exec("with warning_light('light test'):\n    time.sleep(5)")`. The
   light should be on for five seconds, then off. If the console prints
   `could not switch the warning light ON`, the UR driver's
   `io_and_status_controller` is not running
   (`ros2 service list | grep set_io`). Every automated stage will then refuse
   to move, which is deliberate.
4. **Measure the work volume.** The first stage that needs it refuses, and the
   refusal says how. The capture tool does it:
   `capture_touch_off()` (`dgpfiles/modules/cell_capture.dpi`; procedure in
   `docs/cell-data-capture.md`). It records the tip at each touch and writes a
   local override manifest, which the running config picks up at once. It also writes the
   `work_volume` block to fold into
   `tari_cell_configs/config/cells/ex-ur10e-1.yaml` in the site package
   later. By hand instead: touch the probe tip to the bench at two opposite
   corners of the placement area, without compressing the spring, and read
   `ros.t.transform.translation`. Take zmin = bench + 0.0015 and zmax = the
   tallest part + 0.05 (no higher than 1.25), write them under
   `work_volume: hardware: min/max` in that file, and run `work_volume_reload()`.
   Either way, confirm with `describe_work_volume()`.
5. **Check the Azure can see the part from the sweep.** The hardware driver
   runs `depth_mode NFOV_UNBINNED`, whose minimum RANGE is 0.5 m -- a distance
   from the camera to the surface, not a height. The sweep holds that distance
   by standing off to the side and aiming back at the part
   (`SPECIMEN_SWEEP_STYLE = "pan"`), which is what makes it reachable: a
   straight-down sweep needs the camera at 1.455 and probe_end at 1.380, and
   solved against this cell's own kinematics that has **no IK solution at all**
   over the far x column (x >= 1.145) and reaches only 84.6% of the work volume
   anywhere. The pan sweep reaches all of it.
   The plan prints the range it will hold and **refuses** rather than flying
   inside the minimum:

   ```
     camera-to-nearest-surface range (support 0.9300): 530..547 mm; NFOV_UNBINNED works from 500 to 3860 mm
       at 530 mm that mode sees 813 x 675 mm at 1.27 mm/pixel
       camera 500 mm above the part, 176 mm to the side, 27 deg from vertical, from 3 directions
   ```

   Read the first number. If it is under 500 the sweep will not fly at all, and
   the message names every knob that would fix it. Note that the range is
   measured to the **aim plane**, which defaults to the work volume's floor: a
   part standing proud of the bench is that much closer, so pass the part's top
   as `surface_z=` (`yolo_region()` returns it as `surface_z`) when it is more
   than a few centimetres thick.

   Do NOT switch to `WFOV_2X2BINNED` to fly lower. Computed from the rendered
   URDF: the eddy-current probe and its holder sit 65-110 mm from the camera,
   45.5 degrees off the optical axis. In NFOV's 75 x 65 degree frame they clear
   the bottom edge by **11.8 degrees**; in WFOV's 120 x 120 they are **15.7
   degrees inside it** and occupy 2.9% of every frame, at a distance no depth
   mode can measure. The tooling is rigid to the camera, so that does not change
   with the pose and no sweep can be planned around it.
6. **Put the calibration plate in the work volume**, in view from `over_table`.
7. **Velocity calibration — re-measure it over a line as short as a pass.**
   The curve the work-cell manifest carries (`robot_velocity: hardware:`) was measured
   over a **300 mm** line in free space. It records that span because the curve is
   path-length dependent: a move much shorter than the span it was measured over
   is mostly acceleration ramp and never reaches the speed the table promises. An
   eddy-current pass is nothing like 300 mm — the regions this cell scans are
   10–30 mm across (step 9), and a 30 × 30 mm region plans to passes about 25 mm
   long — so **the curve that governs a contact scan has not been measured yet**,
   and the speeds the plan quotes for one are an extrapolation off the end of the
   measurement.

   Measure it before the first scan, with the line set to the length of the passes
   about to be flown: `capture_robot_velocity(length_mm=<pass length in mm>)`
   (scenario 9 of the work cell capture tool, which lists its scenarios when it
   loads; the procedure is `docs/cell-data-capture.md`). It flies a
   ladder of speeds in free space, nothing near the part, and prints a
   `robot_velocity:` block to paste into
   `tari_cell_configs/config/cells/ex-ur10e-1.yaml` in the site package. Then
   `robot_velocity_reload()` and `adopt_measured_velocity_curve()` put it in force
   without restarting the config. The manifest holds one curve per mode, so a
   short-line curve replaces the long-line one for every move, transits between
   passes included; `span_m` in the block records which span it came from, and the
   observed probe speed each scan prints is the check on it.

   It **refuses** a line under 50 mm, on the grounds that such a line is all ramp.
   For passes that short that refusal is the answer rather than an obstacle: no
   table can promise a speed over that distance, and the `observed probe speed`
   the dry run and every scan print is the only number that says what the arm did.

   Either way, compare what was seen against what was commanded. If the observed
   probe speed disagrees with the commanded 10 mm/s by more than 15%, re-measure
   before scanning.
8. **Detect, measure, plan -- no scan.** `run_automated_scan(scan=False)`.
   Section 2 describes each stage's output. It holds the warning light for the
   whole run and parks at `over_table` at the end. If YOLO does not find the
   part, pass the region by hand:
   `run_automated_scan(scan=False, region=(xmin, ymin, xmax, ymax))`.
9. **Choose what to scan.** At 0.3 mm between passes, the whole plate is hours
   of contact: 7.8 h in simulation, over the plan's 1 h limit, so the scan
   refuses it. ClientScannerTest scanned regions 10-30 mm across; a 30 x 30 mm
   region on the plate plans to about 20 minutes. On the fiducial cylinder
   the tilt band sets the length: +-45 degrees (`EC_MAX_TILT_DEG`) plans to
   about 45-50 minutes, and +-60 would be 66 and refused. Set the region and replan
   without flying anything:
   `globals()['EC_SCAN_REGION'] = (x0, y0, x1, y1)`, then `plan_ec_path()`,
   then `preview_ec_scan()`.
10. **Wrist check, no motion.** `ec_joint_report()`. It solves IK along the
    passes and prints the wrist_3 range. Compare that range with the cable
    envelope before flying anything.
11. **Dry run hovering above the part first.**
    `plan_ec_path(preload=-0.005)`, then `dry_run_ec_scan(passes=[0])`. This
    flies the scan's own motion with the tip 5 mm above the measured surface:
    lift, transit, descend, sweep, lift. Watch it in RViz and on the arm. The
    report prints tracking, tilt, roll, speeds, clearance and wrist_3; read
    them against section 2.
12. **Dry run in contact.** `set_ec_preload(<metres>)`, `plan_ec_path()`, then
    `dry_run_ec_scan(passes=[0])`. This is the first time the probe touches the
    part. Watch the spring fixture: it should compress visibly and stay
    compressed along the whole pass. If it lifts off anywhere, the fine mesh is
    high there, or the preload is too small. Section 3 has the whole preload
    procedure and its limits.
13. **Eddy-current setup** -- section 3.
14. **Scan.** `run_ec_scan()`. A pass that cannot be entered is skipped and
    reported; the probe is lifted clear at the end even on error.
15. **Park.** `park_arm()` rises straight up out of the work volume, then goes
    to `over_table`.

If anything goes wrong mid-motion: E-stop first. Then, before commanding
anything, `ec_lift()` to take the probe off the part along its own axis. If the
dgpy process died with the light on, switch output 0 off at the teach
pendant's I/O tab.

---

## 2. Stage by stage

Every stage can be called on its own from the prompt, and every one leaves its
result in a global (`detected_region`, `rough_mesh`, `fine_mesh`,
`ec_scan_path`, `ec_scan_x3d`). A stage that fails can therefore be retried
without redoing the ones before it.

The "healthy" output below is copied from the simulation run on the calibration
plate. On hardware the topics, frames and rates differ where noted.

### detect_part()

Goes to `DETECT_POSE` (a joint move, with the warning light on), enables YOLO,
takes the best detection and deprojects its box at one depth: the median over
the middle of the box. It returns `(xmin, ymin, xmax, ymax)`.

```
  warning light ON (digital output 0) -- automated scan
1/5  detecting the specimen
  yolo enabled
  Specimen score 0.814, bbox 158x315 px at (365, 169)
  region x 0.731..1.035  y -0.284..-0.131 m  (304 x 153 mm)
    from Specimen at 0.535 m, surface z 0.949, frame azure_gazebo_depth
  yolo disabled
```

That is the cal plate to within 1 mm, and `surface z` is its top. On hardware
the frame is `rgb_camera_link` and the image is 2048x1536; check the region
against the part with a ruler once.

| message | meaning | do |
|---|---|---|
| `the arm would not move to 'over_table': MoveIt refused it` | invalid start state, or a planning failure | the `check_state_validity` call it prints; jog clear |
| `no detections. The model is trained on the cal plate's proportions...` | YOLO saw nothing it knows | `run_automated_scan(region=(...))` by hand |
| `every depth pixel inside the detection is empty or non-finite` | the colour camera sees the part, the depth camera does not | NFOV range (step 5): the camera must be at least 0.5 m from the part |
| `the depth image is WxH and camera_info says ...` | depth and intrinsics are different streams | the manifest's hardware topics: `depth_to_rgb` with `rgb/camera_info` |
| a region nowhere near the part | a wrong detection | `region=` by hand; do not fly on it |

### reconstruct_rough()

A **pan sweep** over the region, integrated into a TSDF inside the work volume.
The camera holds the depth mode's minimum range to the part by standing off to
the side and aiming back, and covers the region from several directions, which
is what closes a TSDF. The speed comes from the camera's measured frame rate. It
is a planning surface, not a measurement.

```
reconstructing the specimen with the azure (AzureReconstruction)
    region x 0.7550..0.9000 (145.0 mm)  y -0.2900..-0.2050 (85.0 mm)
  depth encoding 16UC1 on /azure/depth/image_raw_16uc1 -> depth_scale 1000 (2.78 Hz mean, 2.24 Hz worst gap), frame azure_gazebo_depth
    500 mm above the part reaches only 3 directions; trying lower
    450 mm above the part reaches only 2 directions; trying lower
    camera 500 mm above the part, 176 mm clear of the region
    pan directions kept: 135 deg az (27 deg from vertical, roll +180, tooling 467 mm above the part), 180 deg az (26 deg from vertical, ...), 225 deg az (27 deg from vertical, ...)
    directions with no reachable, admissible pose, dropped: 0 deg, 45 deg, 90 deg, 270 deg, 315 deg
  reconstruction sweep: ScanPath(pan sweep (3 directions x 4x2, 530 mm slant, 26-27 deg from vertical)): 24 waypoints, 0.939 m, 6 segments
    integrating inside x 0.600..1.100  y -0.450..0.050  z 0.900..1.250
    camera-to-nearest-surface range (support 0.9300): 530..547 mm; NFOV_UNBINNED works from 500 to 3860 mm
  reconstruction started (tracking azure_gazebo_depth relative to world)
    sweeping at 0.030 m/s, MoveIt scaling 0.0198 (0.015 m gate x 2.44 Hz)
    holding still 17 s while the accumulator warms up
    sweep row 1/6: flew 100% of the row          (and rows 2-6)
    holding still 17 s so the tail of the sweep is not discarded
  reconstruction stopped: /home/user/scandata/specimen_azure.ply
```

**Read the three lines that decide whether this worked.** The range (must start
at or above the mode's minimum, or it refuses); how many directions survived
(one is a height field, not a reconstruction); and how many rows flew 100%.

"reaches only N directions; trying lower" is the height search doing its job,
not a fault: it walks `AZURE_PAN_HEIGHTS` from 0.50 m down, asking `/compute_ik`
with collision checking at every pose, and takes the **highest** rung that gives
enough directions -- highest because obliquity costs depth quality and brings the
tooling down towards the part. Dropped directions are dropped because the arm
cannot get there, and the plan says which.

The roll about the optical axis is chosen, not defaulted: it decides which way
the tooling hangs, and at 30-60 degrees from vertical the difference at the
eddy-current probe's tip is over 100 mm. The plan prints how far the lowest
tooling point ends up above the part.

Planning costs 2-5 minutes in simulation because every candidate pose is a
round trip to `/compute_ik`; on hardware the same search runs against the same
service and costs the same. Flying is 6 rows instead of 5 and about 1.7 m of
sweep instead of 1.3, so roughly a third longer -- except when a row cannot be
entered from where the arm is, when the sweep goes home first and that traverse,
at `JOINT_MOVE_SCALING` 0.015, dominates everything else. End to end in
simulation: 220 s for the cylinder, 401 s for the plate (which needed the home
traverse), against about 100 s for the old dome. On hardware the depth is 16UC1 millimetres on
`/azure/depth_to_rgb/image_raw` stamped `rgb_camera_link` at about 30 Hz. The
sweep then runs at its 50 mm/s ceiling and the two holds shrink to about a
second each.

| message | meaning | do |
|---|---|---|
| `... needs this cell's WORK VOLUME, and ex-ur10e-1 has none for mode 'hardware'` | expected until step 4 | `capture_touch_off()`, or measure by hand, write it and `work_volume_reload()` |
| `no tracking frame: ... published nothing to read a frame_id from` | the depth topic is silent | `ros2 topic hz /azure/depth_to_rgb/image_raw` |
| `the arm would not go to the first pose of the sweep` | the approach was refused | `mesh_path_feasibility()`-style check; start from `over_table` |
| `stop_reconstruction failed: The TSDF is empty` | nothing integrated inside the box | hardware first suspect: the camera closer than 0.5 m (step 5); then the tracking frame, the depth scale and the crop box |
| a later `only N% of ... is inside the work cell's work volume` | the mesh landed somewhere else | the tracking frame |

### reconstruct_fine()

A Noether raster over the rough mesh with the Keyence pointed straight down and
its laser line across the direction of travel. It is flown at 10 mm/s, one
approach per row, and the profiles are cropped to the work volume and meshed by
`tari_pc_reconstruction`.

```
reconstructing the specimen with the keyence (KeyenceReconstructionPC)
  keyence sweep: ScanPath(noether (772 poses) +tool offset): 772 waypoints, 3.745 m, 19 segments (from planner), 19 look like scanning passes (orient fixed)
    laser line against travel: 90.0..90.0 deg over 19 rows (90 = across the pass)
    sensor offset applied in the tool frame: (+0.0752, +0.0022, +0.0393) m
  laser on (pin 1)            <- in simulation: "could not switch the laser ... continuing"
    /profiles: 12 Hz, 32 points per profile (0 of 36 empty), frame laser_link
  keyence capture started (tari_pc_reconstruction)
    sweep row 1/19: flew 100% of the row
    ...
    sweep row 7/19: planner reached 57%, below min_fraction 60%
    ...
    sweep row 13/19: flew 88% of the row
    sweep row 15/19: planner reached 6%, below min_fraction 60%
    ...
    sweep row 19/19: flew 100% of the row
    2 of 19 rows did not fly; the reconstruction will have gaps there
  laser off (pin 1)
  processing the profiles into a mesh; this takes minutes
```

The sweep took about 4 minutes on the plate (19 rows at 20 mm spacing, at
10 mm/s). Meshing takes minutes more; `STOP_TIMEOUT` allows 30. Its progress
is in the `pc_reconstruction_srv` log, not the dgpy console. On the plate:

```
Points after merging: 30211
Points after cropping: 23311
Skipping DBSCAN cluster selection (dbscan_eps_multiplier <= 0). Keeping all 23311 points inside the crop box.
Points after SOR filter: 22237
--- Attempting Ball Pivoting (radii: [0.002, 0.004, 0.008]) ... ---      (46 s)
Cleaning mesh by keeping the largest connected component...
Successfully saved Ball Pivot mesh to .../specimen_keyence_ball_pivot_smoothed.ply
--- Attempting Poisson Mesh Reconstruction ---
```

"Skipping DBSCAN" is the local patch at work. Without it, this line is a
DBSCAN run, and on a plate with holes it keeps a third of the part. Note that
the server already keeps only the largest connected piece of the ball-pivot
mesh.

**The laser line must read 90 degrees against travel.** A line along the
travel sweeps a strip of zero width, and every other line of the output still
looks perfect. On hardware the laser really is switched: `laser on (pin 1)`
must appear, and a `could not switch the laser` there means the profiles will
be empty.

| message | meaning | do |
|---|---|---|
| `nothing publishing on /profiles` | no Keyence driver | on hardware it is started by `ur_moveit.launch.py` (`keyence_driver_node`); check the bring-up |
| `... publishing N Hz of EMPTY profiles from the first pose` | the sensor sees nothing within its +-23 mm window | laser off (DO 1), or the standoff wrong: `ros.keyence_t` |
| `WARNING: on N row(s) the laser line lies within 30 deg of the direction of travel` | the orientation is wrong for those rows | do not trust the coverage; report it |
| `sweep row k: planner reached N%, below min_fraction 60%` | that row could not be followed | a gap in the fine mesh there; acceptable if rare |
| `/stop_and_process failed: DBSCAN did not find any clusters` | the server lacks the local patch | redeploy `tari_pc_reconstruction` from `ros_ws/src` |
| `reported success but wrote no mesh, only point clouds` | every meshing method failed | the node's log says which and why |

### filter_mesh()

`tari_mesh_filter` loose-part removal on the fine mesh: it keeps the largest
connected piece. On the simulated plate it removed nothing (57,215 vertices in,
57,215 out), and the reconstruction server had already done the same to its
ball-pivot mesh.

```
4/5  filtering the mesh
  note: loose-part removal keeps only the BIGGEST connected component. A holey part is several components, and a table is bigger than a part -- check what comes back.
  filtered mesh: .../specimen_keyence_poisson_filtered.ply
```

| message | meaning | do |
|---|---|---|
| `tari_mesh_filter failed on ...` | the service refused | its log |
| a filtered mesh much smaller than its input | the part came back in pieces and the biggest piece won | plan over the unfiltered mesh: `plan_ec_path(mesh=fine_mesh)` |

### plan_ec_path()

Noether over the fine mesh, at 0.3 mm between passes. Then:
- the roll is locked;
- poses tilted more than `EC_MAX_TILT_DEG` are dropped, on the planned normals,
  so a wall's normals are never averaged into the passes beside it;
- the tool axis is averaged along each pass over `EC_AXIS_SMOOTHING_M` (6 mm,
  the probe face) and the roll re-locked. Positions are untouched;
- the tip is put on the measured surface, plus the contact preload along its own
  axis -- into the material, following each pose's normal (section 3);
- the path is cropped to the work volume and to `EC_SCAN_REGION`;
- every pass is resampled to `EC_WAYPOINT_SPACING_M` (1 mm), ends kept.

Before Noether sees the mesh, `clean_mesh_for_planning()` writes a
`*_ecplan.ply` copy without faces on non-manifold edges and without components
under 1% of the faces, and prints what it dropped.
- every pass is turned to travel along `EC_TRAVEL_DIRECTION`;
- the mesh is UV-parameterized for the projection;
- **then** the tooling-clearance gate runs, on the poses that will actually fly.

It moves nothing.

**The order of those last two matters.** The gate must not run inside `plan_mesh_path()`, on
the RAW Noether path — that is before the tilt cap, the work-volume crop, `EC_SCAN_REGION` and
the resample. On a curved specimen that is 911 511 poses in and 180 539 out: four fifths of
the work goes on poses that are about to be thrown away, and the survivors are never
re-checked at the 1 mm spacing they fly at.

**It also samples rather than measuring every pose, and the sampling is certified.** Clearance
along a path is 1-Lipschitz in tooling motion — a pose that is X clear cannot be less than
X − d clear after the tooling has moved d — and `path_step_travel()` computes an upper bound
on d for every step from the path itself (translation plus the turn angle times the tooling's
266 mm reach). Two measured poses therefore bound every pose between them, and a window is
left unmeasured only when that bound is at or above the limit for **every** pose in it;
anything else is measured, and the window is halved and reconsidered. So there is no pose
about which nothing is known, and a violation cannot hide between samples — which a plain
stride cannot promise: the fault this gate exists to catch is a 7.1 mm probe barrel 5 mm
inside a part, and a stride of ten over a 1 mm pitch steps straight across it. Read the line
it prints:

```
    sampled: 99 of 933 poses measured (10.61%) in 2 round(s); every other pose is
    bounded below by its neighbours and clears 5.0 mm. Widest unmeasured gap 12.1 mm of
    tooling travel.
```

`TOOLING_GATE_RESOLUTION_M` (10 mm of tooling travel) only sets how dense the first pass is;
the certificate, not that number, decides what is accepted. A `WARNING: ... NOT certified`
line means the refinement hit `TOOLING_GATE_MAX_ROUNDS` and some window is genuinely unknown —
that is the one case where the gate's answer is incomplete, and it says so.

The contact links (`ec_probe`, `probe_holder`) cannot fail the gate — contact is what this
scan is for — so they are bounded along the path and measured at a sample of poses for the
report, which the same block says.

**Measured** on a calibration plate's own Keyence reconstruction (211 090 triangles)
with a path on the production system's real spacings — 0.30 mm between passes, 1.00 mm along one, the tip
on the surface, the tool upright, contact allowed for the probe and its holder:

| the same 933-pose path | seconds | per pose | peak RSS |
|---|---|---|---|
| before: every pose, 12 candidate triangles, unblocked | 534.2 | 0.573 s | 4 625 MB |
| now, every pose (`sample=False`) | 163.9 | 0.176 s | 415 MB |
| now, sampled (the default above 2 000 poses) | 17.9 | 0.019 s | 402 MB |

30x the speed and a tenth of the memory, with the worst clearance agreeing to 0.06 mm
(−53.148 mm sampled against −53.203 mm measured at every pose; the sampler's number is a
lower bound on the truth by construction, never an optimistic one).

The whole plate at those spacings — **165 141 poses in 531 passes**, which is the shape of a
real path — takes **1 814 s and 477 MB**: 18 385 poses
measured (11.1%) in 7 rounds, widest unmeasured gap 19.6 mm of tooling travel. At the old
0.573 s per pose that same path is **26 hours**, and it would not have fitted in memory to
begin with. Half an hour is a plan you wait for; a day is not.

TBD-SIM-2

**Read the roll and tilt line.** Roll must be 0.00; that is the lock. Tilt is
what the surface asks of the fixture. On the simulated plate's reconstruction
the tilt averaged 33 degrees on a part that is flat, reaching 90 at the 9.6 mm
step around its raised pad. Planned over that without a region or a tilt
limit, IK could solve only 4 of 20 sampled passes, and the dry run could
enter only 1 of 5.

| message | meaning | do |
|---|---|---|
| work-volume refusal | as in reconstruct_rough | step 4 |
| `noether planning failed for ...` | the service refused the mesh | `principal_axis_noether_srv`'s log |
| `WARNING: only N% of ... is inside the work cell's work volume` | the mesh is somewhere else | the reconstruction's tracking frame |
| `N of M waypoints ... are tilted more than EC_MAX_TILT_DEG` | steep poses dropped, passes split | expected at edges and steps; many means a bad mesh |
| `every waypoint of the eddy-current toolpath is outside EC_SCAN_REGION` | the region misses the part | the region's numbers, against `fine_mesh` |
| `no scanning pass is left ... after cropping` | everything was cropped away | region, tilt limit, `PATH_MIN_SEGMENT_M` |
| `WARNING: the projection surface ... would be REJECTED by the projector` | the UV layout will not load | carry on: the scan records every trace with its pose, just without the live image. Scanning with that file would abort the process at the first pass |
| `UV parameterization failed` | Blender could not unwrap the mesh | its output is in the message |

### ec_plan_cost() -- what it costs BEFORE it is built

`preview_ec_scan()` with nothing planned yet, and the first thing
`plan_ec_path()` does. It needs no toolpath, no planner and no robot: the three
technician controls (`SCAN_LINE_SPACING` between passes, `SCAN_STEP_SIZE` along
one, `SCAN_VELOCITY`) plus the extent of the area are enough to say what the
plan will cost.

```
  what this plan costs, before it is built
    area               205 x 149 mm  (the region given)
    line spacing       0.30 mm  ->  497 passes, 101.9 m of travel
    toolpath           102 191 waypoints at 1.00 mm, from about 242 559 the planner returns
                       at the mesh's own 0.42 mm
    planning memory    413.2 MB  (73.1 MB the planner's reply, 84.1 MB path arrays,
                       256.0 MB the clearance gate)
    planning time      0:14  (the clearance gate, sampled)
    then the scan      2:49:41, 102 191 traces, 1.6 MB of trace data
```

Three of those numbers are measured rather than assumed and are written down in
the config next to the constants: Noether returns a waypoint every **0.42 mm**
(911 511 poses over 383.648 m) whatever the scan's own spacing is,
a `geometry_msgs/Pose` costs **316 bytes** of Python objects on this Iron
install (measured over 200 000 of them), and the reply is one per waypoint. At
a 0.15 mm line spacing on a whole plate that reply alone is over a gigabyte,
and the failure it produces is not a slow plan — it is the dgpy process being
OOM-killed with every stage before it thrown away (8.4 GB resident at the kill).

`EC_PLAN_LIMITS` bounds it: 1.5 M planner waypoints, 2 GB of planning memory,
half an hour of planning. `plan_ec_path()` **refuses** a plan over any of them
before calling Noether and says which; `plan_ec_path(force=True)` is the
deliberate override and still prints what is wrong. This is separate from
`SCAN_LIMITS`, which bounds the scan that follows, because they fail at
different times and in different ways.

### preview_ec_scan()

The plan the scan will be held to, in time and memory. It is printed; it
refuses nothing (`run_ec_scan()` does). With no toolpath planned it falls back
to `ec_plan_cost()` above rather than raising.

TBD-SIM-2

The time is a floor. Its per-pass overhead (5 s) was measured on the UT scan;
an eddy-current pass also lifts, crosses and descends.

### ec_joint_report()

IK only, nothing moves. Twenty passes spread across the plan are solved the way
each is entered, far end first. It prints how many solve end to end and the
wrist_3 range. Passes that do not solve here will not be entered by the scan
either.

TBD-SIM-2

### dry_run_ec_scan()

The scan's own motion with no instrument: five passes spread across the plan
by default, `passes=[...]` for chosen ones. The warning light is on and the
probe is lifted clear at the end.

TBD-SIM-2

| line | healthy | if not |
|---|---|---|
| `N passes tried, N entered, N swept` | all | a pass not entered: `no configuration reaches ... above both ends` -- the pass is out of reach at that tilt; drop it with the region or the tilt limit |
| `tracking` | under a millimetre from the planned line | the arm is not following the path |
| `along the normal` | near 0 in simulation. On the production system, what the fixture absorbs | on the production system, negative means into the part |
| `tilt ... follows the planned normal to within` | a degree or less | |
| `roll` | 0.0x degrees | anything more means the lock is not what was flown |
| `speed` | sweep 10, lift and descent at most 10, transit at most 50 mm/s | lower `JOINT_MOVE_SCALING` or `EC_AXIS_SPEED` |
| `clearance ... in transit` | at least the lift height (10 or 50 mm) | a transit that swings the tip near the part; do not scan |
| `wrist_3` | within the cable envelope | set `NOETHER_ROLL_REFERENCE` |

### ec_touch_down(index=0) / ec_lift()

Puts the probe on the start of pass `index` the way every pass is entered:
lift, transit in the air, descend 50 mm along the tool axis at 6 mm/s
(`EC_AXIS_SPEED`). Then it
stops there, in contact. `ec_lift()` backs it off 50 mm along the axis. This
is the pause ClientScannerTest had at its first point ("Null Uniwest?") -- the
scan now issues that null itself at its own first touchdown, so this is where the
fixture and the preload get looked at, and `null_ec_instrument()` is the manual
null if it is wanted here.

### run_ec_scan()

Refuses without `setup_ec()`, and refuses a `SCAN_VELOCITY` above
`NEAR_PART_SPEED`. It enters every pass by lift / transit / descend, sweeps it
at 10 mm/s taking one complex reading per 1 mm of travel (by timestamp), and
lifts the probe clear at the end even on error. Per pass it prints
`pass k/n: N traces (expected M), commanded L m / observed L' m`, and at the end
the observed speed against the commanded one. A pass that cannot be entered is
`SKIPPED` with the coverage lost. **Never run in simulation** (there is no
instrument), so the first run of this is on the production system: start with
`max_passes=1` over good material.

Before it moves it states the preload the path was planned with and what it will
do about the instrument null, and it refuses there if a null is required and
there is no instrument. The instrument is nulled at the first touchdown; see
section 3 for both.

| message | meaning | do |
|---|---|---|
| `eddy current is not set up -- call setup_ec() first` | as it says | section 3 |
| `SCAN_VELOCITY is N mm/s, and the probe is in contact...` | above 10 mm/s | lower `SCAN_VELOCITY` |
| `this scan cannot be run as requested: estimated scan time ... exceeds the 1:00:00 limit` | the plan is too long | `EC_SCAN_REGION`, or `force=True` as a decision |
| `source fault after pass k` | the DAQ stopped producing new data | check the card; the traces up to there are good |
| `there is no eddy-current instrument to null on the part` | refused before moving | section 3, The instrument null |
| `the eddy-current instrument would not null with the probe on the part` | stopped at pass 1, nothing acquired | section 3, The instrument null |
| `preload: UNKNOWN -- this toolpath records none` | the path predates the preload | `plan_ec_path()` again |
| `WARNING: could not lift the probe off the part` | the retract was refused | E-stop, jog clear by hand |

### park_arm()

Rises straight up at the current x/y: the first 50 mm at 10 mm/s, the rest at
`TRANSIT_SPEED` (30 mm/s), up to the work volume's ceiling. Then it makes a joint move to
`PARK_POSE`. `run_automated_scan()` calls it at the end, error or not.

---

## 3. Eddy-current setup

The instrument's settings (frequency, gain, probe drive) are set on the
Uniwest itself; ClientScannerTest had them commented out and set by hand. Its
X and Y outputs go to `PXI1Slot2/ai0` and `ai1`, and its serial port is opened at
startup so the scan can null it — see **The contact preload** and **The
instrument null** below, which are the two settings that decide what the probe
does to the part.

1. **At startup** the banner should read
   `eddy-current DAQ on /PXI1Slot2/ai0 and /PXI1Slot2/ai1`, and a moment
   later `ec_ready()` returns `(True, 'ready')`. If it says
   `the DAQ is open but /PXI1Slot2/ai0 has no data yet`, the card is not
   acquiring. A second line reads
   `eddy-current instrument: Uniwest on port 33 at 9600 baud`; if it says
   `NO eddy-current instrument on port 33`, read **The instrument null**.
   Opening the Uniwest driver changes nothing on the instrument: it reads the
   identity and the current settings and stops. Set the instrument up on its
   front panel in whichever order suits you. The null is the only command this
   config ever sends to it.
2. **Set the preload**, in metres: `set_ec_preload(0.0003)`. Do it before
   `plan_ec_path()` — the toolpath carries the preload it was planned with. See
   **The contact preload**.
3. **Touch down on good material.** `ec_touch_down(0)` puts the probe on the
   first pass's start. Check the spring fixture is compressed. `ec_lift()` backs
   it off again. The scan nulls the instrument at this point by itself, so this
   step is a check that the fixture and the preload are right, not the null —
   `null_ec_instrument()` does the null by hand if you want it here.
4. **Build the source.** `setup_ec()`. It prints the sample pitch, the
   conditioning parameters it is HOLDING, and runs a self-test of the compiled
   conditioning. It is idempotent; `teardown_ec()` releases everything before a
   second setup.

   Two properties of this step are not obvious, and both kill the process when
   they are violated:

   **The conditioning is wired by the SCAN, not by `setup_ec()`.** Its input is
   `/ec/traces`, which is an *output* of `nd_accumulate_once` and does not exist
   until the scan has staged its first reading. Wiring it earlier is fatal:
   under this config's `timed_transaction_manager` a math function naming a
   channel that does not exist throws on the manager's own thread —
   `terminate()`, SIGABRT, the whole dgpy process, after however many clean
   stages preceded it. So `setup_ec()` *holds* the null, rotation and gain, and
   the scan wires them the moment there is a trace list. `set_rotation()` and
   `set_gain()` work either side of that and say which they did.

   **`setup_ec()` self-tests the compiled conditioning, in a child process.**
   `spatialnde2_ec_functions` is a Cython extension linked against spatialnde2's
   C++ **ABI**. Rebuild spatialnde2 and leave the extension alone and the two
   disagree about object layout — and the failure is a *segfault inside the
   extension*, with no traceback and no message. **Importing it does not catch
   this**; only executing the math does. An extension older than the library it
   is linked against kills the scan silently, mid-pass, with the probe on the
   part. If the self-test fails, `setup_ec()` refuses and prints the rebuild:

   ```
   cd spatialnde2-ec-functions
   SETUPTOOLS_USE_DISTUTILS=stdlib python3 setup.py build_ext --inplace
   cp spatialnde2_ec_functions/ec_condition*.so \
      "$(python3 -c 'import spatialnde2_ec_functions as m, os; print(os.path.dirname(m.__file__))')"
   ```

   **Check this on the controlling computer before the visit.** Compare the build dates of
   `ec_condition*.so` and `libspatialnde2.so` in the installed
   `site-packages/`; if the library is newer, the extension is stale and the
   scan will die. `ls -la` on both is the whole check, and `setup_ec()` does the
   definitive version of it in two seconds.
5. **One pass over good material**, painting the rotation-independent value:
   `run_ec_scan(max_passes=1, component="magnitude")`. It prints the preload the
   path carries and what it will do about the instrument null before it moves,
   then nulls the instrument the moment the probe touches down. Read the pass
   line: the traces should match the expected count and the observed length
   should match the commanded one.
6. **Conditioning null** from those traces: `null_here()`. A software offset on
   top of the instrument's own balance — see **The instrument null** for why both
   exist. It prints the `(x, y)` it used.
7. **Rotation.** Lift-off should lie along the horizontal axis. The live place
   to see lift-off is the instrument's own impedance display while touching
   down and lifting (`ec_touch_down()` / `ec_lift()`). `set_rotation(deg)`
   re-renders every trace already acquired, so it can be trimmed against the
   recorded pass: the notch signal should end up vertical. `set_gain(h, v)` if
   the axes need different gains.
8. **Scan**, painting the defect axis: `run_ec_scan(component="vertical")`.

### The contact preload

Eddy-current inspection needs the probe pressed into the sample. The **preload**
is how far past the measured surface the tip is commanded, along the probe's own
axis; the spring fixture takes up the difference. It is the one number in this
config that decides how hard the arm pushes on the part, so it is set in one
place, by one function, in metres.

```
set_ec_preload(0.0003)     # 0.30 mm into the part; prints it back in mm
ec_preload()               # what is in force now
set_ec_preload(0)          # tip exactly on the measured surface
```

**Metres, always.** There is no millimetre form: one function that took either
could be handed `0.3` meaning 0.3 mm and command 300. It prints the millimetre
equivalent so the number can still be read the way the fixture is measured.

**Do not type `EC_PRELOAD_M = ...` at the prompt.** A bare assignment at the dgpy
prompt does not reach the shared globals: it reads back unchanged and the next
plan uses the old value. `set_ec_preload()` is the only way in, and
`plan_ec_path(preload=...)` goes through the same check.

**Set it before `plan_ec_path()`.** The preload is baked into the toolpath's
positions, so a path already planned carries the value it was planned with.
`plan_ec_path()`, `preview_ec_scan()` and `run_ec_scan()` all state it in
millimetres every run, zero included; `run_ec_scan()` reports the *path's* value,
not the current setting, and says `preload: UNKNOWN` for a path that predates it.

**What it does to the poses.** Each pose moves `preload` along **minus its own
tool z**, and on this tooling tool z is the outward surface normal — so the push
is into the material and it follows the surface on a curved part, rather than
going straight down in world z. It is applied before the work-volume and region
crops, so a pose the preload pushes outside the work volume is dropped like any
other. The scan area shrinks for margin; the motion never grows.

**Too small (or zero).** The tip is commanded onto the *measured* surface, so
every error in the reconstructed mesh becomes lift-off where the mesh reads high
and compression where it reads low, one for one. Lift-off is the largest signal
an eddy-current probe sees; a millimetre of mesh error is a millimetre of
lift-off, and it swamps the flaw. This is why the preload exists.

**Too large.** The arm stops pressing a spring and starts pushing the part: the
fixture bottoms out, the reading saturates, and the probe or the specimen is what
gives. There is also nothing left for the fixture to absorb where the mesh reads
low, so the push there is the preload *plus* the mesh error.

**The ceiling** is `EC_PRELOAD_CEILING_M`, 2 mm, and anything above it is refused
with the reason. It comes out of the tooling description
(`tari_ur_description urdf/ec/tooling_macro.xacro` and the meshes it names): the
compliance is `probe_holder`, a 12.5 mm radius plunger, sliding in
`bearing_tube`, a 16.0 mm bore, and both joints are **fixed** in the URDF, so the
spring travel is not modelled at all. The only length in the description that
bears on the slide is the 4.5 mm of plunger inside the tube at rest
(`probe_holder` is mounted 20.0 mm up a 24.5 mm tube); 2 mm is under half of it.
And `probe_end`, the commanded tip, is 68.07 mm along the probe while the probe
mesh ends at 68.32 mm — past 0.25 mm of preload the modelled probe body is
already inside the part, so every clearance number the planner reports is
optimistic by the preload.

**Measure the fixture** and put the probe near the middle of its travel. Until
that is done, a fraction of a millimetre is the defensible setting: enough to
cover the mesh error, far from the ceiling.

| message | meaning | do |
|---|---|---|
| `a preload of -0.300 mm is NEGATIVE, which is LIFT-OFF` | the sign is wrong | it is a distance into the part; pass a positive number |
| `a preload of 3.00 mm is over the 2.00 mm ceiling` | past what the fixture can absorb | measure the fixture's travel; do not raise the ceiling to get a scan out |
| `preload: UNKNOWN -- this toolpath records none` | the path was planned before this existed | `plan_ec_path()` again |

### The instrument null

An eddy-current bridge has to be **balanced with the probe on unflawed
material**. Balanced in air, the operating point is in the wrong place, the gain
is applied about that point, and every trace of the scan rides at an offset. This
is not a scan to repeat — it is data that cannot be recovered.

The scan does it by itself, **once**, at the instant the probe first touches
down, before any trace is read. It is `UniWest.null()`, which sends the
instrument's `NULL` command, and it replaces ClientScannerTest's
`Null Uniwest? (yes=1/n=0)` prompt. It does not repeat per pass: the probe
descends onto the part at the start of every pass, and re-balancing there would
step the baseline between passes and stripe the image.

**This is not `/ec/cond/null`.** Two different things are called a null:

| | what it is | when |
|---|---|---|
| instrument null | a hardware balance of the instrument's own bridge; it sets where the operating point sits | once, at the first touchdown |
| conditioning null | `/ec/cond/null`, a complex number subtracted from recorded traces by the conditioning math | any time, live, re-runnable over traces already acquired (`null_here()`) |

The software one sits **on top of** the hardware one and does not replace it: an
instrument balanced in air has its operating point wrong before any trace exists,
and subtracting a constant afterwards does not move it.

**Turning it off**, for bench work or a simulated source:

```
set_ec_nulling(enabled=False)     # do not null at all
set_ec_nulling(required=False)    # try, but scan anyway if it fails
set_ec_nulling()                  # just print the current state
```

As with the preload, a bare assignment to `EC_NULL_ON_CONTACT` or
`EC_NULL_REQUIRED` at the prompt does not reach the shared globals.
`EC_INSTRUMENT="none"` in the config means there is no instrument at all, which
is the default whenever `EC_SOURCE` is not `hardware`.

**It fails honestly.** With a null required — the default on hardware — an absent
instrument makes `run_ec_scan()` refuse **before the arm moves**, and a `NULL`
command that fails stops the scan at the first pass, before any trace, with the
probe lifted off the part by the scan's own teardown. With `required=False`
the same failure is printed as a warning saying the scan is going ahead
un-nulled, and the scan continues. Either way the outcome comes back in
`report["instrument_null"]`.

| message | meaning | do |
|---|---|---|
| `NO eddy-current instrument on port 33` (at startup) | the serial port did not open | check the cable and `EC_INSTRUMENT_PORT`; `connect_ec_instrument()` retries |
| `there is no eddy-current instrument to null on the part ... and a null is required` | refused before moving | fix the instrument, or `set_ec_nulling(required=False)` as a decision |
| `the eddy-current instrument would not null with the probe on the part` | the `NULL` command failed; stopped at pass 1 | as above |
| `WARNING: the instrument did NOT null` | `required=False`; the scan is running un-nulled | decide whether the data is worth keeping |

---

## 4. What the simulation showed

Gazebo, `ex-ur10e-1`, `CELL_MODE="sim"`, `EC_SOURCE="none"`.
Probe-tip speeds come from TF sampled at 50 Hz; wrist_3 from `/joint_states`.

### Calibration plate (settings before this page's last changes)

| stage | outcome |
|---|---|
| detect | YOLO found it (score 0.81): region x 0.731..1.035, y -0.284..-0.131, against the truth x 0.731..1.035, y -0.284..-0.132; surface z 0.949 against 0.948. 14 s |
| rough (Azure) | 5 rows at 50 mm/s (the ceiling; the 4.4 Hz frame rate implied 66), first row 83%, the rest 100%; 114 kB mesh; 59 s |
| fine (Keyence) | 19 rows at 20 mm spacing, laser line 90.0 degrees to travel on every row; 17 flew (one at 88%), 2 did not (57%, 6%); 30,211 points merged, 23,311 inside the work volume, 22,237 after the outlier filter. The smoothed ball-pivot mesh was 39% non-finite and was skipped for the Poisson mesh. 5 min 20 s |
| fine-mesh accuracy | against the plate's true geometry (a raised pad 252 x 91 mm at z 0.948, a border 9.6 mm lower): pad median **-5.0 mm** (mean -5.6, RMS 7.1, 5-95% -13.7..-0.2); border median -1.1 mm. Half the pad has a vertex within 2 mm of the truth. The simulated Keyence's long outer rays, as on the cylinder |
| filter | removed nothing (57,215 vertices in and out) |
| plan | 285,088 Noether poses, 3,480 segments; lines along **y**, the plate's short side. After cropping, 2,748 passes, 145 m in contact. Roll locked: **0.00 degrees**. Tilt up to 90 degrees, mean 33, on a flat part |
| roll, same normals, `"fixed"` | the same poses with the old mode: 360 degrees of roll across the plan, **997 degrees within a single pass** (unwrapped), 180-degree steps |
| preview | 145,644 traces, **7.8 h** (4.0 h travel + 3.8 h per-pass overhead): over the 1 h limit, so run_ec_scan() refuses it |
| UV surface | the parameterization flagged 33 over-shared UV edges: the projector would have refused it at the scan's first trace (now dropped with a warning) |
| IK report | 4 of 20 sampled passes solvable end to end; wrist_3 132..167 deg, up to 24 within a pass |
| dry run | 1 of 5 passes entered. The other 4: `no configuration reaches 50 mm above both ends` -- 50 mm along a normal tilted 60-90 degrees is off to the side of the part. The one flown: tracking 0.03 mm, along the normal -0.02..+0.03 mm, tilt 19 deg followed to 0.28 deg, roll 0.01 deg. It crawled at 0.3 mm/s: at a 0.0064 scaling every orientation change is slow |
| speeds at MoveIt's joint scaling 0.1 | peak tip speed: **132 mm/s** to `over_table`, **194** in the Azure stage, **245** in the Keyence stage (at z 0.972, 46 mm above the part), 80 in the dry run's transit, 86 parking. Straight-line lift and descent peaked at 12.5 and 11.4 mm/s for a commanded 10 |
| wrist_3 | Azure 142..159 deg; Keyence 159..248 (the line turned across rows that ran along y); eddy-current dry run 148..244, a 96-degree swing on the first approach, out of the Keyence stage's orientation |

The same plan put through the new crops with nothing launched (`_crop_path_by_mask`,
`_crop_path_to_box`, `orient_segments_along`, on the saved 284,514 poses):

| crop | passes | contact | estimate | tilt mean / max |
|---|---|---|---|---|
| whole plate | 2,748 | 145 m | 7.8 h | 33 / 90 deg |
| whole plate, tilt <= 45 deg | 3,319 | 105 m | 7.5 h | 25 / 45 |
| 30 x 30 mm at the pad centre | 158 | 4.0 m | 20 min | 33 / 90 |
| the same, tilt <= 45 deg | 140 | 3.1 m | 17 min | 27 / 45 |

Even in the middle of the flat pad the reconstruction asks for 27 degrees of
tilt on average, and puts the tip 3.7 mm below the pad (median). That is the
simulated Keyence's error, carried straight into the toolpath. The region is
what makes the scan fit the hour; the tilt limit does not.

**Is `"fixed"` locked enough?** On clean geometry that lies along the
reference, nearly. Rastering the true fiducial cylinder along its own axis,
which is 1.6 degrees off x, at 0.3 mm, with analytic normals: `"locked"`
turns the tool 0.000 degrees; `"fixed"` turned it 0.65 degrees over a +-30
degree band, 0.85 over +-45 and 1.04 over +-85, and 0 within any pass. Where
it fails is on reconstructed normals, which are noisy in every direction:
360 degrees across the plate's plan and 997 within one pass, above. It also
fails on anything tilted diagonally to the reference (4 degrees at 30 degrees
of tilt, 18 at 60). A reconstructed mesh is what the scan actually plans over,
so the lock is not optional.

What was changed because of it: every automated joint move now goes at
`JOINT_MOVE_SCALING` 0.02, and the lift and descent at 8 mm/s. The passes run
along the part's longest side, travelling -x. A tilt limit (45 degrees,
placeholder) keeps the probe off step walls. A UV surface the projector would
refuse is dropped. The transit clearance is a straight-line distance, because
along a noisy normal it read 0.8 mm for a probe that was 50 mm clear.

TBD-SIM-2

### The tilt band, measured on the fiducial cylinder

The 85-degree tilt in the first session's analysis was the fiducial cylinder:
a raster of the true cylinder along its own axis, planned over a band of
+-85 degrees, which is almost the whole upper half. To find what the arm can
actually do, the same cylinder (r 31.69 mm, lying along x at y -0.249, top
at z 0.994) was planned at 0.3 mm with analytic normals and every tilt up to
88 degrees: 211 passes, 1.4 to 84.7 degrees from vertical. IK was then solved
along 36 of them, nothing moving (`ec_joint_report`), for two locked roll
references:

| tilt from vertical | -y flank, reference +x | +y flank, reference +x | -y flank, reference -x | +y flank, reference -x |
|---|---|---|---|---|
| 2-25 deg | all solve | all solve | all solve | all solve |
| 29-45 deg | all solve | **none** | all solve | all solve |
| 50-70 deg | all solve | none | all solve | none |
| 83-85 deg | none | none | none | none |
| wrist_3 over the solved passes | 108-160 deg | | 288-343 deg | |

So the band the arm can reach depends on which way the locked probe points.
With the default reference (tool x along +x), the flank facing +y cannot be
entered beyond 25 degrees. The flank facing -y can be entered to 70.
Turned 180 degrees (tool x along -x), both flanks reach 45.

Two more facts bound the band. Noether spaces the passes 0.3 mm apart
horizontally, not along the surface, so on a flank they are further apart:
0.42 mm at 45 degrees, 0.60 at 60, 0.88 at 70. And the scan plan grows with
the band: on the cylinder +-30 degrees plans to 33 min, +-45 to 50 min, and
+-60 to 66 min, which the 1 h limit refuses.

**`EC_MAX_TILT_DEG` is therefore 45.** Inside it, every sampled pass on both
flanks solves (with reference -x), the surface spacing stays within 1.4x the
0.3 mm asked for, and the cylinder fits in the hour. Whether the spring
fixture holds a probe at 45 degrees from vertical is still not known. Lower
it if the fixture test says so.

**The coverage this gives up**, on the cylinder: the band keeps +-45 degrees
of arc, which is half of the upper surface (a quarter of the circumference).
In passes, 149 of the 211 planned over +-85. With the default reference +x,
the +y flank from 25 to 45 degrees cannot be entered either, and the scan
skips and reports those passes: 70 of the band's 90 degrees of arc, 78%.

**Choosing the roll reference is a cell decision, not a code one.** Reference -x
reaches both flanks. But it puts wrist_3 about 180 degrees from where the Azure
and Keyence stages leave it (142-248 degrees, measured), so the first
eddy-current approach turns the wrist half a turn. That is exactly the cable
wrap the rotation lock exists to prevent. Decide with the cable-envelope test
(`capture_cable_envelope()`): keep +x and lose the upper part of one flank, or
set `NOETHER_ROLL_REFERENCE = (-1, 0, 0)` if the cables allow the half turn.
References along y were not tested.

### Fiducial cylinder, second session

| stage | outcome |
|---|---|
| detect | YOLO found it: region x 0.761..0.898, y -0.276..-0.205, against the truth x 0.764..0.892, y -0.281..-0.217 (7 mm long in +y on one side). 7.5 s |
| rough (Azure) | 46 s |
| fine (Keyence) | 8 rows, `orient="fixed"`, laser line across travel, **every row 100%**. 144-179 s |
| filter | `tari_mesh_filter` printed `Parts from separation: 13` then `Error: list index out of range` and wrote the mesh **unchanged** (40,204 vertices in and out); `filter_mesh()` still reported success. The plate's "removed nothing" in the first session may be the same bug |
| plan | **Noether refused the mesh**: `Error during tool path planning: Connecting vertex does not exist at the start or end of a segment`, on the Poisson mesh and the "filtered" one, at both rotation offsets. The mesh was 11 components (the part and ten crumbs of 8-373 faces inside it) with 36 non-manifold edges. The ball-pivot mesh did worse: it **crashed the Noether service** (PCL `kdtree_flann` assertion, exit -6), which then had to be restarted by hand. Fix: `clean_mesh_for_planning()` (below) |

TBD-SIM-2-CYL

---

## 5. What the simulation cannot tell you

- **Contact.** Gazebo has no spring, no contact force and no friction, and the
  specimen is not in MoveIt's planning scene. The dry run's "along the normal"
  number is the arm's tracking error against the planned surface, nothing
  more. On the production system the fixture absorbs the mesh error, the tracking error and
  the preload together. **The preload is unknown** (`EC_PRELOAD_M = 0`): with
  no preload, a fine mesh 1 mm high is 1 mm of lift-off. Measure the fixture's
  travel, put the probe mid-travel, and `set_ec_preload()` it (section 3).
- **The fine mesh.** The simulated Keyence is a `gpu_ray` whose outer rays
  read long (on the cylinder, -2.7 mm mean radial error). The real sensor is
  cleaner but has edge error of its own. Either way, the error goes straight
  into where the probe is commanded.
- **Tilt under drag.** The locked roll and the tilt that follows the normal
  are exactly what the dry run measured. How far the spring fixture leans when
  dragged at 10 mm/s is not, and that is the known tilt problem.
- **The Azure's range and rate.** The simulated camera sees from 0.25 m at
  about 4 Hz; the hardware driver's NFOV_UNBINNED sees from 0.5 m at 30 Hz.
  Step 5.
- **The eddy-current instrument.** `setup_ec()`, the timestamp correlation
  with the real DAQ, `null_here()`, `set_rotation()` and `run_ec_scan()` have
  never run. Neither have the warning light and the laser output: simulation
  has no UR I/O controller. Their on/off logic is unit-tested
  (`warning_light_test.py`) but the pins have not been switched.
- **Speeds on hardware.** Every m/s goes through a calibration measured in
  simulation, and joint-move speeds are MoveIt scalings (section 4 has the
  measured peaks). The UR controller's own safety settings can enforce a TCP
  speed limit independently of any of this, and that is worth doing before
  the first contact.
- **Cables.** There are none in simulation. `ec_joint_report()` and the dry
  run give the wrist_3 range; the cable-envelope test says what is allowed.

---

## 6. What changed from ClientScannerTest, and why

ClientScannerTest's values are what ran on the real cell: its `routine()`,
`newKeyScan()` and `start()`, and the parameter files under
`dgpython_rosmotion/parameters/`. Each difference is either **changed** here to
match the work cell, or **kept**, with the evidence for keeping it.

**Whole run**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| start | `ros.Move("over_table")` | `detect_part()` goes to `DETECT_POSE = "over_table"` | changed to match |
| warning light (DO 0) | on at `start()`, off in its `finally`; also around each Keyence scan | `warning_light()` held by every function that moves the arm (each stage, `park_arm`, `move_to_named_pose`, the dry run, touch-down and lift) and by the whole run; off on error, including Ctrl-C; on hardware a light that will not switch on stops the stage | changed to match, stricter |
| laser (DO 1) | on at Keyence start, off at stop and in `finally` | on in `start()`, off in `stop()`, which runs in the sweep's `finally` | same |
| end | retract +0.05 then +0.1 m, then `over_table` | lift 50 mm along the tool axis, rise out of the work volume, `over_table` (`park_arm()`) | same idea |
| operator | ~25 `input()` prompts | none | by design |
| speeds | `Move(vel=0.1)` in transit; 0.05 for eddy-current points; 0.01 for the Keyence | 10 mm/s near the part (sweeps), 6 mm/s along the probe axis (lift and descent), 30 mm/s straight-line transit and Azure sweep, all converted through the velocity calibration. Every joint move at `JOINT_MOVE_SCALING` 0.015 | changed: at MoveIt's 0.1 the tip reached 132 mm/s going to `over_table`, 194 in the Azure stage and 245 in the Keyence stage 46 mm above the part. At 0.02 it still reached 55. A vertical rise commanded at 50 mm/s peaked at 87.6, and an axial descent commanded at 8 peaked at 11.9: the calibration is from horizontal sweeps and other directions run faster |

**1. detect**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| deprojection | each bbox corner at its own depth, hard-coded 2048x1536 intrinsics, frame `azure_rgb` | one median depth from the middle of the box, intrinsics from `camera_info`, the image's own frame | kept: a corner's depth mixes part and background. Measured within 6 mm on a 304 mm part earlier, and within 1 mm in simulation today |
| which detection | operator picks | `index=0`; `region=` by hand overrides | kept |

**2. rough mesh (Azure)**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| path | 9-pose cross around the detection, 0.25 m above it, point-to-point with 3 s pauses | 5x5 dome sweep, no tilt, continuous, at `SPECIMEN_SWEEP_HEIGHT` 1.35 m (absolute), camera offset corrected | kept (coverage measured 36.8% to 99.6% by the motion gate). **Check the height on hardware**: step 5 |
| sweep speed | `vel=0.1` | gate x measured frame rate, clamped to 50 mm/s | kept; it is the speed that decides coverage |
| voxel / sdf_trunc | 0.001 / 0.04 | 0.006 / 0.024 | kept: a planning surface must be coarse (Noether planned 8 m of lines at 0.002, 2.5 m at 0.006) |
| depth_trunc | 0.75 | 1.0 | kept: both exclude the room; 1.0 leaves room to raise the sweep for the NFOV minimum range |
| depth_scale | 1000 | read from the topic's encoding (1000 on hardware) | same on hardware |
| motion gate | 0 / 0 (every frame) | 0.015 m / 0.02 rad | kept: the sweep speed is derived from the gate |
| tracking frame | `azure_rgb` | the frame the depth images are stamped with (`rgb_camera_link` on hardware) | kept: `azure_rgb` is a hand-placed stand-in a few mm off |
| crop box | from the detection | the work cell's work volume | kept: the detection box both clipped the part and swallowed table |
| normal filter | 90 deg | none | kept: a filter that removes everything kills `industrial_reconstruction` |
| mesh filter after | vertex filter, world z 0.965 | none -- the work volume crops the TSDF | kept: 0.965 deletes the cal plate outright in simulation (its top is 0.948) |

**3. fine mesh (Keyence)**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| planner | fixed-axis Noether, direction +x | principal-axis Noether over the rough mesh | kept: the rows ran along y on the plate, and the laser line is turned across whatever the rows do, so coverage does not depend on it |
| line spacing | 0.032 m | 0.020 m | kept, with a question: 32 mm leaves no overlap if the real line is ~32 mm wide at its reference distance. The simulated line is 43 mm. 20 mm overlaps in both, at ~60% more passes. Check the real swath |
| standoff | noether `z_offset` = the optical frame's z; x/y offset in world | `z_offset` 0, the whole probe_end-to-optical-frame offset applied in the tool frame | same for a vertical tool; right for a tilted one |
| orientation | one fixed quaternion (identity: laser line along y, across x-lines) | `KEYENCE_ORIENTATION = "fixed"`: tool z vertical, tool x along the mean travel, so the line is across travel whatever the raster direction | changed to match: the default was `"surface"`, which could not reach the fiducial cylinder in simulation |
| snake | yes | no | kept: with snake, 9 of 16 rows could be entered; without, 16 of 16 |
| speed | `Move(vel=0.01)` | 10 mm/s through the calibration (MoveIt scaling 0.0064) | changed: 0.01 as a raw scaling is 15.6 mm/s on this cell |
| approach | 0.1 m above the first point, then down | `ApproachInConfiguration` for every row | kept: 6 of 14 rows failed without it |
| crop | a point clicked in RViz plus typed dimensions | the work volume | kept, so the Keyence now rasters the whole part (longer), not a clicked region |
| clustering | DBSCAN, eps x4 | off | kept: DBSCAN kept 35% of the plate's occupied cells, the work-volume crop 98%. Needs the local patch |
| merge / ball-pivot radii | 0.5 mm / 1-4 mm | 1 mm / 2, 4, 8 mm | kept: each pair brackets its own point spacing |
| SOR outlier filter | off | on (20 neighbours, 2 sigma) | kept, unverified: removes isolated points only |
| mesh passed on | `_ball_pivot` | `_ball_pivot_smoothed`, then `_poisson`, then `_ball_pivot`, skipping any with non-finite vertices | kept: smoothing produced NaN vertices on the plate once; smoothed normals steady the probe's tilt |

**4. filter**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| what is filtered | the Azure mesh (z plane 0.965); the Keyence mesh not at all | the Keyence mesh, loose-part removal (largest connected piece) | kept: it removed nothing on the plate (57,215 vertices in and out); the reconstruction server already keeps only the largest piece of its ball-pivot mesh |

**5. plan (eddy-current toolpath)**

| | ClientScannerTest | ECAutomatedScan before | now | verdict |
|---|---|---|---|---|
| line spacing | 0.0003 m (0.0001 "higher resolution") | 0.010 (UT's) | **0.0003** | changed |
| z offset | 0.0 | -0.05 (UT's 50 mm standoff) | **0.0**: probe_end is the tip on the ec tooling | changed |
| preload | none, set by hand at the pendant | none | 0 at startup, `set_ec_preload()` in metres, refused above 2 mm, stated in mm every run | changed: the value is still not known (section 5), but it is now one owned, bounded control |
| instrument null | operator prompt at the first point ("Null Uniwest?") | none at all | issued once by the scan at the first touchdown, before any trace | changed: an un-nulled scan is data that cannot be recovered |
| snake | no | yes | **no** | changed: a probe dragged in contact leans against its travel, so alternating passes stripe |
| roll | one commanded orientation | `"fixed"`: twists with diagonal tilt | **`"locked"`**: tilt follows the normal, never twists | changed |
| line direction | fixed axis: along x, travelling -x | principal axis: along the part's second-longest side (y on the plate, measured) | principal axis turned pi/2: along the longest side (x on both specimens), travelling -x | changed to match: along y, every pass on the cylinder would tilt the probe through +-60 degrees |
| tilt limit | none | none | `EC_MAX_TILT_DEG` 45 (placeholder) | new: the plate's reconstruction planned tilts to 90 degrees at its pad step |
| probe axis along a pass | every waypoint's own normal, stopped at each | every waypoint's own normal, moving | averaged over 6 mm of travel (`EC_AXIS_SMOOTHING_M`, the probe face is 5.8 mm), roll re-locked, positions untouched | new: on the plate's simulated reconstruction the planned axis turned 6.0 degrees per mm of pass, and the swept pass crawled at 0.3 mm/s against 10 because each orientation change is slow at the sweep's MoveIt scaling. Averaged: 2.7 degrees per mm (host, on the saved 284,514 poses). Stopping at every point, ClientScannerTest never paid this |
| projection surface | `x3d_converter` | whatever Blender wrote | dropped, with a warning, when its layout will not load | new: the plate's reconstruction produced one that would abort the process at the first pass |
| samples along a pass | at every Noether waypoint (~1 mm), stopped | 1 mm, position-triggered, moving | same | kept |
| region | the Keyence crop | whole part | `EC_SCAN_REGION` | new |
| work volume | not applied | not applied | cropped to it; refuses without one | new |
| short segments dropped | -- | under 20 mm | under 5 mm | changed: ClientScannerTest's regions were 10-30 mm |

**6. scan motion**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| along a pass | point-to-point, stopping at each point (`Move(0.05, 0.01)`) | continuous at 10 mm/s | by design |
| between passes | point-to-point in contact, back across the part | lift, transit in the air, descend along the tool axis at 10 mm/s | changed: the contact rules |
| first approach | 100 mm, then 50 mm above the first point, then down | the same lift / transit / descend from 50 mm | same heights |
| end | +50, +100 mm, `over_table` | lift 50 mm along the axis, then `park_arm()` | same idea |

**7. eddy-current data**

| | ClientScannerTest | ECAutomatedScan | verdict |
|---|---|---|---|
| null | operator nulls the Uniwest at the first point | `ec_touch_down(0)`, null on the instrument, then `null_here()` in software | same step, plus a software null |
| contact force | the UR force/torque reading stored with every point (`/msg_sync/force_msg`) | **not recorded** | gap: `msg_sync` is started by neither bring-up mode. Worth adding once the work cell's wrench topic is confirmed |
| output | a pickle per scan, with flaw metadata typed in | the recording database (`/ec/traces`, `/ec/trace_poses`), file-backed under `/home/user/scandata` | gap: no export file yet |
| projection | `x3d_converter` + `mapScan()` | Blender planar UV + `SurfaceProjector`, live | kept: the old exporter's 4-decimal UVs broke the loader |
