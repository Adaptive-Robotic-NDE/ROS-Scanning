# Cell steps

Cell ex-ur10e-1, workspace `~/Desktop/tyler_test9_ws`.

The parts accumulate, and the highest-numbered one is the most complete. Reading
backwards is the useful order:

| part | what it covers |
|---|---|
| **5** | the eddy-current scan; the Keyence display and RViz going unresponsive |
| **4** | two work volumes and how to check the new one, coverage is a warning, the Keyence planning surface |
| 3 | the ROI, the gauge, the toolpath normals, the cable guard |
| 0-2 | the Azure depth correction, and its verification |

If you are going to the cell once, **do Part 4.** It subsumes Part 3's run steps.

**The three things this procedure exists to hold in place.** The Azure reads every surface
12.45 mm too far (8 measurements over 4 days, sd 0.68 mm), and a bring-up adapter subtracts
it. The table model was 14 mm low and is raised. `run_automated_scan()` refuses if the mesh
disagrees with the Keyence by more than 2 mm.

---

# Part 0 — deploy (once, ~15 min)

### 0.1 Copy to the cell

| from | to |
|---|---|
| `ROS-Scanning/` | `~/Desktop/tyler_test9_ws/ROS-Scanning/` |
| `ros_ws/src/tari_site/` | `~/Desktop/tyler_test9_ws/src/tari_site/` |
| `ros_ws/src/tari_ros/tari_bringup/` | `~/Desktop/tyler_test9_ws/src/tari_ros/tari_bringup/` |
| `ros_ws/src/tari_ros/tari_ur_description/` | `~/Desktop/tyler_test9_ws/src/tari_ros/tari_ur_description/` |
| `ros_ws/src/tari_ros/tari_pc_reconstruction/` | `~/Desktop/tyler_test9_ws/src/tari_ros/tari_pc_reconstruction/` |
| `ros_ws/src/tari_ros/tari_key_reconstruction/` | `~/Desktop/tyler_test9_ws/src/tari_ros/tari_key_reconstruction/` |

**Copy all six or none.** A partial copy stops the pipeline: `ECAutomatedScan.dgp`
includes `azure_gauge.dpi`, and dgpy will not start without it. `tari_pc_reconstruction`
and `tari_key_reconstruction` must be at the same revision as each other: they once
published two different message types on one topic name (`/accumulated_cloud`), which
breaks `ros2 topic echo` and `hz` on it.

### 0.2 Rebuild — required, not optional
```bash
cd ~/Desktop/tyler_test9_ws && colcon build --symlink-install --packages-select tari_bringup tari_cell_configs tari_cell_ex_ur10e_1 tari_ur_description
```
Two new executables ship here and console scripts are *generated* — `--symlink-install` alone
will not expose them.

### 0.3 Re-source, in every shell you launch from
```bash
source ~/Desktop/tyler_test9_ws/install/setup.bash
```

### 0.4 Check the build before you touch the arm
```bash
ros2 pkg executables tari_bringup | grep -E "depth_bias_correct|depth_cloud"
```
**Both must be listed.** If not, 0.2 or 0.3 did not take. Everything below will read
−12 mm and you will have moved the arm for nothing.

### 0.5 Restart the bring-up, then confirm the adapter is live
```bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```
The topic audit should now name **6** topics, with depth as
`/azure/depth_to_rgb/image_raw_corrected`. Then:
```bash
ros2 node info /depth_bias_correct
```

### 0.6 Start dgpy
```bash
cd ~/Desktop/tyler_test9_ws/ROS-Scanning && dataguzzler-python --ipython dgpfiles/ECAutomatedScan.dgp
```

---

# Part 1 — did it work? (~20 min)

**Target: a sheet of plain matte white A4 or Letter paper, flat on the table.**
Must be **at least 150 mm across** — the tilt check now pans the Keyence to ±55 mm corners and
a smaller target is refused by name. Not the black breadboard (ToF cannot measure it), not
glossy or sprayed-white stock (it saturates and loses most of its pixels).

### 1.1 Teach the point — in the MIDDLE of the paper
Jog the EC probe tip down until it just touches, then:
```
teach_point("cal_flat")
```
**Expect z = 0.921 to 0.922.** Precision barely matters — verify compares the Keyence against
the Azure, and neither depends on your touch-off.

### 1.2 Look at RViz

Two things, both new:

- **The touching probe should sit ON the modelled table**, not ~14 mm above it.
- **There are now two Azure clouds.** `Azure cloud CORRECTED` (green) is what the pipeline
  uses. `Azure cloud RAW` (red) is the driver's own, still 12 mm long, kept for comparison.

| what you see | meaning |
|---|---|
| green on the table, red ~12 mm below it | **correct** |
| no green cloud at all | `depth_cloud` did not start — go back to 0.4 |
| green and red on top of each other | the correction is not being applied |
| green 14 mm *below* the table mesh | the table raise did not take — rebuild `tari_cell_ex_ur10e_1` |

### 1.3 The check that matters
```
verify_calibration()
```
Confirm each motion with Enter. **~3.5 min, 7 moves** (2 more than before — the tilt check
needs the wider footprint).

| line | before | **expect now** |
|---|---|---|
| `AZURE IS ... OFF THE KEYENCE` | −11.62 mm | **about −0.4 mm** |
| `PASS` / `FAIL` | FAIL | **PASS** |
| `TILT ... OFF THE KEYENCE (gate ...)` | 15.04 mrad, failed | **about 1.7 ± 2.0, gate ~10.2** |
| `patch_grey` in the returned dict | `None` | **a number, 0–255** |

**−0.4 mm is the right answer, not 0.0.** Depth is integer millimetres, so the adapter
subtracts 12, not 12.45. The 0.45 mm left over is smaller than the measurement's own sd.

If it still reads **−11.6 mm**, the adapter is not in the chain (0.4/0.5).
If it reads about **−24 mm**, the correction went on with the wrong sign.

### 1.4 Twice more, in different places
Move the paper, re-teach, re-run. **All three should PASS.** Spread beats repetition.

### 1.5 Gauge a real mesh
Needs a rough Azure mesh, so run it after `detect_part()` / `reconstruct_rough()`:
```
gauge_mesh_against_keyence()
```
Nothing touches the part — it is optical, from ~100 mm off. 3 poses, about a minute.
**Expect `THE MESH READS +x mm LOW` with x under 2.0**, and `ok: True`.
The number to act on is `mesh_correction_mm`.

### 1.6 A real scan
```
run_automated_scan(scan=False)
```
A new stage `2a/5` gauges the mesh between the Azure sweep and the ROI window. If it refuses,
the message names the measured dz and the override — **do not use `allow_azure_bias=True`
unless you have decided the number is wrong.**

### 1.7 Two minutes of measurement while you are there
What the always-on adapter costs the camera driver is not measured anywhere:
```bash
top -b -n 30 -d 1 -p $(pgrep -f k4a_ros_device) | grep k4a
```
Then again with `depth_bias_correct` stopped. Procedure is in `CELL_CHECKLIST.md` section 7.

---

# Part 2 — optional

### 2.1 Raw vs registered depth (worth doing)
```
verify_calibration(depth="both")
```
One extra grab at the same pose, no extra motion.
- the two disagree → the fault is `depth_to_rgb` resampling, fixable in configuration
- the two agree → it is the depth sensor itself, and the adapter is the right answer

### 2.2 Does backing off help? (low value)
```
verify_calibration(distance=0.55)
verify_calibration(distance=0.59)
verify_calibration(distance=0.62)
```
**0.651 m is the ceiling** — the camera sits 79.5 mm above the probe tip and the tip's motion
envelope tops out at 1.493 m. Do not ask for 0.75 or 0.95; it will refuse. A 70 mm lever
proves little either way, so skip unless 2.1 comes back clean and you still want an answer.

---

# What to send back

The whole take directory:
`/data/scandata/capture/<date>_<time>_ex-ur10e-1_hardware/`

It now carries `azure_grabs/*.npz` (the full depth image behind every number) **and**
`keyence_patches/verify.npz` (the Keyence points, grouped by grab), so every number above can
be re-derived offline. Plus `dataguzzler.log` if you ran under `script`.

---

# Part 3 — the ROI, the gauge, the toolpath normals and the cable guard

Everything above is the Azure depth correction and still stands. This part covers the ROI, the
gauge, the toolpath normals and the cable guard. **None of it has been exercised on hardware**
— it was built against saved reconstructions and a stand-in for the Noether service.

## 3.0 Deploy

Same copy list and rebuild as Part 0. ROS-Scanning only this time, unless you also want the
Part 0 items:
```bash
cd ~/Desktop/tyler_test10_ws && colcon build --symlink-install --packages-select tari_bringup tari_cell_configs tari_cell_ex_ur10e_1 tari_ur_description
```
Restart the bring-up and dgpy as in 0.5/0.6.

**Do not set `KEYENCE_ALLOW_BAD_STANDOFF`.** Leave it unset and let the gates speak.

## 3.1 Run it

```
run_automated_scan(scan=False)
```

**You should not be prompted before any move.** The gauge at stage 2a flies the pipeline's own
motion path. If it asks `[Enter=go, s=skip, q=abort]` before each of its four moves, that is
an older copy and the deploy did not take.

## 3.2 The four lines to read

### a) stage 2a, the gauge — does it sample the PART?
```
gauge 1/3 at (x, y), head at 100 mm
```
**Expect all three x,y inside the detected region.** A gauge pose outside it is aimed at the
bench, not the part, and means the region is not reaching the gauge — the symptom is gauge x,y
tens of millimetres beyond the region's own x bounds.

### b) the ROI crop — did z do anything?
After you draw the region, in the mesh-gate block:
```
the ROI's own z ... is provenance, not a crop; this mesh is at z ...
```
and a line reporting how many faces the z band excluded. **Expect a non-zero exclusion count if
your selection reaches a plate edge, and zero if it sits well inside the top.** A line of the
shape `left only 0 of N faces ... the restriction was NOT applied` means the crop is failing.

### c) the postcondition — is the probe normal to the surface?
```
POSTCONDITION HOLDS -- probe axis against the surface normal over N waypoint(s):
  worst 0.000 deg, p99.9 0.000, mean 0.0000 (tolerance 1.00); 0 over it, 0 facing into the part
```
**This is the most important line in the run.** It is the direct test of "probe tilt is zero
with respect to the surface normal". Worst should be ~0.000. Anything above a degree, or any
count of waypoints "facing into the part", means the toolpath is not doing what it claims.

### d) tilt from vertical — a reachability number, NOT an error
```
tilt from VERTICAL (a reachability question, NOT a normal error): up to X deg ...
```
On a flat plate expect a couple of degrees. On a curved part a large number here is **correct** —
it means the poses genuinely follow the surface. If waypoints are excluded you get an
`UNREACHABLE REGION` block with where and what it cost. Do not reach for `EC_MAX_TILT_DEG`;
the block says why.

## 3.3 If the cable guard refuses

You may see, on a park or a transit move:
```
CABLE GUARD REFUSED CartesianMove: ... wrist 3 to -189.9 deg ...
```
The message now ends with a line saying which of two things it is. **Read the last line, not
the first:**
- *"THIS IS THE BRANCH THE MOVE WOULD HAVE TAKEN"* — a real envelope problem. The toolpath wants
  a pose the cables cannot hold.
- *"BUT THE PREDICTION IS UNVERIFIED"* — the guard could not match the executing node's IK mode.
  Fix that before touching the toolpath or the band.

## 3.4 Then the real thing

```
run_automated_scan()
```
Stage 2a refuses if the mesh disagrees with the Keyence by more than 2 mm. If it refuses, the
message names the measured dz and the override — **do not use `allow_azure_bias=True` unless you
have decided the number is wrong.**

## What to send back

The whole take directory as before. It now also carries `keyence_patches/verify.npz` (the
Keyence points grouped by grab) alongside `azure_grabs/`, so the tilt and the scatter can both be
re-derived offline.


---

# Part 4 — two work volumes, and the pipeline start to finish

All five stages complete in simulation on three geometries: a 300 mm plate with a 10 mm bow,
the fiducial cylinder, and `cut_test.ply` (a 303 x 337 x 55 mm reconstruction of a real part).
Each ends at the 60-minute scan-time ceiling, which is arithmetic at a 0.30 mm line spacing,
not a fault.

**None of this has been exercised on hardware.**

## 4.0 Deploy

| from | to |
|---|---|
| `ROS-Scanning/` | `~/Desktop/tyler_test9_ws/ROS-Scanning/` |
| `ros_ws/src/tari_site/` | `~/Desktop/tyler_test9_ws/src/tari_site/` |

```bash
cd ~/Desktop/tyler_test9_ws && colcon build --symlink-install --packages-select tari_cell_configs
source ~/Desktop/tyler_test9_ws/install/setup.bash
```

The manifest gained a `motion_volume:` block — without the rebuild the arm is still held
to the old single box and the Keyence raster will fail to enter its first rows.

Restart the bring-up and dgpy as in 0.5/0.6.

## 4.1 Check the banner before you move anything

`ECAutomatedScan.dgp` should print **two** volumes now:

```
work volume (bounds the PROBE TIP): x ... y ... z ...
motion volume (bounds the TOOLING, not the part): x ... y ... z ...
```

If the second line is missing, 4.0 did not take: the arm is still held to the single box of
Part 3 and everything below will behave accordingly.

**Why there are two.** The work volume is where a sample may be placed and where the
eddy-current probe TIP must stay — that is still the touched-off box and nothing about it
has changed. The motion volume is where the arm and its tooling may navigate: the tooling
stands ~192 mm above the tip and reaches ~75 mm to the side, and holding all of it inside
the part box put probe_end on the wall while the laser was correctly over the part. It is
150 mm wider in x-y, reaches the old ceiling in z, and has the **same floor**.

It is not a licence to hit anything: collision is still the planner's to refuse, against
the environment model and the specimen in the planning scene.

## 4.2 Check the motion volume before you trust it — NO MOTION, ~1 min

**The hardware numbers in `motion_volume:` are the one thing in that manifest that was
never measured.** The work volume is touched off: an operator drives the tooling to each
corner and the number is where the arm actually was. The motion volume cannot be — its
corners are in free space, and the whole point of it is that the tooling goes where the tip
does not. So it is DERIVED (the work volume widened by 150 mm in x-y, the old ceiling in z)
and has to be checked instead.

```
check_motion_volume()
```

Nothing moves. It asks the same two services the pipeline asks — `/compute_ik` for reach and
`/check_state_validity` for collision — at a 3×3×3 grid over the box, corners included.

**The number that matters is `in collision`, and it must be 0.** This is what the simulated
cell prints:
```
motion volume x 0.450..1.250 y -0.600..0.200 z 0.930..1.500: 27 point(s) checked,
  19 unreachable, 0 in collision, 0 unknown
PASS: nothing in the box is in collision. NOTE what that does NOT cover: anything in the
cell that is not in the URDF.
(19 point(s) have no IK solution with the tool down. That is NOT a fault: ...)
```

**Unreachable corners are normal — ignore the count.** An arm's reach is roughly a sphere
and a box is not, so the box's corners fall outside first. A pose the arm cannot reach fails
at the planner whether or not this box allowed it, so a permissive corner costs nothing. The
count is not even stable: the same box gave 14 one run and 19 the next, because IK is
randomised. Do not chase it.

Two things that DO fail, and both send you to the manifest:

```
3 point(s) in collision, spanning x ... y ... z ...
    y: 3 on the -0.659 face, 0 on the 0.604 face
PULL THOSE FACES IN in the manifest's motion_volume block and run this again.
```

- **`in collision`** — the box reaches into something the model knows about. Pull that face in.
- **`unknown`** — the validity service did not answer. Not permission; find out why first.

Edit `ros_ws/src/tari_site/tari_cell_configs/config/cells/ex-ur10e-1.yaml`, rebuild
`tari_cell_configs`, restart dgpy, run it again. `check_motion_volume(samples=5)` gives a
finer grid once the coarse one is clean.

The box must still CONTAIN the work volume — a host test fails if it stops doing so. If a
face cannot be pulled in that far, the **work** volume is the thing that is too big for this
cell, and that is a different and more serious conversation.

One caveat on the reach number, since it is printed: `/compute_ik` may be collision-aware, so
a pose that is really blocked can come back as "no IK solution" rather than as a collision. A
**large** unreachable count next to zero collisions is worth one look for that reason, even
though it does not fail.

### Two things this check does NOT cover

1. **Anything in the cell that is not in the URDF.** This is the planner's collision model
   and nothing else. The standing assumption is that the environment is properly modelled —
   if there is a fixture, a cable tray or a bench stop that the model does not carry, this
   will call the space free and the arm will be allowed to try it. **Walk the cell and look
   along each widened face before the first real run.**
2. **Orientations other than tool-down.** It checks the orientation every sweep and raster
   uses. A box the arm can reach with the tool pointing sideways is not the box this is
   about.

## 4.3 Run it

```
run_automated_scan(scan=False)
```

## 4.4 The five lines to read, in order

### a) stage 3, the planning surface — is the bench gone?
```
ROI crop: specimen_azure.ply -> specimen_azure_roi.ply, N of M faces kept
planning-mesh depth trim: ... -> ..._top.ply, N of M faces kept
```
**Both lines should appear.** Without them the Keyence raster is planned over the whole
reconstruction — 500 x 500 mm of bench for a 300 mm part — so the region crop throws away
half the raster and the clearance gate shreds what is left. With both crops the clearance
loss is 2.3% on a flat plate and **0.4%** on a complex part, against 14.0% without them.

If `depth trim` says *"NOT trimming the planning mesh by depth"*, it found most of the
mesh deep and left it alone deliberately — read that line, it says why.

### b) the raster candidates — four, and which one was chosen
```
raster candidates (waypoints outside the work volume, cable-band violations sampled, -clearance m):
  0 deg rows, as planned   (0, 0, -0.049)  ...
  ...
  chose 90 deg rows, flipped
```
Two levers are searched now: which way the rows run, and which side of the laser line the
sensor body hangs on. They are not independent — turning the tool 180 degrees about its own
axis moves the body ~150 mm AND turns wrist 3 by 180 degrees. Nothing to do here; it is the
line to quote if a sweep later refuses.

### c) the standoff gate — curvature is not a mesh error
On a curved part you should see:
```
** the laser line spans NN mm of height here (p5..p95), which is a CURVED surface across
   the line rather than a mesh error: the commanded 80.0 mm lies inside the A..B mm the
   sensor actually returns.
```
The laser line is 31 mm wide, so on a curve it returns a range of heights and the commanded
point sits at one END of it. The gate therefore tests the commanded standoff against that
whole range, not against its median: comparing against the median reads the sagitta of the
curve as a mesh error — on a 67 mm-radius cylinder, 17 mm of "error" on a mesh accurate to
0.4 mm — and refuses a sound sweep.

**A genuinely wrong mesh still refuses.** If the commanded standoff is NOT inside what the
sensor returned, the old message stands and you should believe it.

### d) the postcondition — unchanged, and still the line that matters
```
POSTCONDITION HOLDS -- probe axis against the surface normal over N waypoint(s):
  worst 0.000 deg ... 0 facing into the part
```
Worst should be 0.000 on every part, flat or curved. This is the direct test of "probe tilt
is zero with respect to the surface normal".

### e) the gate's warnings — three of them are now warnings, not refusals
```
VERDICT: plan: nothing blocks it (3 warning(s): holes, region_coverage, cable_tilt)
```
- **`region_coverage`** does not block. Most of the shortfall is the Keyence's 73 mm lateral
  offset at a part's edge, which is fixed in millimetres, so the smaller the part the bigger
  the fraction it costs (~6% of a 300 mm part, ~14% of a 200 mm one) — a sound sweep of a
  200 mm plate covers about 93% of it. Read the number and decide whether to sweep again; it
  will not stop you.
- **`cable_tilt`** says what fraction of the surface needs a tool tilt outside the measured
  cable envelope. On the cylinder it was 48.5%. See 4.5.
- **`tilt_loss`** still BLOCKS above 40%, and on a steeply curved part that is the honest
  answer: 41.8% of the 67 mm fiducial cylinder leans past 45 degrees from vertical. Scan the
  crest instead — a narrower region planned cleanly.

## 4.5 If stage 3 will not enter its first rows

```
the arm would not go to the first pose of the sweep ([x y z])
```
Check 4.1 and 4.2 first — this is the symptom the motion volume fixes. If the banner shows both
volumes and it still happens, quote the 4.4(b) candidate table: the chooser picks by
(waypoints outside the motion volume, cable-band violations, clearance) and the table says
what it was choosing between.

## 4.6 The measurement that is still owed

The cable envelope on record was swept at five tilts — no tilt, x +/-40, y +/-20 — and only
over wrist_3 16..327 degrees. Two consequences, both of which show up as warnings rather than
mysteries:

- the arc of wrist_3 **through zero has never been probed**, and the arm naturally wants to
  sit there for some poses;
- any part with more than ~20 degrees of surface slope asks for tilts outside the hull, and
  the envelope refuses rather than extrapolating. The cylinder needs 48.5% of its surface
  there; a 30 mm bow over 300 mm reaches 21.8 degrees and is already past it.

```
capture_cable_envelope(full_circle=True)
```
at more tilts is what closes both. It is ~40 minutes and it is the single highest-value
thing left that only the cell can do.

## What to send back

The whole take directory, as before, plus the dgpy console log — the candidate table and the
standoff lines in 4.4 are printed there and nowhere else.


---

# Part 5 — RViz, and the eddy-current scan

## 5.1 RViz goes unresponsive after the Keyence measurement

**Fixed; this is the confirmation.** Full write-up, measurements and the
troubleshooting tree: **`docs/rviz-performance.md`**. The tool is
`dgpfiles/utils/rviz_probe.py`.

The cause is not a leak and not a rate. The fine-mesh pipeline publishes each
of its stage clouds **once**, at `/stop_and_process`, and an RViz PointCloud2
display with `Decay Time: 0` **keeps the last message forever** — "decay 0"
means *never expire*, not *clear*. Seven displays that cost nothing all session
start costing every frame at the Keyence measurement and never stop. The worst,
`/sanitized_pointcloud`, carries the **RAW** cloud despite its name: about
2.85 M points for a 250 mm plate at this profiler's pitch. Measured: **+864 MB
of RSS and +140 points of CPU from one message, never released.**

Six stage displays are now **off by default**. Every stage is still written to
disk as a `.ply`, so nothing is lost — tick one on when debugging that stage and
restart RViz afterwards.

**Three commands, in this order:**

```bash
# 1. before anything: are the six stage displays off?
python3 ROS-Scanning/dgpfiles/utils/rviz_probe.py displays \
    src/tari_ros/tari_ur_description/rviz/view_robot.rviz
```

```bash
# 2. in its own terminal, BEFORE run_automated_scan(): leave it running
python3 ROS-Scanning/dgpfiles/utils/rviz_probe.py watch --csv /tmp/rviz_run.csv
```

```bash
# 3. only if RSS still steps up -- run immediately after /stop_and_process
python3 ROS-Scanning/dgpfiles/utils/rviz_probe.py census --secs 30
```

**The number that matters is RSS across the Keyence measurement.** Flat means
fixed. A step that never comes down means something is still holding a cloud,
and `census` says which topic. A *gradual* climb is a real leak and a different
fault from the one above — note when it starts and send the CSV back.

Two things to know before you trust the result:

* **The fix only arrives with `tari_ur_description`**, and **RViz must be
  restarted** to read a changed config.
* **If RViz has ever been used to save the config** (File > Save Config), the
  cell's copy may have the displays back on. Step 1 catches that.

## 5.2 The eddy-current scan

The scan runs end to end in simulation (318 of 320 traces, every pass at 100 %
of commanded distance). Three faults that used to stop it at its first line are
fixed; each is worth knowing because each is silent or misattributed:

1. the conditioning was wired onto `/ec/traces` during `setup_ec()`, before the
   accumulator that creates that channel had run — SIGABRT under the timed
   transaction manager, not an exception;
2. `set_trigger_pitch()` returned `None` in this config, so the trace pitch was
   `None`;
3. the eddy-current scan never declared its contact to the planner, so **every
   descent onto the part was truncated and every pass skipped** — 0 traces of
   720, with no error.

**Before the scan, `setup_ec()` now self-tests the compiled conditioning in a
child process.** If it refuses, read the message: it means
`spatialnde2_ec_functions` was built against a different spatialnde2 than the
one installed, and the rebuild command is in the error. Check it before the
visit — compare the build dates of `ec_condition*.so` and `libspatialnde2.so` in
`site-packages/`. See `docs/ec-automated-scan-first-run.md` §3.

## 5.3 Robot speed

`ROBOT_VELOCITY_CALIBRATION` and `JOINT_MOVE_SCALING` were both measured **in
simulation** and never on the arm. `capture_robot_velocity()` measures them
properly — a speed ladder on a 300 mm line in free space, plus the joint-move
tip peak, printing a table to paste into `ECAutomatedScan.dgp`. `TRANSIT_SPEED`
is raised 30 → 40 mm/s meanwhile, which is the only change the existing
measurements support.
