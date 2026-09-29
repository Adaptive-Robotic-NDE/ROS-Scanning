# Sensor calibration at ex-ur10e-1

Where the Azure Kinect and the Keyence LJ-V7080 actually are on the end of the
arm, how to measure it, how to get the answer into the robot description as
**files you copy**, and — the part that matters most — how to check in three
minutes that nothing has drifted.

Run it at the cell. Everything below is typed at the `dgpy>` prompt of
`ECAutomatedScan.dgp` except where it says shell.

---

## Why

An uncalibrated Azure puts the rough mesh low, and the whole fine scan is
planned over that mesh. Three measurements of one surface-ground flat plate, at
a mount error this procedure exists to remove:

| | z | against the touch-off |
|---|---|---|
| contact probe touch-off | 0.9402 | — |
| Keyence LJ-V7080 | 0.9409 | +0.7 mm |
| Azure Kinect | 0.9273 | **−12.9 mm** (−22 mm at one end) |

The Keyence raster is planned by Noether **over the Azure's mesh**, so a 13.6 mm
error there flies the whole scan 13.6 mm closer to the plate than intended. The
head measures over 57–103 mm; where that error is worst it flies at ~58 mm
against the 57 mm hard limit, the probe comes within visible distance of the
part, and **26–39% of a 250 mm plate comes back as garbage**.

The Azure's fixture is known to be physically loose. So this is two problems:
a calibration, and a check that survives the fixture moving again.

---

## 1. The routine check — run this, and mostly only this

```python
verify_calibration()
```

Four moves, four grabs, about three minutes. **No artifact, no printed board, no
touch-off.** Teach roughly where a flat patch is, and this measures it with the
Azure and with the Keyence and subtracts.

> **Aim it at the specimen, not at the bench.** "Anything flat" is wrong here.
> The bench is a black anodised optical breadboard drilled on a 25 mm grid. The
> Keyence reads it to 0.015 mm rms, because a laser triangulator brings its own
> light. The Azure reads it **13 to 18 mm low**, because a time-of-flight camera
> gets almost nothing back from it. Aimed at the bench, this check reports the
> Azure about 13 mm out no matter how good the mount is — and it keeps
> reporting it after a clean re-solve, because the mount was never the problem.
>
> Measured against heights known from touch-off:
>
> | patch | grey level | Azure depth error |
> |---|---|---|
> | white sprayed specimen | 250 / 255 | −3.6 mm |
> | bare black breadboard | 103 / 255 | −18.1 mm |
>
> `verify_calibration()` prints the patch's grey level and says so when it is
> too dark to believe. Teach `cal_flat` **on the part**.

```
  VERIFICATION over a flat at (0.850, -0.200), +/-45 mm
    keyence  z = 0.9409 m   0.102 mm rms flat, 144000 points from 180 profiles
                            at 3 positions -20 +0 +20 mm across the line
    azure    z = 0.9273 m   1.480 mm rms flat, 3100 points
    AZURE IS -13.60 mm AND 2.10 mrad OFF THE KEYENCE
    CONSEQUENCE: a raster planned over an Azure mesh at the 80 mm reference
    distance would fly at 66.4 mm, which is 9.4 mm clear of the head's 57 mm
    hard limit.
    FAIL. Run calibrate_azure(), then this again.
```

Tolerance is 2 mm and 10 mrad. **The Keyence is the reference, not the average
of the two**: it agreed with a contact touch-off to 0.7 mm on the same plate,
and its 0.102 mm noise on a flat is an order below the Azure's.

**Why three Keyence positions and not one.** One static profiler pose measures
one *line* — 31 mm long, microns wide, every point at y = 0 in the sensor's own
frame. A plane fitted to a line still comes back, with a normal that is chosen
by nothing: roughly horizontal, at a residual of exactly zero. The tell is a
report reading `0.000 mm rms flat` with a tilt near 1571 mrad (90°) and a z
nowhere near the surface — numbers that say nothing at all about the Azure. The
head is therefore panned ±20 mm across its own line so the points are a patch,
and a point set that still cannot determine a plane is **refused** rather than
fitted.
Both sensors' grabs are also checked for being a measurement at all — scatter
above the head's noise floor, and an implied standoff inside 57–103 mm — before
anything is subtracted.

`verify_calibration()` and `calibrate_keyence()` record `/profiles` in their
bag, so a Keyence number that turns out to be nonsense can be looked at, and
re-solved from, afterwards. Without it, a take with a bad Keyence number is
unsalvageable. It costs **2.82 MB/s** on this cell — 217.6 Hz × 800 columns ×
16 bytes — so it goes on those two takes and nowhere else: about **0.7 GB** for
a verification (172 s of flying plus two extra moves) and **3.8 GB** for a full
`calibrate_keyence()` (about 1340 s), against 36 MB and 372 MB without it.
`calibrate_azure()` does not get it: the laser is off and there is nothing to
record.

The pre-take size estimate counts it too. The estimate comes from a three-second
probe that runs *before* the recorder and therefore before the laser, so
`/profiles` is silent while it listens; without a floor the number it prints —
and refuses the take on — would have been an order of magnitude low.

**Run it at the start of every scanning session and after anything is unbolted
from the arm.** Everything below exists for when it fails.

---

## 2. What you need at the cell

Two artifacts, both made here. Both live in `ROS-Scanning/docs/calibration/`.

### 2.1 The checkerboard — for the Azure

`docs/calibration/tari_calibration_board_20mm_letter.pdf`, generated by
`make_board.py` in the same directory.

| | |
|---|---|
| pattern | **9 × 6 inner corners** = 10 × 7 squares |
| square | **20 mm**, not 25 |
| printed board | **200.00 × 140.00 mm** |
| sheet | US Letter **landscape**, 39.7 mm side margins |

20 mm and not 25 because 9 × 6 inner corners is *ten* squares along the long
axis; at 25 mm that is 250 mm, which does not clear a Letter sheet's
unprintable margin. `CAL_BOARD_SQUARE` in `calibration.dpi` is already `0.020`
to match this sheet.

**Print at 100%.** Turn off "fit to page" / "shrink to fit" / "scale to fit".
Then bond it flat to something rigid — the calibration plate is ideal — and
leave it there. The caliper fixes scale; nothing fixes a bubble.

#### Then measure it, with calipers, before anything else

The sheet has four border bars whose **inner faces are exactly 200.00 × 140.00
mm apart**. That is the only crisp physical edge a checkerboard has — the outer
squares alternate, so half the nominal boundary is white paper with nothing
printed on it. Put the caliper's **inside jaws** against them.

```python
set_board_square(0.2013)       # the long axis, in METRES
  board square pitch measured 20.1300 mm over 10 squares (nominal 20.0000, +0.65%)
```

The default divisor is **ten squares**, which is the long axis of the whole
board. Pass `n=7` if you measured the short axis instead.

**Why this is not optional.** A hand-eye solve turns board scale error straight
into camera standoff error, in proportion. 1.5% — ordinary for a consumer
printer, and different along and across the paper — is **8 mm of camera standoff
at the 0.55 m working distance**. That is most of the 13.6 mm you are trying to
remove, arriving through the artifact instead of through the fixture, and
nothing in the residuals will show it.

A certified aluminium-composite target (Calib.io, Edmund Optics) is $200–300
and good to 0.01–0.1 mm. It is worth buying if this becomes routine. Printing
plus a caliper makes the printed route *survivable*, not *safe*.

### 2.2 The stepped artifact — for the Keyence

`docs/calibration/tari_calibration_artifact.stl`, generated by
`make_artifact.py` in the same directory, which carries the design reasoning in
its docstring. 4196 triangles, prints on any FDM machine.

| | |
|---|---|
| footprint | 160 × 160 × 24 mm |
| base | 4 mm thick, one **25 mm chamfered corner** so the outline cannot be flipped |
| pads | nine, Ø22 mm, 45 mm pitch |
| pad tops | **7, 14, 21 / 24, 10, 17 / 14, 7, 21 mm** above the table |
| cones | four, Ø26 mm at the base plane, **apex at 13 mm** |

No rotation of the pattern maps those nine heights onto itself, and the chamfer
breaks the outline's symmetry, so the artifact's *pose* is unambiguous from
shape alone. A flat plate gives a profiler nothing to locate laterally at all.

**It does not need to be accurate, it needs to be KNOWN.** A printer is good to
a few tenths of a millimetre and the answer being solved for is half of one. So
print it, then characterise the actual pad tops **once** — a height gauge, or a
Keyence pass with the current calibration — and record them:

```python
use_printed_artifact(pad_tops=[0.0071, 0.0139, 0.0208,
                               0.0242, 0.0101, 0.0171,
                               0.0141, 0.0069, 0.0212])
```

From that moment the measured geometry is the reference and the printer's
tolerance stops mattering. Without the argument it uses the design heights,
which is fine for a first pass and not for a result you rely on.

#### What the height spread buys, and what it does not

Nine surfaces span 0–24 mm. With the head at its 80 mm reference distance over
the artifact's mid-height, one pass sees them at

    68  71  75  78  79  82  85  88  92 mm

all inside the head's 57–103 mm window, with 11 mm of placement margin at each
end. `describe_calibration_artifact()` prints exactly this for whatever heights
are declared.

**It does not, on its own, break the single-standoff degeneracy** — §6 has the
measurement and the reason. It is worth knowing that up front because it is the
natural assumption and it is wrong.

#### The Azure targets are design intent, not code

`make_artifact.py`'s docstring describes stick-on Creaform targets so the Azure
can locate the same body in colour and in IR. **Nothing implements that.**
`calibrate_azure()` detects a checkerboard and only a checkerboard. Treat the
target story as the reason the artifact is shaped as it is, not as a procedure.

---

## 3. The procedure, start to finish

```python
calibration_status()                   # what is on record, and the artifact in use
```

### 3.1 Azure — hand-eye, no touch-off

```python
set_board_square(0.2013)               # §2.1. Once per printed board.
teach_point("cal_board")               # roughly the centre of the board
azure_calibration_poses()              # optional: look before you fly
check_calibration_poses(azure_calibration_poses(), "azure")
calibrate_azure()                      # ~24 poses, 20-25 min
```

Teaching `cal_board` is not a touch-off in the sense that is ruled out for the
routine pipeline — nothing downstream depends on it, it only says where to
point the camera, and being 10 mm out costs nothing.

What good looks like: every pose reports corners, the amplification is ≤ 2×,
the verdict is `GOOD`, residuals ≤ 1 mm.

### 3.2 Keyence — surface fit

```python
use_printed_artifact(pad_tops=[...])   # §2.2, or omit for the ground flat
teach_point("cal_flat")                # on the artifact, near its middle
calibrate_keyence()                    # 30 poses, 3 standoffs, 15-20 min
```

`calibrate_keyence()` **refuses a single-standoff pose set outright**, whatever
artifact is in use. §6 says why.

### 3.3 Deploy — §4. Then

```python
verify_calibration()                   # must pass before you believe any of it
```

### 3.4 The one touch-off, once

Hand-eye needs none and the routine pipeline must not have one. But hand-eye is
blind to any error that commutes with the arm's motion, and it believes the
robot's own kinematics without question because it is *told* the flange pose.

So **once, at calibration time**, touch the plate off with the contact probe and
compare against the Keyence at the same spot. Expect agreement within about
0.7 mm on a ground flat. If they do not agree, the problem is upstream of
everything in this document — the tool definition or the robot's kinematic
calibration — and no amount of sensor calibration will fix it.

---

## 4. Deploying it — the files, and what to do with them

Each `calibrate_*()` writes two files and prints both. **Nothing applies
itself**: a mount lives in the robot description, it is consumed by xacro at
bring-up, and it is what MoveIt plans collisions against, so changing it under a
running stack would leave TF and the planning scene disagreeing about where a
camera on the arm is.

| file | what it is |
|---|---|
| `<CAPTURE_ROOT>/cell_override/ex-ur10e-1.sensor_calibration.json` | the record — both sensors, full report, conditioning, pose set. The source of truth. |
| `<CAPTURE_ROOT>/cell_override/sensor_calibration.urdf.xacro` | **the drop-in.** Regenerated in full from the record every time, so it always carries both sensors. |

`CAPTURE_ROOT` is `/data/scandata/capture` on the cell PC. The printout gives
the real paths — copy what it prints.

### What the drop-in contains

```xml
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- azure -->
  <xacro:property name="azure_mount_cal_xyz" value="-0.00761360 0.00601012 0.01512352"/>
  <xacro:property name="azure_mount_cal_rpy" value="-0.01720665 0.05226814 0.03055803"/>
  <!-- keyence -->
  <xacro:property name="keyence_mount_cal_xyz" value="-0.00104854 -0.00065902 0.00299593"/>
  <xacro:property name="keyence_mount_cal_rpy" value="0.00399700 0.00200599 -0.00299600"/>
  <!-- Superseded by keyence_mount_cal above. Keep it zero. -->
  <xacro:property name="keyence_ref_dist_cal" value="0.00000000"/>
</robot>
```

Six numbers per sensor, identity when uncalibrated. They set a fixed joint that
`tari_ur_description` already carries, above the nominal bracket geometry:

* `mounting_plate → azure_mount_cal`, above all four Azure bracket variants;
* `keyence_metal → lj_v7080_mount_cal`, above the head's call-site origin.

The correction is **in the parent link's frame**: `T_actual = Cal · T_nominal`.
Written on the other side it is a different transform whenever the nominal
origin rotates — and the `redo` Azure bracket rotates by two right angles — so
the wrong side is out by about twice the offset and reads perfectly plausibly
in a diff. `tari_ur_description/test/test_sensor_mount_cal.py` renders the
description with a known correction and checks the composed transform, which is
the only thing that tells the two apart.

Unlike `keyence_ref_dist_cal`, a mount correction **moves the bracket link and
its collision geometry**. That is the point: a camera the planner cannot see the
real position of is a camera it will drive through something.

### The deploy step

`write_calibration()` prints exactly this, with the real paths filled in:

```
  DEPLOY IT -- copy one file, rebuild one package, restart ROS
    cp /data/scandata/capture/cell_override/sensor_calibration.urdf.xacro \
       ~/ros_ws/src/tari_site/tari_cell_ex_ur10e_1/urdf/
    colcon build --packages-select tari_cell_ex_ur10e_1
    restart the bring-up, then:  check_calibration_applied()
```

Or, from the `dgpy>` prompt, the same copy:

```python
deploy_calibration()
```

Then in a shell: `colcon build --packages-select tari_cell_ex_ur10e_1`, and
restart the bring-up. (With a `--symlink-install` workspace the build is a
no-op, because the installed file *is* the source file. It takes a second
either way — run it.)

The cell's `urdf/environment.urdf.xacro` already includes
`sensor_calibration.urdf.xacro`, and the file already exists with identity
values, so **deploying is a copy and never an edit**. That is deliberate: an
edit is the step that gets skipped at 6 pm at a cell.

**This one time only**, `tari_ur_description` itself changed too (it gained the
correction joints), so also:

```bash
colcon build --symlink-install --packages-select tari_ur_description
```

### The read-back — this is what closes the loop

After the restart:

```python
check_calibration_applied()
```

```
  azure    APPLIED
           correction joint mounting_plate <- azure_mount_cal   off by 0.000 mm, 0.000 mrad
           probe_end -> rgb_camera_link             off by 0.000 mm, 0.000 mrad
  keyence  APPLIED
           correction joint keyence_metal <- lj_v7080_mount_cal   off by 0.000 mm, 0.000 mrad
           probe_end -> lj_v7080_optical_frame      off by 0.000 mm, 0.000 mrad
  keyence_ref_dist_cal is zero -- the mount correction owns the standoff, as it should

  PASS (tolerance 0.50 mm, 1.00 mrad). Now run verify_calibration(): this said
  the numbers ARRIVED, that says they were RIGHT.
```

It reads TF, moves nothing, needs no target, and takes a second. **Three
statements per sensor and the second is the one that matters** — the correction
joint being right says the file arrived; `probe_end → sensor` being right says
the whole chain now gives the transform that was solved, which can fail even
when the first passes if something else in the chain changed underneath.

If it fails it says so and tells you what to check:

```
  azure    NOT APPLIED
           correction joint mounting_plate <- azure_mount_cal   off by 17.967 mm, 63.159 mrad
           probe_end -> rgb_camera_link             off by 16.377 mm, 63.159 mrad
           The file did not reach the description in force. Was it copied into
           the cell's tari_site package, rebuilt, and the bring-up restarted?
```

### Two files in `tari_site` that are NOT this

* **`tari_cell_ex_ur10e_1/etc/calibration.yaml`** is the **UR arm's kinematic
  calibration** from the robot vendor — joint origins to eleven decimal places,
  plus the TRAC-IK solver settings. Nothing to do with sensor extrinsics. Do
  not touch it.
* **`tari_cell_configs/config/cells/ex-ur10e-1.yaml`** has `sensors.azure.mount`
  and `sensors.keyence.mount` blocks. **Editing them changes nothing.** Both
  carry `in_robot_description: true`, and `tari_bringup/tari_bringup/sensors.py`
  returns early on that flag and publishes no transform — because the URDF
  already has the joint, and a second publisher for the same frame would give
  TF two parents for one child. Those blocks are inventory. The description is
  the authority.

---

## 5. Who owns the Keyence standoff

`keyence_ref_dist_cal` is a single number that moves the optical frame along the
head's own axis. The z of `keyence_mount_cal` moves the head, including along
that axis. **They are the same physical quantity reached two ways, and their sum
is a clean constant offset that looks exactly like a real surface** — which is
how an 85 mm reference distance survived for months and put a whole
reconstruction 5.27 mm low.

**The rule: the 6 DOF solve supersedes.** Three things enforce it, and none of
them is a note in a document:

1. `calibrate_keyence()` reads whatever `keyence_ref_dist_cal` is in force out
   of live TF and **folds it into the mount correction** before emitting
   anything. Nothing is lost in the changeover. It says so:

   ```
       keyence_ref_dist_cal was -5.27 mm and is FOLDED INTO the correction
       above. The emitted file sets it to zero -- one owner for the standoff,
       and the description refuses to render if both are set.
   ```

2. The emitted file **always** writes `keyence_ref_dist_cal`, and writes `0.0`.
3. `lj_v7080_macro.xacro` calls `xacro.fatal()` and **refuses to render** if
   `keyence_ref_dist_cal` is non-zero while a mount correction is set. The
   bring-up stops with a message naming both properties rather than coming up
   quietly with the standoff counted twice.

`check_calibration_applied()` also reports the value in force and fails if it is
not zero.

On this cell `keyence_ref_dist_cal` is unset, i.e. `0.0`: a −5.27 mm standoff
error here was a wrong `keyence_ref_dist` (0.085 against the head's specified
0.080), fixed at the source rather than by a per-cell correction. So the
changeover costs nothing here. The folding exists so it stays free on a cell
where the correction is non-zero.

---

## 6. Why the pose set is what it is

### The failure both solvers are built around

The translation of a hand-eye solve only enters through `(R_A − I) t_X`, and
`(R_A − I)` is **singular**: it annihilates the axis of `R_A`, and its other two
singular values are `2·sin(θ/2)`. So a pose set that barely rotates, or rotates
about only one axis, determines the offset barely or not at all — **and fits the
data perfectly while doing it**, because a wrong offset in the null direction is
invisible.

**A small residual is not evidence.** So the report leads with the evidence that
is: rms rotation about each of the pose set's three principal axes, the
amplification `1/σ_min` stated as *"a 0.5 mm error in locating the target
becomes N mm of error in where the sensor is"*, per-observation residuals with
outliers named rather than averaged, and 1σ on all six degrees of freedom.

`calibrate_azure()` **will not write a result** whose pose set it judges
inadequate. Measured on simulated pose sets (`calibration_test.py`):

| pose set | amplification | verdict |
|---|---|---|
| 13 tilts × 4 rolls | 1.6× | GOOD |
| 7 tilts × 2 rolls | 1.6× | GOOD |
| 5 tilts × 1 roll | ∞ | INADEQUATE |
| no tilt, 4 rolls | ∞ | **INADEQUATE** |

The conditioning is flat from about fourteen poses; the default 24 are margin
against poses that do not see the board. **Rolling the wrist without tilting is
worthless** however many poses you take — and it is the pose set an operator
naturally produces.

### The cable envelope is a hard bound

This cell cannot do textbook rotations: the end-of-arm loom binds. The measured
envelope on record (`cable_envelope.dpi`) allows tool tilts no further than
±40° about world x and ±20° about world y, with a wrist_3 band **270° wide
untilted and 75° wide at x −40°**.

So the pose set is generated *from* the measured band: the roll budget at each
tilt comes from `wrist3_band_deg()` **for that tilt** less a 10° margin, every
candidate goes through IK, the work-volume check and `check_wrist3()` before it
is offered, what was dropped is reported by reason, and the survivors are judged
by `check_calibration_poses()` **before the arm moves**.

One orientation is **not** flipped: `probe_end`'s z is world **+z** when the
probe looks down, so identity is the working orientation. Adding the obvious
180° makes every pose read as a 180° tilt, outside the measured hull, and the
envelope refuses the whole set.

### The Keyence degeneracy, and what actually fixes it

Hold the head at its reference distance at every pose — the obvious way to take
the data, because that is where the head is happiest — and **rotation of the
head about its own laser line moves no measured point at first order**. The fit
converges, reports a residual at the sensor's noise floor, and returns whatever
the URDF said.

It is tempting to think the stepped artifact fixes this by itself, since its
features span 24 mm and one pass sees nine ranges. It does not, and the reason
is worth knowing: to first order that rotation displaces each point along the
**sensor's own y** by `−z·δ`, and on a horizontal top face that displacement is
*tangential* — it changes no measured height whatever z is. The artifact is a
stack of horizontal top faces. Only its four cone flanks have a y gradient.

Measured, same poses, same 0.1 mm point noise
(`test_the_printed_artifact_helps_the_weak_direction_but_does_not_replace_the_standoffs`):

| artifact | standoffs | poses | error | conditioning |
|---|---|---|---|---|
| bare flat | 1 | 16 | 0.079 mm, 0.46 mrad | **free direction**, σ 0.032 |
| printed, stepped | 1 | 16 | 0.085 mm, 0.86 mrad | **free direction**, σ 0.067 |
| bare flat | 3 | 48 | 0.021 mm, 0.69 mrad | determined |
| printed, stepped | 3 | 48 | 0.110 mm, 0.49 mrad | determined |

The artifact roughly **doubles** the leverage on the weak direction and still
does not determine it. **Varying the standoff across the pose set is what
does**, which is why `keyence_calibration_poses()` refuses a single one whatever
the artifact is, and `CAL_KEYENCE_STANDOFFS` is `(−15, 0, +15) mm`.

What the artifact is for is everything else: it is an asymmetric rigid body with
a pose, where a flat is a body nothing can locate laterally; it exercises the
head's whole measuring window in every pass; and it is the one object the
Keyence, the Azure and the contact probe could all be made to agree on.

### What comes from the factory, and what needs solving

* `depth_camera_link → rgb_camera_link` is the Azure's **per-device factory
  extrinsic**, read off the device and published by the driver. Trust it.
* `camera_base → depth_camera_link` is **nominal**, not per-device — the driver
  hard-codes it and its own source calls it "purely cosmetic". Do not trust it,
  and do not need to: it is inside the chain being solved.
* On hardware this cell's depth topic is **`depth_to_rgb`** — depth already
  resampled into the *colour* camera — so depth points and the colour
  checkerboard are in **one frame**, `rgb_camera_link`, and solving where that
  frame sits places both.

**One thing is solved by none of this: the depth *values*.** Hand-eye locates the
camera; it does not check that the camera measures distance correctly. A
time-of-flight range bias looks exactly like a mount error — the mesh low by a
constant — and survives calibration untouched. That is what §1 is for. Three
ceramic balls of known
diameter would separate a range error from a pose error
(`fit_sphere()` with the radius free); they are not fitted and are not needed
for the 6 DOF solve.

---

## 7. What to expect, and what to do when it isn't

| | expected | how you know |
|---|---|---|
| Azure, 24 poses | **0.5–1 mm**, 2–5 mrad | 1σ printed per axis; amplification ≤ 2×; residuals ≤ 1 mm |
| Keyence, 30 poses | **0.2–0.5 mm**, ≤ 2 mrad | rms at the head's 0.1 mm noise floor; **no null directions** |
| verification | **< 2 mm**, < 10 mrad | `verify_calibration()["pass"]` |
| read-back | **0.000 mm** | `check_calibration_applied()["pass"]` |

| symptom | what it means | what to do |
|---|---|---|
| residual < 1 mm but `VERDICT: INADEQUATE` | the pose set, not the sensor | add **tilts**, not poses |
| residual > 3 mm at one or two poses | a misdetection; they are named | `calibrate_azure(poses=...)` minus those, re-solve |
| "the move is 25 mm" on a fixture nobody touched | a bad artifact, most likely board scale | check `set_board_square()` was run and the plate did not shift |
| `NOT SAVED: the pose set does not determine this transform` | correct refusal | widen the tilts, or move the board so more are reachable |
| `no transform ... <- azure_mount_cal` | the description predates the mechanism | rebuild `tari_ur_description`, restart the bring-up |
| `check_calibration_applied()` says NOT APPLIED | the file did not reach the running description | `deploy_calibration()`, `colcon build`, restart |
| bring-up dies naming `keyence_ref_dist_cal` | both standoff owners are set | set it to `0.0`; the mount correction carries it |
| `only N of M poses saw the board` | glare, or the board out of view | relight it, or move `cal_board` nearer the middle of the work volume |

`write_calibration()` says **loudly** when a solve moves a sensor by more than
2 mm:

```
    THAT IS A REAL MOVE, not a refinement. 18.0 mm means the fixture has
    shifted or the description never matched it. Look at the bracket before you
    believe it, and check the board pitch was measured (set_board_square).
```

That is the interesting case, not the alarming one — the Azure fixture is loose
and an 18 mm move is exactly what this exists to find. But look at the bracket.

### When a pose will not fly

**A run that loses a pose does not lose the run.** A 26-to-34-pose set on
hardware will lose individual poses — a cable-guard refusal by a couple of
degrees, a 150 mm straight line with a 180 degree wrist turn that plans to 51%
— and abandoning the run at the first one throws away the poses already
collected, the bag and the arm time.

What each kind of failure costs, per pose:

| what happened | what it costs |
|---|---|
| the straight line will not plan, or does not complete | **a retry in joint space to the same pose.** Usually it flies |
| no route reaches it (refused, failed, no IK, cable band) | **that pose.** The reason is printed and named in the take |
| the profiles are silent, or come back a constant | **that pose.** Same |
| the operator skips it at the prompt | **that pose** — a skip is a skip, not a retry |
| **the monitor stops the arm** (overspeed, overforce) | **the rest of the flying.** What was collected is still solved and saved, then the stop is re-raised — go and look at the cell |
| the operator aborts (`q`) | **the rest of the flying.** Same: you still get the calibration |

The retry is a *joint* move and not the same Cartesian request again, because
`computeCartesianPath` is deterministic: a leg that reached 51% reaches 51%
again. The joint move is checked exactly as hard — the interpolated path is
sampled every 2 degrees, FK'd, and put through the work volume and the cable
band at every sample — it just does not require the tip to travel in a straight
line, and that requirement is the whole of what fails.

### If the run dies anyway

**Every pose is written to disk as it is collected**, inside the take:

```
<take>/keyence_poses/
    index.json                    rewritten after every pose
    000_keyence_t+0,+0_r-90_d-15.npz    the pose, its G, its points
    ...
    keyence_calibration.npz       the combined file, at the end
```

`calibrate_keyence()` prints the path when it starts, before the arm moves:

```
  collecting into /data/scandata/capture/.../02_calibrate_keyence_.../keyence_poses
    if this run is interrupted, nothing is lost:  solve_keyence_from_take('...')
```

So a power cut, a `kill -9`, or an exception nobody anticipated still leaves
everything the solve needs. Afterwards, **with no arm and no cell**:

```python
solve_keyence_from_take("/data/.../keyence_poses")     # or the take dir,
                                                       # or the session dir,
                                                       # or the .npz
solve_azure_from_take("/data/.../azure_poses")
```

It applies the *same* guards as the live run — it is the same function, minus
the moving — so a set it accepts is a set the cell would have accepted. An npz
that `dgpfiles/utils/cal_extract.py` lifts out of a bag has the same shape
and loads the same way, which is how takes from before this existed are
recovered.

### It still refuses

Resilience is "finish the run and tell the truth about what it got", not
"always emit a number". Whatever survived is still held to every guard: at
least three poses, at least **two distinct standoffs** (at one, rotation of the
head about its own laser line is unmeasured), `check_plane_support`, and the
recovery check. A set that does not determine the transform is not written, and
the refusal names the poses that were lost and why.

The standoffs are flown **interleaved** — `-15, 0, +15` at one tilt, then the
next tilt — precisely so that a run cut short still has the spread. Flown a
rung at a time, a run that dies early leaves every surviving pose at one
standoff, and such a set cannot be solved however many poses it has.

### Time

| | first time | thereafter |
|---|---|---|
| `verify_calibration()` | 4 min | **3 min** |
| `calibrate_azure()` | 45 min (incl. printing and measuring the board) | 20–25 min |
| `calibrate_keyence()` | 25 min | 15–20 min |
| deploy, build, restart, read back | 15 min | 15 min |

Full calibration from cold: **about two hours.** The routine check: three minutes.

---

## 8. Files

| | |
|---|---|
| `docs/calibration/make_board.py`, `tari_calibration_board_20mm_letter.pdf` | the checkerboard, and the script that makes it |
| `docs/calibration/make_artifact.py`, `tari_calibration_artifact.stl` | the stepped artifact, and the script that makes it |
| `dgpfiles/modules/calibration.dpi` | the operator's module: poses, motion, capture, detection, reporting, the drop-in file |
| `dgpfiles/modules/calibration_solve.py` | the arithmetic. numpy only, no cv2, no ROS — runs on the host |
| `dgpfiles/modules/calibration_test.py` | the arithmetic, host-runnable: `python3 calibration_test.py` |
| `dgpfiles/modules/calibration_resilience_test.py` | what a run does when a pose will not fly, host-runnable |
| `tari_ur_description/urdf/azure_bracket.urdf.xacro`, `lj_v7080_macro.xacro` | the correction joints, identity by default |
| `tari_ur_description/test/test_sensor_mount_cal.py` | the composition-order and double-count tests |
| `tari_cell_ex_ur10e_1/urdf/sensor_calibration.urdf.xacro` | **where the drop-in goes** |

Both generator scripts reproduce their output byte for byte — regenerate rather
than hand-edit the STL or the PDF.

`calibration.dpi` is loaded by `ECAutomatedScan.dgp` after `cell_capture.dpi`.

### Every callable

| | |
|---|---|
| `calibration_status()` | the artifact, the record and the cable envelope in force |
| `set_board_square(m, n=10)` | record the printed board's real pitch |
| `use_printed_artifact(pad_tops=...)` | switch the Keyence to the stepped artifact |
| `azure_calibration_poses()` / `keyence_calibration_poses()` | the pose sets, no motion |
| `check_calibration_poses(poses)` | judge a pose set before flying it |
| `calibrate_azure()` / `calibrate_keyence()` | solve, record, write the drop-in |
| `solve_azure_from_take(p)` / `solve_keyence_from_take(p)` | **re-solve a take that was collected but not solved. No arm** |
| `deploy_calibration()` | copy the drop-in into `tari_site` |
| `check_calibration_applied()` | did the file take? reads TF |
| `verify_calibration()` | are the two sensors right? flies the arm |
| `describe_calibration()` / `load_calibration()` | what is on record |

---

## 9. Still open

1. **Buy the checkerboard, or keep printing and measuring?** $200–300 for a
   certified target versus free plus a caliper measurement that must not be
   skipped. `set_board_square()` makes the printed route survivable, not safe.
2. **Fit three ceramic balls?** ~$150–250 plus machining three seats. Not needed
   for the 6 DOF solve — measured, §6 — but the only way to tell an Azure
   *range* error from an Azure *pose* error, and a range error looks exactly
   like a mount error while surviving calibration. Recommendation: not yet. Run
   the flat-and-artifact calibration and `verify_calibration()` first. If the
   Azure still disagrees with the Keyence after a clean hand-eye solve, that
   *is* a range error, and then they are worth fitting.
3. **Should `verify_calibration()` gate the automated scan?** It returns
   pass/fail and takes three minutes, so it would catch a bad calibration before
   a scan is flown over it. Making `run_automated_scan()` refuse on a failure is
   a one-line change, deliberately not made — it turns a bad calibration from a
   wasted scan into a session that cannot start.
4. **Creaform targets on the artifact for the Azure**, so all three sensors
   locate one body. Designed for, not implemented; the Azure route is the
   checkerboard.
