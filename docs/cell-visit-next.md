# Cell visit, next session

Written 2026-10-10 from `results22` and `results23`. **Supersedes the
2026-10-08 version**, whose section 2 assumed the scan would simply run; it
did not, twice, and the reasons are now understood and fixed.

Undated because I do not know when you are next at the cell. Nothing here
goes stale with time — only with the next run.

**Section 1 comes before the arm moves at all.** It is a repair, not a
measurement.

---

## 0. What changed since 10-09, and how much to trust it

| change | evidence behind it | flown? |
|---|---|---|
| **Azure calibration on record is WRONG** | a 1299 mm solve overwrote a 0.39 mm one | section 1 |
| A bad solve can no longer be saved | `adequate` graded the pose set, not the fit | no |
| Replacing a calibration asks first | nothing asked on 10-09 | no |
| Keyence sweep no longer refuses over an overhanging line | three refusals on a flat plate | no |
| Clearance sign taken from the height field | 141.8 mm "inside" a 2 mm plate | no |
| Turn rate ignores coincident waypoints | 49810 deg/mm from one duplicated point | no |
| Correlator cannot wait forever for a globalrev | both freezes, 3517 DAQ overruns | no |
| mtc drains its goal threads before shutdown | the SIGINT segfault | no |

Everything except the first is derived from saved files and none of it has
flown.

---

## 1. REPAIR THE AZURE CALIBRATION — before anything else

### 1.1 What happened

The calibration run at 18:08 on 10-09 produced a solve that is physically
impossible: the camera 0.977 m **below** the flange, the mount moved 1299.17
mm and 2984 mrad (171°), 76.0 mm rms residual, 1 sigma 30.4 mm, and 13 of 24
detections 180° flipped. Its own report said `THIS IS NOT GOOD ENOUGH TO SCAN
FROM`.

It was written anyway, over the 2026-09-18 calibration that was good to 0.39
mm, to **both** of these:

```
/data/scandata/capture/cell_override/ex-ur10e-1.sensor_calibration.json
/data/scandata/capture/cell_override/sensor_calibration.urdf.xacro
```

It was almost certainly **not deployed** — deploying needs a copy into the
cell package plus a colcon build, which is a separate step you would have had
to run. Section 1.2 settles that rather than assuming it.

### 1.2 First: find out what is actually in force

Start the cell normally, then in dgpy:

```
check_calibration_applied()
```

Read the `azure` line. There are two outcomes:

- **It reports a correction near (+0.67, +0.52, +0.08) mm** — the good 09-18
  numbers are still in the robot description. The damage is confined to the
  recorded file and 1.3 fixes it. **This is what I expect.**
- **It reports anything near (+44, +62, +1297) mm** — the bad solve reached
  the URDF. Do 1.3 *and* re-deploy (1.4) before the arm goes anywhere near
  the part, because every planned standoff is 1.3 m wrong.

Either way, write down what it said.

### 1.3 Restore the recorded calibration

The numbers to restore are inside the bad file itself — the calibration
machinery records what it replaced, as `mount_cal.old_xyz` and
`mount_cal.old_rpy`. Confirm them first:

```bash
python3 -c "import json;d=json.load(open('/data/scandata/capture/cell_override/ex-ur10e-1.sensor_calibration.json'));m=d['hardware']['azure']['mount_cal'];print('old_xyz',m['old_xyz']);print('old_rpy',m['old_rpy'])"
```

That must print:

```
old_xyz [0.00066854, 0.00051648, 8.197e-05]
old_rpy [-0.005456480000000001, -0.009277000000000004, -0.008982210000000004]
```

Keep the bad file rather than deleting it — it is the only record of the run:

```bash
cd /data/scandata/capture/cell_override && cp -a ex-ur10e-1.sensor_calibration.json ex-ur10e-1.sensor_calibration.json.bad-2026-10-09 && cp -a sensor_calibration.urdf.xacro sensor_calibration.urdf.xacro.bad-2026-10-09 && ls -la
```

Now put the good numbers back into the drop-in. Edit
`/data/scandata/capture/cell_override/sensor_calibration.urdf.xacro` so the
two azure lines read **exactly**:

```xml
  <xacro:property name="azure_mount_cal_xyz" value="0.00066854 0.00051648 0.00008197"/>
  <xacro:property name="azure_mount_cal_rpy" value="-0.00545648 -0.00927700 -0.00898221"/>
```

Leave the keyence properties and `keyence_ref_dist_cal` alone — they are
zero and they were not touched.

Check it took:

```bash
grep azure_mount_cal /data/scandata/capture/cell_override/sensor_calibration.urdf.xacro
```

The `z` value must be **0.00008197**, not 1.29705095. That one digit is the
whole bug.

### 1.4 Only if 1.2 said the bad numbers reached the URDF

```bash
cp /data/scandata/capture/cell_override/sensor_calibration.urdf.xacro /home/labuser/tari/ros_ws/src/tari_site/tari_cell_ex_ur10e_1/urdf/
```

```bash
cd /home/labuser/tari/ros_ws && colcon build --packages-select tari_cell_ex_ur10e_1
```

Restart the bring-up, then in dgpy:

```
check_calibration_applied()
```

It must now report the (+0.67, +0.52, +0.08) mm correction.

### 1.5 The JSON record

The JSON still holds the bad entry. It is not read by the robot description —
the xacro is what deploys — so it does not endanger anything, but the next
`calibrate_azure()` will compare against it and report a 1.3 m "move" that is
not real. Simplest honest fix is to let the next good solve replace it, which
now asks you first (1.6).

### 1.6 What is different if you re-calibrate

Two changes, both aimed at this failure:

- **A solve this bad can no longer be saved at all.** `adequate` used to grade
  only the pose set — which was textbook that day, 24 poses with rotation
  about all three axes — and said nothing about whether the observations
  actually pinned anything down. It now also requires the fit's own 1 sigma
  position uncertainty to be under 2.0 mm, which is the threshold the report
  has always printed `THIS IS NOT GOOD ENOUGH TO SCAN FROM` at. The 10-09
  solve was 30.4 mm.
- **Replacing a calibration on record asks you, in as many words.** If the new
  solve moves the mount more than 5 mm or 50 mrad, or is less certain than the
  one it would replace, you get a side-by-side and a prompt:

  ```
  REPLACING THE AZURE CALIBRATION ON RECORD -- confirmation needed
      on record   2026-09-18T10:20  1 sigma 0.39 mm
      this solve  2026-10-10T...    1 sigma 30.43 mm
      the move    1299.2 mm and 2984.5 mrad
      ** it moves the mount 1299.2 mm, and a fixture that has not been
         touched re-solves to a millimetre or two
      ** the solve on record is more certain than this one

      Type OVERWRITE to replace it.
  ```

  You have to type `OVERWRITE` in full. Enter, `y`, and anything else all keep
  what is on record. It is deliberately not the `confirm()` you use during a
  capture run, because that has an `r` key that stops it asking for the rest
  of the take — and an operator who pressed that an hour ago has not agreed to
  this.

**If you do re-calibrate, watch the flip count.** The 10-09 run reported `13
detection(s) were 180 degrees flipped and were corrected`. A 6x9 checkerboard
is genuinely 180°-ambiguous and some correction is normal, but more than half
the set is not, and it is the best single indicator that the detections
themselves were bad. The machine was running an on-demand antivirus scan at
the time; if you re-run, make sure it is not.

---

## 2. The scan over the 0.040/0.060/0.080" notches (~25-35 min)

Settings are unchanged from the last plan — 0.25 mm pitch, comb, preload 2 mm,
probe rotation 0.

```
set_probe_rotation(0)
set_ec_preload(0.002)
set_scan_step_size(0.00025)
```

Confirm before anything moves:

```
ec_serpentine()
probe_rotation()
ec_preload()
scan_velocity()
scan_step_size()
projection_mode()
```

### 2.1 Set the conditioning rotation BEFORE the scan, not after

This is new and it is the single biggest improvement available to this run.

The 10-09 scan ran with conditioning rotation **0.0**, which is no lift-off
rejection at all, so the live image was almost entirely lift-off: the
per-pass baseline wandered over 0.765 V while the notch itself is 0.9 V. Re-
conditioning the saved traces off-line at the measured **86.5°** dropped that
drift to 0.088 V — 8.7× — and the notch came out at SNR 20.1 against a flat
field.

86.5° is close to the 83.8° measured on 10-08, so it is a property of the
probe and the material and not of that scan. Set it up front:

```
set_rotation(86.5)
```

```
ec_rotation()
```

`set_rotation()` already existed; what is new is that it now **sticks**. It
used to set only the live conditioning, and `run_automated_scan()` calls
`setup_ec()` on the first scan of a session, which defaults the rotation back
to `EC_ROTATION_DEG` — 0.0. So a rotation dialled in at the prompt was
silently replaced by zero the moment the scan started, and the only evidence
was `"rotation_deg": 0.0` in the saved metadata. That is why 10-09 ran with no
lift-off rejection. `ec_rotation()` reads back what the next scan will use.

### 2.2 Read these lines in the plan before confirming

| line | what it should say | if it doesn't |
|---|---|---|
| `projection surface checked at plan time: … loads` | the mesh gate ran here | send me the refusal; the arm has not moved |
| `tool axis smoothed (linear, N mm): turn rate X -> Y deg/mm` | **Y under 0.40**, and **no** "still over the bound" | see 2.4 |
| `… step(s) are shorter than 0.010 mm` | **absent**. If present, the planner is emitting coincident waypoints again | tell me the count |
| `IMPOSSIBLE PENETRATION` | **absent** | tell me; the sign repair did not take |
| `coverage given up to keep the tool clear` | **0%**, or a few percent at the edges | more than ~5% means the clearance sign is still wrong |

The last two are the ones to watch. On 10-09 the clearance check reported the
Keyence frame 141.8 mm *inside* a 2 mm plate, dropped 376 of 2925 waypoints,
skipped whole passes, split 101 passes into 136 and gave up 17.1% of the path.
Each mid-pass split is a lift and an approach, which is where the stop-start
toolpath came from.

### 2.3 Watch the pass times — this is the real test

On 10-09 the same 29 mm pass took **50 s** where a clean mesh took **16 s**,
with the UR's own speed scaling pinned at 100% the whole time. The arm did
exactly what it was told; TOTG had priced a path whose p99 tool-axis turn rate
was 3.4× higher, and the smoothing that should have fixed it was chasing a
49810 deg/mm reading that came from two waypoints at the same place.

**Note the elapsed time on a dozen passes.** A ~29 mm pass should be around
16–20 s. Three outcomes:

- **~16-20 s, no rocking** → the turn-rate metric fix and the mesh are both
  good. This is what I expect.
- **40 s or more, but the plan reported a turn rate under 0.40 and no
  degenerate steps** → it is not the turn rate. Tell me; that is a real and
  useful negative.
- **40 s or more and the plan said "still over the … bound"** → read which of
  the two messages it printed. "A wider window stopped reducing it" means the
  surface genuinely turns that fast; "the mesh is noisier than this can fix"
  means re-run `reconstruct_fine()`.

### 2.4 If the smoothing says it could not meet the bound

Nothing is refused — the scan runs either way. Do not raise
`EC_AXIS_SMOOTHING_MAX_M` to force it under: past 20 mm the axis is being
averaged over a distance comparable to real features.

---

## 3. If it freezes again

Both 10-09 freezes were the same thing and it is fixed, but if it recurs the
symptom will be different, so here is what to look for.

The cause was the correlator thread waiting forever for a globalrev that was
never going to exist, on the one thread that drains the DAQ — which is why the
first run ended in 3517 copies of `NI-DAQmx -200279`. That wait is now bounded
at 20 s (normal is 50–150 ms).

**What you should see now instead of a freeze:**

```
** staging timed out after 20.0 s on trace N -- the globalrev for that pose
   never arrived. The trace is given up and the scan continues; 1 lost so far.
```

The scan carries on with a hole. If forty time out in a row it stops with a
fault naming the count, because by then it is the pipeline and not a trace.

**If you see that message even once, tell me.** It means the underlying
globalrev problem is still there and only its consequence has been contained.
Bring back `dataguzzler.log`; the per-pass `trigger_report` in each
`pass*.npz` now carries `staging_timeouts`.

---

## 4. Condition the image — do not skip this (~2 min)

If 2.1 worked, this is already done and you can skip it. If you ran at
rotation 0 again:

```
out["report"].get("rotation_measured_deg")
```

```
replay_ec_scan(rotation_deg=out["report"]["rotation_measured_deg"])
```

**Write down the measured rotation.** 10-08 gave 83.8°, 10-09 gave 86.5°.
A third number tells us how stable it is, which decides whether it can be a
configured constant rather than a per-scan measurement.

---

## 5. The acquisition lag — still one speed, still open (~25 min, optional)

Unchanged and still the oldest open question: **0.140 s at 3.93 mm/s, from one
scan.** One speed cannot separate a time lag from a fixed distance offset.

```
set_scan_velocity(0.004)
scan_velocity()
out2 = run_automated_scan(line_spacing=0.0003)
```

**Check the printed plan's `velocity` line before confirming.** Note that the
10-09 runs commanded 2.5 mm/s and the arm averaged 0.41 mm/s — but that was
TOTG pricing the path, not a velocity-calibration error, so if section 2 comes
in at 16-20 s per pass the commanded speed is being honoured and this
experiment is finally worth running.

---

## 6. Optional

**RViz memory.** It grew 5.28 → 6.82 GB over 70 minutes on 10-09 and then
plateaued when the scan stopped, so the growth tracks scan activity. It did
not cause either freeze. Still worth a log:

```bash
( while true; do date +%s; \ps -eo rss,comm --sort=-rss | head -6; sleep 60; done ) > ~/mem.log 2>&1 &
```

(Note the `\ps` — there is an alias on this machine that breaks `-o`.)

**Keyence comb return leg.** Still not investigated.

---

## 7. What to bring back

1. What `check_calibration_applied()` said, before and after section 1.
2. The five plan lines from 2.2, verbatim.
3. Elapsed times for a dozen passes, and whether the arm rocked anywhere.
4. The measured conditioning rotation.
5. The `ec_scan_*` directories, `dataguzzler.log`, `ros.log` — **separate
   files** if you can.
6. Anything that printed `staging timed out`, `IMPOSSIBLE PENETRATION`, or
   `step(s) are shorter than`.
