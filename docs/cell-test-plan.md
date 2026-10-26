# Cell test plan — run the pipeline end to end

A run sheet for a site visit whose objective is `run_automated_scan()` through
all five stages on the real arm.

Stage 2 — the Azure reconstruction that produces the rough planning surface — is
the one that has not succeeded on hardware, and everything downstream of it
(Noether, the Keyence raster, the eddy-current scan) therefore has not either.
Reaching stage 5 is the objective; everything else is setup for it.

Budget about 3½ hours. Steps 0–2 are setup and take an hour; step 3 is the one
that matters.

The one rule that saves the visit: **run the stages separately first** (step 3).
Each is its own callable, so a failure costs one stage instead of the whole run.

---

## 0. Before the robot moves — 20 minutes

### 0.1 Where you put the part decides whether stage 2 can run

Stage 2 no longer orbits the part. It flies a **straight-down raster at one fixed
orientation** — the orientation the arm is already standing in — so it asks for
**no wrist rotation at all**. Cable room is not what decides whether it can fly.
**Reach is.**

The camera has to get 530 mm above the part's top, which puts probe_end at about
1.38 m, and the arm cannot do that everywhere: over the measured work volume it
manages 87.5%, failing at **x ≥ 1.061**.

**Put the specimen near the middle of the table: centre it around x 0.85, y 0.00.**
Keep it well clear of x 1.06.

The raster is now centred on the part rather than started from its corner, and a
grid whose corners do not reach is **pulled in toward its centre instead of
refusing**. At this standoff the camera sees 813 × 675 mm and the part is
135 × 286 mm, so every stop frames the whole part — the raster buys extra viewing
rays, not coverage, and losing some of its span costs nothing you need.

### 0.2 Bring-up

Follow `tari_bringup/CELL_CHECKLIST.md`. Unchanged.

### 0.3 Three checks, 30 seconds each

In dgpy, before anything moves:

```python
supervisor_status()          # want: watching: yes
describe_cable_envelope()    # want: the date of your own re-measurement, below
```

```bash
ros2 topic echo --once /keyence_status
```

`streaming` = data flowing. `connected_no_data` = laser off or nothing in range,
which is healthy. `disconnected` = the controller is unreachable, and that is a
real fault.

If `supervisor_status()` says `watching: NO`, stopping External Control will
**not** pause a scan. Note it and be aware you have lost that safety net.

---

## 1. Cable envelope — 30 minutes, do this first

Everything that moves is gated on this, and the existing measurement does not
cover what the Azure sweep needs.

```python
pan_tilts_needed()           # what THIS part will ask for
capture_cable_envelope()     # 9 tilts, ±40° on both axes
describe_cable_envelope()    # confirm it shows TODAY'S date
```

`pan_tilts_needed()` prints the tilt each sweep direction wants and whether the
loaded envelope covers it. If it says NOT MEASURED anywhere, add those tilts to
`capture_cable_envelope(tilts=...)`.

Step size 15°, each step previewed. **Answer `t` one step early rather than
late** — the recorded limit is the last step you confirmed good, so "tight" means
"the one before this was the last safe one".

Two things worth knowing:

- **The envelope is not repeatable.** Successive measurements of the same cell
  have differed by 30–75° on some tilts, in the slacker direction. Measure; do
  not assume the band on record.
- A fresh measurement can be written and then ignored in favour of the
  manifest's block. `describe_cable_envelope()` showing the date you just
  measured is the one-second way to be sure it took.

---

## 2. Work volume and taught points — 20 minutes

```python
capture_touch_off()          # only if the table or fixturing moved
teach_point('flat')          # probe tip ON a flat face of the specimen
list_taught()
```

**You always teach with the PROBE TIP.** The Keyence sits 75 mm to the side of it
and the Azure further still. Every scenario takes your taught point as *where the
sensor must end up* and solves for where the arm has to be — you do not
pre-compensate, and you should not try to.

Previews now print both: `probe_end -> (x, y, z)  Keyence measures at (x, y, z)`.
If the laser figure is not on the feature you taught, stop and say so.

Re-teach anything that moved, including after the specimen has been picked up and
put back.

---

## 3. Walk the pipeline one stage at a time — 60 minutes

**This is the session.** Run each stage on its own, in order, and look at the
result before going on. A failure here costs one stage; a failure inside
`run_automated_scan()` costs the run.

| # | Call | What good looks like |
|---|---|---|
| 1 | `region = detect_part()` | a region roughly the size of the part, surface z within ~10 mm of the real top |
| 2 | `check_rough_plan(region)` | the raster, the standoff, and how many stops reach. Moves nothing |
| 3 | `rough = reconstruct_rough(region=region)` | a `.ply`, and the reported top within ~10 mm of the known top |
| 4 | `fine = reconstruct_fine(rough=rough, region=region)` | a denser `.ply`; the Keyence at 218 Hz, 800 points/profile |
| 5 | `mesh = filter_mesh(fine)` | a mesh that still covers the part |
| 5b | `require_plannable_mesh(mesh)` | verdict `plan`. `repair` fixes it for you; `refuse` says why |
| 6 | `path = plan_ec_path(mesh)` | passes covering the region, no huge wrist travel |
| 6b | `ec_joint_report()` | `cable band: every solved pass is inside …`. If it names passes outside, do not fly it |
| 7 | `dry_run_ec_scan()` | flies the path with the probe clear of the part |

### Why stage 2 is a straight-down raster and not an orbit

**What an orbit costs.** A `"pan"` sweep orbits the part: eight candidate
azimuths around it, and to face each azimuth the camera has to be *rolled* to it.
That roll is wrist_3. What the orbit buys is views of the **sides** of the part —
and the sides of a thin flat plate cannot be reached from this cell at all, so
the whole wrist cost is paid for data that is never collected. On a flat plate
it is refused at all 8 azimuths and all 6 heights.

**What stage 2 commands instead.** One orientation, held for every stop:
whatever the arm is already in, which is inside the cable band by construction
because the arm is standing in it. The stops differ only in x and y, and the
parallax comes from translating. Measured across the whole raster: **wrist_3
moves 0°.**

**Why a plan can pass and the run then fail, and why it no longer can.** Two
causes, both fixed, both of which read as an intermittent Azure fault:

- the dome laid its grid out for the **camera**, and probe_end — which is what
  the arm holds and what IK is asked about — hangs 77 mm off it, so the plan and
  the run were not screening the same poses. The reach screen runs on probe_end,
  after the camera offset.
- `check_sweep_plan()` was a *separate implementation* of the thing it checked,
  and IK is randomised, so it would eventually disagree — a check reporting "2
  directions kept" against a run that found 0 of 8 minutes later, over the same
  region and the same cable file. `check_rough_plan()` is `reconstruct_rough()`
  with `plan_only=True` — the same source, the same kwargs, the same planner. It
  cannot drift.

**If stage 2 still refuses**, the message distinguishes the two causes:

1. `the arm cannot reach even a single pose over the middle of the region` →
   reach. Move the part toward the middle of the table, or lower the standoff
   with `height=`. The raster was already tried at 100%, 70%, 45% and 25% of the
   region span before you saw this.
2. `REFUSED by the cable envelope` → the orientation the arm is standing in is
   itself outside the band. Move wrist_3 back toward the middle of the band
   (`over_table` is 154°, the untilted band is 16…301°) and try again.

`style="pan"` gets the orbit back, for a part whose sides matter and can be
reached.

---

---

## 3b. If stage 2 or 3 will not work — go round them

**Do not let a broken Azure cost you stages 5 and 6.** They have not run on
hardware either, and they do not need the Azure: they need *a mesh*, and any
earlier Keyence reconstruction of a real specimen will serve. The controlling computer keeps
them under `/data/scandata/<timestamp>/specimen_keyence_poisson.ply`.

```python
path = plan_ec_path("/data/scandata/<timestamp>/specimen_keyence_poisson.ply")
preview_ec_scan(path)        # look at it in RViz before flying it
dry_run_ec_scan()            # flies the path, measuring instead of acquiring
```

That exercises Noether planning, the clearance gate, and the contact motion on
real geometry, whatever the Azure did. **If you do nothing else, do this** — it
is the only route that moves the arm through those stages without a working
stage 2.

**Expect a repair message, and do not stop for it.** A real reconstruction
typically carries a handful of non-manifold edges and Noether refuses it as-is
at 0.3 mm spacing. The path repairs it automatically and says so. A verdict of
`repair` is normal; `refuse` is not, and it will tell you why.

Put the specimen where that mesh was taken, or pass `region=` to crop the path to
where the part actually is now.

---

## 3c. Noether over a real reconstruction — four things to expect

Planning against a real Keyence reconstruction, rather than a synthesised
specimen, brings out four behaviours that a clean mesh never shows.

**A real mesh is REFUSED at the production line spacing**, and the cause is a
couple of non-manifold edges — on a 23000-face reconstruction, 8 faces. Worse,
the spacing sweep is not monotone: 1.5 mm fails, 1 mm works, 0.5 mm fails,
0.4 mm works, 0.3 mm fails. It is a lottery, not a resolution limit. Dropping
those faces makes every spacing plan. `require_plannable_mesh()` catches this
before the planner is asked, and `EC_CLEAN_PLANNING_MESH` repairs it for the EC
stage specifically.

**A NaN in the mesh kills the Noether service process outright.** Open3D
smoothing can write NaN vertices — a handful in eight thousand is enough — and
`ball_pivot_smoothed.ply` is a reconstruction output that has carried them.
`principal_axis_noether_srv` aborts on a PCL assert, every plan after that fails
with **"no principal_axis_noether service"**, which blames the wrong thing
entirely, and nothing restarts it.

> **If a plan fails with "no principal_axis_noether service", do not restart the
> bring-up.** Check the service log for `Invalid (NaN, Inf) point coordinates`
> first. Restart just that node.

**Do not touch `NOETHER_TOOL_ROLL`.** It ships as `"locked"`, which gives exactly
0.00° of tool twist across the whole path. The alternatives, measured on the same
74100-pose path: Noether's own roll gives 406° of twist *within a single pass*,
`path` gives 182°, `fixed` gives 60°. Locked is why the sawtooth does not bite.

**`EC_MAX_TILT_DEG` and the cable envelope disagree by more than 2×.** The cap
lets waypoints through at up to 44° of tilt about y; the cables on record have
only been swept to ±20° in y. On a real mesh that leaves a few percent of flying
waypoints at tilts the envelope refuses outright. Step 1 (sweeping y to ±40°) is
what closes this. If you skip step 1, drop `EC_MAX_TILT_DEG` to 20 instead —
otherwise the EC stage refuses moves partway through the scan rather than at
planning.

**Plan on the mesh's own normals whenever it carries them**, which is the
default. The planner's own normal estimate puts about 16% of poses facing *into*
the part; the mesh's normals put 0% wrong, halve the tilt crop, and cut the
cable-refused waypoints from 5.9% to 1.0%.

---

## 4. The full automated run — 30 minutes

Once every stage has worked on its own:

```python
run_automated_scan(scan=False)   # everything except the EC scan itself
```

Then, if that completes:

```python
run_automated_scan(scan=True)
```

This is the deliverable. If it runs start to finish, the visit succeeded
whatever else did or did not work.

---

## 5. Stopping safely

**Stopping External Control on the pendant is a pause, not a failure.** The scan
holds at the next row boundary and says so. To carry on, restart External Control
and then, from any terminal:

```bash
ros2 topic pub -t 5 -r 5 /scan/command std_msgs/msg/String "{data: resume}"
```

`{data: abort}` abandons instead, without moving the arm. Bags and
reconstructions are closed properly either way, so an aborted run is still usable
data.

`-t 5 -r 5` and not `--once`: `--once` publishes and exits before discovery
reaches the subscriber, and the command is silently lost.

A pause nobody lifts becomes an abort after 30 minutes.

---

## 6. Two traps

**Changing a setting at the prompt.** `global NAME` then `NAME = value` **does
not work** and looks like it did. Use:

```python
globals().update(SPECIMEN_SWEEP_STYLE="dome")
```

The same applies to list comprehensions at the prompt — they cannot see the
prompt's own locals. Put the value in `globals()` first.

**A refused move inside the pipeline is a failed move, not an exception.** The
row is dropped and the sweep carries on with a gap. So a scan that *finishes* is
not proof that nothing was refused — read `session.log`, and run the wrist-3
audit below.

---

## 7. What to bring back

- the whole capture session directory (bags, `session.log`, `scenario.json`,
  `taught_points.json`)
- `cable_limits_ex-ur10e-1_hardware.json` — **especially this**
- every `.ply` the run produced, at every stage
- `ros.log`, the `roslog/` directory including `mtc_node_*.log`, and
  `dataguzzler.log`
- what you typed, and anything surprising, in your own words

The single most useful check when it comes back:

```bash
python3 ROS-Scanning/dgpfiles/utils/wrist3_audit.py \
    --limits cable_limits_ex-ur10e-1_hardware.json \
    ros.log roslog/mtc_node_*.log
```

It flags every commanded wrist-3 angle outside the measured band. It should print
`OK`.

---

## Reference, not procedure

- `cell-data-capture.md` — the individual capture scenarios in detail
- `ec-automated-scan-first-run.md` — what each pipeline stage does and why
- `next-visit-plan.md` — a superseded run sheet for one earlier visit; keep it
  for what that visit measured, not for what to do
