# The three-point plane scan

Teach a scan area by driving the probe to three points and capturing the pose at
each. No mesh, no reconstruction, no registration — which is the whole reason
this path exists and why it is the one to reach for on a new part or a new cell.

Verified end to end in simulation on 2026-08-17, by the procedure below rather
than by a convenience wrapper: three points captured from `ros.t` at three
manually reached poses, then `Scan2DPlaneTriggered` — 183 of 183 traces staged,
every pass `move_ok` at `fraction 1.0`, 0 stager errors, 0 late transactions.

## What the three points mean

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
  coplanar with the surface you meant, the scan is flat and the surface is not;
  that is fine for a quick look and wrong for a gated C-scan (see "when the
  plane is a lie" below).
- **Orientation comes from t1 alone** and is held for every point. Whatever
  attitude the probe is in when you capture t1 is the attitude it scans in, so
  square the probe up before capturing the first point, not the second.

## The procedure

Drive the probe to each corner however you normally do — pendant, servoing,
`ros.MoveJoints`, anything — and capture at each:

```python
pt1 = ros.t        # origin corner; probe squared up as you want it to scan
# ... drive to the far end of the first scan axis ...
pt2 = ros.t        # sets the u direction and length
# ... drive to the far corner of the other axis ...
pt3 = ros.t        # sets the v extent
```

**`ros.t` is a snapshot, not a live handle.** The TF thread rebinds it to a new
`TransformStamped` on every poll rather than mutating the one you hold
(`motion.py`, `self.t = self._tf_buffer.lookup_transform(...)`), so the three
captures stay distinct while the arm keeps moving. Verified: `pt1` still read
x = 0.7496 after the arm had moved on to x = 1.0499 and `pt2` had been taken. No
`deepcopy` needed.

Then choose the grid and scan:

```python
m, n = plane_grid_size(0.30)                  # from the technician settings
res = Scan2DPlaneTriggered(pt1, pt2, pt3, m, n)
```

`m` is the number of points ALONG a row (u), `n` is the number of ROWS (v). For
a rectangle the sizes differ, so derive them from the two lengths rather than
one:

```python
import numpy as np
def _len(a, b):
    pa = np.array([a.transform.translation.x, a.transform.translation.y, a.transform.translation.z])
    pb = np.array([b.transform.translation.x, b.transform.translation.y, b.transform.translation.z])
    return float(np.linalg.norm(pb - pa))

m = int(round(_len(pt1, pt2) / SCAN_STEP_SIZE)) + 1
n = int(round(_len(pt1, pt3) / SCAN_LINE_SPACING)) + 1
```

`SCAN_M` / `SCAN_N` in the config override the derivation if you would rather
pin the grid; `plane_grid_size()` honours them.

## Which of the three to call

All three take the same `(t1, t2, t3, m, n)` and build identical geometry.

| call | motion | acquisition | use it when |
|---|---|---|---|
| `Scan2DPlane(...)` | stops at every point | one A-scan while stationary | the geometry is in doubt, or the probe needs to settle. Costs a MoveIt plan-and-execute cycle per point — 0.3–0.5 s regardless of step size |
| `Scan2DPlaneRaster(...)` | one continuous trajectory per row | free-running digitizer, samples binned to the nearest grid column | a quick continuous scan where the digitizer has no position trigger |
| `Scan2DPlaneTriggered(...)` | one continuous trajectory per row | **one trace per position trigger** | the normal choice. Trigger k IS column k — nothing is binned, nothing is filled from a neighbour, and the pose on a trace is the interpolated trigger crossing rather than whatever TF last published |

`Scan2DPlaneTriggered` requires `ASCAN_TRIGGER_MODE = "position"` (or
`"timestamped"`) and refuses otherwise. The trigger pitch is set from the grid
at the start of the scan — trigger k must be column k, so the pitch is not
yours to choose independently of `m`.

Both continuous forms take `max_rows=` (stop after that many rows) and
`force=`; the point-to-point form takes `max_points=`. The volume is allocated
at full size either way, so a short run is a dry run of the real geometry rather
than a different scan.

## Reading the result

The returned dict is the scan's own account of itself. The fields that matter:

| field | meaning |
|---|---|
| `staged` vs `expected_traces` | traces that reached the database against traces planned |
| `per_segment[k].move_ok` / `.fraction` | did the row execute, and how much of it the planner covered |
| `skipped` | rows that did not run at all |
| `short_passes` | rows whose trigger count disagreed with the grid |
| `stager_errors`, `stager_last_error` | staging failures — should be 0 |
| `late_transactions`, `max_lateness_secs` | the timestamped-acquisition window — should be 0 |
| `stager_max_queue_depth` | how close the staging pipeline came to falling behind |

A healthy 3-row run looks like:

    segments 3, expected_traces 183, staged 183, stager_errors 0,
    late_transactions 0, elapsed 32.8 s

Then `/ut_scan_volume` holds `(m, n, nsamples)` and `/ut_cscan` the gated
reduction, retunable live from the panel (`show_scan_panel()`) because the gate
reads `/cscan_start` and `/cscan_end` as channels rather than constants.

## Traps

**A pass that runs short is a fabricated C-scan, not a gap.** The volume is
indexed by grid position, so a row that stops three quarters of the way along
does not leave a hole — the acquisition fills the rest from a neighbour and the
image looks complete. That is why `RASTER_MIN_FRACTION` is 1.0 and why the
Cartesian stage refuses a short path outright instead of logging it. Do not
lower it for a grid scan; `PATH_MIN_FRACTION` exists for the surface-following
scans, where a short pass costs coverage rather than correctness.

**Pin the origin for anything you intend to compare.** `scan_corners()` with
`origin=None` starts from wherever the probe happens to be, which drifts between
sessions; a scan started from a wandered position can run off the edge of the
workspace and every point past the limit only logs "Ensure point is reachable".
Capturing t1 by hand is the pinned version of that, which is the point of this
workflow.

**When the plane is a lie.** A flat scan plane over a curved part changes the
standoff across the scan, so the gate walks off the feature and the C-scan is
wrong even though every trace is real. That is acceptable for confirming the
mechanism and for a flat part; for a curved one either follow the surface
(`run_mesh_scan()`, which needs a mesh) or use a tracking gate — `cscan_gate`
takes a reference recording precisely so a second gate can sit a fixed distance
behind a front-wall crossing.

**Velocity is derived, and the derivation can be wrong.** The continuous forms
build a scan plan that clamps velocity against the measured trace-rate ceiling.
If the arm's observed speed disagrees with the commanded one the scan says so
and names the correction, e.g. *"observed probe speed 8.4 mm/s over 30 passes
(asked for 50.0) ... ROBOT_VELOCITY_CALIBRATION is wrong here. Replace or add
the entry (0.0424, 0.0084)"*. Believe it; the calibration is a measured curve,
not a constant, and it is per cell.

## Saving a taught area

The three transforms are ordinary messages. To keep an area across sessions,
store the nine numbers and the orientation and rebuild them — do not pickle the
message:

```python
def area_to_dict(t):
    tr, r = t.transform.translation, t.transform.rotation
    return {"xyz": [tr.x, tr.y, tr.z], "quat": [r.x, r.y, r.z, r.w]}
```

Rebuild with a `TransformStamped` whose `header.frame_id` is `world` and whose
`child_frame_id` is `probe_end`; only `transform` is read by `_plane_grid()`,
plus `t1.transform.rotation` for the scan orientation.
