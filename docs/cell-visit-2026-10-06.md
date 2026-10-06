# Cell visit, Tuesday 2026-10-06

Written 2026-10-05 from `results19` — the first run that went start to finish and
produced a readable image. Everything here follows from measurements on that
data, and each section says **what to run, what you should see, and what to
write down**.

Unlike the 10-05 plan, two of these sections are *measurements whose answers we
do not have*, not verifications of a fix. Those are section 2 (acquisition
latency) and section 4 (Keyence waviness). They are the two unknowns standing
between the pipeline and a trustworthy image.

**Time budget.** Section 1 is unavoidable (~15 min). Section 2 is ~20 min and
unblocks the most. Section 3 is free — it happens inside a scan you were going
to run anyway. Section 4 is ~15 min. Section 5 is the real measurement.

Sections are independent and can be run in any order after section 1.

---

## 0. What changed since the last run

Four code changes landed. All of them are on `main` in `ROS-Scanning` and in
`tari_ros`, so `tari.bash` picks them up — see section 1.

| change | why | what you will notice |
|---|---|---|
| Projection spot radius now follows **both** sample spacings | it only ever saw the trace pitch, so a 0.30 mm line spacing still got a 1.6 mm radius — 26 traces averaged into every pixel | the plan prints `8 trace(s) averaged per pixel`, and the radius drops from 1.60 mm to 0.87 mm at a 1.00 × 0.30 mm grid |
| Pinched vertices are repaired before the surface is loaded | the 14:51 scan died in `x3d_load_geometry` **after** the probe was on the part and the instrument nulled | either nothing, or one line saying a face was removed |
| `mtc_node` drains its goal threads before shutting down | it segfaulted on the way out three times in one session | a clean exit when you stop the cell |
| The preload ceiling note records that 1–2 mm is the intended operating point | so nobody "fixes" it back | nothing; it is a comment |
| `r` at a capture prompt runs the rest of that take without asking | a tilt × rotation matrix is ~50 prompts and by pass two you are confirming the same descent for the ninth time | a new key in the prompt — see below |
| `capture_contact()` declares specimen contact for its own take | the probe is pressed into a collision object, so moves made *from* the part were refused at 14–86% | a `contact with 'specimen' is now ALLOWED` line at the start of each take, `now refused` at the end |

### `r` — stop confirming, for the rest of this take

Every capture prompt now reads:

```
  [Enter=go, s=skip, q=abort, r=run the rest of this take without asking] >
```

Press `r` once you have watched a pass and are satisfied. Every later move in
**that take** proceeds without asking — still previewed, still printed, still
recorded (the take's artifact marks them `auto_by: operator`, so the record
distinguishes a move you looked at from one you armed through).

Three things worth knowing:

- **It is scoped to the take, by identity.** The next `capture_contact()` asks
  again from its first move, even minutes later. Nothing has to be cleared.
- **There is no way back except Ctrl-C**, which the supervisor turns into a stop
  at the next checkpoint without moving the arm. The line printed when you arm
  it says so.
- **It is not `CAPTURE_AUTO_CONFIRM`**, which is still refused outside
  simulation and still should be. That one is set before anything runs by
  someone who has not seen the arm move; `r` is a decision made mid-run by the
  person watching it.

**Nothing here changes a number you were using.** A square sampling grid gets
exactly the radius it got before; only a raster whose line spacing differs from
its trace pitch changes, and that change is the fix.

---

## 1. Bring the cell up (~15 min)

### 1.1 Update the workspace

**This is the whole deployment.** `tari.bash` fetches every repository, fast-
forwards it, runs `dgpy_install.sh --with-extras` (which builds and installs
spatialnde2 *and* rebuilds `spatialnde2-ec-functions` against it in the same
pass), runs one colcon build, and verifies the result. There is nothing to
install by hand.

```bash
cd ~/tari && ./tari.bash --layer site
```

Expect it to report the repositories it fast-forwarded. If any says
`cannot be fast-forwarded ... it has diverged from origin`, **stop and send me
the line** — it means there is a local commit on the cell that I do not have.

Watch for these two lines near the end:

```
==> spatialnde2_ec_functions installed
==> all packages current
```

The extras are rebuilt every time with no staleness check, deliberately: they
link spatialnde2's C++ ABI, and an extra left in place across a spatialnde2
rebuild does not fail to import — it segfaults when the EC math runs.

If the verify phase fails, it will print `dgpy_install.sh --check` output naming
the package and the state. Send me that.

### 1.2 Source and launch

```bash
source ~/tari/tari_setup.bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```

### 1.3 Start dgpy, in a second terminal

```bash
source ~/tari/tari_setup.bash
cd ~/tari/ROS-Scanning && dataguzzler-python --ipython dgpfiles/ECAutomatedScan.dgp
```

### 1.4 Confirm the state before moving anything

At the `dgpy>` prompt:

```
ec_ready()
probe_rotation()
ec_preload()
```

`ec_ready()` should report the DAQ live. The other two print what is in force and
the wrist angle the rotation implies.

---

## 2. Acquisition latency — the measurement that matters most (~20 min)

### What we know and what is missing

On `results19` the **same notch lands in a different place depending on which way
the probe drove over it**:

| run | speed | a→b vs b→a separation | implied lag |
|---|---|---|---|
| contact line over three notches | 10.0 mm/s | 2.49 mm | 0.125 s |
| 59-pass raster | 3.93 mm/s | 1.10 mm | 0.140 s |

Two speeds 2.5× apart giving the same *time* means this is a **latency**, not a
mechanical offset — if the probe were leaning in its fixture, the raster would
have shown 1.25 mm too. On a serpentine raster it puts alternate lines ~1 mm
apart, and that is a large part of why the image washes out.

Two speeds is enough to say "time, not distance". It is not enough to pin the
number, and the number is what a correction needs.

### 2.1 Teach a line across one notch

Use the **0.005 / 0.010 / 0.020"** group — the notches parallel to the short
edge, in the lower-left corner. Teach `contact_a` before the first notch and
`contact_b` past the third, with the probe **compressed as it will be when
scanning** and the **same orientation** at both points.

Jog the probe tip onto the plate at the start of the line, then:

```
teach_point("contact_a")
```

Jog to the far end of the line, same orientation, then:

```
teach_point("contact_b")
```

```
taught_points.keys()
```

### 2.2 Contact — nothing to do, but know what you will see

`capture_contact()` now declares the probe's contact with the specimen itself,
for the duration of the take, and withdraws it after. You should see this when
a take starts:

```
planning scene: contact with 'specimen' is now ALLOWED for ec_probe, probe_holder.
```

and the matching `now refused` line when it ends. If instead it says

```
  no specimen in the planning scene, so there is no contact to declare --
  and nothing is checked against the part either.
```

that is also fine for a capture take, and worth noting: it means nothing on the
tooling is being collision-checked against the part at all.

This is what was behind the refusals last time — `onto contact_a along the tool
axis` at 86%, `off contact_a` at 14%, and `over contact_a: lift off` reporting a
100% plan that then moved 1.0 mm of the 10 it was commanded. All three read as
planner faults; all three were the planner correctly refusing to interpolate out
of a start state it had been told was illegal. If you see one of those again,
it is **not** this, and I want to know.

A declaration you made by hand is left alone and not withdrawn.
`capture_contact(..., declare_contact=False)` keeps every protection, for the
case where the refusal is the thing being tested.

### 2.3 Set the probe to its sensitive orientation

These notches run parallel to the short edge, so **90°** is the orientation that
sees them (section 5 has the numbers).

```
set_probe_rotation(90)
```

It prints the wrist_3 angle and refuses before anything moves if the cables
cannot hold it. +90 puts wrist 3 at +256.1°, inside the measured +16.1..+301.3.

### 2.4 Fly the ladder — six passes

Three speeds, both directions. Each call is one pass and prompts before every
move.

```
capture_contact(points=("contact_a","contact_b"), tilts_deg=(0,), rotations_deg=(90.0,), speed_mm_s=4)
capture_contact(points=("contact_b","contact_a"), tilts_deg=(0,), rotations_deg=(90.0,), speed_mm_s=4)
capture_contact(points=("contact_a","contact_b"), tilts_deg=(0,), rotations_deg=(90.0,), speed_mm_s=10)
capture_contact(points=("contact_b","contact_a"), tilts_deg=(0,), rotations_deg=(90.0,), speed_mm_s=10)
capture_contact(points=("contact_a","contact_b"), tilts_deg=(0,), rotations_deg=(90.0,), speed_mm_s=20)
capture_contact(points=("contact_b","contact_a"), tilts_deg=(0,), rotations_deg=(90.0,), speed_mm_s=20)
```

Each one prints the take directory it wrote, like

```
recording 01_contact -> /data/scandata/capture/2026-10-06_..._hardware/01_contact_...
```

**Write down which take directory went with which speed and direction**, or just
note the order — the timestamps are in order anyway.

### 2.5 Read the answer on the cell

Save this as `/tmp/lag.py` on the cell:

```python
import numpy as np, glob, os, sys, json

def centroids(path, n_notches=3):
    """Where along the line each notch sits, in mm, in a->b coordinates."""
    z = np.load(path, allow_pickle=True)
    s = z["s_m"]*1e3
    v = z["impedance"].imag
    o = np.argsort(s); s, v = s[o], v[o]
    base = np.median(v)
    # the n strongest excursions, each taken as an amplitude-weighted centroid
    out, used = [], np.zeros(len(s), bool)
    for _ in range(n_notches):
        d = np.abs(v - base); d[used] = 0.0
        if not d.max(): break
        c = s[np.argmax(d)]
        m = (s > c - 6) & (s < c + 6)
        w = np.abs(v[m] - base); w = np.where(w > 0.4*w.max(), w, 0.0)
        out.append(float((s[m]*w).sum()/w.sum()))
        used |= m
    return sorted(out)

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tari/scandata/capture")
takes = sorted(glob.glob(os.path.join(root, "2026-10-06_*/0*_contact_*")))
rows = []
for t in takes:
    f = sorted(glob.glob(os.path.join(t, "ec_*tilt+0.npz")))
    if not f: continue
    sc = json.load(open(os.path.join(t, "scenario.json")))
    p = sc.get("params", sc)
    speed = 1e3*float(p.get("speed_mps", 0))
    fwd = p.get("points", ["a", "b"])[0].endswith("a")
    c = centroids(f[0])
    if not fwd:                       # put b->a on the a->b axis
        L = np.load(f[0], allow_pickle=True)["s_m"].max()*1e3
        c = sorted(L - x for x in c)
    rows.append((speed, fwd, c, os.path.basename(t)))
    print("%6.1f mm/s  %-4s  %s  %s" % (speed, "a->b" if fwd else "b->a",
          " ".join("%7.2f" % x for x in c), os.path.basename(t)))

print()
print("speed    separation (a->b minus b->a)    implied latency")
for sp in sorted({r[0] for r in rows}):
    f = [r[2] for r in rows if r[0] == sp and r[1]]
    b = [r[2] for r in rows if r[0] == sp and not r[1]]
    if not f or not b: continue
    n = min(len(f[0]), len(b[0]))
    sep = float(np.mean([a - c for a, c in zip(f[0][:n], b[0][:n])]))
    print("%6.1f   %+7.3f mm                      %.4f s" % (sp, sep, sep/2/sp))
```

Run it:

```bash
python3 /tmp/lag.py
```

### 2.6 What the answer means

**Write down the three latencies.**

- **All three within ~0.02 s of each other** → it is a fixed time lag. That is
  correctable in software and I will do it: the correction is to shift each
  trace's position back along its own travel direction by `latency × speed`.
- **Latency falls as speed rises** (separation roughly constant in mm) → it is a
  geometric offset after all and the fix is a tooling calibration, not a
  timestamp.
- **Latency rises with speed** → something is queueing, and the trigger report's
  `max_correlation_wait_s` is the place to look.

Expect something near **0.13 s** on all three.

---

## 3. The projection radius — free, inside the first real scan

Nothing to set. The rule now derives the radius from the sampling cell rather
than the trace pitch alone. When you run a scan, the plan prints:

```
    projection         live, spot radius 0.87 mm (auto), 2048 px (auto)
                       8 trace(s) averaged per pixel over a 1.00 mm step x 0.30 mm line spacing
```

**Check that second line.** If it says a number much larger than 8, the line
spacing did not reach the planner and I want to know.

Measured on your `results19` data, reprojected offline:

| radius | notch SNR |
|---|---|
| 1.60 mm (what you flew) | 15.0 / 13.6 |
| 0.87 mm (what it does now) | 33.1 / 26.2 |
| 0.70 mm | 39.4 / 25.4 |

You can push further by hand if you want to see it:

```
PROJECTION_SPOT_RADIUS = 0.0007
```

— but that is a bare assignment at the prompt and **will not take**, the same
trap as `EC_PRELOAD_M`. The supported way is the settings panel, or pass it to
the plan. Leave it on auto unless you are deliberately experimenting; 0.87 mm is
within noise of the best value measured.

---

## 4. Keyence waviness — does it belong to the part or to the scan? (~15 min)

### What we know

The fine meshes are flat to **18–21 µm rms** after removing the plate's own tilt.
That is a good surface measurement. The problem is their **slope**: the normals
wobble by p99 **1.1–1.3°**, and the probe axis is set to the mesh normal at every
waypoint. At the measured lift-off sensitivity that is **~0.9 V of signal at
rotation 0**, against notches that read 0.10–0.62 V there.

Two meshes from the same session ripple in **different directions** — the 13:45
one bands along x, the 14:36 one streaks along y. That hints the ripple belongs
to the scan rather than the part, but they also covered different regions, so it
is a hint and not a result.

This experiment settles it: raster the **same** patch twice with the rows turned
90°, and see whether the ripple turns with them.

### 4.1 Pick the region and get a rough surface

Put the probe over the flat area you want to test — a patch with no notches is
best, so nothing real is in the picture.

```
detect_part()
reconstruct_rough()
```

`reconstruct_rough()` flies the Azure and leaves the rough surface in
`rough_mesh`. If detection misses, give the region explicitly instead:

```
reconstruct_rough(region=(xmin, ymin, xmax, ymax))
```

(metres, world frame, in that order — the order has cost us 696 mm before.)

### 4.2 Fly the same patch twice, rows 90° apart

```
import math
a = reconstruct_fine(rotation_offset=0.0)
b = reconstruct_fine(rotation_offset=math.pi/2)
print(a); print(b)
```

Each call prints and returns the mesh path, in its own timestamped directory
under `~/tari/scandata/`. **Write both paths down**; the script below takes them
as arguments.

`rotation_offset=` pins the raster axis — passing it explicitly stops the planner
choosing the axis for you, which it otherwise does on cable grounds.

If the second one refuses on cable or reach grounds, say so and stop; that is
itself the answer to "can we do this", and I will plan around it.

### 4.3 Compare them

Save as `/tmp/ripple.py` on the cell:

```python
import numpy as np, sys

def read_ply(path):
    with open(path, "rb") as fh:
        hdr = b""
        while b"end_header" not in hdr: hdr += fh.readline()
        lines = hdr.decode("ascii", "replace").splitlines()
        nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        props = [(l.split()[-1], l.split()[1]) for l in lines
                 if l.startswith("property ") and "list" not in l]
        dt = np.dtype([(p, {"double": "<f8", "float": "<f4", "uchar": "u1",
                            "int": "<i4", "uint": "<u4"}[t]) for p, t in props])
        return np.frombuffer(fh.read(nv*dt.itemsize), dtype=dt, count=nv)

def report(path, label):
    v = read_ply(path)
    P = np.c_[v["x"], v["y"], v["z"]]
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    c, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    r = (P[:, 2] - A.dot(c))*1e3
    X, Y = (P[:, 0] - P[:, 0].min())*1e3, (P[:, 1] - P[:, 1].min())*1e3
    out = {"label": label, "n": len(P), "rms": r.std()}
    for name, axis in (("x", X), ("y", Y)):
        nb = 80
        e = np.linspace(axis.min(), axis.max(), nb + 1)
        k = np.clip(np.digitize(axis, e) - 1, 0, nb - 1)
        prof = np.array([r[k == i].mean() if (k == i).any() else np.nan
                         for i in range(nb)])
        g = np.isfinite(prof)
        y = prof[g] - prof[g].mean()
        xg = 0.5*(e[1:] + e[:-1])[g]
        f = np.fft.rfftfreq(len(y), d=np.median(np.diff(xg)))
        amp = np.abs(np.fft.rfft(y*np.hanning(len(y))))
        kk = int(np.argmax(amp[1:])) + 1
        out[name] = (1.0/f[kk] if f[kk] > 0 else float("inf"), 2*amp[kk]/len(y))
    if "nx" in v.dtype.names:
        N = np.c_[v["nx"], v["ny"], v["nz"]]
        N = N/np.linalg.norm(N, axis=1, keepdims=True)
        t = np.degrees(np.arccos(np.clip(np.abs(N[:, 2]), 0, 1)))
        out["tilt_p99"] = float(np.percentile(t, 99))
    return out

for path, label in ((sys.argv[1], "rows at 0 deg"), (sys.argv[2], "rows at 90 deg")):
    o = report(path, label)
    print("%-14s  n=%-6d  flatness %.3f mm rms   normal tilt p99 %.2f deg"
          % (o["label"], o["n"], o["rms"], o.get("tilt_p99", float("nan"))))
    for ax in ("x", "y"):
        print("                 ripple along %s: period %6.1f mm, amplitude %.4f mm"
              % (ax, o[ax][0], o[ax][1]))
```

```bash
python3 /tmp/ripple.py <path printed by a> <path printed by b>
```

### 4.4 What the answer means

**Write down the two ripple directions and amplitudes.**

- **The dominant ripple axis swaps between the two meshes** → it is an artefact
  of the Keyence sweep, and the fix is to combine the two scans (or to stop
  following per-vertex normals on a part we know is flat). This is the outcome I
  expect.
- **The ripple stays on the same axis in both** → it is 20 µm of real relief on a
  surface-ground 7075 block, which would be surprising and worth knowing, and the
  answer is to keep the mesh and live with a 1° normal wobble.
- **Amplitudes differ by more than ~2×** → one sweep direction is better than the
  other, which is itself worth having.

### 4.5 And while you are here — about that standoff drift

On the 14:35 sweep the median standoff ramped **81.4 → 81.9 → 82.4 → 83.7 →
83.9 mm** across five rows while each row was internally clean to 0.3–0.7 mm. The
between-row variation was four times the within-row variation.

That is **not a Keyence fault**. The raster was planned over the *Azure* rough
mesh at a commanded 80 mm, so the measured standoff is the disagreement between
the Azure surface and the real plate. 2.5 mm over ~80 mm of raster is about
**1.8° of tilt error in the Azure mesh** — and the 12:38 run shows the same thing
at the same magnitude (+3.4 mm over 90 mm, 2.2°).

It is a different scale from the waviness above (2.5 mm versus 20 µm) and a
different cause. It did no harm this time because the profiler stayed well inside
its 57–103 mm window. It is worth a line in your notes each run: if the ramp ever
gets to where a row leaves the window, the mesh is what to fix, not the margin.

---

## 5. The large notches at rotation 0 — the hard case

### Why this is hard, in numbers

Measured on your own tilt ladders from `results19`:

| | rotation 0 | rotation 90 |
|---|---|---|
| lift-off sensitivity to tilt | **0.78 V/deg** | **0.08 V/deg** |
| ±6° tilt | **rails at exactly 5.000 V for 95–98% of the trace** | no saturation |
| the three notches on the taught line | 0.10–0.62 V p-p | **1.5–2.8 V p-p** |
| mesh ripple (p99 1.2°) contributes | ~0.9 V | ~0.10 V |

At rotation 0 the artefact is 2–10× the flaw. At rotation 90 the flaw wins by
15–30×. The 0.040/0.060/0.080" notches run parallel to the long edge, which is
what forces rotation 0 on you — that is the unlucky part, not a mistake.

### 5.1 Measure before you commit a scan to it

Teach a line across the large notches, then:

```
set_probe_rotation(0)
capture_contact(points=("contact_a","contact_b"), tilts_deg=(0, 1, -1, 2, -2), rotations_deg=(0.0,), speed_mm_s=10)
```

**Note the ±1 and ±2° ladder instead of ±3 and ±6.** At 0.78 V/deg, ±6° is past
full scale and the trace is clipped — four of your ladders last time were
unusable for that reason. ±2° gives 1.6 V, comfortably inside the rail, and a
clean sensitivity slope.

Then look at the trace:

```
import numpy as np
z = np.load("<the ec_rot+0_tilt+0.npz the take printed>", allow_pickle=True)
print("peak-to-peak %.3f V" % np.ptp(z["impedance"].imag))
```

- **Over about 0.5 V p-p** → there is a response to find; fly the scan.
- **Under ~0.2 V p-p** → the scan cannot show what the contact line cannot, and
  you have saved 25 minutes. Do section 5.2 instead.

### 5.2 If rotation 0 shows nothing: take the mesh normal out of it

The probe is spring-loaded and conforms to the surface anyway, so following
1.2° of mesh ripple with the commanded tool axis buys nothing and costs ~0.9 V.
Planning over a *flattened* mesh would remove that. We do not have a one-line
switch for it yet — tell me what section 4 says and I will add one, because the
right form of it depends on whether the ripple is real.

### 5.3 If it does show a response, fly it

```
run_automated_scan(line_spacing=0.0003)
```

Check the plan before confirming:

- `probe rotation: +0.0 deg ... wrist 3 at +166.1 deg` — in band
- `preload: 2.00 mm past the measured surface` — intended, see below
- `8 trace(s) averaged per pixel` — the projection fix is live
- `projection surface unpinched` or nothing — the mesh gate

---

## 6. Preload — no action, just so it is written down

`EC_PRELOAD_CEILING_M` is 2 mm and running at or near it is the **intended**
operating point. The description's two numbers (4.5 mm of plunger overlap, the
probe mesh ending 0.25 mm past `probe_end`) understate the real retract travel,
because the plunger is a fixed joint in the URDF and the model cannot show
travel it does not have. Against that, the measured surface has its own error of
up to half a millimetre, and at zero preload all of it becomes lift-off, one for
one. Lift-off loses the measurement; extra preload costs clearance margin that
the planner has already been told about.

What it costs, so it is a decision and not a surprise: past 0.25 mm the modelled
probe body is inside the part, so the clearance the gate reports **for
`ec_probe`** is optimistic by the preload — it read −4.7 mm at 2.00 mm on the
10-05 scan, and contact for `ec_probe` and `probe_holder` is declared to the
planner, so nothing is refused by it. Every other link is still checked honestly.

The thing that would genuinely fix this one day is a prismatic joint for the
plunger in `tari_ur_description`, so the planner can model the retract instead of
the pipeline working around its absence. Not needed now.

---

## 7. What to bring back

1. The three latencies from 2.6, and whether they agree.
2. The two ripple directions and amplitudes from 4.4.
3. The rotation-0 peak-to-peak from 5.1.
4. Any `scandata/` directory from a scan that ran, plus `dataguzzler.log` and
   `ros.log`.
5. Anything that refused with a message you did not expect — the messages are
   written to be diagnostic and a surprising one is a bug in the message.
