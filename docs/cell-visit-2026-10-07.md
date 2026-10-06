# Cell visit, Wednesday 2026-10-07

Written 2026-10-06 from `results20`. The previous plan asked for two
measurements and got neither, for reasons that were mostly defects in this
pipeline rather than in the cell. Those are fixed. **This plan has one
objective — a working scan over the 0.040/0.060/0.080" notches — and everything
else is either free or optional.**

Read section 0 before anything else: one of yesterday's recommendations is
withdrawn, and acting on it would cost you a morning.

---

## 0. Corrections to yesterday's plan

**Do NOT re-measure the cable envelope.** Yesterday's emergency stop looked like
the wrist band being optimistic at +263°, and a `capture_cable_envelope(
full_circle=True)` was about to be recommended. It was **the Azure Kinect's power
cable about to pull out**, which has nothing to do with wrist_3 travel. The band
stands. (The Azure cable is its own hazard — the planner has no model of it, so
no clearance check or preview can see it. Your item, noted here so it is written
down.)

**The latency ladder cannot be run from a taught line.** On the larger-notch
region, repeats of the *same* line at the *same* speed and direction put the
response up to 8 mm apart, while the arm repeats the path to 0.2 mm and trace
spacing is exactly 1.000 mm. The floor is several times the effect. Section 2
measures it from a raster instead, which is what you said.

**I have no explanation for that scatter.** The first one offered was the sprung
probe sticking; you were watching and saw no dragging, so it is withdrawn rather
than kept as a leading theory.

**±π/2 is not a sign choice for a raster.** A raster axis is a line direction, so
`rotation_offset=-math.pi/2` and `+math.pi/2` give identical rows. Your 0-vs-90
comparison was still valid.

---

## 1. What changed, and the lines that prove it

Five fixes. All are on `main`, so `tari.bash` brings them.

| change | what you will see |
|---|---|
| **The projection surface is proved loadable at PLAN time** | a new line during planning (below) — and a refusal there instead of after the null |
| The pinched-vertex repair runs *after* the winding repair | the repair now fires on the meshes that were killing runs |
| The preview says what the arm does | `wrist 3 +173 -> +263 deg, 38 deg from the cable limit; largest joint 90 deg; the TOOL turns 90 deg` |
| The near-part speed clamp announces itself | `20.0 mm/s was asked for; 10.0 mm/s is the near-part limit` |
| `capture_contact()` declares its own specimen contact | `contact with 'specimen' is now ALLOWED` at the start of a take |

### 1.1 Bring up

```bash
cd ~/tari && ./tari.bash --layer site
source ~/tari/tari_setup.bash
ros2 launch tari_bringup bringup.launch.py cell:=ex-ur10e-1 mode:=hardware
```

Second terminal:

```bash
source ~/tari/tari_setup.bash
cd ~/tari/ROS-Scanning && dataguzzler-python --ipython dgpfiles/ECAutomatedScan.dgp
```

### 1.2 The line that matters

During planning, after `parameterizing … -> …_uv.x3d`, you should now see **one
of**:

```
  projection surface checked at plan time: ..._uv.x3d loads
```

or, if the mesh needed work, the repair lines followed by that same `loads`, or
a refusal naming what is wrong. **Any of the three is the gate working.** What
must not happen any more is reaching `nulling the INSTRUMENT` and *then* failing.

If it refuses at plan time: the arm has not moved and nothing is lost. Re-run
`reconstruct_fine()` and plan again, and send me the refusal text.

---

## 2. The scan over the larger notches (~40 min) — the objective

### 2.1 Set up

```
set_probe_rotation(0)
ec_preload()
```

Rotation 0 because those notches run parallel to the long edge. Preload should
read 1–2 mm; set it with `set_ec_preload(0.002)` if it does not.

### 2.2 Run it

```
run_automated_scan(line_spacing=0.0003)
```

Check these in the plan before confirming:

| line | expect |
|---|---|
| `probe rotation: +0.0 deg … wrist 3 at +166.1 deg` | in band |
| `8 trace(s) averaged per pixel over a 1.00 mm step x 0.30 mm line spacing` | the projection fix is live |
| `projection surface checked at plan time: … loads` | section 1.2 |
| `preload: 2.00 mm past the measured surface` | intended |

### 2.3 Write down the achieved speed, not the commanded one

**New, and it needs an eye.** The `results19` scan was commanded at 10 mm/s and
flew at **3.9 mm/s** — every pass, consistently: 77 mm of travel in 19.7 s with
78 traces spread evenly across it, so it is the arm that was slow, not overhead
at the ends. The plan estimated 24 minutes on the strength of 10 mm/s.

Each pass prints `commanded 0.095 m / observed 0.095 m` and the elapsed time.
**Note the elapsed time for a few passes.** If it is again ~2.5× the estimate,
the velocity calibration is off for this path class and that is worth a separate
look — it also decides whether section 3 can get two distinguishable speeds.

### 2.4 If the notches do not appear

They may not, and the reason is measured rather than speculative: at rotation 0
the lift-off sensitivity to tilt is **0.78 V/deg**, and these fine meshes carry a
p99 normal tilt of **2.5–2.9°**, so mesh noise alone contributes about **2 V**.
On the *small* notches at rotation 0 the response was 0.10–0.62 V. The larger
notches are 2–16× deeper so they may well win — but if the image is a wash, that
arithmetic is why, and the fix is to stop following per-vertex normals on a part
we know is flat to 25 µm rms. Tell me and I will add the flat-normal planning
option; the shape of it depends on what this scan shows.

---

## 3. The acquisition lag — free, if section 2 ran (~25 min)

One number is missing and one raster supplies it.

From `results19`: at **3.93 mm/s** the same notch lands **1.10 mm** apart
depending on travel direction — 0.55 mm each way, **0.140 s**. One speed cannot
separate a time lag from a fixed distance offset, and that distinction decides
whether it is correctable in software.

So: **run one more raster over the same notch at a different speed.**

```
set_scan_velocity(0.004)
run_automated_scan(line_spacing=0.0003)
```

**Slower, not faster, and deliberately.** Section 2's scan was commanded at 10
and flew at 3.9, so asking for 20 may well produce 3.9 again and the whole
25 minutes buys nothing. Commanding 4 mm/s is a speed the arm has demonstrably
delivered, and if it then flies at ~2 you have two speeds that differ by a real
factor. Read the per-pass elapsed time, not the setting.

`set_scan_velocity()` is new and **unproven at the cell** — a bare
`SCAN_VELOCITY = 0.004` at the prompt is the same trap as `EC_PRELOAD_M`, which
is why it exists, but its reader lives in a different module from the two
setters that are proven. So check it rather than trusting it, which costs five
seconds:

```
set_scan_velocity(0.004)
scan_velocity()
```

then **read the `velocity` line in the printed scan plan before confirming the
run**. If the plan still says 10.0 mm/s, the setting did not reach the planner —
tell me, and use the scan settings panel instead, which writes it the way the
GUI always has.

**Stop and tell me rather than spending the time if:** the achieved speed in
section 2.3 was close to the commanded 10 mm/s after all (then ask for 20
instead), or if this run also lands near 3.9 (then the arm is doing the same
thing whatever it is asked, and the velocity question has to be answered before
the lag one).

I will extract the lag from the saved `pass*.npz`; nothing to analyse at the
cell. **Bring back the `ec_scan_*` directory and the log.**

---

## 4. Optional, if there is time

**RViz memory.** The session ground to a halt yesterday. Your yolo hypothesis
does not match the log — I see three `yolo enabled` / four `yolo disabled` pairs,
so it is being released after each detection. To find the real one, in a third
terminal before you start:

```bash
( while true; do date +%s; ps -o rss=,comm= -C rviz2 -C gzserver -C move_group -C python3 --sort=-rss | head -8; sleep 60; done ) > ~/mem.log 2>&1 &
```

Kill it at the end and bring `~/mem.log` back. One number per process per minute
is enough to say which one grows.

**Keyence comb return leg.** You noticed the return movement accumulating points
while moving faster. I have not looked at it yet. If it is easy to confirm — does
`/stop_recording` land before or after the return? — that would tell me where to
start.

---

## 5. What to bring back

1. The planning-stage surface line (section 1.2), whatever it said.
2. The achieved pass times from section 2.3.
3. The `ec_scan_*` directories, `dataguzzler.log`, `ros.log`.
4. `~/mem.log` if you ran section 4.
5. Any refusal whose wording did not tell you what to do — those are bugs in the
   message and I would rather fix them than have you decode them.
