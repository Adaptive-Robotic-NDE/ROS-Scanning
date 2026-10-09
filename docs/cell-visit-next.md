# Cell visit, next session

Written 2026-10-08 from `results21`. **Supersedes `cell-visit-2026-10-07.md`**,
whose section 2 assumed a serpentine raster and no axis smoothing; both of those
have changed and the old numbers in it no longer describe what the arm will do.

Undated because I do not know when you are next at the cell. Nothing here goes
stale with time — only with the next run.

`results21` answered three questions and the answers are in the code now. **This
visit is mostly about confirming that on hardware**, because every one of those
fixes was derived from saved files and none has flown.

---

## 0. What changed, and how much to trust it

| change | evidence behind it | flown? |
|---|---|---|
| **Comb instead of serpentine** | per-pass baseline vs travel direction r = +0.952, a 0.502 V step on a 1.46 V range | **no** |
| **Tool-axis turn rate bounded and smoothed** | pass speed vs turn rate r = −0.855; 3.120 → 0.350 deg/mm on your own path | **no** |
| Projection surface proved loadable at plan time | three runs lost to it after the null | **no** |
| Projection spot radius from both spacings | notch SNR 15.0 → 33.1 reprojected | yes, 10-08 |
| `capture_contact()` declares its own contact | the 14–86% truncations | partly |

**The two in bold are the ones to watch.** They are the largest behavioural
changes this pipeline has had in a fortnight and both were tuned against one
scan's worth of saved data.

---

## 1. Bring up

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

Confirm the settings before anything moves:

```
ec_serpentine()
probe_rotation()
ec_preload()
scan_velocity()
scan_step_size()
projection_mode()
```

`ec_serpentine()` must say **comb**. If it says serpentine, `tari.bash` did not
pick up the change and nothing below is testing what it claims to.

---

## 2. The scan over the 0.040/0.060/0.080" notches (~25-35 min)

Same objective as last time, with two changes: a comb pays for the return, and
the sampling pitch drops from 1.00 mm to 0.25 mm because 1.00 mm cannot resolve
a notch signature (2.1). Deferred projection roughly cancels the cost of the
second (2.2).

### 2.1 Set up

```
set_probe_rotation(0)
set_ec_preload(0.002)
set_scan_step_size(0.00025)
```

**The step size must be 0.25 mm, not 1.00 mm.** This is new since the last plan
and it is the difference between finding the notch and reading it.

The 10-08 scan found and correctly located the 0.080" notch — feature 1, at
(11, 9) mm in the patch — but the response is **≤1.0 mm FWHM along the pass**
and the pitch was **1.00 mm**, so the whole feature got one sample above
half-maximum. You watched a clean figure of eight on the instrument; the
recorded data has the same diagonal (+114°, upper-left to lower-right, matching
your instrument's rotation) but as a line, because a loop cannot be drawn
through two points. The direction survives undersampling, the shape does not.

`set_scan_step_size()` is new, and prints the speed ceiling the new pitch
implies. Metres, like `set_ec_preload()`.

### 2.2 Choose live or deferred, then run

Four times the resolution is not four times the time, and **deferred projection
buys most of it back** — the staging ceiling is 22 traces/s instead of 10:

| pitch | live | deferred |
|---|---|---|
| 1.00 mm *(what you flew)* | 10.0 mm/s, 11.6 min | 22.0 mm/s, 9.4 min |
| **0.25 mm** | 2.5 mm/s, **23.8 min** | 5.5 mm/s, **14.9 min** |

7.6 min of each of those is per-pass overhead that the pitch does not change.
**0.25 mm deferred costs about what 1.00 mm live did.** The trade is that the
image is built at the end rather than as it goes, so you lose the live view —
which is worth it once, to see whether the figure of eight comes through.

```
set_projection_mode("deferred")
out = run_automated_scan(line_spacing=0.0003)
```

Capture the return — section 3 needs it. Skip the first line if you would
rather watch the image build and can spare the extra nine minutes.

**One caveat on the deferred numbers.** 22 traces/s is a *staging* ceiling, not
an arm speed. At 0.25 mm it asks for 5.5 mm/s, and on 10-05 the arm delivered
3.9 mm/s when commanded 10 — so the arm may bind first and the real time may
land nearer the live column. Section 2.4's elapsed times will say.

### 2.3 Four lines to read in the plan, before confirming

| line | what it should say | if it doesn't |
|---|---|---|
| `projection surface checked at plan time: … loads` | the mesh gate ran **here**, not after the null | send me the refusal; the arm has not moved |
| `tool axis smoothed (linear, N mm): turn rate X -> Y deg/mm` | **Y under 0.40** | see 2.5 |
| `8 trace(s) averaged per pixel over a 1.00 mm step x 0.30 mm line spacing` | the projection fix is live | — |
| `probe rotation: +0.0 deg … wrist 3 at +166.1 deg` | in band | — |

You will also see the normal postcondition read something like **1.9 deg**
instead of the `0.000` it used to. **That is expected and the next line explains
it**: the postcondition measures the probe axis against the *mesh* normal, and
the smoothing departs from the mesh normal deliberately, because the mesh normal
is the noisy quantity. It warns; it does not refuse.

### 2.4 Watch for the stall — this is the real test

On 2026-10-08, passes 18–27 crawled at **1.74–2.98 mm/s** against a **4.68
median**, and you watched the arm rock. The turn-rate repair is supposed to have
removed the cause.

**Note the elapsed time on a dozen passes**, especially any where the arm looks
unhappy. Three outcomes:

- **No slow band, no rocking** → the diagnosis and the fix both hold. This is
  what I expect.
- **Still slow, but the plan reported a turn rate under 0.4** → the turn rate
  was not the whole story. Tell me; that is a real and useful negative, and it
  means something else is loading the joints in that region.
- **Slow and the plan reported `still over the … bound`** → read which of the
  two messages it printed. "A wider window stopped reducing it" means the
  surface genuinely turns that fast there; "the mesh is noisier than this can
  fix" means re-run `reconstruct_fine()`.

### 2.5 If the smoothing says it could not meet the bound

Nothing is refused — the scan runs either way. The message distinguishes the two
causes. Do not raise `EC_AXIS_SMOOTHING_MAX_M` to force it under: past 20 mm the
axis is being averaged over a distance comparable to real features, and on a
genuinely sharp part that would flatten the part to satisfy a number.

---

## 3. Condition the image — do not skip this (~2 min, at the prompt)

`results21` ran with **conditioning rotation 0.0**, which is no lift-off
rejection at all. `EC_AUTO_ROTATION` *measures* the rotation after the scan but
does not re-project, so the image you looked at was unconditioned.

After the scan finishes:

```
out["report"].get("rotation_measured_deg")
```

Then re-project with it:

```
replay_ec_scan(rotation_deg=out["report"]["rotation_measured_deg"])
```

(If `rotation_measured_deg` is `None`, the measurement failed and
`rotation_error` says why — send me that; the traces are unaffected and on
disk.)

**Write down the measured rotation.** On the 10-08 data the direction-dependent
lift-off sat at +83.8° in the impedance plane and the notch features at +96.0°
— **12.2° apart**, which is close to the 8.5° the simulation table predicted and
is the number that decides how much of the flaw survives conditioning. A second
measurement of it is worth as much as the scan.

---

## 4. The acquisition lag — still one speed, still open (~25 min, optional)

Unchanged from the last plan and still the oldest open question: **0.140 s at
3.93 mm/s, from one scan.** One speed cannot separate a time lag from a fixed
distance offset.

It needs a second raster at a different speed, and the contact-line route is
dead — on the larger-notch region, repeats of one line at one speed and
direction disagreed by up to 8 mm.

```
set_scan_velocity(0.004)
scan_velocity()
out2 = run_automated_scan(line_spacing=0.0003)
```

**Check the printed plan's `velocity` line before confirming.** If it still says
10.0 mm/s the setter did not reach the planner — tell me and use the settings
panel. `set_scan_velocity()` is still unproven at the cell.

Slower rather than faster, deliberately: the 10-05 scan was commanded at 10 mm/s
and flew at **3.9**, so asking for 20 would probably give 3.9 again. If this run
also lands near 3.9 whatever is asked, stop — the velocity question has to be
answered before the lag one.

I extract the lag from the saved `pass*.npz`; nothing to analyse at the cell.

---

## 5. Optional

**RViz memory.** The 10-06 session ground to a halt. Your yolo hypothesis does
not match the log (three `yolo enabled` / four `yolo disabled` pairs — it is
being released). Before you start:

```bash
( while true; do date +%s; ps -o rss=,comm= -C rviz2 -C gzserver -C move_group -C python3 --sort=-rss | head -8; sleep 60; done ) > ~/mem.log 2>&1 &
```

Kill it at the end and bring `~/mem.log` back.

**Keyence comb return leg.** You noticed the return accumulating points while
moving faster. Still not investigated. If it is easy to see whether
`/stop_recording` lands before or after the return, that tells me where to look.

---

## 6. What to bring back

1. The four plan lines from 2.3, verbatim, whatever they said.
2. Elapsed times for a dozen passes, and whether the arm rocked anywhere.
3. The measured conditioning rotation from section 3.
4. The `ec_scan_*` directories, `dataguzzler.log`, `ros.log`.
5. `~/mem.log` if you ran section 5.

**And please keep `dataguzzler.log` and `ros.log` separate if you can** — last
time they were interleaved into one file. It cost nothing on 10-08 because I
worked from the `.npz` files, but the ROS half is where the joint-level evidence
would be if the stall comes back.

---

## 7. If the stall comes back

The one thing I could not get from `results21` is what the *joints* were doing,
because the EC trigger fires on distance: when the arm stops advancing there are
no traces, so a stall is a region with no samples and the rocking is unsampled
by construction. The EC scan path also records no bag and, unlike a capture
take, runs no in-flight speed monitor.

So if it recurs, the next measurement is instrumentation rather than analysis:

```bash
ros2 bag record -o ~/stall_bag /joint_states /tf /speed_scaling_state_broadcaster/speed_scaling
```

started before the scan and stopped after. At 500 Hz that makes the rocking
directly visible — amplitude, frequency, which joints move, and whether the UR's
own speed scaling is pulling it down. That is a ~2 GB bag for a 50-minute scan,
so only run it if the stall is still there.
