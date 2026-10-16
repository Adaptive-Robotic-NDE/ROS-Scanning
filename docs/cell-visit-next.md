# Cell visit, next session

Written 2026-10-15 from `results24`. **Supersedes the 2026-10-10 version**,
whose section 1 was the Azure calibration repair — that is done and verified,
and nothing of it is left to do.

Undated because I do not know when you are next at the cell. Nothing here goes
stale with time, only with the next run.

**Section 1 is not about the robot and should happen before you are at the
cell at all.**

---

## 0. What 10-15 settled, and what it did not

| | |
|---|---|
| **Azure calibration** | SOLVED. 0.822 mm rms over 23 of 25 poses, 1.18 mm from the 09-29 solve, deployed, and `check_calibration_applied()` says APPLIED to 0.000 mm. It is in the cell package now (§5). |
| **The mesh** | Clean. 0 non-manifold edges, 3 faces rewound, 0 of 1177 normals repaired, normals turning 0.72° at p99.9, 100% of the ROI inside the work volume. |
| **The toolpath** | Clean. 126 passes, turn rate 0.7 deg/mm at p99, 0 degenerate steps, 0 waypoints facing away, clearance gave up nothing. The 10-09 failures are all gone. |
| **Pass times** | 34.0 s, dead steady, for passes 4 through 10 — against 50 s on 10-09 for a comparable pass. The turn-rate work held. |
| **The correlator** | Healthy. 2791 globalrev waits, worst 99 ms, **0 staging timeouts** over 31 passes. The 10-09 freeze mechanism did not recur. |
| **THE STALL** | **NOT SOLVED.** Different place, and now localised. §2. |
| The planning pause | Explained and fixed. §4. |
| The machine | Throwing a corrected PCIe error every six seconds, all day. §1. |

---

## 1. BEFORE THE CELL: the PC itself

Two things were wrong with the computer on 10-15, independent of any of our
code, and both of them are in the path of a real-time scan. Neither needs the
robot, so do them first and separately.

### 1.1 The NVMe link is erroring continuously

`dmesg.log` from that session holds **943 corrected PCIe receiver errors** on

```
nvme 0000:89:00.0    vendor 0x1dbe  device 0x5220  class 010802
```

— `RxErr`, Physical Layer, Receiver ID — spread evenly from 17:27 to 18:54 at
between 3 and 30 a minute, with no gap anywhere. 312 of them came up through
APEI/GHES, which means the firmware took an SMI for each one, and an SMI
freezes every core while it runs. They are *corrected*, so nothing is
mis-reading data; what they cost is retries and latency, and they were the
only thing in that kernel log in any quantity. They had pushed the ring buffer
down to 86 minutes of history.

```bash
# which device, and is it still counting
for d in /sys/bus/pci/devices/*/aer_dev_correctable; do
  grep -qvE ' 0$' "$d" && { echo "== ${d%/*}"; cat "$d"; }
done
lspci -s 89:00.0 -vv | grep -iE 'LnkSta|LnkCap|Correctable'
```

What to try, roughly in order of cost: reseat the drive, try a different M.2
socket, check the slot is not sharing lanes with something newly installed,
and look at whether the link has fallen below its capable width or speed
(`LnkSta` vs `LnkCap`). If it is a riser or a cable, that is the usual answer.
If it will not come clean, **move `/data/scandata` to a different disk** —
it is one path and nothing else depends on where it is.

### 1.2 The antivirus on-demand scan

It was running during the scan, consuming "considerable CPU" by your account,
over the same tree the scan writes to.

Two separate costs, and the second is the one that can stall a scan outright:

* **CPU.** The clearance gate runs 48 threads and the trace staging has a
  10 traces/s budget it has to meet trace by trace. Anything competing eats
  directly into both.
* **On-access hooks.** A scanner that registers fanotify *permission* events
  intercepts file opens and page faults and holds them until it decides. That
  wait is in the kernel, uninterruptible, and invisible to Python — exactly
  the shape of what §2 describes, and it applies to `/dev/shm` as much as to a
  real filesystem, because tmpfs is a filesystem.

So: **exclude `/data/scandata`, `/dev/shm` and the ROS/dgpy install trees from
on-access scanning, and schedule on-demand scans outside scanning hours.** If
the product cannot be configured that way, stopping its service for the
duration of a run is a legitimate answer and worth proving once.

---

## 2. THE STALL — what we now know, and the one thing to do when it happens

### 2.1 It is not where it was

It stalled at **pass 32 of 74**, 18 minutes in, and printed:

```
** STAGING STALLED: still waiting after 120 s -- 37 trace(s) queued,
   2845 staged, 0 error(s)
   recdb: spatialnde2 recording database latest ready/defined = 30074/30074
   no staging error has been recorded, so this is a stage that has not
   returned rather than one that failed.
```

Read against the per-pass metadata, that narrows a long way:

* **It is the trace STAGER, not the correlator.** `trigger_report` for pass 30
  says `staging_timeouts: 0`, `max_consecutive_staging_timeouts: 0`,
  `max_correlation_wait_s: 0.099`, `errors: 0`, `fault: None`. The 10-09
  mechanism — a correlator waiting forever on a globalrev — is bounded now and
  never fired. Something inside `stage_ec_reading()` did not come back.
* **It had not got as far as committing.** `ready == defined == 30074`. A
  transaction that had been ended would have pushed `defined` past `ready`. So
  it was parked *before* `end_transaction()`, which rules out the globalrev
  wait on line 907 and points at `start_transaction()`, the storage
  allocation, or the array write.
* **0 errors**, so it is not an exception retained mid-transaction. That trap
  is fixed and it stayed fixed.
* **The cost per trace was already creeping.** Passes 4–10 took 34.0 s each,
  to within 0.4 s. Pass 11 took **45.9 s**. Passes 12 onward then settled into
  a slow climb — 35.5, 36.3, 36.4, 37.7 — and never came back to 34. That is
  something growing with the number of traces already staged, and it is the
  same shape as the "minor slowdowns" you saw.

### 2.2 Ctrl-C cannot get you out, and that is not a surprise any more

You pressed it three times and then fell back to `^Z` and `kill %1`. That is
the only thing that works. dataguzzler-python runs the GUI loop on the
process's **real** main thread and the console in another one
(`main_thread.py`), so SIGINT is delivered to a thread sitting inside Qt's C++
event loop with the GIL released: the Python handler never runs and no
`KeyboardInterrupt` is ever raised, anywhere. The stall note now says this
instead of letting you find out.

`^Z` then `kill %1` costs **the pass in flight and nothing else** — every
completed pass is already written to `ec_scan_<time>/passNNN.npz`, and 31 of
them were, which is why the notch image could still be made afterwards.

### 2.3 What to run when it stalls

```bash
ROS-Scanning/tools/capture_stall.sh
```

No arguments: it finds the dataguzzler-python process itself. It collects
everything non-invasive first and attaches gdb last, because attaching gdb
suspends the process — `--no-gdb` skips that step, and if it is interrupted
anyway the script sends SIGCONT on the way out. It writes a directory and a
tarball and prints what to read first. **Run it before killing anything.**

The stall note prints the exact command with the pid filled in.

Install `py-spy` first if you can — it reads Python stacks out of a live
process with no debugger and no cooperation, and it is the single most useful
thing the script can collect:

```bash
pip3 install --user py-spy
```

The script's own `threads.txt` is the discriminator even without it:

| the staging worker's `wchan` | what it means |
|---|---|
| `io_schedule`, `folio_wait_bit`, `wait_on_page_bit`, state `D` | **storage.** §1 is the cause, not our code |
| `fanotify_*`, and an AV daemon alive in `scanners.txt` | **the scanner is holding a page fault.** §1.2 |
| `futex_wait` / `rwsem` | a lock another thread holds — then `native.txt` says which |
| `do_sys_poll`, `ep_poll` | it is idle, and the fault is upstream of it |

### 2.4 What the process now tells you by itself

Three changes, all in the scan's own output, so the next stall is diagnosable
from `dataguzzler.log` alone even if nobody is at the keyboard:

* the note **repeats** every 120 s, and each repeat says whether anything was
  staged since the last one — a crawl and a death look identical in one note
  and obvious in two;
* each note prints the **staging worker's own Python stack**, which separates
  `start_transaction()` from the globalrev wait from the projection — the
  three candidates §2.1 leaves open;
* the note names `capture_stall.sh` with the pid, and says Ctrl-C will not
  work.

**If it stalls again, the stack in the log is the answer.** Bring back
`dataguzzler.log` and the `capture_stall.sh` tarball and that is enough.

---

## 3. The scan

Settings unchanged, and 10-15 confirmed they are right. Rotation 86.5° again
(83.8° on 10-08, 86.5° on 10-09, 86.5° set deliberately on 10-15 — it is a
property of the probe and the material, and a third agreeing number would let
it become a configured constant).

```
set_probe_rotation(0)
set_ec_preload(0.002)
set_scan_step_size(0.00025)
set_rotation(86.5)
```

```
ec_serpentine()
probe_rotation()
ec_preload()
scan_velocity()
scan_step_size()
projection_mode()
ec_rotation()
```

Then:

```
run_automated_scan(line_spacing=0.00025)
```

### 3.1 Note the pass times, in writing

This is still the measurement that matters most, and now there is a baseline
to compare against: **34.0 s for a 94-trace pass**, from passes 4 through 10
of the 10-15 run.

* **34 s and flat** → nothing has regressed and §2's creep is gone.
* **34 s climbing past 37** → the creep is real and reproducible, which is a
  result. Note the pass numbers where it steps.
* **One pass far longer than its neighbours** → that is pass 11's 45.9 s. Note
  when it happens and whether anything was printed around it.

### 3.2 Lines to read in the plan before confirming

| line | expect |
|---|---|
| `planning time` | now honest — a few minutes for a small region, not 3 s. §4 |
| `the traces are in shared memory while the scan runs` | new, and true. It used to claim "file-backed", which it was not |
| `tool axis smoothed … turn rate X -> Y deg/mm` | on 10-15 this said `0.844 -> 2.047` and warned "the mesh is noisier than this can fix" — and the scan still ran at 34 s a pass and produced a clean path. Note it; do not act on it |
| `IMPOSSIBLE PENETRATION` | absent |
| `coverage given up to keep the tool clear` | absent (10-15 gave up nothing) |
| `region_coverage` warning | 10-15 reported 83%. The ROI was drawn slightly wider than the fine sweep measured. Harmless; draw inside the swept area or sweep wider |

---

## 4. The planning pause is explained

The 10-15 run went silent for **231 seconds** between

```
[18:25:45] resampling to 1.0 mm: 3552 -> 1749 waypoints, 76 passes in 0.2 s
[18:29:36] applying the planning scene diff: starting
```

after a plan that had priced the whole of planning at **3 s**. It was the
tooling clearance gate, and it was two thresholds that cannot both be
satisfied:

* `enforce_path_clearance()` SAMPLES only above `TOOLING_GATE_SAMPLE_ABOVE`
  (2000) poses. 1749 is under it, so every pose was measured exactly, with
  nothing certified — about 80× the per-pose cost the estimate was calibrated
  on, which came from a run where 16.2% of poses were measured and the rest
  certified. 1749 × 0.132 s = 231 s. The arithmetic closes.
* `path_clearance()` turned its progress lines on above `4 × TOOLING_POSE_BLOCK`
  — 16384 poses. Nothing that takes the exact path can ever reach that, so the
  exact gate was silent **by construction**, at every length. And a path
  shorter than one block is one block, so it would have ticked once, at zero,
  even if the threshold had been met.

Both are fixed: progress is a pose count now (120), the block is divided so
there is something to tick between, and the estimate picks its rate from which
gate will actually run and says which. The counter-intuitive part is worth
knowing at the cell: **a smaller region can plan for far longer than a big
one**, because a big one gets the sampler.

Nothing was wrong with the gate's answer. It gave up no coverage and found no
collision.

---

## 5. The Keyence lateral correction was zeroed, and is back

`deploy_calibration()` writes a file **regenerated in full from the JSON
record**, and the Keyence's lateral pair is not in that record — it was solved
offline and written into the cell file by hand. So deploying the Azure
calibration replaced

```
keyence_mount_cal_xyz = 0.00056576 0.00172072 0
```

with `0 0 0`, and the fine Keyence mesh of 10-15 was reconstructed with the
head 1.8 mm from where it is. Nothing downstream could see it:
`check_calibration_applied()` iterates the record, so a sensor with no record
is not checked at all.

Two changes. `calibration_xacro()` now carries a correction it has no record
for forward out of TF, saying in the file where the number came from; and the
cell package's own `sensor_calibration.urdf.xacro` carries the new Azure
numbers **and** the Keyence block, with the reasoning for both.

**So the file in `tari_cell_ex_ur10e_1/urdf/` is correct and complete now.**
It is not what the cell is running: the cell is running the generated file
that `deploy_calibration()` copied there on 10-15, which has the Keyence
zeroed. Put the repository's version back:

```bash
cd ~/tari && git -C ros_ws/src/tari_site pull
git -C ros_ws/src/tari_site diff --stat         # expect the one urdf file
cd ros_ws && colcon build --packages-select tari_cell_ex_ur10e_1
# restart the bring-up, then in dgpy:
```

```
check_calibration_applied()
```

Expect `azure APPLIED, off by 0.000 mm`. Then confirm the Keyence came with
it — it has no record, so nothing checks it for you:

```bash
ros2 run tf2_ros tf2_echo keyence_metal lj_v7080_mount_cal
```

Expect translation `[0.00057, 0.00172, 0.000]`. Zeros mean the generated file
is still in force.

### 5.1 And the `--` is gone

The generated xacro's header said `Do not hand-edit -- re-run the
calibration`. XML forbids `--` inside a comment, so **every file that
generator has ever written was malformed**, and deploying it took the whole
robot description down — which is why 10-15 cost a vim session, a rebuild and
a second restart of the bring-up. The generator now rewrites hyphens in every
comment it emits and parses the finished file before returning it, so a
malformed one cannot leave the function.

---

## 6. Still open, unchanged

**The acquisition lag.** 0.140 s at 3.93 mm/s, from one scan, one speed. One
speed cannot separate a time lag from a fixed distance offset. Worth 25
minutes only once §2 is closed:

```
set_scan_velocity(0.004)
scan_velocity()
out2 = run_automated_scan(line_spacing=0.0003)
```

**The rare spatialnde2 background-thread interpreter crash.** Two sightings,
never reproduced in ~35 attempts. `capture_stall.sh` is not the tool for it;
a core is. It did not happen on 10-15.

**The Keyence comb return leg.** Still not investigated.

---

## 7. What to bring back

1. Whether `check_calibration_applied()` says APPLIED, and what
   `tf2_echo keyence_metal lj_v7080_mount_cal` prints (§5).
2. Elapsed time for every pass, or at least the first fifteen and any that
   stood out (§3.1).
3. **If it stalls: the `capture_stall.sh` tarball, and `dataguzzler.log`.**
   The log now carries the staging worker's own stack, which is the answer.
4. The plan lines from §3.2, verbatim.
5. The `ec_scan_*` directory, `dataguzzler.log`, `ros.log`, `dmesg.log` —
   separate files if you can.
6. Whether the AER counters in §1.1 were still climbing, and what you changed.
