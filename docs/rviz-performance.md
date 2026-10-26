# RViz goes unresponsive after the Keyence measurement

RViz runs at a usable frame rate for a whole session until the fine-mesh
pipeline finishes, and from that moment on it renders about one frame every few
seconds and never recovers until it is restarted.

It is neither a leak nor a message rate. **It is seven displays that hold onto a
cloud forever, and they all receive that cloud at the Keyence measurement.** The
fix is in `tari_ur_description/rviz/view_robot.rviz`; §3 is how to confirm it on
the arm, because it was verified in the simulation image, which renders through
software GL over VNC while the work cell has a discrete GPU. Absolute frame rates do
not transfer between the two. The *shape* — a step change at a known moment that
never comes back — does.

---

## 1. What is happening

`reconstruction_srv.py` publishes each stage of the fine-mesh pipeline **once**,
at `/stop_and_process`:

| topic | what it carries |
|---|---|
| `/sanitized_pointcloud` | **the RAW cloud** — every point the profiler returned |
| `/voxel_pointcloud` | after the merge |
| `/cropped_pointcloud` | after the crop box |
| `/cluster_pointcloud` | after clustering |
| `/sor_filtered_pointcloud` | after outlier removal |
| `/uniform_downsampled_pointcloud` | after downsampling |
| `/final_mesh` | the mesh, as a Marker |

An RViz `PointCloud2` display with **`Decay Time: 0` keeps the last message
forever**. "Decay 0" means *never expire*, not *clear*. So each of those displays
is empty and free all session, and from the instant `/stop_and_process` returns
it holds a cloud and is re-rendered every frame until RViz is restarted.

**`/sanitized_pointcloud` is the expensive one, and its name is a lie.**
`reconstruction_srv.py:384` is `np.vstack(self.accumulated_points)` under the
comment *"Publish raw pointcloud"* — the same array written to `_raw.ply`. A
24 × 66 mm ROI puts about 340,000 points on it; a 250 mm plate at this
profiler's pitch puts about 2.85 million.

### Measured

In the simulation image, one 2.85 M-point message on that topic, nothing else
publishing, display enabled:

| | CPU | RSS |
|---|---|---|
| before | 241.6 % | 921 MB |
| after the one message | 381.7 % | **1785 MB** |
| 60 s later, still nothing publishing | 389.1 % | **1785 MB** |

**+864 MB and +140 points of CPU, from a single message, never released.** RViz
spends roughly **300 bytes per point** building billboard geometry for a 12-byte
point. That is both the "memory leak" appearance and the permanent frame-rate
collapse, and it starts at exactly the moment the stage clouds are published.

With the display disabled, the same message changes nothing: 245.4 → 244.3 →
246.9 % CPU, 900 → 873 → 887 MB. A disabled display does not even subscribe, so
the 34 MB never crosses the wire either.

### What it is NOT

* **Not the live `/profiles` firehose.** Measured with the sim profiler at 50 Hz:
  rviz2 burned **236.2 %** CPU with that display enabled and **237.8 %** with it
  disabled. No difference — do not spend time there.
* **Not a leak in the usual sense.** Nothing grows over time. It steps once and
  stays.

---

## 2. The fix

In `tari_ur_description/rviz/view_robot.rviz`, six stage displays are **off by
default**: `/sanitized_pointcloud`, `/voxel_pointcloud`, `/cluster_pointcloud`,
`/cropped_pointcloud`, `/sor_filtered_pointcloud`,
`/uniform_downsampled_pointcloud`.

Left **on**: `/accumulated_cloud` (the scan as it builds — the publisher caps it
at `preview_max_points` 200000) and `/final_mesh` (the result).

Nothing is lost. `/stop_and_process` writes **every one of those stages to disk
as a `.ply`** and the dgpy log prints the stage table with exact point counts.
The displays were a convenience; the files are the record. Tick one on when you
are debugging that stage, and restart RViz afterwards.

> **This reaches the work cell only when `ros_ws/src` is copied across**, and RViz
> must be restarted to pick up a config change. If RViz has ever been used to
> *save* the config (File > Save Config), the work cell's copy may have been
> overwritten with displays re-enabled — check with `rviz_probe.py displays`.

---

## 3. Confirming it on the production system

`dgpfiles/utils/rviz_probe.py` does all three steps. It reads `/proc`, so it
needs no extra packages; only `census` needs ROS sourced.

### Step 1 — before you start, confirm what is enabled

```bash
python3 ROS-Scanning/dgpfiles/utils/rviz_probe.py displays \
    ros_ws/src/tari_ros/tari_ur_description/rviz/view_robot.rviz
```

Expect the six stage topics to read `off`. If any reads `ON`, the fix has not
reached this machine, or the config was re-saved from RViz.

### Step 2 — watch RViz across a whole run

Start RViz and the work cell as usual, then, in its own terminal, **before** you
start the scan:

```bash
python3 ROS-Scanning/dgpfiles/utils/rviz_probe.py watch --csv /tmp/rviz_run.csv
```

It prints elapsed seconds, CPU % and RSS once a second, and flags any step
growth over 50 MB. Leave it running for the whole of `run_automated_scan()` and
note the wall-clock time of the Keyence stage so the two can be lined up.

**What the outcomes mean:**

| what `watch` shows | reading |
|---|---|
| RSS flat across the Keyence measurement, CPU unchanged | **Fixed.** The cause was the stage displays. |
| RSS steps up at `/stop_and_process` and stays | Something is still holding a cloud — go to step 3. |
| RSS climbs *gradually* through the scan | A genuine leak, which is a different fault from the one described here. Note when it starts. |
| CPU high from the start, no step | A different problem: baseline cost, not the Keyence event. See §4. |

### Step 3 — if it still steps, find which topic did it

Immediately after `/stop_and_process` returns:

```bash
python3 ROS-Scanning/dgpfiles/utils/rviz_probe.py census --secs 30
```

It lists every display topic with the point count and estimated RViz cost of the
last message it saw.

**Read `(N pub)` as "probably too big to sample", not as "empty".** A
multi-million-point cloud is a 30 MB sample fragmented across thousands of UDP
datagrams, and a subscriber that misses one fragment never reassembles it.
Against a 2,500,000-point cloud: a node with one subscription received 4 of 6
messages, the same node with a second subscription on the 30 Hz `/profiles`
received 2, and the full census with 16 subscriptions received 0. RViz itself
does receive them — it is a C++ subscriber with its own buffers that has been
running for minutes.

So **the authoritative sizes are on disk**, in the scan folder's `_raw.ply`,
`_merged.ply`, `_sor_filtered.ply` … and in the dgpy log's stage table:

```
  raw                  every point the profiler returned        340680 points
  merged               voxel-merged to the merge distance       4851 points
```

Multiply the biggest by ~300 bytes to estimate what a display holding it costs.

---

## 4. If the fix does not hold, these are next, in order

1. **A display that was re-enabled by hand or by a saved config.** Step 1 catches
   it. This is the likeliest cause of a recurrence.
2. **`/accumulated_cloud` exceeding its cap.** It is capped by the *publisher*
   (`preview_max_points`, default 200000, `preview_stride` 8). Check with
   `ros2 param get /pc_reconstruction_srv preview_max_points`. At 200 k the
   display costs ~60 MB; if the cap were raised it would scale linearly.
3. **The Azure clouds.** `/azure/points2` and `/azure/points2_corrected` are both
   enabled and both live at ~30 Hz through the whole session. They are a
   *baseline* cost, not a step at the Keyence stage, so they cannot explain the
   failure described here — but if step 2 shows high CPU from the start, disable
   one and re-measure.
4. **MoveIt's `MotionPlanning` display.** It re-renders the robot twice and is
   the heaviest single display in most MoveIt configs. Again baseline, not a
   step.
5. **GPU rather than RViz.** If `watch` shows CPU flat and modest while the frame
   rate is still terrible, the cost is in the driver, not in RViz's own loop.
   `nvidia-smi` during a scan will say. The simulation image cannot test this.

---

## 5. What cannot be tested off the work cell

The container renders with software GL (llvmpipe) over KasmVNC, so rviz2 sits at
~240 % CPU there before anything is displayed. That number is a property of the
container and says nothing about the work cell. **RSS growth and its permanence are
GPU-independent**, which is why the diagnosis above rests on those and why
`watch` measures them.
