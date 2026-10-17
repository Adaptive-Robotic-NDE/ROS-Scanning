# NORMS metadata plug-in for eddy-current scans

`ec_scan_plugin.py` turns one `ec_scan_*` directory written by `run_ec_scan()` into the
metadata NORMS records for the measurement, plus two figures that NORMS hashes and backs up
with the data.

It is the only path by which anything *inside* the data files reaches the `.norms` record.

---

## What it produces

Three sections of metadata, and two generated files.

| Section | What it carries |
|---|---|
| `Eddy Current Scan` | identity and timing, passes and traces, trace pitch, **measured** line spacing and extent, travel in contact, **measured** mean pass speed, the digitizer channels, and the acquisition health counters (duplicate traces, dropped crossings, over-budget triggers, staging timeouts, errors, fault) |
| `Eddy Current Conditioning` | the null, rotation and gain the scan ran with, and the things only measurement can give: the **measured lift-off angle**, the lift-off to flaw ratio, the flaw-channel noise, and the peak flaw response, its SNR and where it is |
| `Eddy Current Surface` | the planning mesh, the projection surface, the spot radius and the image size |

### The two figures

| File | Panes | What it is for |
|---|---|---|
| `<scan dir>_ec_raw.png` | **Horizontal**, **Vertical** | the instrument's own X and Y analog outputs, exactly as it delivered them. Every other picture of the scan is derived from this one, and it is the only one that cannot be wrong about what was measured. |
| `<scan dir>_ec_conditioned.png` | **Lift-off**, **Flaw** | the same traces rotated onto the lift-off axis the plug-in measures from the data. This is where a flaw is visible. |

The panes are in the same order in both figures: the flaw channel is the component
perpendicular to lift-off, so it sits where vertical sits.

They are declared as `Plugin File[1]` and `Plugin File[2]`, which is what makes NORMS hash
them into `.hfiles` and copy them to the backup; a generated file that is not declared is
silently left behind.

### The axes are the scan's own, not world x and y

`along the pass (mm)` and `across the passes (mm)`, and that is a correctness choice rather
than a presentation one.

A plan view is a projection, and a projection of a surface scan is wrong twice. On curved
geometry it foreshortens — equal steps along the surface become unequal steps in x-y, and a
steep flank collapses towards a line. And even on a *flat* part it **aliases**, because the
samples are not on the grid it bins them onto: a pass wanders along its own length and the
across spacing scatters, so two neighbouring traces snap into one cell and the next cell gets
none. Measured on the 2026-10-16 scan, which is flat to 0.11 mm:

| | world x-y grid at the pitch | the scan's own coordinates |
|---|---|---|
| cells holding 2+ traces | 838 (10.2% of all traces) | **0** |
| empty cells inside the coverage | 1012 (11.7%) | **0** |
| peak retained | 0.944 V | 0.944 V |

Those two numbers in the first column are the same thousand events counted from both ends.

The scan already has better coordinates and already saves them. **Along** is `s_m`, the arc
length the probe travelled along the surface since the pass began — exact by construction,
because the trigger fires every `pitch` of travel, whatever the surface does underneath.
**Across** is the cumulative distance from pass to pass, measured perpendicular to the pass
direction. Both follow the surface, so distances on the figure are arc lengths over it, which
is the unit you would size an indication in anyway.

The image is then **one row per pass**. That is not an approximation — it is what the scan
did — and it is what makes an across-collision impossible: there is exactly one row per pass,
so two passes can never land on each other and none can be skipped. Nothing is interpolated,
smoothed, or given a footprint, which matters for a record figure where an invented value is
indistinguishable from a measured one. A blank cell means no trace reached it, and after this
it only ever means that: a pass cropped short by the clearance gate leaves a real hole and
the figure shows it.

Rows are drawn with `pcolormesh` over their **real** edges, not `imshow` onto an even grid:
the across spacing scatters a few percent, and evening it out would misplace every indication
by up to half a line with nothing on the figure saying so.

A serpentine is folded out first. Alternate passes run the other way, so their `s_m`
increases in the opposite world direction; taken at face value the rows would mirror against
each other and one flaw would appear twice a few millimetres apart — which is the shape of a
real fault, and must not be manufactured by the plotting.

`Scan Coordinates`, `Pass Straightness`, `Image Rows`, `Image Columns`, `Image Gaps` and
`Image Coverage` in the record say which coordinates were used, how far the arm's own
positions departed from them (the pass's wander, not an error), and what is blank.

**The two kinds of blank mean opposite things.** `Image Gaps` is blanks *inside* a pass's own
span — it should read 0, and anything else says the binning is aliasing again.
`Image Coverage` is how much of the whole rectangle was measured at all, and a low number is
a real result rather than a defect in the figure:

| scan | gaps | coverage | |
|---|---|---|---|
| 2026-10-16 | 0% | 97% | a clean run |
| 2026-10-15 | 0% | 93% | partial — stalled at pass 31 of 74 |
| 2026-10-09 | 0% | **60%** | the clearance gate cut most of the passes short |

**Where this still has to grow.** The along axis is exact for any geometry. The across axis
is a cumulative perpendicular distance between pass centroids, which is right while the
passes are roughly parallel and degrades on a surface that twists between them. And on a
surface that genuinely *folds* — a fillet, a blade root — no single 2-D figure is honest at
all; the plug-in should then emit patches rather than a plausible flattened picture. It does
not detect that case yet.

### What is on them

**These are meant to go into a report as they are.** Each pane carries what the figure *is* —
a title naming the component, the two axes with their unit, `V` on the colorbar — and
nothing the plug-in merely *thinks* about it. No circled peak, no rotation angle written
across the top, no callouts.

That is a deliberate line. A reading drawn onto the picture is indistinguishable from the
measurement once the picture is in a report, and these go into permanent NDE records. The
readings are still made and still recorded, in the properties, where they are named and
attributed `A` for auto-generated: `Peak Position Across` / `Along`, `Peak Flaw Response`,
`Peak Flaw SNR`, `Flaw Channel Noise`, `Measured Lift-off Angle`. Highlighting, cropping and
callouts get added afterwards, deliberately, by whoever is making the argument.

`Peak Position Along` and `Peak Position Across` are in those same axes now. In the old world
grid they meant x and y, and which of those was "along" depended on how the part happened to
be lying.

**Read the flaw channel, not the raw panes.** Lift-off
is what the probe does riding over an imperfect surface and it is far larger than any flaw, so
in the raw figure a notch is a small feature on a large varying background. On the 2026-10-15
scan the flaw was 0.84 V against ±0.65 V of raw banding and a millimetre across in a 23 × 8 mm
field — present, and not findable by eye. In the flaw pane it is the only thing there, and
`Peak Position Across` / `Along` in the record say where to look.

There used to be **one** figure, with its panes labelled `Real` and `Imaginary` — and it
plotted the *conditioned* components under those names. So the pane an operator read as the
in-phase signal was the lift-off channel, and the raw data had no picture at all.

**The raw pair is horizontal and vertical, not real and imaginary.** What arrives on
`/PXI1Slot2/ai0` and `ai1` is the instrument's X and Y analog outputs, and the instrument has
already applied its **own** phase rotation to produce them — set on its front panel, not here.
They are not in-phase and quadrature against any reference this plug-in holds. They are packed
into a complex number because that is the convenient container for a point in the impedance
plane; `.real` is horizontal and `.imag` is vertical, and that is all the packing means. The
rest of the tree has always used horizontal/vertical — `set_gain(h, v)`, the null pair,
`/ec/cond/{horizontal,vertical}` — and these panes now agree with it.

It follows that `Measured Lift-off Angle` is a rotation **on top of** the instrument's own,
measured from its horizontal output, not an absolute phase. Comparing it between scans is only
meaningful while the instrument's front-panel rotation is unchanged.

The two DAQ channel paths are recorded as `Horizontal Channel` and `Vertical Channel`. They
were `Real Channel` and `Imaginary Channel` until 2026-10-15 and carried the same misreading;
the `.norms` written at the cell on that date still uses the old names, and nothing rewrites
it. The scan's own saved metadata still calls them `real_path` and `imag_path` — that is
deliberately left alone, so the plug-in keeps reading scans recorded before the rename, and
the translation happens on the way out.

Each pane is scaled to the 99.5th percentile of |value − median| rather than to the extremes,
so one strong indication cannot flatten the field around it into a single colour: the texture
of the noise stays readable and the indication saturates, which is the right way round for
deciding whether an indication is real.

Both figures are drawn with `aspect="equal"` — a scan is a picture of a part, and squashing it
lies about where things are — and the figure height follows the scanned area's own shape.

### What only the plug-in knows

Four values are measured by the plug-in and exist nowhere in the scan's own output: the
lift-off angle, the flaw-channel noise, the peak SNR and the line spacing. The 2026-10-09 scan
ran with conditioning rotation 0, so its recorded image was almost entirely lift-off; the
plug-in's rotation to the measured 86.5 deg is what shows the notch.

The rotation in the conditioned figure is **measured from the data** and is not necessarily
the one the instrument was set to during the scan. The record carries both — `Measured
Lift-off Angle` against `Conditioning Rotation` — and they should agree to a degree or two.
Measured: 86.5 deg on 2026-10-09, 84.6 deg on 2026-10-15 against a set 86.5.

---

## Requirements

* The **system** `python3`, with `numpy` and `matplotlib` importable by it.
* **No ROS.** Nothing here imports `rclpy`, `dataguzzler_python` or `spatialnde2`, and the
  shebang is `#!/usr/bin/python3` rather than `/usr/bin/env python3` precisely so that it does
  not matter what is on `PATH` when NORMS starts it — a sourced ROS overlay, a conda root or a
  colcon workspace all put a different `python3` first, and none of them carry this machine's
  numpy.

Check both in one line:

```bash
/usr/bin/python3 -c "import numpy, matplotlib; print('ok')"
```

---

## Installing it

The copy in this repository is the authoritative one. Put a copy where EASEL can reach it by
absolute path:

```bash
sudo install -D -m 0755 norms/ec_scan_plugin.py /opt/norms-plugins/ec-scan/ec_scan_plugin.py
```

Any directory works. Give it one of its own rather than `/usr/local/bin`: when a run fails,
NORMS writes `<plugin name>.log` **next to the executable**, and that is the first place to
look.

---

## Configuring EASEL

EASEL Setup form, tab **Plugin & Monitoring**:

| Field | Value |
|---|---|
| Plugin Executable | `/opt/norms-plugins/ec-scan/ec_scan_plugin.py` |
| Extension(s) | `npz` |
| Arguments | `--files %s` |
| Restriction | **Only Oldest** |

**Why Only Oldest.** A scan writes one `pass000.npz` per pass — 77 of them on 10-09, and a
long raster will write hundreds. "Only Oldest" passes just the first, which keeps the command
line short and makes the `.norms` name deterministic. The plug-in does not need the rest on
the command line: it takes the *folder* from whatever it is given and globs `pass*.npz` itself.

That tolerance is deliberate, so the choice can change without touching the plug-in. All of
these work and all produce the same record:

* one `pass*.npz` (Only Oldest) — recommended
* every `pass*.npz` (Restriction None)
* any other file in the scan folder — a trigger or sidecar file, if a future version of the
  scan writes one and it is easier to point EASEL at that
* the scan folder itself

So if it later suits us to have `run_ec_scan()` drop a small sidecar file at the end and have
EASEL watch *that* extension instead, only the two EASEL fields change.

**After changing the property list, restart EASEL.** It runs the plug-in once with no
arguments when the Setup form is accepted, and caches that answer per executable path for the
life of the process; the cached list is what the documentation forms offer on their
"Auto-Generated" tab.

This applies to the 2026-10-15 rename of `Real Channel` / `Imaginary Channel` to
`Horizontal Channel` / `Vertical Channel`: a running EASEL will keep offering the old pair
until it is restarted.

---

## Checking it by hand

```bash
/opt/norms-plugins/ec-scan/ec_scan_plugin.py | python3 -m json.tool | head -20
```

Every `Value` must be `""` and the units present: that is the no-argument contract.

```bash
/opt/norms-plugins/ec-scan/ec_scan_plugin.py --files /data/scandata/ec_scan_*/pass000.npz
```

One JSON object on stdout, the values filled in, and the PNG written into the scan folder
*before* the process exits — NORMS looks for declared files immediately after writing the
`.norms` and never comes back for one that was not there.

Things that make a run fail, in NORMS' own terms:

| Symptom in `Software/Metadata Plug-in Result` | Cause |
|---|---|
| `Plug-in Not Defined` | the executable bit is missing |
| `JSON Missing` | no shebang, or a CRLF in the shebang line |
| `JSON Invalid` | anything on stdout that is not the single JSON object |
| `Plug-in Error` | stdout empty and stderr written |

The plug-in is built so the last three cannot happen from bad data: an unreadable file, a
corrupt `.npz`, a scan with too few traces to rotate, or no data at all each produce the
**empty** record on stdout and the reason on stderr, which NORMS records as `JSON Valid` with
empty values and a completed measurement. Diagnostics never go to stdout.

---

## The longer term

NORMS records will eventually be written directly, without EASEL driving a plug-in. Nothing
here depends on EASEL beyond the argument handling and the stdout contract: `extract()`
returns a plain dictionary and `render_raw()` / `render_conditioned()` each write one file, so
the same module can be imported and called when that happens.
