# NORMS metadata plug-in for eddy-current scans

`ec_scan_plugin.py` turns one `ec_scan_*` directory written by `run_ec_scan()` into the
metadata NORMS records for the measurement, plus a two-pane Real/Imaginary image of the
conditioned response that NORMS hashes and backs up with the data.

It is the only path by which anything *inside* the data files reaches the `.norms` record.

---

## What it produces

Three sections of metadata, and one generated file.

| Section | What it carries |
|---|---|
| `Eddy Current Scan` | identity and timing, passes and traces, trace pitch, **measured** line spacing and extent, travel in contact, **measured** mean pass speed, the digitizer channels, and the acquisition health counters (duplicate traces, dropped crossings, over-budget triggers, staging timeouts, errors, fault) |
| `Eddy Current Conditioning` | the null, rotation and gain the scan ran with, and the things only measurement can give: the **measured lift-off angle**, the lift-off to flaw ratio, the flaw-channel noise, and the peak flaw response, its SNR and where it is |
| `Eddy Current Surface` | the planning mesh, the projection surface, the spot radius and the image size |

`<scan directory name>_ec_real_imag.png` — the response rotated so that lift-off lies along
the real axis. **Real** is then the lift-off channel and **Imaginary** is the flaw channel.
It is declared as `Plugin File[1]`, which is what makes NORMS hash it into `.hfiles` and
copy it to the backup; a generated file that is not declared is silently left behind.

Four of those values are measured by the plug-in and exist nowhere in the scan's own output:
the lift-off angle, the flaw-channel noise, the peak SNR and the line spacing. The 2026-10-09
scan ran with conditioning rotation 0, so its recorded image was almost entirely lift-off;
the plug-in's rotation to the measured 86.5 deg is what shows the notch.

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
returns a plain dictionary and `render_real_imag()` writes a file, so the same module can be
imported and called when that happens.
