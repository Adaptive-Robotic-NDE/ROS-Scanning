#!/usr/bin/python3
"""NORMS/EASEL metadata plug-in for an eddy-current path scan.

Turns one ec_scan_* directory written by run_ec_scan() into
  * the documentation metadata EASEL merges into the measurement's .norms, and
  * a two-pane Real/Imaginary image of the conditioned response, declared as a
    Plugin File so EASEL hashes and backs it up with the data.

WHY /usr/bin/python3 AND NOT env. This runs from EASEL, which starts it with no
shell and whatever environment EASEL happens to have. Pinning the interpreter
means it does not matter whether ROS, a conda root or a colcon overlay is on
PATH -- all of which put a different python3 first and none of which carry this
machine's numpy. The only requirement is that the SYSTEM python3 can import
numpy and matplotlib; nothing here imports rclpy, dataguzzler or spatialnde2,
and nothing here needs ROS sourced.

See README.md next to this file for how to configure EASEL to call it.
"""
import argparse
import datetime
import glob
import json
import os
import sys
import traceback

PLUGIN_NAME = "TARI eddy-current scan"

# EVERY PROPERTY THIS PLUG-IN CAN EMIT, in one place, as (section, name, unit).
# The no-argument run and a real run both build their output from this list, so
# the "Auto-Generated" tab EASEL caches after Setup always matches what arrives
# later. Section names avoid the ones EASEL owns -- Software, Overview, Data
# File Format, Notes, Sample, Probe, Scan Classification, Remote Backup, JSON
# Parameters, Plugin File -- because a plug-in property that collides with a
# form field REPLACES what the user typed.
SCAN = "Eddy Current Scan"
COND = "Eddy Current Conditioning"
SURF = "Eddy Current Surface"

PROPERTIES = [
    (SCAN, "Scan Name", ""),
    (SCAN, "Started", ""),
    (SCAN, "Elapsed", "min"),
    (SCAN, "Passes", ""),
    (SCAN, "Traces", ""),
    (SCAN, "Trace Pitch", "mm"),
    (SCAN, "Pass Spacing", "mm"),
    (SCAN, "Scanned Extent", "mm"),
    (SCAN, "Travel In Contact", "mm"),
    (SCAN, "Mean Pass Speed", "mm/s"),
    (SCAN, "Trigger Mode", ""),
    (SCAN, "Real Channel", ""),
    (SCAN, "Imaginary Channel", ""),
    (SCAN, "Samples Per Trace", ""),
    (SCAN, "Sample Reduction", ""),
    (SCAN, "Duplicate Traces", ""),
    (SCAN, "Dropped Crossings", ""),
    (SCAN, "Over Budget Triggers", ""),
    (SCAN, "Staging Timeouts", ""),
    (SCAN, "Acquisition Errors", ""),
    (SCAN, "Max Correlation Wait", "s"),
    (SCAN, "Fault", ""),
    (SCAN, "Partial Scan", ""),
    (SCAN, "Toolpath Source", ""),

    (COND, "Component Recorded", ""),
    (COND, "Conditioning Rotation", "deg"),
    (COND, "Null Horizontal", "V"),
    (COND, "Null Vertical", "V"),
    (COND, "Gain Horizontal", ""),
    (COND, "Gain Vertical", ""),
    (COND, "Measured Lift-off Angle", "deg"),
    (COND, "Lift-off To Flaw Ratio", ""),
    (COND, "Flaw Channel Noise", "V"),
    (COND, "Peak Flaw Response", "V"),
    (COND, "Peak Flaw SNR", ""),
    (COND, "Peak Position Across", "mm"),
    (COND, "Peak Position Along", "mm"),

    (SURF, "Planning Mesh", ""),
    (SURF, "Projection Surface", ""),
    (SURF, "Projection Spot Radius", "mm"),
    (SURF, "Projection Pixels", ""),
]

IMAGE_SUFFIX = "_ec_real_imag.png"


def log(msg):
    """Diagnostics go to stderr. EASEL reads stderr only when stdout is empty,
    so this is free when the run succeeds and is the error text when it does
    not."""
    print("%s: %s" % (PLUGIN_NAME, msg), file=sys.stderr)


def empty_output():
    out = {}
    for section, name, unit in PROPERTIES:
        out.setdefault(section, {"Properties": {}})
        out[section]["Properties"][name] = {"Value": "", "Unit": unit}
    return out


def placeholder(value):
    """EASEL passes %n, %f, %h and %j through unchanged in the scan flow,
    because none of those files exist yet when the plug-in runs."""
    return value is None or value.startswith("%")


def clean(value):
    """Every Value must be a STRING, and must not contain '|' -- that is
    EASEL's internal field separator and corrupts the merge."""
    if value is None:
        return ""
    if isinstance(value, float):
        text = ("%.6g" % value)
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    else:
        text = str(value)
    return text.replace("|", "/").strip()


# ---------------------------------------------------------------------------
# finding the scan
# ---------------------------------------------------------------------------

def resolve_passes(paths):
    """Every pass*.npz of the scan, from whatever EASEL handed us.

    DELIBERATELY TOLERANT, because there are several reasonable ways to point
    EASEL at this data and the plug-in should not care which is chosen:
      * the whole list of pass*.npz           (Extension npz, Restriction None)
      * one of them                           (Restriction "Only Oldest" -- the
                                               recommended setting, see README)
      * any other file inside the scan folder (a trigger or sidecar file)
      * the scan folder itself
    In every case the scan is the folder, so the folder is what is resolved and
    then globbed.
    """
    folders = []
    for p in paths:
        if os.path.isdir(p):
            folders.append(os.path.abspath(p))
        elif os.path.isfile(p):
            folders.append(os.path.dirname(os.path.abspath(p)))
    seen, ordered = set(), []
    for f in folders:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    passes = []
    for f in ordered:
        passes.extend(sorted(glob.glob(os.path.join(f, "pass*.npz"))))
    return ordered, passes


def load_scan(passes):
    """(positions N x 3, impedance N complex, per-pass metadata list)."""
    import numpy as np
    pos, imp, metas = [], [], []
    for p in passes:
        try:
            with np.load(p, allow_pickle=True) as z:
                meta = json.loads(str(z["meta"]))
                metas.append(meta)
                if "position" in z and "impedance" in z:
                    pos.append(np.asarray(z["position"], dtype=float))
                    imp.append(np.asarray(z["impedance"], dtype=complex))
        except Exception as exc:                                   # noqa: BLE001
            log("could not read %s: %s" % (os.path.basename(p), exc))
    if pos:
        return np.vstack(pos), np.concatenate(imp), metas, pos
    return np.zeros((0, 3)), np.zeros(0, dtype=complex), metas, []


def pass_spacing_m(per_pass):
    """Distance between neighbouring passes, measured ACROSS the pass.

    Not the distance between pass centroids: passes do not all cover the same
    stretch of the part -- a clearance crop or a hole shortens some of them --
    so their centroids differ ALONG the pass by far more than the line spacing.
    On the 2026-10-09 scan that reads 3.85 mm against a true 0.25 mm pitch.

    So the common travel direction is fitted first and only the component
    perpendicular to it is measured, which is the thing the plan calls line
    spacing. Signs are normalised before averaging, or a comb's uniform
    direction and a serpentine's alternating one would not be told apart.
    """
    import numpy as np
    dirs = []
    for p in per_pass:
        if len(p) >= 2:
            d = p[-1] - p[0]
            n = np.linalg.norm(d)
            if n > 1e-9:
                dirs.append(d/n)
    centres = [p.mean(axis=0) for p in per_pass if len(p)]
    if not dirs or len(centres) < 2:
        return None
    D = np.vstack(dirs)
    D = D*np.sign(D.dot(D[0]))[:, None]
    t = D.mean(axis=0)
    n = np.linalg.norm(t)
    if n < 1e-9:
        return None
    t = t/n
    step = np.diff(np.vstack(centres), axis=0)
    across = step - np.outer(step.dot(t), t)
    value = float(np.median(np.linalg.norm(across, axis=1)))
    return value if np.isfinite(value) and value > 0 else None


# ---------------------------------------------------------------------------
# conditioning: the lift-off axis, measured from the data
# ---------------------------------------------------------------------------

def liftoff_axis(impedance):
    """(angle_deg, along, perp, ratio) -- the impedance plane rotated so that
    lift-off lies along the real axis.

    Lift-off is the dominant direction of the trace cloud: it is what the probe
    does as it rides over an imperfect surface, and it is far larger than any
    flaw. A trimmed principal direction finds it without being dragged by the
    flaw itself, and the component PERPENDICULAR to it is the flaw channel.

    Measured on the 2026-10-09 scan: 86.5 deg, against 83.8 on 10-08, and the
    rotation cut the pass-to-pass baseline drift from 0.765 V to 0.088 V.
    """
    import numpy as np
    X = np.c_[impedance.real, impedance.imag]
    if X.shape[0] < 8:
        return None, None, None, None
    keep = np.ones(X.shape[0], dtype=bool)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    mu = X.mean(axis=0)
    for _ in range(6):
        if keep.sum() < 8:
            break
        mu = X[keep].mean(axis=0)
        A = X[keep] - mu
        w, V = np.linalg.eigh(A.T.dot(A)/max(1, len(A)))
        e1 = V[:, int(np.argmax(w))]
        e2 = V[:, int(np.argmin(w))]
        off = np.abs((X - mu).dot(e2))
        med = np.median(off)
        mad = np.median(np.abs(off - med))
        keep = off < med + 4.0*1.4826*mad + 1e-12
    along = (X - mu).dot(e1)
    perp = (X - mu).dot(e2)
    angle = float(np.degrees(np.arctan2(e1[1], e1[0])) % 180.0)
    spread = float(perp.std())
    ratio = float(along.std()/spread) if spread > 0 else None
    return angle, along, perp, ratio


def grid_of(u_mm, v_mm, values, cell_mm):
    """Mean of `values` per cell, as a 2-D array with NaN where nothing landed."""
    import numpy as np
    nu = int(np.ceil(u_mm.max()/cell_mm)) + 1
    nv = int(np.ceil(v_mm.max()/cell_mm)) + 1
    iu = np.clip(np.round(u_mm/cell_mm).astype(int), 0, nu - 1)
    iv = np.clip(np.round(v_mm/cell_mm).astype(int), 0, nv - 1)
    acc = np.zeros((nu, nv))
    cnt = np.zeros((nu, nv))
    np.add.at(acc, (iu, iv), values)
    np.add.at(cnt, (iu, iv), 1.0)
    out = np.full((nu, nv), np.nan)
    hit = cnt > 0
    out[hit] = acc[hit]/cnt[hit]
    return out


def render_real_imag(path, along, perp, u_mm, v_mm, cell_mm):
    """The two-pane figure: Real is the lift-off axis, Imaginary is the flaw
    axis perpendicular to it. One colormap, each pane on its own symmetric
    scale, units on the axes, and nothing else on the plot."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GR = grid_of(u_mm, v_mm, along, cell_mm)
    GI = grid_of(u_mm, v_mm, perp, cell_mm)
    nu, nv = GR.shape

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    for ax, G, title in ((axes[0], GR, "Real"), (axes[1], GI, "Imaginary")):
        mid = np.nanmedian(G)
        lim = np.nanpercentile(np.abs(G - mid), 99.5)
        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0
        im = ax.imshow(G.T, origin="lower", aspect="equal", cmap="RdBu_r",
                       extent=[0, nu*cell_mm, 0, nv*cell_mm],
                       vmin=mid - lim, vmax=mid + lim)
        ax.set_title(title)
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        fig.colorbar(im, ax=ax, label="V", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def first(metas, *keys, **kw):
    """The first value present at metas[i][k0][k1]..., or the default."""
    default = kw.get("default")
    for m in metas:
        node = m
        ok = True
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and node is not None:
            return node
    return default


def total(metas, *keys):
    """Sum of a per-pass number across the scan, or None if none carried it."""
    got, acc = False, 0.0
    for m in metas:
        node = m
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            acc += float(node)
            got = True
    return acc if got else None


def last(metas, *keys):
    """The LAST pass's value: the cumulative counters in trigger_report are
    running totals, so the final pass carries the figure for the whole scan."""
    value = None
    for m in metas:
        node = m
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node is not None:
            value = node
    return value


def extract(folder, passes, image_path):
    import numpy as np
    pos, imp, metas, per_pass = load_scan(passes)
    v = {}
    if not metas:
        return v, None

    v["Scan Name"] = os.path.basename(folder.rstrip(os.sep))
    starts = [m.get("wall_time") for m in metas if isinstance(m.get("wall_time"), (int, float))]
    if starts:
        v["Started"] = datetime.datetime.fromtimestamp(min(starts)).isoformat(timespec="seconds")
        sweep_last = first(reversed(metas), "segment_report", "sweep_s", default=0.0) or 0.0
        v["Elapsed"] = (max(starts) - min(starts) + float(sweep_last))/60.0

    v["Passes"] = len(metas)
    v["Traces"] = int(len(imp))
    pitch = first(metas, "pitch_m")
    if pitch:
        v["Trace Pitch"] = 1e3*float(pitch)

    if len(pos):
        v["Scanned Extent"] = "%.1f x %.1f" % (1e3*np.ptp(pos[:, 0]), 1e3*np.ptp(pos[:, 1]))

    travelled = total(metas, "segment_report", "observed_m")
    swept = total(metas, "segment_report", "sweep_s")
    if travelled is not None:
        v["Travel In Contact"] = 1e3*travelled
        if swept:
            v["Mean Pass Speed"] = 1e3*travelled/swept

    v["Trigger Mode"] = first(metas, "trigger_report", "mode")
    v["Real Channel"] = first(metas, "trigger_report", "real_path")
    v["Imaginary Channel"] = first(metas, "trigger_report", "imag_path")
    v["Samples Per Trace"] = first(metas, "trigger_report", "block_samples")
    v["Sample Reduction"] = first(metas, "trigger_report", "reduce")
    v["Duplicate Traces"] = last(metas, "trigger_report", "duplicate_traces")
    v["Dropped Crossings"] = last(metas, "trigger_report", "dropped_crossings")
    v["Over Budget Triggers"] = last(metas, "trigger_report", "over_budget")
    v["Staging Timeouts"] = last(metas, "trigger_report", "staging_timeouts")
    v["Acquisition Errors"] = last(metas, "trigger_report", "errors")
    v["Max Correlation Wait"] = last(metas, "trigger_report", "max_correlation_wait_s")
    v["Fault"] = last(metas, "trigger_report", "fault") or "none"
    v["Partial Scan"] = bool(first(metas, "partial", default=False))
    v["Toolpath Source"] = first(metas, "path_source")

    v["Component Recorded"] = first(metas, "component")
    v["Conditioning Rotation"] = first(metas, "rotation_deg")
    null = first(metas, "null") or []
    if len(null) >= 2:
        v["Null Horizontal"], v["Null Vertical"] = float(null[0]), float(null[1])
    gain = first(metas, "gain") or []
    if len(gain) >= 2:
        v["Gain Horizontal"], v["Gain Vertical"] = float(gain[0]), float(gain[1])

    mesh = first(metas, "mesh")
    x3d = first(metas, "x3d")
    if mesh:
        v["Planning Mesh"] = os.path.basename(mesh)
    if x3d:
        v["Projection Surface"] = os.path.basename(x3d)
    radius = first(metas, "projection_radius_m")
    if radius:
        v["Projection Spot Radius"] = 1e3*float(radius)
    v["Projection Pixels"] = first(metas, "projection_pixels")

    # The measured part: the lift-off axis, the flaw channel, and the image.
    made_image = None
    angle, along, perp, ratio = liftoff_axis(imp) if len(imp) else (None, None, None, None)
    if angle is not None:
        v["Measured Lift-off Angle"] = angle
        if ratio is not None:
            v["Lift-off To Flaw Ratio"] = ratio
        u_mm = 1e3*(pos[:, 0] - pos[:, 0].min())
        v_mm = 1e3*(pos[:, 1] - pos[:, 1].min())
        cell = 1e3*float(pitch) if pitch else 0.25
        try:
            GI = grid_of(u_mm, v_mm, perp, cell)
            med = float(np.nanmedian(GI))
            mad = float(np.nanmedian(np.abs(GI - med)))*1.4826
            dev = np.abs(GI - med)
            if np.isfinite(dev).any():
                ju, jv = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(dev), dev, -1))),
                                          dev.shape)
                v["Flaw Channel Noise"] = mad
                v["Peak Flaw Response"] = float(GI[ju, jv] - med)
                if mad > 0:
                    v["Peak Flaw SNR"] = float(dev[ju, jv]/mad)
                v["Peak Position Across"] = ju*cell
                v["Peak Position Along"] = jv*cell
            render_real_imag(image_path, along, perp, u_mm, v_mm, cell)
            made_image = image_path
        except Exception:                                          # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
    else:
        log("fewer than 8 traces, or no impedance: no rotation and no image")

    # Line spacing, measured from the traces rather than taken from the plan.
    try:
        spacing = pass_spacing_m(per_pass)
        if spacing is not None:
            v["Pass Spacing"] = 1e3*spacing
    except Exception:                                              # noqa: BLE001
        traceback.print_exc(file=sys.stderr)

    return v, made_image


# ---------------------------------------------------------------------------

def main(argv):
    if not argv:
        # The no-argument run EASEL makes right after Setup is accepted. It
        # caches this per executable path for the life of the process, so the
        # property list here has to be the one real runs will fill in.
        sys.stdout.write(json.dumps(empty_output(), indent=2, ensure_ascii=False) + "\n")
        return 0

    parser = argparse.ArgumentParser(description=PLUGIN_NAME, add_help=False)
    parser.add_argument("--norms", default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--hfiles", default=None)
    parser.add_argument("--job", default=None)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("positional", nargs="*")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        log("could not parse arguments: %r" % (argv,))
        sys.stdout.write(json.dumps(empty_output(), indent=2, ensure_ascii=False) + "\n")
        return 0

    given = [p for p in (args.files + args.positional) if not placeholder(p)]
    folders, passes = resolve_passes(given)

    out = empty_output()
    if not passes:
        log("no pass*.npz found from %r -- emitting the empty record" % (given,))
        sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        return 0

    folder = folders[0]
    # The .norms lands in the data folder, so the image goes there too: a
    # generated file is only hashed and backed up if it is inside that folder
    # AND exists before this process exits.
    if not placeholder(args.folder):
        folder = args.folder
    elif not placeholder(args.norms):
        folder = os.path.dirname(os.path.abspath(args.norms))
    stem = os.path.basename(folders[0].rstrip(os.sep)) or "ec_scan"
    image_path = os.path.join(folder, stem + IMAGE_SUFFIX)

    try:
        values, made_image = extract(folders[0], passes, image_path)
    except Exception:                                              # noqa: BLE001
        # Emit the empty record rather than a partial one: EASEL records the
        # run as "JSON Valid" with empty values and the measurement still
        # completes, with the reason in the log beside this script.
        traceback.print_exc(file=sys.stderr)
        values, made_image = {}, None

    for section, name, unit in PROPERTIES:
        if name in values:
            out[section]["Properties"][name] = {"Value": clean(values[name]), "Unit": unit}

    if made_image and os.path.isfile(made_image):
        out["Plugin File[1]"] = {"Properties": {
            "File Name": {"Value": os.path.basename(made_image)},
            "Relative Path": {"Value": ""},
        }}

    sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    sys.exit(main(sys.argv[1:]))
