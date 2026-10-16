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

# TWO FIGURES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS. The raw pair is what
# the instrument put out and is the record of the measurement; the conditioned
# pair is the same data rotated so that lift-off and flaw are separated, and is
# where a flaw is actually visible. There used to be one figure, labelled "Real"
# and "Imaginary", which showed the CONDITIONED components under the raw ones'
# names -- so the pane an operator read as the in-phase signal was the lift-off
# channel, and the raw data had no picture at all.
RAW_IMAGE_SUFFIX = "_ec_raw.png"
COND_IMAGE_SUFFIX = "_ec_conditioned.png"


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


def _figure_height(G):
    """Inches, from the scanned area's own shape.

    The panes are drawn with aspect="equal" because a scan is a picture of a
    part and squashing it lies about where things are. A fixed figure height
    then leaves most of the figure empty for a long thin strip -- 23 x 8 mm is
    the usual shape here -- so the height follows the data instead, bounded so
    that a nearly-square scan does not produce a poster.
    """
    nu, nv = G.shape
    aspect = float(nu)/float(max(1, nv))
    return max(3.0, min(9.0, 1.8 + 4.3/max(aspect, 0.4)))


def _pane(fig, ax, G, title, cell_mm, mark=None, note=None):
    """One pane: robust symmetric scale about the median, units on both axes.

    The scale comes from the 99.5th percentile of |G - median| rather than from
    the extremes, so a single flaw indication cannot flatten the field around
    it into one colour -- the texture of the noise stays readable and the flaw
    saturates, which is the right way round for deciding whether an indication
    is real. `mark` is the measured peak, which is small enough on a real part
    to be missed if it is not pointed at.
    """
    import numpy as np
    nu, nv = G.shape
    mid = float(np.nanmedian(G))
    lim = float(np.nanpercentile(np.abs(G - mid), 99.5))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    im = ax.imshow(G.T, origin="lower", aspect="equal", cmap="RdBu_r",
                   extent=[0, nu*cell_mm, 0, nv*cell_mm],
                   vmin=mid - lim, vmax=mid + lim)
    ax.set_title(title)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    if mark is not None:
        x, y = mark
        ax.plot([x], [y], marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor="k", markeredgewidth=1.2)
        if note:
            # INSIDE THE AXES WHEREVER THE PEAK IS. A flaw at the edge of the
            # scanned strip is the normal case, not the exception -- the one on
            # the 2026-10-15 scan sat 0.55 mm from the top -- and a label
            # pinned above-right of it lands outside the frame and is cropped.
            above = y < 0.8*nv*cell_mm
            right = x < 0.75*nu*cell_mm
            ax.annotate(note, xy=(x, y),
                        xytext=(9 if right else -9, 10 if above else -14),
                        textcoords="offset points", fontsize=8, color="k",
                        ha="left" if right else "right",
                        va="bottom" if above else "top")
    fig.colorbar(im, ax=ax, label="V", fraction=0.046, pad=0.04)


def render_raw(path, impedance, u_mm, v_mm, cell_mm):
    """The impedance as the instrument delivered it: in-phase and quadrature.

    No rotation, no null beyond whatever the instrument itself was carrying.
    This is the figure that belongs with the record, because every other
    picture of this scan is derived from it and it is the only one that cannot
    be wrong about what was measured.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GR = grid_of(u_mm, v_mm, np.asarray(impedance).real, cell_mm)
    GI = grid_of(u_mm, v_mm, np.asarray(impedance).imag, cell_mm)
    fig, axes = plt.subplots(1, 2, figsize=(11, _figure_height(GR)))
    _pane(fig, axes[0], GR, "Real (in-phase)", cell_mm)
    _pane(fig, axes[1], GI, "Imaginary (quadrature)", cell_mm)
    fig.suptitle("raw impedance, as acquired")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_conditioned(path, along, perp, angle_deg, u_mm, v_mm, cell_mm,
                       mark=None, note=None):
    """The same traces rotated onto the measured lift-off axis.

    Lift-off is the dominant direction of the trace cloud and is far larger
    than any flaw, so until it is separated out a flaw is a small feature on a
    large varying background -- which is exactly what the single old figure
    showed. `along` is lift-off, `perp` is the flaw channel, and every
    conditioning number in the record (noise, peak, SNR, peak position) is
    measured on `perp`, so this figure and those numbers describe one thing.

    The rotation here is MEASURED FROM THE DATA and is not necessarily the one
    the instrument was set to during the scan; the record carries both
    ("Measured Lift-off Angle" against "Conditioning Rotation") and they should
    agree to a degree or two.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    GA = grid_of(u_mm, v_mm, along, cell_mm)
    GP = grid_of(u_mm, v_mm, perp, cell_mm)
    fig, axes = plt.subplots(1, 2, figsize=(11, _figure_height(GP)))
    _pane(fig, axes[0], GA, "Lift-off channel", cell_mm)
    _pane(fig, axes[1], GP, "Flaw channel", cell_mm, mark=mark, note=note)
    fig.suptitle("conditioned: rotated %.1f deg onto the measured lift-off axis"
                 % (angle_deg,))
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


def extract(folder, passes, raw_path, cond_path):
    """(values, [paths of the figures written]).

    Both figures or neither: a declared file that is missing is reported as an
    error by EASEL and written into the .hfiles as one, so the list this
    returns is what main() declares and nothing else.
    """
    import numpy as np
    pos, imp, metas, per_pass = load_scan(passes)
    v = {}
    if not metas:
        return v, []

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

    # The measured part: the lift-off axis, the flaw channel, and the figures.
    made = []
    angle, along, perp, ratio = liftoff_axis(imp) if len(imp) else (None, None, None, None)
    if angle is not None:
        v["Measured Lift-off Angle"] = angle
        if ratio is not None:
            v["Lift-off To Flaw Ratio"] = ratio
        u_mm = 1e3*(pos[:, 0] - pos[:, 0].min())
        v_mm = 1e3*(pos[:, 1] - pos[:, 1].min())
        cell = 1e3*float(pitch) if pitch else 0.25
        mark = note = None
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
                # THE SAME PEAK THE RECORD REPORTS, drawn on the pane it was
                # measured on. On a real part it is a millimetre across in a
                # field of tens, and the number alone does not make it findable.
                mark = (ju*cell, jv*cell)
                note = ("%.2f V" % v["Peak Flaw Response"]
                        + ("  SNR %.0f" % v["Peak Flaw SNR"]
                           if "Peak Flaw SNR" in v else ""))
        except Exception:                                          # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
        try:
            render_raw(raw_path, imp, u_mm, v_mm, cell)
            render_conditioned(cond_path, along, perp, angle, u_mm, v_mm, cell,
                               mark=mark, note=note)
            made = [raw_path, cond_path]
        except Exception:                                          # noqa: BLE001
            # NEITHER IS DECLARED IF EITHER FAILED, and the half that did get
            # written is removed. A file left in the scan folder and not
            # declared is not hashed and not backed up -- it just stays in
            # EASEL's modified-files list and is offered to the NEXT
            # measurement on its wait-for-files form, which is how one scan's
            # debris ends up attached to another's record.
            traceback.print_exc(file=sys.stderr)
            made = []
            for orphan in (raw_path, cond_path):
                try:
                    if os.path.isfile(orphan):
                        os.remove(orphan)
                except OSError:
                    log("could not remove %s" % (orphan,))
    else:
        log("fewer than 8 traces, or no impedance: no rotation and no figures")

    # Line spacing, measured from the traces rather than taken from the plan.
    try:
        spacing = pass_spacing_m(per_pass)
        if spacing is not None:
            v["Pass Spacing"] = 1e3*spacing
    except Exception:                                              # noqa: BLE001
        traceback.print_exc(file=sys.stderr)

    return v, made


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
    raw_path = os.path.join(folder, stem + RAW_IMAGE_SUFFIX)
    cond_path = os.path.join(folder, stem + COND_IMAGE_SUFFIX)

    try:
        values, made = extract(folders[0], passes, raw_path, cond_path)
    except Exception:                                              # noqa: BLE001
        # Emit the empty record rather than a partial one: EASEL records the
        # run as "JSON Valid" with empty values and the measurement still
        # completes, with the reason in the log beside this script.
        traceback.print_exc(file=sys.stderr)
        values, made = {}, []

    for section, name, unit in PROPERTIES:
        if name in values:
            out[section]["Properties"][name] = {"Value": clean(values[name]), "Unit": unit}

    # ONE "Plugin File[n]" SECTION PER FIGURE, numbered from 1 with no gaps --
    # a key without [n] is index 1, so a gap would silently overwrite. Only
    # files that are on disk NOW are declared: EASEL looks for a declared file
    # immediately after it writes the .norms and records a missing one as an
    # error in the .hfiles.
    try:
        on_disk = [p for p in (made or []) if os.path.isfile(p)]
    except Exception:                                              # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        on_disk = []
    for n, image in enumerate(on_disk, start=1):
        out["Plugin File[%d]" % n] = {"Properties": {
            "File Name": {"Value": os.path.basename(image)},
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
