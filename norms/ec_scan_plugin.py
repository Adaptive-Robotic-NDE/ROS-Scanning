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
#: How much longer the trace cloud's first principal axis has to be than its
#: second before the angle between them is called a lift-off direction.
#:
#: THE SAME NUMBER THE PIPELINE USES (MIN_LIFTOFF_ANISOTROPY in
#: ec_acquisition.dpi). It has to be: on 2026-10-16 the scan's own
#: ec_liftoff_rotation_from_traces() refused -- "anisotropy 2.86, under 3.00,
#: so there is no lift-off direction to find and any angle would be arbitrary"
#: -- and this plug-in wrote 86.7729 deg into the permanent record of the same
#: measurement, with no qualification. One of those two statements had to go,
#: and the one to keep is the refusal.
MIN_LIFTOFF_ANISOTROPY = 3.0

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
    (SCAN, "Horizontal Channel", ""),
    (SCAN, "Vertical Channel", ""),
    (SCAN, "Samples Per Trace", ""),
    (SCAN, "Sample Rate", "Hz"),
    (SCAN, "Block Span", "ms"),
    (SCAN, "Block Smear", "mm"),
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
    (SCAN, "Scan Coordinates", ""),
    (SCAN, "Pass Straightness", "mm"),
    (SCAN, "Image Rows", ""),
    (SCAN, "Image Columns", ""),
    (SCAN, "Image Gaps", "%"),
    (SCAN, "Image Coverage", "%"),

    (COND, "Component Recorded", ""),
    (COND, "Conditioning Rotation", "deg"),
    (COND, "Null Horizontal", "V"),
    (COND, "Null Vertical", "V"),
    (COND, "Gain Horizontal", ""),
    (COND, "Gain Vertical", ""),
    (COND, "Measured Lift-off Angle", "deg"),
    (COND, "Lift-off To Flaw Ratio", ""),
    (COND, "Lift-off Angle Determinate", ""),
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
#
# AND THE RAW PAIR IS HORIZONTAL AND VERTICAL, NOT REAL AND IMAGINARY. What
# arrives on /PXI1Slot2/ai0 and ai1 is the instrument's own X and Y analog
# outputs, and the instrument has already applied its OWN phase rotation to
# produce them -- set on its front panel, not here. So they are not in-phase
# and quadrature in any absolute sense, and calling them that claims a
# reference phase nobody holds. They are packed into a complex number because
# that is the convenient container for a point in the impedance plane; .real
# is horizontal and .imag is vertical, and that is all the packing means. The
# rest of this tree has always used horizontal/vertical (set_gain(h, v), the
# null pair, /ec/cond/{horizontal,vertical}); these panes now agree with it.
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
    """(positions N x 3, impedance N complex, metadata list, per-pass pos,
    per-pass arc length).

    The per-pass arc length is `s_m`, the distance the probe travelled ALONG
    the surface since the pass began, which the trigger measures and the trace
    store saves. It is the along axis of the scan's own coordinates -- see
    scan_coordinates(). A pass recorded before it was saved comes back None
    and the geometry falls back to a projection.
    """
    import numpy as np
    pos, imp, metas, arc = [], [], [], []
    for p in passes:
        try:
            with np.load(p, allow_pickle=True) as z:
                meta = json.loads(str(z["meta"]))
                metas.append(meta)
                if "position" in z and "impedance" in z:
                    pos.append(np.asarray(z["position"], dtype=float))
                    imp.append(np.asarray(z["impedance"], dtype=complex))
                    arc.append(np.asarray(z["s_m"], dtype=float)
                               if "s_m" in z else None)
        except Exception as exc:                                   # noqa: BLE001
            log("could not read %s: %s" % (os.path.basename(p), exc))
    if pos:
        return np.vstack(pos), np.concatenate(imp), metas, pos, arc
    return np.zeros((0, 3)), np.zeros(0, dtype=complex), metas, [], []


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
    lift-off lies along the horizontal axis.

    The angle is measured from the instrument's own horizontal output, which
    already carries whatever rotation its front panel was set to -- so it is a
    rotation ON TOP of the instrument's, not an absolute phase.

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


def scan_coordinates(per_pass_pos, per_pass_arc):
    """The scan's OWN two coordinates. -> (per-pass along mm, across mm, info)

    WHY NOT WORLD x AND y. A plan view is a projection, and a projection of a
    surface scan is wrong twice over. On curved geometry it foreshortens --
    equal steps along the surface become unequal steps in x-y, and a steep
    flank collapses towards a line -- and even on a flat part it ALIASES,
    because the samples are not on the grid it bins them onto. Measured on the
    2026-10-16 scan, which is flat to 0.11 mm: binning at the 0.25 mm pitch put
    838 cells on top of each other (10.2% of all traces) and left 1012 cells
    (11.7%) empty. Those are the same thousand events counted from both ends.

    THE SCAN ALREADY HAS BETTER COORDINATES AND SAVES THEM.

      along   `s_m`, the arc length the probe travelled ALONG the surface
              since this pass began. Exact by construction -- the trigger
              fires every `pitch` of travel -- so traces sit on a uniform
              lattice whatever the surface does underneath them. Each pass is
              shifted by where its own start sits along the pass direction, so
              the rows register to each other.
      across  the cumulative distance from pass to pass, measured
              perpendicular to the pass direction between consecutive pass
              centroids. It follows the surface too, and it is monotone.

    In those coordinates the samples ARE on the lattice: the same scan grids
    with 0.00% empty cells and 0 collisions. Nothing has to be interpolated,
    smoothed or given a footprint -- which matters for a record figure, where
    an invented value is indistinguishable from a measured one.

    Distances are therefore arc lengths over the surface, which is the right
    unit for sizing an indication anyway.

    `info["residual_mm"]` is how far the world positions depart from
    (pass start + s along the pass direction). It is the pass's own wander,
    not an error in the coordinates, and it is reported so that a pass that
    curved is visible rather than silently straightened.
    """
    import numpy as np
    dirs, centres = [], []
    for p in per_pass_pos:
        p = np.asarray(p, dtype=float)
        if len(p) >= 2:
            d = p[-1] - p[0]
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                dirs.append(d/n)
        if len(p):
            centres.append(p.mean(axis=0))
    if not dirs or len(centres) < 2:
        return None, None, {"why": "need two passes with length to find the axes"}

    # SERPENTINE FOLDED OUT FIRST. Alternate passes run the opposite way, so
    # averaging their raw directions would cancel to nothing.
    D = np.vstack(dirs)
    D = D*np.sign(D.dot(D[0]))[:, None]
    t = D.mean(axis=0)
    nt = float(np.linalg.norm(t))
    if nt < 1e-9:
        return None, None, {"why": "the passes do not share a direction"}
    t = t/nt

    C = np.vstack(centres)
    step = np.diff(C, axis=0)
    acr = step - np.outer(step.dot(t), t)
    nvec = acr.sum(axis=0)
    nn = float(np.linalg.norm(nvec))
    if nn < 1e-9:
        return None, None, {"why": "the passes do not step across"}
    nvec = nvec/nn

    ref = C[0]
    # IN SPATIAL ORDER, not file order. They are usually the same and the
    # cumulative across distance is wrong if they are not.
    order = np.argsort((C - ref).dot(nvec))
    across_of, prev, cum = {}, None, 0.0
    for k in order:
        k = int(k)
        if prev is not None:
            d = C[k] - C[prev]
            cum += float(np.linalg.norm(d - np.dot(d, t)*t))
        across_of[k] = cum
        prev = k

    along, across, resid = [], [], []
    for k, p in enumerate(per_pass_pos):
        p = np.asarray(p, dtype=float)
        if not len(p):
            along.append(np.zeros(0))
            continue
        arc = per_pass_arc[k] if k < len(per_pass_arc) else None
        if arc is None or len(arc) != len(p):
            # No saved arc length: project onto the pass direction instead.
            # Right on a flat pass, foreshortened on a curved one, and said so.
            a = (p - ref).dot(t)
        else:
            sgn = 1.0
            if len(p) >= 2 and np.dot(p[-1] - p[0], t) < 0:
                sgn = -1.0          # this pass runs the other way
            off = float(np.dot(p[0] - ref, t))
            a = off + sgn*np.asarray(arc, dtype=float)
            resid.append((p - ref).dot(t) - a)
        along.append(1e3*a)
        across.append(across_of.get(k, 0.0))

    lo = min(float(a.min()) for a in along if a.size)
    along = [a - lo for a in along]
    across = np.asarray([1e3*v for v in across], dtype=float)
    across = across - across.min()
    r = np.concatenate(resid) if resid else np.zeros(0)
    info = {
        "along_axis": [float(v) for v in t],
        "across_axis": [float(v) for v in nvec],
        "from_arc_length": bool(len(resid) == len(per_pass_pos)),
        "serpentine": bool(len({
            1 if np.dot(np.asarray(p, float)[-1] - np.asarray(p, float)[0], t) >= 0
            else -1
            for p in per_pass_pos if len(np.asarray(p, float)) >= 2}) > 1),
        "residual_mm": float(1e3*np.sqrt(np.mean(r**2))) if r.size else None,
        "worst_residual_mm": float(1e3*np.abs(r).max()) if r.size else None,
        "why": "",
    }
    return along, across, info


def scan_image(per_pass_along, per_pass_values, across_mm, pitch_mm):
    """One image ROW PER PASS. -> (image, along edges, across edges, info)

    A pass is a row. That is not an approximation -- it is what the scan did --
    and it is what makes the across axis incapable of aliasing: there is
    exactly one row per pass by construction, so two passes can never land on
    each other and no row can be skipped. Along the pass the traces are on an
    exact `pitch` lattice (see scan_coordinates), so the same holds there.

    Measured on the 2026-10-16 scan: 0.00% empty cells inside the measured
    span and 0 cells holding more than one trace, against 11.7% and 838 for
    the same traces binned into a world x-y grid.

    The rows are NOT evenly spaced -- the across spacing scatters a few percent
    -- so the edges are returned rather than assumed, and the caller draws with
    pcolormesh. An imshow would stretch them onto an even grid and quietly
    misplace every indication by up to half a line.

    NaN means no trace landed there, and after this it only ever means that:
    a pass that was cropped short leaves a real hole and the figure shows it.
    """
    import numpy as np
    n_rows = len(per_pass_along)
    spans = [a for a in per_pass_along if a is not None and len(a)]
    if not n_rows or not spans:
        return None, None, None, {"why": "no passes with traces"}
    hi = max(float(a.max()) for a in spans)
    n_along = int(np.ceil(hi/pitch_mm)) + 1
    acc = np.zeros((n_along, n_rows))
    cnt = np.zeros((n_along, n_rows))
    covered = np.zeros((n_along, n_rows), dtype=bool)
    for k, (a, vals) in enumerate(zip(per_pass_along, per_pass_values)):
        if a is None or not len(a):
            continue
        ia = np.clip(np.rint(np.asarray(a)/pitch_mm).astype(int), 0, n_along - 1)
        np.add.at(acc, (ia, np.full(len(ia), k)), np.asarray(vals, dtype=float))
        np.add.at(cnt, (ia, np.full(len(ia), k)), 1.0)
        covered[int(ia.min()):int(ia.max()) + 1, k] = True
    img = np.full((n_along, n_rows), np.nan)
    hit = cnt > 0
    img[hit] = acc[hit]/cnt[hit]

    along_edges = (np.arange(n_along + 1) - 0.5)*pitch_mm
    a = np.asarray(across_mm, dtype=float)
    if a.size == n_rows and n_rows > 1:
        mid = 0.5*(a[1:] + a[:-1])
        across_edges = np.concatenate((
            [a[0] - 0.5*(a[1] - a[0])], mid, [a[-1] + 0.5*(a[-1] - a[-2])]))
    else:
        across_edges = np.arange(n_rows + 1, dtype=float)
    holes = int((covered & ~hit).sum())
    info = {"rows": n_rows, "columns": n_along,
            "traces_binned": int(cnt.sum()),
            "cells_with_more_than_one": int((cnt > 1).sum()),
            "holes_inside_coverage": holes,
            "hole_fraction": (holes/float(covered.sum())
                              if covered.any() else None),
            # TWO DIFFERENT BLANKS, and they mean opposite things.
            # `hole_fraction` is blanks INSIDE a pass's own span -- after the
            # move to scan coordinates it should be zero, and anything else
            # says the binning is aliasing again. `coverage` is how much of
            # the whole rectangle was measured at all, and a low number is a
            # real result: on the 2026-10-09 scan the clearance gate cut most
            # of the passes short and the figure is half blank because the
            # part was half scanned.
            "coverage": float(hit.sum())/float(img.size) if img.size else None,
            "why": ""}
    return img, along_edges, across_edges, info


def _figure_height(along_edges, across_edges):
    """Inches, from the scanned area's own shape.

    The panes are drawn with aspect="equal" because a scan is a picture of a
    part and squashing it lies about where things are. A fixed figure height
    then leaves most of the figure empty for a long thin strip -- 23 x 8 mm is
    the usual shape here -- so the height follows the data instead, bounded so
    that a nearly-square scan does not produce a poster.
    """
    along_mm = float(along_edges[-1] - along_edges[0])
    across_mm = float(across_edges[-1] - across_edges[0])
    aspect = along_mm/max(across_mm, 1e-9)
    return max(3.0, min(9.0, 1.8 + 4.3/max(aspect, 0.4)))


def _pane(fig, ax, img, along_edges, across_edges, title):
    """One pane: the image, titled, with complete axes and a colorbar.

    DRAWN WITH pcolormesh OVER THE REAL EDGES, not imshow over an even grid.
    The rows are the scan's own passes and their spacing scatters a few
    percent; stretching them onto an even grid would misplace every indication
    by up to half a line and nothing on the figure would say so.

    What belongs on it is what the figure IS -- which component, which axis,
    what unit. What does not is what the plug-in THINKS of it: no marked peak,
    no rotation angle written across the top, no callouts. Those are readings,
    they are already in the .norms properties where they are named and
    attributed, and a reading drawn onto the picture is indistinguishable from
    the measurement once the picture is in a report. These figures are meant to
    go into one as they are; anything else is added deliberately, later, by
    whoever is making the argument.

    The scale is the 99.5th percentile of |value - median| rather than the
    extremes, so a single strong indication cannot flatten the field around it
    into one colour: the texture of the noise stays readable and the
    indication saturates, which is the right way round for judging whether an
    indication is real.
    """
    import numpy as np
    finite = img[np.isfinite(img)]
    mid = float(np.median(finite)) if finite.size else 0.0
    lim = (float(np.percentile(np.abs(finite - mid), 99.5))
           if finite.size else 1.0)
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    # Masked, so a cell no trace reached is left blank rather than coloured as
    # if it had been measured.
    masked = np.ma.masked_invalid(img.T)
    mesh = ax.pcolormesh(along_edges, across_edges, masked, cmap="RdBu_r",
                         vmin=mid - lim, vmax=mid + lim, shading="flat")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("along the pass (mm)")
    ax.set_ylabel("across the passes (mm)")
    fig.colorbar(mesh, ax=ax, label="V", fraction=0.046, pad=0.04)


def render_raw(path, per_along, per_values, across_mm, pitch_mm):
    """The impedance as the instrument delivered it: horizontal and vertical.

    The instrument's own X and Y analog outputs, read on two DAQ channels.
    Whatever rotation and null the instrument was set to on its front panel is
    already in them; nothing here has been applied on top. This is the figure
    that belongs with the record, because every other picture of this scan is
    derived from it and it is the only one that cannot be wrong about what was
    measured.

    `per_values` is complex only as a container: .real is the horizontal output
    and .imag the vertical one. Neither is in-phase or quadrature against any
    reference this plug-in knows.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, ae, ce, info = scan_image(
        per_along, [np.asarray(v).real for v in per_values], across_mm, pitch_mm)
    V, _ae, _ce, _i = scan_image(
        per_along, [np.asarray(v).imag for v in per_values], across_mm, pitch_mm)
    if H is None or V is None:
        raise ValueError("no image: %s" % (info.get("why"),))
    fig, axes = plt.subplots(1, 2, figsize=(11, _figure_height(ae, ce)))
    _pane(fig, axes[0], H, ae, ce, "Horizontal")
    _pane(fig, axes[1], V, ae, ce, "Vertical")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return info


def render_conditioned(path, per_along, per_lift, per_flaw, across_mm,
                       pitch_mm):
    """The same traces rotated onto the measured lift-off axis.

    Lift-off is the dominant direction of the trace cloud and is far larger
    than any flaw, so until it is separated out a flaw is a small feature on a
    large varying background. `per_lift` is lift-off, `per_flaw` is the flaw
    channel, and every conditioning number in the record (noise, peak, SNR,
    peak position) is measured on the flaw channel, so this figure and those
    numbers describe one thing.

    The rotation is MEASURED FROM THE DATA and is not necessarily the one the
    instrument was set to during the scan; the record carries both -- "Measured
    Lift-off Angle" against "Conditioning Rotation" -- and they should agree to
    a degree or two.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, ae, ce, info = scan_image(per_along, per_lift, across_mm, pitch_mm)
    F, _a, _c, _i = scan_image(per_along, per_flaw, across_mm, pitch_mm)
    if L is None or F is None:
        raise ValueError("no image: %s" % (info.get("why"),))
    fig, axes = plt.subplots(1, 2, figsize=(11, _figure_height(ae, ce)))
    # Same order as the raw figure's horizontal and vertical: the flaw channel
    # is the component perpendicular to lift-off, so it sits where vertical
    # sits. The rotation that produced them is "Measured Lift-off Angle" in the
    # record, and is deliberately not written across the figure.
    _pane(fig, axes[0], L, ae, ce, "Lift-off")
    _pane(fig, axes[1], F, ae, ce, "Flaw")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return info


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
    pos, imp, metas, per_pass, per_arc = load_scan(passes)
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

    # READ BEFORE THE FIRST USE. The smear below needs the block span and the
    # achieved speed, and the two are gathered in different places.
    fs = first(metas, "trigger_report", "sample_rate_hz")
    span = first(metas, "trigger_report", "block_span_s")

    travelled = total(metas, "segment_report", "observed_m")
    swept = total(metas, "segment_report", "sweep_s")
    if travelled is not None:
        v["Travel In Contact"] = 1e3*travelled
        if swept:
            v["Mean Pass Speed"] = 1e3*travelled/swept
            if span:
                # The smear is the measured speed times the block span, which
                # is why it is computed here and not at plan time: the speed
                # the arm ACHIEVES is the one that smears, and on 2026-10-16
                # that was 0.43x the speed the plan commanded.
                v["Block Smear"] = 1e3*float(span)*(travelled/swept)

    v["Trigger Mode"] = first(metas, "trigger_report", "mode")
    # The scan's own metadata still calls these real_path and imag_path, and
    # that is not changed here: old runs have to keep reading. What is recorded
    # is what they are -- the instrument's horizontal and vertical outputs.
    v["Horizontal Channel"] = first(metas, "trigger_report", "real_path")
    v["Vertical Channel"] = first(metas, "trigger_report", "imag_path")
    v["Samples Per Trace"] = first(metas, "trigger_report", "block_samples")
    v["Sample Reduction"] = first(metas, "trigger_report", "reduce")
    # HOW MUCH GROUND ONE TRACE AVERAGES OVER. block_samples alone cannot say:
    # at 0.25 mm pitch and 1.1 mm/s, 100 samples is 4% of the pitch at 10 kHz
    # and 43% of it at 1 kHz, and until 2026-10-17 the record held the 100 and
    # not the rate. Scans recorded before that carry no sample_rate_hz and
    # these three come back empty, which is the honest answer for them.
    if fs:
        v["Sample Rate"] = float(fs)
    if span:
        v["Block Span"] = 1e3*float(span)
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
        # DETERMINATE, OR NOT RECORDED AT ALL. A cloud that is not elongated
        # has no principal direction, and the angle between its axes is then
        # whatever the noise says. Writing it into a permanent record with no
        # qualification is worse than leaving it out -- it would be read later
        # as a measurement of the probe and the material, which is exactly
        # what the number is NOT when the ratio is this low.
        determinate = (ratio is not None and ratio >= MIN_LIFTOFF_ANISOTROPY)
        if ratio is not None:
            v["Lift-off To Flaw Ratio"] = ratio
        v["Lift-off Angle Determinate"] = "yes" if determinate else "no"
        if determinate:
            v["Measured Lift-off Angle"] = angle
        else:
            log("lift-off to flaw ratio %.2f is under %.2f, so the angle "
                "(%.2f deg) is arbitrary and is not recorded"
                % (ratio if ratio is not None else float("nan"),
                   MIN_LIFTOFF_ANISOTROPY, angle))
        cell = 1e3*float(pitch) if pitch else 0.25
        # THE SCAN'S OWN COORDINATES, not a plan view. See scan_coordinates().
        per_along, across_mm, geo = scan_coordinates(per_pass, per_arc)
        if per_along is None:
            log("could not build the scan coordinates (%s)" % (geo.get("why"),))
        else:
            v["Scan Coordinates"] = ("arc length along the pass"
                                     if geo.get("from_arc_length")
                                     else "projected along the pass direction")
            if geo.get("residual_mm") is not None:
                v["Pass Straightness"] = geo["residual_mm"]

        # SPLIT PER PASS, because the figure is one row per pass and the peak
        # has to be reported in the same coordinates the figure is drawn in.
        sizes = [len(np.asarray(p)) for p in per_pass]
        bounds = np.cumsum([0] + sizes)
        per_imp = [imp[bounds[k]:bounds[k+1]] for k in range(len(sizes))]
        per_lift = [along[bounds[k]:bounds[k+1]] for k in range(len(sizes))]
        per_flaw = [perp[bounds[k]:bounds[k+1]] for k in range(len(sizes))]

        try:
            # MEASURED ON THE TRACES, not on a grid of them. Binning used to
            # average a tenth of the traces into a neighbour before the noise
            # and the peak were taken off it; the traces are what was
            # measured. The difference on the 2026-10-16 scan is 0.7% of the
            # noise and 0.5% of the peak, so this is for honesty rather than
            # for accuracy.
            med = float(np.median(perp))
            mad = float(np.median(np.abs(perp - med)))*1.4826
            dev = np.abs(perp - med)
            if dev.size:
                k = int(np.argmax(dev))
                v["Flaw Channel Noise"] = mad
                v["Peak Flaw Response"] = float(perp[k] - med)
                if mad > 0:
                    v["Peak Flaw SNR"] = float(dev[k]/mad)
                # WHERE THE PEAK IS, in the record and not on the picture, and
                # in the figure's own axes -- "across" and "along" now mean
                # across the passes and along a pass, which is what they say.
                # In the world x-y grid they meant x and y, and which of those
                # was which depended on how the part happened to be lying.
                if per_along is not None:
                    which = int(np.searchsorted(bounds, k, side="right") - 1)
                    v["Peak Position Along"] = float(
                        per_along[which][k - bounds[which]])
                    v["Peak Position Across"] = float(across_mm[which])
        except Exception:                                          # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
        try:
            if per_along is None:
                raise ValueError("no scan coordinates")
            shape = render_raw(raw_path, per_along, per_imp, across_mm, cell)
            render_conditioned(cond_path, per_along, per_lift, per_flaw,
                               across_mm, cell)
            made = [raw_path, cond_path]
            v["Image Rows"] = shape.get("rows")
            v["Image Columns"] = shape.get("columns")
            if shape.get("hole_fraction") is not None:
                v["Image Gaps"] = 100.0*shape["hole_fraction"]
            if shape.get("coverage") is not None:
                v["Image Coverage"] = 100.0*shape["coverage"]
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
