#########################################################################
# AdaptiveNDE Project                                                   #
# ROS-Scanning Package                                                  #
#########################################################################
# Copyright 2023-2024 University of Dayton                              #
#                                                                       #
# Redistribution and use in source and binary forms, with or without    #
# modification, are permitted provided that the following conditions    #
# are met:                                                              #
#                                                                       #
# 1. Redistributions of source code must retain the above copyright     #
#    notice, this list of conditions and the following disclaimer.      #
# 2. Redistributions in binary form must reproduce the above copyright  #
#    notice, this list of conditions and the following disclaimer in    #
#    the documentation and/or other materials provided with the         #
#    distribution.                                                      #
# 3. Neither the name of the copyright holder nor the names of its      #
#    contributors may be used to endorse or promote products derived    #
#    from this software without specific prior written permission.      #
#                                                                       #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   #
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,#
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT      #
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY #
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE #
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  #
#########################################################################
# This material is based in part upon work supported by the U.S. Air    #
# Force under Contract FA8650-19-D-5230.                                #
#                                                                       #
# DISTRIBUTION A. Approved for public release: distribution unlimited.  #
#########################################################################
# File:  ec_liftoff.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-02
# Description:
#   Offline lift-off characterisation from capture_contact()'s npz
#   files. Finds the lift-off axis from the data rather than from the
#   dialled-in rotation, and reports how much lift-off a pass actually
#   suffers in units of a known tilt step.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-02:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""How much lift-off does a contact pass actually suffer?

THIS IS THE MEASUREMENT THAT DECIDES WHETHER FORCE CONTROL IS WORTH BUILDING,
and it needs no new hardware, no force sensor and no contact beyond what the
cell already does.

The argument. An eddy-current reading cares about LIFT-OFF, not about force --
force is only the means by which a spring holds lift-off steady. The cell
already rotates lift-off onto its own axis (`/ec/cond/horizontal`; setting that
rotation IS the setup procedure for an EC inspection), and the probe is already
spring-loaded, so the controlled variable is already measured and the mechanism
meant to hold it is already fitted. What has never been measured is how much
lift-off actually varies along a pass. If it is small, a force loop has nothing
to fix. If it is large, the number also says how much softer the spring needs
to be.

WHY THE ROTATION IS RE-DERIVED HERE RATHER THAN TRUSTED. The npz stores RAW
complex impedance, which is better for this purpose: the rotation dialled in at
setup was set by eye, against a display, by turning it until lift-off looked
flat. That is the right procedure for an operator and the wrong thing to build
a measurement on. capture_contact() flies the same taught line at several
deliberate tilts about the probe tip, and tilt is a lift-off perturbation for a
split-D probe -- so the direction in the impedance plane along which the
PER-PASS MEANS move, as tilt is stepped, is the lift-off direction, derived from
the data. The dialled-in angle is reported beside it as a cross-check, and a
large disagreement is itself a finding.

WHY THE TILT STEPS ARE THE RULER. Impedance is in volts and nobody has a volts
per millimetre figure for this probe, so a bare standard deviation is
uninterpretable. But the tilt steps are KNOWN, in degrees, and they move the
signal along the same axis. So the within-pass wander is reported as a fraction
of the tilt-induced step: "the lift-off wander along this pass is 40% of what
3 degrees of tilt does to it" is a statement someone can act on, and it needs
no probe calibration at all.

WHAT THE SHAPE OF THE WANDER MEANS, and this is the part that chooses the fix:

    a TREND along the pass      registration or calibration error -- the mesh
                                is tilted or offset relative to the part. Fix
                                the calibration; compliance only hides it.
    NOISE at short wavelength   mesh roughness, per-vertex. This is what
                                compliance is for, and it is also the component
                                an active force loop at 1-2 Hz CANNOT reach at
                                10 mm/s (it corrects wavelengths above about
                                5-10 mm; a spring reaches below 1 mm).
    a STEP                      something discrete -- a lost contact, a seam,
                                a plunger sticking.

So the autocorrelation length of the residual is reported too, in millimetres
of travel.

NO ROS, NO CELL, NO ARM, NO spatialnde2. Plain numpy over the npz files
capture_contact() already writes. The conditioning math is reproduced from
spatialnde2-ec-functions/spatialnde2_ec_functions/ec_condition_cpp.hpp and is
checked against it by ec_liftoff_test.py, so the two cannot drift apart
silently.

    python3 ec_liftoff.py /path/to/take_dir
    python3 ec_liftoff.py /path/to/ec_tilt0.npz /path/to/ec_tilt3.npz ...
"""

import glob
import json
import os
import sys

import numpy as np


# ==============================================================================
# The conditioning math, reproduced
# ==============================================================================

def condition(impedance, null=(0.0, 0.0), rotation_deg=0.0, gain=(1.0, 1.0)):
    """Null, rotate, then per-axis gain. Returns (horizontal, vertical).

    THE ORDER IS NOT INTERCHANGEABLE and the sign of the rotation is not
    obvious, so both are taken from the C++ rather than re-derived:

        theta = -rotation_deg * pi/180
        h = re*cos(theta) - im*sin(theta)
        v = re*sin(theta) + im*cos(theta)

    which is `(z - null) * exp(i*theta)`. Rotating the DATA by -theta is what
    puts a feature currently lying at +theta onto the horizontal axis, because
    that is how the angle is quoted on an instrument: "lift-off at 40 degrees"
    means rotate by -40 to flatten it.

    Nulling before rotating matters: rotate first and the null point itself
    becomes a moving offset, which is the classic way to get a display that
    drifts as the rotation is adjusted.
    """
    z = np.asarray(impedance, dtype=np.complex128)
    z = z - complex(float(null[0]), float(null[1]))
    z = z * np.exp(1j * (-float(rotation_deg) * np.pi / 180.0))
    return np.real(z) * float(gain[0]), np.imag(z) * float(gain[1])


# ==============================================================================
# Loading
# ==============================================================================

def saturation(impedance, fraction=0.02, repeats=3):
    """Is this pass railed against the digitiser's input range?

    -> a dict; `['saturated']` is the only field most callers need.

    WHY THIS EXISTS. The Uniwest's analogue outputs run into a +/-5 V DAQ
    range, and a pass flown at enough tilt pins one component against the rail
    and stays there. A railed pass does not report a small lift-off variation
    -- it reports NO variation, because every sample is the same number -- so
    its mean is not a measurement of anything and its standard deviation is a
    measurement of the rail. Measured on the 2026-10-02 take: the +6 deg pass
    had 64 of 64 samples at exactly -5.0000 on the imaginary component, giving
    it a standard deviation of exactly zero and the smallest "lift-off spread"
    of the five passes. Without this check that pass looks like the BEST one.

    The test is for a REPEATED extremal value rather than for a particular
    voltage, so it does not need to know the DAQ range: a hard rail produces
    many samples at bit-identical extremes, which a real signal does not.
    Both conditions are needed. Every pass reaches its own minimum and maximum
    exactly once, so a fraction threshold alone fires on every pass ever
    recorded -- at 64 traces a single extremal sample is already 1.6% -- which
    is why `repeats` exists and why it is the condition that does the work.
    `fraction` then keeps a handful of identical samples near a turning point,
    which ADC quantisation can produce honestly, from counting as a rail.
    """
    z = np.asarray(impedance)
    out = {"saturated": False, "components": {}, "n": int(z.size)}
    if z.size == 0:
        return out
    for label, part in (("real", np.real(z)), ("imag", np.imag(z))):
        info = {}
        for end, value in (("low", float(np.min(part))),
                           ("high", float(np.max(part)))):
            # Bit-identical, not approximately equal: a clipped sample is the
            # same float every time, a real extremum is reached once.
            count = int(np.count_nonzero(part == value))
            info[end] = {"value": value, "count": count,
                         "fraction": count / float(part.size)}
        end = ("low" if info["low"]["count"] >= info["high"]["count"]
               else "high")
        info["fraction"] = info[end]["fraction"]
        info["count"] = info[end]["count"]
        info["value"] = info[end]["value"]
        info["railed"] = (info["count"] >= repeats
                          and info["fraction"] >= fraction)
        out["components"][label] = info
        if info["railed"]:
            out["saturated"] = True
    return out


def load_pass(path):
    """One npz from capture_contact(), as a dict. Tilt comes from the filename.

    The filename is the only place the tilt is recorded -- `_ec_disarm()` names
    the file `ec_<label>.npz` and the label carries the tilt the pass was flown
    at. A file whose name does not carry one gets tilt None and is still usable
    for within-pass statistics; it just cannot contribute to finding the axis.
    """
    data = np.load(path, allow_pickle=False)
    out = {"path": path, "name": os.path.basename(path)}
    for key in ("wall_time", "impedance", "position", "s_m", "index"):
        out[key] = data[key] if key in data else None
    if "trigger_report" in data:
        try:
            out["trigger_report"] = json.loads(str(data["trigger_report"]))
        except (ValueError, TypeError):
            out["trigger_report"] = None
    out["tilt_deg"] = _tilt_from_name(out["name"])
    out["saturation"] = (saturation(out["impedance"])
                         if out["impedance"] is not None else None)
    return out


def _tilt_from_name(name):
    """The tilt in degrees from `ec_tilt-3.npz`, or None."""
    stem = name
    for suffix in (".npz",):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    marker = "tilt"
    at = stem.find(marker)
    if at < 0:
        return None
    rest = stem[at + len(marker):].replace("p", ".").replace("_", "")
    try:
        return float(rest)
    except ValueError:
        return None


def load_take(*paths):
    """Every pass named, expanding a directory into its ec_*.npz files."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "ec_*.npz"))))
        else:
            files.append(p)
    if not files:
        raise ValueError(
            "no ec_*.npz files found in %s. capture_contact() writes one per "
            "pass into the take directory; a take with none means the EC source "
            "was never armed (pass ec=True)." % (", ".join(paths),))
    return [load_pass(f) for f in files]


# ==============================================================================
# Finding the lift-off axis from the data
# ==============================================================================

def _usable(passes):
    """The passes that can contribute to the axis and the ruler, and why not.

    -> (usable, excluded) where `excluded` is a list of (name, reason).

    A saturated pass is excluded outright: see saturation().  Its mean sits
    wherever the rail is rather than where the signal would have been, so
    including it bends both the fitted axis and the fitted ruler toward the
    rail.
    """
    usable = []
    excluded = []
    for p in passes:
        if p.get("impedance") is None or len(p["impedance"]) == 0:
            excluded.append((p.get("name", "?"), "no impedance samples"))
        elif p.get("tilt_deg") is None:
            excluded.append((p.get("name", "?"),
                             "filename carries no tilt, so it cannot be placed "
                             "on the tilt axis"))
        elif (p.get("saturation") or {}).get("saturated"):
            sat = p["saturation"]
            which = ", ".join(
                "%s %d of %d samples at %+.4f"
                % (name, info["count"], sat["n"], info["value"])
                for name, info in sat["components"].items() if info["railed"])
            excluded.append((p["name"],
                             "SATURATED (%s) -- a railed mean is not a "
                             "measurement" % which))
        else:
            usable.append(p)
    return usable, excluded


def liftoff_axis_from_tilts(passes):
    """The lift-off angle in degrees, derived from how tilt moves the signal.

    -> (rotation_deg, detail) where `rotation_deg` can be handed straight to
    condition() as its `rotation_deg`, or None when there is not enough to go
    on (fewer than two distinct tilts).

    THE METHOD. Each pass is flown along the same taught line at a different
    deliberate tilt about the probe tip. For a split-D, tilt is a lift-off
    perturbation -- it is differential lift-off across the two coils. So the
    per-pass MEAN impedance moves along the lift-off direction as tilt is
    stepped, and a straight-line fit of those means gives that direction.

    Using the means, not the samples, is deliberate: within a pass the signal
    carries the part as well as lift-off, and the part is what we are trying not
    to measure. Averaging over a whole pass leaves what the tilt changed.

    The fit is a total-least-squares line through the means in the complex
    plane (the principal axis of their scatter), not a regression of one
    coordinate on the other, because neither axis is the independent variable
    -- both are measured and either can be the one that barely moves.
    """
    usable, excluded = _usable(passes)
    tilts = sorted({p["tilt_deg"] for p in usable})
    if len(tilts) < 2:
        return None, {"reason":
                      "need at least two distinct tilts to find the axis; got %d "
                      "(%s) after exclusions. Fly capture_contact() with its "
                      "default tilt sweep."
                      % (len(tilts), tilts or "none"),
                      "excluded": excluded}

    means = np.array([np.mean(p["impedance"]) for p in usable])
    pts = np.column_stack([np.real(means), np.imag(means)])
    centred = pts - pts.mean(axis=0)
    # Principal axis of the scatter of per-pass means.
    _u, sv, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))
    # The axis is a line, so +theta and theta+180 are the same axis. Report it
    # in (-90, 90] so two runs of this never disagree by a half turn.
    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg <= -90.0:
        angle_deg += 180.0

    # How well a LINE describes those means. A lift-off perturbation should move
    # the signal along one direction; if the means scatter in a blob, tilt moved
    # something other than lift-off and the axis is not trustworthy.
    spread = float(sv[0])
    off_axis = float(sv[1]) if len(sv) > 1 else 0.0
    linearity = (1.0 - off_axis / spread) if spread > 0.0 else 0.0

    return angle_deg, {
        "tilts_deg": tilts,
        "passes_used": len(usable),
        "along_axis_spread": spread,
        "off_axis_spread": off_axis,
        "linearity": linearity,
        "means": means,
        "excluded": excluded,
    }


def tilt_step_size(passes, rotation_deg):
    """Signal change on the lift-off axis per degree of tilt, as the RULER.

    -> (per_degree, detail) or (None, detail).

    Impedance is in volts and there is no published volts-per-millimetre figure
    for this probe, so a bare standard deviation of the lift-off axis cannot be
    acted on. The tilt steps ARE known, so they calibrate the axis in units
    somebody can reason about: a wander of 0.4 tilt-degrees-equivalent is a
    statement, 0.0013 V is not.

    Fitted as a straight line of per-pass mean lift-off against tilt. Returns
    the slope's magnitude; the sign is a convention of which way the axis points
    and carries no information.
    """
    usable, excluded = _usable(passes)
    if len({p["tilt_deg"] for p in usable}) < 2:
        return None, {"reason": "fewer than two distinct tilts after exclusions",
                      "excluded": excluded}
    tilt = np.array([p["tilt_deg"] for p in usable], dtype=float)
    h_mean = np.array([np.mean(condition(p["impedance"],
                                         rotation_deg=rotation_deg)[0])
                       for p in usable])
    slope, intercept = np.polyfit(tilt, h_mean, 1)
    residual = h_mean - (slope * tilt + intercept)
    return abs(float(slope)), {
        "slope_per_deg": float(slope),
        "intercept": float(intercept),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "tilt_deg": tilt,
        "mean_liftoff": h_mean,
        "excluded": excluded,
    }


# ==============================================================================
# What one pass suffers
# ==============================================================================

def analyse_pass(p, rotation_deg, per_degree=None):
    """Lift-off statistics for one pass, and the shape of its variation.

    The shape is what chooses the fix, so the trend and the residual are
    separated rather than lumped into one standard deviation:

      trend      a straight line of lift-off against distance along the pass.
                 A real slope here is registration or calibration error -- the
                 mesh sits tilted or offset relative to the part -- and no
                 amount of compliance fixes it, it only hides it.
      residual   what is left. THIS is what compliance is for, and it is also
                 the part an active force loop cannot reach: at 10 mm/s a 1-2 Hz
                 loop corrects wavelengths above roughly 5-10 mm, while the
                 spring already fitted reaches below 1 mm.

    `per_degree` from tilt_step_size() converts both into tilt-degrees
    equivalent, which is the only unit here anybody can act on.
    """
    h, v = condition(p["impedance"], rotation_deg=rotation_deg)
    s = p.get("s_m")
    n = len(h)
    out = {"name": p["name"], "tilt_deg": p.get("tilt_deg"), "traces": n}
    if n < 2:
        out["reason"] = ("%d trace(s) -- nothing to characterise. A pass with "
                         "one trace is the position trigger never firing, not a "
                         "lift-off problem." % n)
        return out

    sat = p.get("saturation") or {}
    if sat.get("saturated"):
        # Refuse rather than report. Every statistic below would be a statistic
        # about the rail, and the railed pass is the one that comes out looking
        # quietest -- see saturation() for the measured case.
        out["saturated"] = True
        worst = ", ".join(
            "%s component %d of %d samples (%.0f%%) pinned at %+.4f"
            % (name, info["count"], n, 100.0 * info["fraction"], info["value"])
            for name, info in sat.get("components", {}).items()
            if info.get("railed"))
        out["reason"] = (
            "%d traces, tilt %s -- SATURATED, nothing reported: %s. The pass "
            "ran the instrument into the digitiser's input range, so its "
            "spread measures the rail and not lift-off. Reduce the "
            "instrument's gain (or re-null it) until the whole tilt sweep "
            "stays inside range, and fly the take again."
            % (n,
               "%+g deg" % p["tilt_deg"] if p.get("tilt_deg") is not None
               else "unknown",
               worst))
        return out

    out["liftoff_mean"] = float(np.mean(h))
    out["liftoff_std"] = float(np.std(h))
    out["liftoff_p5_p95"] = float(np.percentile(h, 95) - np.percentile(h, 5))
    out["liftoff_pk_pk"] = float(np.max(h) - np.min(h))
    out["defect_std"] = float(np.std(v))
    # If lift-off wanders more than the part varies, the image is a picture of
    # the mesh error rather than of the specimen.
    out["liftoff_over_defect"] = (out["liftoff_std"] / out["defect_std"]
                                  if out["defect_std"] > 0 else float("inf"))

    # --- trend vs residual, against distance travelled -----------------------
    if s is not None and len(s) == n and np.ptp(s) > 0:
        x = np.asarray(s, dtype=float)
        slope, intercept = np.polyfit(x, h, 1)
        residual = h - (slope * x + intercept)
        out["travel_m"] = float(np.ptp(x))
        out["trend_per_m"] = float(slope)
        out["trend_total"] = float(slope * np.ptp(x))
        out["residual_std"] = float(np.std(residual))
        out["residual_corr_length_m"] = _corr_length(x, residual)
        # Which of the two dominates decides where to spend the effort.
        total = abs(out["trend_total"]) + 2.0 * out["residual_std"]
        out["trend_share"] = (abs(out["trend_total"]) / total
                              if total > 0 else 0.0)
    else:
        out["reason_shape"] = (
            "no usable s_m, so trend and residual cannot be separated and only "
            "the aggregate spread is reported")

    # --- in units of tilt degrees -------------------------------------------
    if per_degree:
        for key in ("liftoff_std", "liftoff_p5_p95", "liftoff_pk_pk",
                    "residual_std", "trend_total"):
            if key in out:
                out[key + "_tiltdeg"] = abs(out[key]) / per_degree
    return out


def _corr_length(s, residual):
    """Distance in metres over which the residual stays correlated.

    The first lag at which the autocorrelation falls below 1/e, converted to
    metres using the mean sample spacing. Short means per-vertex mesh noise,
    which is compliance's job; long means a smooth excursion, which is
    calibration's.

    Returns None when the pass is too short for the estimate to mean anything.
    """
    n = len(residual)
    if n < 8:
        return None
    r = residual - residual.mean()
    denom = float(np.dot(r, r))
    if denom <= 0.0:
        return None
    ac = np.correlate(r, r, mode="full")[n - 1:] / denom
    below = np.nonzero(ac < np.exp(-1.0))[0]
    if len(below) == 0:
        return None
    spacing = float(np.ptp(s)) / float(n - 1)
    return float(below[0] * spacing)


# ==============================================================================
# The report
# ==============================================================================

def characterise(*paths, dialled_rotation_deg=None):
    """Everything above, over a take. -> a dict; `report()` prints it."""
    passes = load_take(*paths)
    axis_deg, axis_detail = liftoff_axis_from_tilts(passes)
    rotation = axis_deg if axis_deg is not None else (dialled_rotation_deg or 0.0)
    per_degree, step_detail = (tilt_step_size(passes, rotation)
                               if axis_deg is not None else (None, {}))
    return {
        "passes": [analyse_pass(p, rotation, per_degree) for p in passes],
        "axis_deg": axis_deg,
        "axis_detail": axis_detail,
        "rotation_used_deg": rotation,
        "dialled_rotation_deg": dialled_rotation_deg,
        "per_degree": per_degree,
        "step_detail": step_detail,
    }


def report(result):
    """Lines a person can read. Returns the text and prints nothing."""
    out = []
    add = out.append
    add("EC LIFT-OFF CHARACTERISATION")
    add("")

    axis = result["axis_deg"]
    if axis is None:
        add("  lift-off axis: COULD NOT BE DERIVED -- %s"
            % result["axis_detail"].get("reason", "unknown"))
        add("    Using rotation %.2f deg instead, so every number below is in "
            "whatever frame that is." % result["rotation_used_deg"])
    else:
        d = result["axis_detail"]
        add("  lift-off axis: %.2f deg, from %d pass(es) at tilts %s"
            % (axis, d["passes_used"],
               ", ".join("%g" % t for t in d["tilts_deg"])))
        add("    linearity %.3f (1.0 = the per-pass means lie on a line, which "
            "is what a lift-off perturbation should do)" % d["linearity"])
        if d["linearity"] < 0.8:
            add("    WARNING: the means scatter rather than lining up, so tilt "
                "moved something other than lift-off and this axis is not "
                "trustworthy. Treat everything below as provisional.")
        if result["dialled_rotation_deg"] is not None:
            delta = abs(axis - result["dialled_rotation_deg"])
            add("    dialled in at setup: %.2f deg, a %.2f deg disagreement%s"
                % (result["dialled_rotation_deg"], delta,
                   " -- worth investigating, the scan display has been "
                   "separating lift-off from the part along the wrong axis"
                   if delta > 10.0 else ""))

    if result["per_degree"]:
        add("")
        add("  the ruler: %.6g signal units per degree of tilt (fit residual "
            "%.3g)" % (result["per_degree"],
                       result["step_detail"].get("residual_rms", 0.0)))
        add("    Every '(= N tilt-deg)' below is a spread divided by this. It "
            "needs no probe calibration: the tilt steps were commanded, so "
            "they are a known perturbation along the same axis.")
    else:
        add("")
        add("  the ruler: UNAVAILABLE -- spreads below are in raw signal units "
            "and cannot be compared to anything.")

    excluded = (result["axis_detail"].get("excluded")
                or result.get("step_detail", {}).get("excluded") or [])
    if excluded:
        add("")
        add("  EXCLUDED from the axis and the ruler (%d of %d pass(es)):"
            % (len(excluded), len(result["passes"])))
        for name, why in excluded:
            add("    %-20s %s" % (name, why))

    add("")
    add("  per pass:")
    for p in result["passes"]:
        add("")
        if "reason" in p:
            add("    %-20s %s" % (p["name"], p["reason"]))
            continue
        add("    %-20s %d traces%s"
            % (p["name"], p["traces"],
               ", tilt %+g deg" % p["tilt_deg"]
               if p["tilt_deg"] is not None else ""))
        add("      lift-off spread  std %.6g%s, 5-95%% %.6g%s, pk-pk %.6g"
            % (p["liftoff_std"], _tilt(p, "liftoff_std"),
               p["liftoff_p5_p95"], _tilt(p, "liftoff_p5_p95"),
               p["liftoff_pk_pk"]))
        add("      lift-off / defect axis  %.2f%s"
            % (p["liftoff_over_defect"],
               "  -- lift-off varies MORE than the part does, so the image is "
               "largely a picture of mesh error"
               if p["liftoff_over_defect"] > 1.0 else ""))
        if "trend_total" in p:
            add("      over %.1f mm of travel: trend %.6g%s end to end, "
                "residual std %.6g%s"
                % (1000.0 * p["travel_m"], p["trend_total"],
                   _tilt(p, "trend_total"), p["residual_std"],
                   _tilt(p, "residual_std")))
            add("      trend is %.0f%% of the variation -- %s"
                % (100.0 * p["trend_share"], _verdict(p["trend_share"])))
            cl = p.get("residual_corr_length_m")
            if cl:
                add("      residual correlation length %.1f mm%s"
                    % (1000.0 * cl,
                       " -- short-wavelength, so a 1-2 Hz force loop at 10 mm/s "
                       "cannot reach it (it corrects above ~5-10 mm); a spring "
                       "can" if cl < 0.005 else ""))
        elif "reason_shape" in p:
            add("      %s" % p["reason_shape"])
    add("")
    add("  WHAT TO DO WITH THIS. The decision is not 'is there lift-off "
        "variation' -- there always is. It is whether the variation is big "
        "enough to matter against the defect signal, and whether it is a trend "
        "(fix the calibration) or short-wavelength residual (soften the spring; "
        "an active loop cannot reach it at scan speed).")
    return "\n".join(out)


#: Where the trend/residual split stops being a coin toss.
#:
#: DELIBERATELY NOT A SINGLE THRESHOLD AT 50%. Two passes flown back to back
#: differ by a few percent of share by chance, and a hard boundary then prints
#: "fix the calibration" on one and "soften the spring" on the other from
#: statistically identical data -- which is worse than saying nothing, because
#: both readings look authoritative. Anything in between is reported as what it
#: is: comparable, and both worth attention.
TREND_DOMINATES = 0.65
RESIDUAL_DOMINATES = 0.35


def _verdict(share):
    """What the trend/residual split says to do, with an honest middle."""
    if share > TREND_DOMINATES:
        return ("REGISTRATION dominates. A tilted or offset mesh relative to "
                "the part. Fix the calibration; compliance would only hide it.")
    if share < RESIDUAL_DOMINATES:
        return ("the RESIDUAL dominates, which is what compliance is for -- and "
                "what an active force loop cannot reach at scan speed.")
    return ("trend and residual are COMPARABLE, so neither explanation stands "
            "alone. Expect the calibration fix and a softer spring to each buy "
            "roughly half, and do not read a few percent either side of this as "
            "a verdict.")


def _tilt(p, key):
    """' (= N tilt-deg)' when the ruler is available, else ''."""
    v = p.get(key + "_tiltdeg")
    return "" if v is None else " (= %.2f tilt-deg)" % v


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    print(report(characterise(*argv[1:])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
